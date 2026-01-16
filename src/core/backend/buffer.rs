use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) u64);

impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Physical GPU memory storage (the raw allocation)
#[derive(Debug)]
pub struct Storage<R: Runtime> {
    /// The underlying GPU handle from cubecl
    handle: Handle,
    size_bytes: usize,
    id: BufferId,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Storage<R> {
    /// Create storage from an existing handle
    pub fn new(handle: Handle, size_bytes: usize) -> Self {
        Storage {
            handle,
            size_bytes,
            id: BufferId::new(),
            _runtime: PhantomData,
        }
    }

    /// Allocate uninitialized storage
    pub fn empty(client: &ComputeClient<R>, size_bytes: usize) -> Self {
        let handle = client.empty(size_bytes);
        Self::new(handle, size_bytes)
    }

    /// Allocate storage from raw bytes
    pub fn from_bytes(client: &ComputeClient<R>, data: &[u8]) -> Self {
        let bytes = Bytes::from_bytes_vec(data.to_vec());
        let handle = client.create(bytes);
        Self::new(handle, data.len())
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn id(&self) -> BufferId {
        self.id
    }
}

/// A view into a Storage with shape, strides, and offset.
/// Multiple BufferViews can share the same underlying Storage.
#[derive(Debug, Clone)]
pub struct BufferView {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub len: usize,
}

impl BufferView {
    /// Create a contiguous view from shape
    pub fn contiguous(shape: Vec<usize>) -> Self {
        let strides = Self::compute_contiguous_strides(&shape);
        let len = shape.iter().product();
        BufferView {
            shape,
            strides,
            offset: 0,
            len,
        }
    }

    /// Create a view with explicit strides and offset
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        let len = shape.iter().product();
        BufferView {
            shape,
            strides,
            offset,
            len,
        }
    }

    /// Calculate row-major (C-contiguous) strides from shape
    pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Check if the view is contiguous in memory (row-major)
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let expected = Self::compute_contiguous_strides(&self.shape);
        self.strides == expected
    }

    /// Check if the view is Fortran-contiguous (column-major)
    pub fn is_fortran_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected_stride = 1;
        for (i, &dim) in self.shape.iter().enumerate() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Create a reshaped view
    /// Only valid for contiguous views
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<Self> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return None;
        }
        if !self.is_contiguous() {
            return None;
        }
        Some(BufferView {
            shape: new_shape.clone(),
            strides: Self::compute_contiguous_strides(&new_shape),
            offset: self.offset,
            len: self.len,
        })
    }

    /// Create a transposed view (swap last two dimensions)
    pub fn transpose(&self) -> Option<Self> {
        if self.ndim() < 2 {
            return None;
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        let n = self.ndim();
        new_shape.swap(n - 2, n - 1);
        new_strides.swap(n - 2, n - 1);
        Some(BufferView {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            len: self.len,
        })
    }

    /// Create a permuted view with arbitrary axis order
    pub fn permute(&self, axes: &[usize]) -> Option<Self> {
        if axes.len() != self.ndim() {
            return None;
        }
        // Verify axes is a valid permutation
        let mut seen = vec![false; self.ndim()];
        for &axis in axes {
            if axis >= self.ndim() || seen[axis] {
                return None;
            }
            seen[axis] = true;
        }

        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&i| self.strides[i]).collect();

        Some(BufferView {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            len: self.len,
        })
    }

    /// Create a sliced view along one dimension
    /// start..end with optional step
    pub fn slice(&self, dim: usize, start: usize, end: usize, step: usize) -> Option<Self> {
        if dim >= self.ndim() || start >= end || end > self.shape[dim] || step == 0 {
            return None;
        }

        let new_dim_size = (end - start + step - 1) / step;
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape[dim] = new_dim_size;
        new_strides[dim] *= step;

        let new_offset = self.offset + start * self.strides[dim];
        let new_len: usize = new_shape.iter().product();

        Some(BufferView {
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            len: new_len,
        })
    }

    /// Squeeze (remove) dimensions of size 1
    pub fn squeeze(&self, dim: Option<usize>) -> Self {
        match dim {
            Some(d) => {
                if d < self.ndim() && self.shape[d] == 1 {
                    let mut new_shape = self.shape.clone();
                    let mut new_strides = self.strides.clone();
                    new_shape.remove(d);
                    new_strides.remove(d);
                    BufferView {
                        shape: new_shape,
                        strides: new_strides,
                        offset: self.offset,
                        len: self.len,
                    }
                } else {
                    self.clone()
                }
            }
            None => {
                let (new_shape, new_strides): (Vec<_>, Vec<_>) = self
                    .shape
                    .iter()
                    .zip(self.strides.iter())
                    .filter(|(s, _)| **s != 1)
                    .map(|(&s, &st)| (s, st))
                    .unzip();
                BufferView {
                    shape: new_shape,
                    strides: new_strides,
                    offset: self.offset,
                    len: self.len,
                }
            }
        }
    }

    /// Unsqueeze a dimension of size 1 at position
    pub fn unsqueeze(&self, dim: usize) -> Option<Self> {
        if dim > self.ndim() {
            return None;
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        let stride = if dim == self.ndim() {
            1
        } else {
            self.strides[dim] * self.shape[dim]
        };

        new_shape.insert(dim, 1);
        new_strides.insert(dim, stride);

        Some(BufferView {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            len: self.len,
        })
    }
}

/// A reference-counted backend context that wraps the ComputeClient.
pub struct Backend<R: Runtime> {
    client: Arc<ComputeClient<R>>,
}

impl<R: Runtime> Clone for Backend<R> {
    fn clone(&self) -> Self {
        Backend {
            client: Arc::clone(&self.client),
        }
    }
}

impl<R: Runtime> Backend<R> {
    pub fn new(client: ComputeClient<R>) -> Self {
        Backend {
            client: Arc::new(client),
        }
    }

    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    // ========================
    // Buffer Creation Methods
    // ========================

    /// Create a buffer from host data with this backend attached
    pub fn from_data<E: CubeElement + CubePrimitive>(
        &self,
        data: &[E],
        shape: Vec<usize>,
    ) -> Buffer<R, E> {
        Buffer::from_data_with_backend(self, data, shape)
    }

    /// Create an uninitialized buffer with this backend attached
    pub fn empty<E: CubeElement + CubePrimitive>(&self, shape: Vec<usize>) -> Buffer<R, E> {
        Buffer::empty_with_backend(self, shape)
    }

    /// Create a zero-filled buffer with this backend attached
    pub fn zeros<E: CubeElement + CubePrimitive + cubecl::num_traits::Zero + Clone>(
        &self,
        shape: Vec<usize>,
    ) -> Buffer<R, E> {
        Buffer::zeros_with_backend(self, shape)
    }

    /// Create a buffer filled with ones with this backend attached
    pub fn ones<E: CubeElement + CubePrimitive + cubecl::num_traits::One + Clone>(
        &self,
        shape: Vec<usize>,
    ) -> Buffer<R, E> {
        let len: usize = shape.iter().product();
        let data = vec![E::one(); len];
        self.from_data(&data, shape)
    }

    /// Create a buffer filled with a specific value with this backend attached
    pub fn full<E: CubeElement + CubePrimitive + Clone>(
        &self,
        shape: Vec<usize>,
        value: E,
    ) -> Buffer<R, E> {
        let len: usize = shape.iter().product();
        let data = vec![value; len];
        self.from_data(&data, shape)
    }
}

impl<R: Runtime> std::fmt::Debug for Backend<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend").finish_non_exhaustive()
    }
}

/// A GPU buffer with reference counting for zero-copy operations.
#[derive(Debug)]
pub struct Buffer<R: Runtime, E: CubeElement + CubePrimitive> {
    storage: Arc<Storage<R>>,
    view: BufferView,
    backend: Option<Backend<R>>,
    _element: PhantomData<E>,
}

impl<R: Runtime, E: CubeElement + CubePrimitive> Clone for Buffer<R, E> {
    fn clone(&self) -> Self {
        Buffer {
            storage: Arc::clone(&self.storage),
            view: self.view.clone(),
            backend: self.backend.clone(),
            _element: PhantomData,
        }
    }
}

impl<R: Runtime, E: CubeElement + CubePrimitive> Buffer<R, E> {
    /// Create a new buffer from storage and shape (contiguous)
    pub fn new(storage: Arc<Storage<R>>, shape: Vec<usize>) -> Self {
        let view = BufferView::contiguous(shape);
        Buffer {
            storage,
            view,
            backend: None,
            _element: PhantomData,
        }
    }

    /// Create a new buffer from storage, view, and optional backend
    pub fn from_parts(
        storage: Arc<Storage<R>>,
        view: BufferView,
        backend: Option<Backend<R>>,
    ) -> Self {
        Buffer {
            storage,
            view,
            backend,
            _element: PhantomData,
        }
    }

    /// Create a buffer from host data
    pub fn from_data(client: &ComputeClient<R>, data: &[E], shape: Vec<usize>) -> Self {
        let bytes = E::as_bytes(data);
        let storage = Arc::new(Storage::from_bytes(client, bytes));
        Self::new(storage, shape)
    }

    /// Create a buffer from host data with backend context
    pub fn from_data_with_backend(backend: &Backend<R>, data: &[E], shape: Vec<usize>) -> Self {
        let mut buffer = Self::from_data(backend.client(), data, shape);
        buffer.backend = Some(backend.clone());
        buffer
    }

    /// Create an uninitialized buffer of given shape
    pub fn empty(client: &ComputeClient<R>, shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        let size_bytes = len * std::mem::size_of::<E>();
        let storage = Arc::new(Storage::empty(client, size_bytes));
        Self::new(storage, shape)
    }

    /// Create an uninitialized buffer with backend context
    pub fn empty_with_backend(backend: &Backend<R>, shape: Vec<usize>) -> Self {
        let mut buffer = Self::empty(backend.client(), shape);
        buffer.backend = Some(backend.clone());
        buffer
    }

    /// Create a buffer filled with zeros
    pub fn zeros(client: &ComputeClient<R>, shape: Vec<usize>) -> Self
    where
        E: cubecl::num_traits::Zero + Clone,
    {
        let len: usize = shape.iter().product();
        let data = vec![E::zero(); len];
        Self::from_data(client, &data, shape)
    }

    /// Create a buffer filled with zeros with backend context
    pub fn zeros_with_backend(backend: &Backend<R>, shape: Vec<usize>) -> Self
    where
        E: cubecl::num_traits::Zero + Clone,
    {
        let mut buffer = Self::zeros(backend.client(), shape);
        buffer.backend = Some(backend.clone());
        buffer
    }

    pub fn shape(&self) -> &[usize] {
        &self.view.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.view.strides
    }

    pub fn offset(&self) -> usize {
        self.view.offset
    }

    pub fn len(&self) -> usize {
        self.view.len
    }

    pub fn is_empty(&self) -> bool {
        self.view.len == 0
    }

    pub fn ndim(&self) -> usize {
        self.view.ndim()
    }

    pub fn storage_id(&self) -> BufferId {
        self.storage.id()
    }

    pub fn handle(&self) -> &Handle {
        self.storage.handle()
    }

    pub fn storage(&self) -> &Arc<Storage<R>> {
        &self.storage
    }

    pub fn view(&self) -> &BufferView {
        &self.view
    }

    pub fn backend(&self) -> Option<&Backend<R>> {
        self.backend.as_ref()
    }

    /// Attach a backend context to this buffer
    pub fn with_backend(mut self, backend: Backend<R>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set the backend context
    pub fn set_backend(&mut self, backend: Backend<R>) {
        self.backend = Some(backend);
    }

    pub fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
    }

    pub fn is_fortran_contiguous(&self) -> bool {
        self.view.is_fortran_contiguous()
    }

    /// Check if this buffer shares storage with another
    pub fn shares_storage_with<E2: CubeElement + CubePrimitive>(
        &self,
        other: &Buffer<R, E2>,
    ) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Check if this is the only reference to the storage
    pub fn is_storage_unique(&self) -> bool {
        Arc::strong_count(&self.storage) == 1
    }

    /// Check if this view can be safely modified in-place
    /// (unique storage reference AND contiguous layout)
    pub fn can_modify_inplace(&self) -> bool {
        self.is_storage_unique() && self.is_contiguous()
    }

    /// Create a view with different shape (must be contiguous and same total elements)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<Self> {
        let new_view = self.view.reshape(new_shape)?;
        Some(Self::from_parts(
            Arc::clone(&self.storage),
            new_view,
            self.backend.clone(),
        ))
    }

    /// Create a transposed view (swap last two dimensions)
    pub fn transpose(&self) -> Option<Self> {
        let new_view = self.view.transpose()?;
        Some(Self::from_parts(
            Arc::clone(&self.storage),
            new_view,
            self.backend.clone(),
        ))
    }

    /// Create a view with permuted axes
    pub fn permute(&self, axes: &[usize]) -> Option<Self> {
        let new_view = self.view.permute(axes)?;
        Some(Self::from_parts(
            Arc::clone(&self.storage),
            new_view,
            self.backend.clone(),
        ))
    }

    /// Create a sliced view with zero-copy
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Option<Self> {
        self.slice_with_step(dim, start, end, 1)
    }

    /// Create a sliced view with step with zero-copy
    pub fn slice_with_step(
        &self,
        dim: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Option<Self> {
        let new_view = self.view.slice(dim, start, end, step)?;
        Some(Self::from_parts(
            Arc::clone(&self.storage),
            new_view,
            self.backend.clone(),
        ))
    }

    /// Squeeze dimensions of size 1
    pub fn squeeze(&self, dim: Option<usize>) -> Self {
        let new_view = self.view.squeeze(dim);
        Self::from_parts(Arc::clone(&self.storage), new_view, self.backend.clone())
    }

    /// Unsqueeze (add dimension of size 1)
    pub fn unsqueeze(&self, dim: usize) -> Option<Self> {
        let new_view = self.view.unsqueeze(dim)?;
        Some(Self::from_parts(
            Arc::clone(&self.storage),
            new_view,
            self.backend.clone(),
        ))
    }

    /// Read data back to host (requires client or backend)
    pub fn to_data(&self, client: &ComputeClient<R>) -> Vec<E> {
        let bytes = client.read_one(self.storage.handle().clone());
        let all_elements = E::from_bytes(&bytes);

        if self.is_contiguous() {
            all_elements[self.view.offset..self.view.offset + self.len()].to_vec()
        } else {
            self.gather_elements(&all_elements)
        }
    }

    /// Read data using attached backend
    pub fn to_data_auto(&self) -> Option<Vec<E>> {
        self.backend.as_ref().map(|b| self.to_data(b.client()))
    }

    /// Gather elements from a flat array according to view's strides
    fn gather_elements(&self, data: &[E]) -> Vec<E> {
        let mut result = Vec::with_capacity(self.len());
        self.gather_recursive(data, &mut result, 0, self.view.offset);
        result
    }

    fn gather_recursive(&self, data: &[E], result: &mut Vec<E>, dim: usize, offset: usize) {
        if dim == self.ndim() {
            if offset < data.len() {
                result.push(data[offset].clone());
            }
            return;
        }

        for i in 0..self.view.shape[dim] {
            let new_offset = offset + i * self.view.strides[dim];
            self.gather_recursive(data, result, dim + 1, new_offset);
        }
    }

    /// Create a contiguous copy of this buffer on the GPU
    pub fn contiguous(&self, client: &ComputeClient<R>) -> Self {
        if self.is_contiguous() && self.view.offset == 0 {
            return self.clone();
        }

        if !self.is_contiguous() && self.view.offset != 0 {
            let data = self.to_data(client);
            let mut buffer = Self::from_data(client, &data, self.view.shape.clone());
            buffer.backend = self.backend.clone();
            return buffer;
        }

        let output = Self::empty(client, self.view.shape.clone());
        let num_elements = self.len();

        if num_elements > 0 {
            let cube_dim = CubeDim::new_1d(256);
            let cube_count = CubeCount::new_1d((num_elements as u32 + 255) / 256);

            unsafe {
                let _ = contiguous_kernel::launch_unchecked::<E, R>(
                    client,
                    cube_count,
                    cube_dim,
                    self.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    ScalarArg::new(self.view.offset as u32),
                );
            };
        }

        if let Some(backend) = &self.backend {
            output.with_backend(backend.clone())
        } else {
            output
        }
    }

    /// Create a contiguous copy using attached backend
    pub fn contiguous_auto(&self) -> Option<Self> {
        self.backend.as_ref().map(|b| {
            let mut buffer = self.contiguous(b.client());
            buffer.backend = Some(b.clone());
            buffer
        })
    }

    /// Create a TensorHandleRef for use with cubecl kernels
    /// Note: offset is encoded in the handle, strides account for the view
    pub fn as_tensor_handle_ref(&self) -> TensorHandleRef<'_, R> {
        unsafe {
            TensorHandleRef::from_raw_parts(
                self.storage.handle(),
                &self.view.strides,
                &self.view.shape,
                std::mem::size_of::<E>(),
            )
        }
    }

    /// Create a TensorArg for kernel launching
    pub fn as_tensor_arg<'a>(&'a self, line_size: u8) -> TensorArg<'a, R> {
        unsafe {
            TensorArg::from_raw_parts::<E>(
                self.storage.handle(),
                &self.view.strides,
                &self.view.shape,
                line_size as usize,
            )
        }
    }
}

///!TODO: move this kernel to a proper place
#[cube(launch_unchecked)]
pub fn contiguous_kernel<E: CubePrimitive>(input: &Tensor<E>, output: &mut Tensor<E>, offset: u32) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS + offset as usize];
    }
}

pub trait DynBufferOps<R: Runtime>: Send + Sync {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn ndim(&self) -> usize;
    fn storage_id(&self) -> BufferId;
    fn is_contiguous(&self) -> bool;
    fn element_size(&self) -> usize;
    fn element_type_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<R: Runtime, E: CubeElement + CubePrimitive + 'static> DynBufferOps<R> for Buffer<R, E> {
    fn shape(&self) -> &[usize] {
        &self.view.shape
    }

    fn strides(&self) -> &[usize] {
        &self.view.strides
    }

    fn offset(&self) -> usize {
        self.view.offset
    }

    fn len(&self) -> usize {
        self.view.len
    }

    fn is_empty(&self) -> bool {
        self.view.len == 0
    }

    fn ndim(&self) -> usize {
        self.view.ndim()
    }

    fn storage_id(&self) -> BufferId {
        self.storage.id()
    }

    fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
    }

    fn element_size(&self) -> usize {
        std::mem::size_of::<E>()
    }

    fn element_type_name(&self) -> &'static str {
        std::any::type_name::<E>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Type-erased buffer for heterogeneous collections
pub struct DynBuffer<R: Runtime> {
    inner: Box<dyn DynBufferOps<R>>,
}

impl<R: Runtime> DynBuffer<R> {
    /// Create a new DynBuffer from a typed buffer
    pub fn new<E: CubeElement + CubePrimitive + 'static>(buffer: Buffer<R, E>) -> Self {
        DynBuffer {
            inner: Box::new(buffer),
        }
    }

    /// Try to downcast to a specific buffer type
    pub fn downcast_ref<E: CubeElement + CubePrimitive + 'static>(&self) -> Option<&Buffer<R, E>> {
        self.inner.as_any().downcast_ref()
    }

    /// Try to downcast to a specific buffer type (mutable)
    pub fn downcast_mut<E: CubeElement + CubePrimitive + 'static>(
        &mut self,
    ) -> Option<&mut Buffer<R, E>> {
        self.inner.as_any_mut().downcast_mut()
    }

    /// Get the element type name
    pub fn element_type_name(&self) -> &'static str {
        self.inner.element_type_name()
    }
}

impl<R: Runtime> std::ops::Deref for DynBuffer<R> {
    type Target = dyn DynBufferOps<R>;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<R: Runtime> std::ops::DerefMut for DynBuffer<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

impl<R: Runtime> std::fmt::Debug for DynBuffer<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynBuffer")
            .field("element_type", &self.element_type_name())
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("offset", &self.offset())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Allocate exactly what's needed
    Exact,
    /// Allocate with some padding for potential reuse
    Padded {
        /// Alignment in bytes for padding
        alignment: usize,
    },
    /// Use a memory pool for frequent allocations
    Pooled,
}

/// Memory pool for buffer allocation and reuse
pub struct BufferPool<R: Runtime> {
    /// Available buffers organized by size
    available: std::collections::HashMap<usize, Vec<Handle>>,
    backend: Backend<R>,
    strategy: AllocationStrategy,
}

impl<R: Runtime> BufferPool<R> {
    /// Create a new buffer pool with a Backend context
    pub fn new(backend: Backend<R>, strategy: AllocationStrategy) -> Self {
        BufferPool {
            available: std::collections::HashMap::new(),
            backend,
            strategy,
        }
    }

    /// Create a new buffer pool from a raw ComputeClient (wraps in Backend)
    pub fn from_client(client: ComputeClient<R>, strategy: AllocationStrategy) -> Self {
        Self::new(Backend::new(client), strategy)
    }

    /// Get the backend context
    pub fn backend(&self) -> &Backend<R> {
        &self.backend
    }

    /// Get or allocate a buffer of the given size
    pub fn get_or_alloc(&mut self, size: usize) -> Handle {
        let aligned_size = match self.strategy {
            AllocationStrategy::Exact => size,
            AllocationStrategy::Padded { alignment } => {
                (size + alignment - 1) / alignment * alignment
            }
            AllocationStrategy::Pooled => size.next_power_of_two(),
        };

        if let Some(buffers) = self.available.get_mut(&aligned_size) {
            if let Some(handle) = buffers.pop() {
                return handle;
            }
        }

        self.backend.client().empty(aligned_size)
    }

    /// Get or allocate a typed buffer with the backend attached
    pub fn get_or_alloc_buffer<E: CubeElement + CubePrimitive>(
        &mut self,
        shape: Vec<usize>,
    ) -> Buffer<R, E> {
        let len: usize = shape.iter().product();
        let size_bytes = len * std::mem::size_of::<E>();
        let handle = self.get_or_alloc(size_bytes);

        let aligned_size = match self.strategy {
            AllocationStrategy::Exact => size_bytes,
            AllocationStrategy::Padded { alignment } => {
                (size_bytes + alignment - 1) / alignment * alignment
            }
            AllocationStrategy::Pooled => size_bytes.next_power_of_two(),
        };

        let storage = Arc::new(Storage::new(handle, aligned_size));
        Buffer::from_parts(
            storage,
            BufferView::contiguous(shape),
            Some(self.backend.clone()),
        )
    }

    pub fn return_buffer(&mut self, handle: Handle, size: usize) {
        let aligned_size = match self.strategy {
            AllocationStrategy::Exact => size,
            AllocationStrategy::Padded { alignment } => {
                (size + alignment - 1) / alignment * alignment
            }
            AllocationStrategy::Pooled => size.next_power_of_two(),
        };

        self.available.entry(aligned_size).or_default().push(handle);
    }

    pub fn clear(&mut self) {
        self.available.clear();
    }
}
