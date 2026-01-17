use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::core::backend::buffer::backend::Backend;
use crate::core::backend::buffer::id::BufferId;
use crate::core::backend::buffer::kernels;
use crate::core::backend::buffer::storage::Storage;
use crate::core::backend::buffer::view::{BroadcastMetadata, BufferView};

/// A GPU buffer with reference counting for zero-copy operations.
#[derive(Debug)]
pub struct Buffer<R: Runtime, E: CubeElement + CubePrimitive> {
    pub(crate) storage: Arc<Storage<R>>,
    pub(crate) view: BufferView,
    pub(crate) backend: Option<Backend<R>>,
    pub(crate) _element: PhantomData<E>,
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
                let _ = kernels::contiguous_kernel::launch_unchecked::<E, R>(
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

    /// Fill the buffer with a constant value
    pub fn filled(client: &ComputeClient<R>, shape: Vec<usize>, value: E) -> Self
    where
        E: Clone,
    {
        let len: usize = shape.iter().product();
        let data = vec![value; len];
        Self::from_data(client, &data, shape)
    }

    /// Fill the buffer with a constant value (with backend)
    pub fn filled_with_backend(backend: &Backend<R>, shape: Vec<usize>, value: E) -> Self
    where
        E: Clone,
    {
        let mut buffer = Self::filled(backend.client(), shape, value);
        buffer.backend = Some(backend.clone());
        buffer
    }

    /// Copy data from another buffer (GPU-to-GPU copy)
    /// Both buffers must be contiguous
    pub fn copy_from(&mut self, client: &ComputeClient<R>, other: &Self) {
        assert!(
            self.is_contiguous() && other.is_contiguous(),
            "copy_from requires both buffers to be contiguous"
        );
        assert_eq!(
            self.len(),
            other.len(),
            "copy_from requires buffers of the same length"
        );

        let num_elements = self.len();
        if num_elements == 0 {
            return;
        }

        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d((num_elements as u32 + 255) / 256);

        unsafe {
            let _ = kernels::copy_kernel::launch_unchecked::<E, R>(
                client,
                cube_count,
                cube_dim,
                other.as_tensor_arg(1),
                self.as_tensor_arg(1),
                ScalarArg::new(other.view.offset as u32),
                ScalarArg::new(self.view.offset as u32),
            );
        };
    }

    /// Copy using attached backend
    pub fn copy_from_auto(&mut self, other: &Self) -> bool {
        if let Some(backend) = self.backend.clone() {
            self.copy_from(backend.client(), other);
            true
        } else {
            false
        }
    }

    /// Create a new buffer with aligned memory layout for better GPU performance
    pub fn empty_aligned(client: &ComputeClient<R>, shape: Vec<usize>, alignment: usize) -> Self {
        let element_size = std::mem::size_of::<E>();
        let storage_elements =
            BufferView::compute_aligned_storage_size(&shape, element_size, alignment);
        let size_bytes = storage_elements * element_size;
        let storage = Arc::new(Storage::empty(client, size_bytes));
        let view = BufferView::aligned(shape, element_size, alignment);
        Buffer {
            storage,
            view,
            backend: None,
            _element: PhantomData,
        }
    }

    /// Create an aligned buffer with backend context
    pub fn empty_aligned_with_backend(
        backend: &Backend<R>,
        shape: Vec<usize>,
        alignment: usize,
    ) -> Self {
        let mut buffer = Self::empty_aligned(backend.client(), shape, alignment);
        buffer.backend = Some(backend.clone());
        buffer
    }

    /// Create a Fortran-contiguous (column-major) buffer
    pub fn empty_fortran(client: &ComputeClient<R>, shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        let size_bytes = len * std::mem::size_of::<E>();
        let storage = Arc::new(Storage::empty(client, size_bytes));
        let view = BufferView::fortran_contiguous(shape);
        Buffer {
            storage,
            view,
            backend: None,
            _element: PhantomData,
        }
    }

    /// Create a Fortran-contiguous buffer with backend context
    pub fn empty_fortran_with_backend(backend: &Backend<R>, shape: Vec<usize>) -> Self {
        let mut buffer = Self::empty_fortran(backend.client(), shape);
        buffer.backend = Some(backend.clone());
        buffer
    }

    /// Broadcast this buffer's view to a target shape (zero-copy)
    /// Returns a new buffer with modified strides where broadcasted dimensions have stride 0
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Option<Self> {
        let new_view = self.view.broadcast_to(target_shape)?;
        Some(Self::from_parts(
            Arc::clone(&self.storage),
            new_view,
            self.backend.clone(),
        ))
    }

    /// Check if this buffer can be broadcast to the target shape
    pub fn is_broadcastable_to(&self, target_shape: &[usize]) -> bool {
        self.view.is_broadcastable_to(target_shape)
    }

    /// Compute broadcast metadata for a binary operation with another buffer
    pub fn broadcast_metadata_with<E2: CubeElement + CubePrimitive>(
        &self,
        other: &Buffer<R, E2>,
    ) -> Option<BroadcastMetadata> {
        BroadcastMetadata::compute(&self.view, &other.view)
    }

    /// Creaxte a TensorArg with broadcasted strides for kernel launching
    /// This allows kernels to handle broadcasting without explicit copies
    /// Note: Returns the broadcasted view that must be kept alive while the TensorArg is in use
    pub fn broadcast_view_to(&self, target_shape: &[usize]) -> Option<BufferView> {
        self.view.broadcast_to(target_shape)
    }
}
