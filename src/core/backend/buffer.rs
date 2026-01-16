use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
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

/// Inner buffer data that holds the actual GPU handle
/// This is wrapped in Arc for reference counting
#[derive(Debug)]
pub struct BufferInner<R: Runtime> {
    pub handle: Handle, /// The underlying GPU handle from cubecl
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub len: usize,
    pub id: BufferId,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> BufferInner<R> {
    /// Calculate strides from shape (row-major order)
    pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// A GPU buffer handle with reference counting for zero-copy 
#[derive(Debug)]
pub struct Buffer<R: Runtime, E: CubeElement> {
    pub(crate) inner: Arc<BufferInner<R>>,
    _element: PhantomData<E>,
}

impl<R: Runtime, E: CubeElement> Clone for Buffer<R, E> {
    /// Cheap clone (only increments reference count)
    fn clone(&self) -> Self {
        Buffer {
            inner: Arc::clone(&self.inner),
            _element: PhantomData,
        }
    }
}

impl<R: Runtime, E: CubeElement> Buffer<R, E> {
    /// Create a new buffer from a GPU handle and shape
    pub fn new(handle: Handle, shape: Vec<usize>) -> Self {
        let strides = BufferInner::<R>::calculate_strides(&shape);
        let len = shape.iter().product();

        Buffer {
            inner: Arc::new(BufferInner {
                handle,
                shape,
                strides,
                len,
                id: BufferId::new(),
                _runtime: PhantomData,
            }),
            _element: PhantomData,
        }
    }

    /// Create a buffer from host data
    pub fn from_data(client: &ComputeClient<R>, data: &[E], shape: Vec<usize>) -> Self {
        let bytes = Bytes::from_bytes_vec(E::as_bytes(data).to_vec());
        let handle = client.create(bytes);
        Self::new(handle, shape)
    }

    /// Create an uninitialized buffer of given shape
    pub fn empty(client: &ComputeClient<R>, shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        let handle = client.empty(len * std::mem::size_of::<E>());
        Self::new(handle, shape)
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

    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.inner.strides
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn id(&self) -> BufferId {
        self.inner.id
    }

    pub fn handle(&self) -> &Handle {
        &self.inner.handle
    }

    /// Check if this is the only reference to the buffer
    /// Used for in-place operations
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Try to get a mutable reference if this is the only reference
    /// Returns None if there are other references
    pub fn try_get_unique(&mut self) -> Option<&mut BufferInner<R>> {
        Arc::get_mut(&mut self.inner)
    }

    /// Read data back to host
    pub fn to_data(&self, client: &ComputeClient<R>) -> Vec<E> {
        let bytes = client.read_one(self.inner.handle.clone());
        E::from_bytes(&bytes).to_vec()
    }

    /// Create a TensorHandleRef for use with cubecl kernels
    pub fn as_tensor_handle_ref(&self) -> TensorHandleRef<'_, R> {
        unsafe {
            TensorHandleRef::from_raw_parts(
                &self.inner.handle,
                &self.inner.strides,
                &self.inner.shape,
                std::mem::size_of::<E>(),
            )
        }
    }
}

impl<R: Runtime, E: CubeElement + CubePrimitive> Buffer<R, E> {
    /// Create a TensorArg for kernel launching
    pub fn as_tensor_arg<'a>(&'a self, line_size: u8) -> TensorArg<'a, R> {
        unsafe {
            TensorArg::from_raw_parts::<E>(
                &self.inner.handle,
                &self.inner.strides,
                &self.inner.shape,
                line_size as usize,
            )
        }
    }
}


