use cubecl::prelude::*;
use cubecl::server::Handle;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, Weak};

use crate::core::backend::buffer::backend::Backend;
use crate::core::backend::buffer::buffer::Buffer;
use crate::core::backend::buffer::storage::{Storage, PooledStorage, HandleRecycler};
use crate::core::backend::buffer::view::BufferView;

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
    pub(crate) available: HashMap<usize, Vec<Handle>>,
    pub(crate) backend: Backend<R>,
    pub(crate) strategy: AllocationStrategy,
}

impl<R: Runtime> BufferPool<R> {
    /// Create a new buffer pool with a Backend context
    pub fn new(backend: Backend<R>, strategy: AllocationStrategy) -> Self {
        BufferPool {
            available: HashMap::new(),
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

    /// Get statistics about the pool
    pub fn stats(&self) -> PoolStats {
        let total_handles: usize = self.available.values().map(|v| v.len()).sum();
        let total_bytes: usize = self
            .available
            .iter()
            .map(|(size, handles)| size * handles.len())
            .sum();
        PoolStats {
            available_handles: total_handles,
            available_bytes: total_bytes,
            bucket_count: self.available.len(),
        }
    }
}

/// Statistics about the buffer pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub available_handles: usize,
    pub available_bytes: usize,
    pub bucket_count: usize,
}

/// Inner state for ManagedBufferPool
pub(crate) struct ManagedPoolInner<R: Runtime> {
    pub(crate) pool: BufferPool<R>,
}

/// A thread-safe, reference-counted buffer pool that automatically recycles handles
/// When buffers allocated from this pool are dropped, their handles are automatically
/// returned to the pool for reuse.
pub struct ManagedBufferPool<R: Runtime> {
    pub(crate) inner: Arc<Mutex<ManagedPoolInner<R>>>,
}

impl<R: Runtime> Clone for ManagedBufferPool<R> {
    fn clone(&self) -> Self {
        ManagedBufferPool {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<R: Runtime> ManagedBufferPool<R> {
    pub fn new(backend: Backend<R>, strategy: AllocationStrategy) -> Self {
        ManagedBufferPool {
            inner: Arc::new(Mutex::new(ManagedPoolInner {
                pool: BufferPool::new(backend, strategy),
            })),
        }
    }

    /// Create a managed pool from a raw ComputeClient
    pub fn from_client(client: ComputeClient<R>, strategy: AllocationStrategy) -> Self {
        Self::new(Backend::new(client), strategy)
    }

    /// Get the backend context
    pub fn backend(&self) -> Backend<R> {
        self.inner.lock().unwrap().pool.backend.clone()
    }

    /// Allocate a pooled buffer that will automatically return its handle when dropped
    pub fn alloc<E: CubeElement + CubePrimitive>(&self, shape: Vec<usize>) -> PooledBuffer<R, E> {
        let len: usize = shape.iter().product();
        let size_bytes = len * std::mem::size_of::<E>();

        let (handle, aligned_size, backend) = {
            let mut inner = self.inner.lock().unwrap();
            let handle = inner.pool.get_or_alloc(size_bytes);
            let aligned_size = match inner.pool.strategy {
                AllocationStrategy::Exact => size_bytes,
                AllocationStrategy::Padded { alignment } => {
                    (size_bytes + alignment - 1) / alignment * alignment
                }
                AllocationStrategy::Pooled => size_bytes.next_power_of_two(),
            };
            (handle, aligned_size, inner.pool.backend.clone())
        };

        let recycler: Weak<dyn HandleRecycler> =
            Arc::downgrade(&self.inner) as Weak<dyn HandleRecycler>;

        let storage = Arc::new(PooledStorage::new(handle, aligned_size, recycler));
        PooledBuffer {
            storage,
            view: BufferView::contiguous(shape),
            backend: Some(backend),
            _element: PhantomData,
        }
    }

    pub fn stats(&self) -> PoolStats {
        self.inner.lock().unwrap().pool.stats()
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap().pool.clear();
    }
}

impl<R: Runtime> HandleRecycler for Mutex<ManagedPoolInner<R>> {
    fn recycle(&self, handle: Handle, size_bytes: usize) {
        if let Ok(mut inner) = self.lock() {
            inner.pool.return_buffer(handle, size_bytes);
        }
    }
}

/// A buffer whose storage is managed by a pool.
/// When this buffer is dropped, its GPU handle is automatically returned to the pool.
pub struct PooledBuffer<R: Runtime, E: CubeElement + CubePrimitive> {
    pub(crate) storage: Arc<PooledStorage<R>>,
    pub(crate) view: BufferView,
    pub(crate) backend: Option<Backend<R>>,
    pub(crate) _element: PhantomData<E>,
}

impl<R: Runtime, E: CubeElement + CubePrimitive> std::fmt::Debug for PooledBuffer<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("shape", &self.view.shape)
            .field("strides", &self.view.strides)
            .field("offset", &self.view.offset)
            .finish()
    }
}

impl<R: Runtime, E: CubeElement + CubePrimitive> Clone for PooledBuffer<R, E> {
    fn clone(&self) -> Self {
        PooledBuffer {
            storage: Arc::clone(&self.storage),
            view: self.view.clone(),
            backend: self.backend.clone(),
            _element: PhantomData,
        }
    }
}

impl<R: Runtime, E: CubeElement + CubePrimitive> PooledBuffer<R, E> {
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

    pub fn handle(&self) -> &Handle {
        self.storage.handle()
    }

    pub fn backend(&self) -> Option<&Backend<R>> {
        self.backend.as_ref()
    }

    pub fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
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

    /// Read data back to host
    pub fn to_data(&self, client: &ComputeClient<R>) -> Vec<E> {
        let bytes = client.read_one(self.storage.handle().clone());
        let all_elements = E::from_bytes(&bytes);

        if self.is_contiguous() {
            all_elements[self.view.offset..self.view.offset + self.len()].to_vec()
        } else {
            let mut result = Vec::with_capacity(self.len());
            self.gather_recursive(&all_elements, &mut result, 0, self.view.offset);
            result
        }
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

    /// Read data using attached backend
    pub fn to_data_auto(&self) -> Option<Vec<E>> {
        self.backend.as_ref().map(|b| self.to_data(b.client()))
    }
}
