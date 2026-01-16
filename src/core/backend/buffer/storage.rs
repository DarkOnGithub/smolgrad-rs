use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;
use std::sync::Weak;
use crate::core::backend::buffer::id::BufferId;

/// Physical GPU memory storage (the raw allocation)
#[derive(Debug)]
pub struct Storage<R: Runtime> {
    pub(crate) handle: Handle,
    pub(crate) size_bytes: usize,
    pub(crate) id: BufferId,
    pub(crate) _runtime: PhantomData<R>,
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
        use cubecl::bytes::Bytes;
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

    /// Take ownership of the handle, consuming the storage
    pub fn into_handle(self) -> Handle {
        self.handle
    }
}

/// A handle recycler that can be used to return handles to a pool
pub trait HandleRecycler: Send + Sync {
    fn recycle(&self, handle: Handle, size_bytes: usize);
}

/// Storage that returns its handle to a pool when dropped
pub struct PooledStorage<R: Runtime> {
    pub(crate) handle: Option<Handle>,
    pub(crate) size_bytes: usize,
    pub(crate) id: BufferId,
    pub(crate) recycler: Weak<dyn HandleRecycler>,
    pub(crate) _runtime: PhantomData<R>,
}

impl<R: Runtime> std::fmt::Debug for PooledStorage<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledStorage")
            .field("size_bytes", &self.size_bytes)
            .field("id", &self.id)
            .field("has_handle", &self.handle.is_some())
            .finish()
    }
}

impl<R: Runtime> PooledStorage<R> {
    /// Create pooled storage with a recycler reference
    pub fn new(handle: Handle, size_bytes: usize, recycler: Weak<dyn HandleRecycler>) -> Self {
        PooledStorage {
            handle: Some(handle),
            size_bytes,
            id: BufferId::new(),
            recycler,
            _runtime: PhantomData,
        }
    }

    pub fn handle(&self) -> &Handle {
        self.handle.as_ref().expect("Handle already taken")
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn id(&self) -> BufferId {
        self.id
    }
}

impl<R: Runtime> Drop for PooledStorage<R> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            if let Some(recycler) = self.recycler.upgrade() {
                recycler.recycle(handle, self.size_bytes);
            }
        }
    }
}
