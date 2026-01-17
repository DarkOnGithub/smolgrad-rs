use rustc_hash::FxHashMap;
use std::any::TypeId;
use std::sync::{Arc, RwLock};

use crate::core::backend::Buffer;
use crate::core::backend::executor::error::{DType, ExecutorError, ExecutorResult};
use crate::core::backend::executor::graph::NodeId;
use cubecl::prelude::*;

pub trait AnyBuffer: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn shape(&self) -> &[usize];
    fn len(&self) -> usize;
    fn type_id(&self) -> TypeId;
    fn dtype(&self) -> DType;
}

impl<R: Runtime, E: CubeElement + CubePrimitive + 'static> AnyBuffer for Buffer<R, E> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn shape(&self) -> &[usize] {
        Buffer::shape(self)
    }

    fn len(&self) -> usize {
        Buffer::len(self)
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<E>()
    }

    fn dtype(&self) -> DType {
        DType::of::<E>()
    }
}

#[derive(Clone)]
pub struct CacheEntry {
    buffer: Arc<dyn AnyBuffer>,
    shape: Vec<usize>,
    dtype: DType,
}

impl std::fmt::Debug for CacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheEntry")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl CacheEntry {
    fn new(buffer: Arc<dyn AnyBuffer>) -> Self {
        let shape = buffer.shape().to_vec();
        let dtype = buffer.dtype();
        Self {
            buffer,
            shape,
            dtype,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn buffer(&self) -> &Arc<dyn AnyBuffer> {
        &self.buffer
    }
}

#[derive(Default)]
pub struct BufferCache {
    cache: RwLock<FxHashMap<NodeId, CacheEntry>>,
}

impl BufferCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn insert<R: Runtime + 'static, E: CubeElement + CubePrimitive + 'static>(
        &self,
        node_id: NodeId,
        buffer: Buffer<R, E>,
    ) {
        let entry = CacheEntry::new(Arc::new(buffer));
        let mut cache = self.cache.write().unwrap();
        cache.insert(node_id, entry);
    }

    pub fn insert_validated<R: Runtime + 'static, E: CubeElement + CubePrimitive + 'static>(
        &self,
        node_id: NodeId,
        buffer: Buffer<R, E>,
        expected_shape: &[usize],
        expected_dtype: &DType,
    ) -> ExecutorResult<()> {
        let buffer_shape = buffer.shape();
        let buffer_dtype = DType::of::<E>();

        if buffer_shape != expected_shape {
            return Err(ExecutorError::ShapeMismatch {
                node_id,
                expected: expected_shape.to_vec(),
                actual: buffer_shape.to_vec(),
            });
        }

        if &buffer_dtype != expected_dtype {
            return Err(ExecutorError::DTypeMismatch {
                node_id,
                expected: expected_dtype.clone(),
                actual: buffer_dtype,
            });
        }

        self.insert(node_id, buffer);
        Ok(())
    }

    pub fn get<R: Runtime + 'static, E: CubeElement + CubePrimitive + 'static>(
        &self,
        node_id: NodeId,
    ) -> Option<Buffer<R, E>> {
        let cache = self.cache.read().unwrap();
        cache.get(&node_id).and_then(|entry| {
            entry
                .buffer
                .as_any()
                .downcast_ref::<Buffer<R, E>>()
                .cloned()
        })
    }

    pub fn get_validated<R: Runtime + 'static, E: CubeElement + CubePrimitive + 'static>(
        &self,
        node_id: NodeId,
        expected_shape: &[usize],
        expected_dtype: &DType,
    ) -> ExecutorResult<Buffer<R, E>> {
        let cache = self.cache.read().unwrap();
        let entry = cache
            .get(&node_id)
            .ok_or(ExecutorError::CacheMiss { node_id })?;

        if entry.shape() != expected_shape {
            return Err(ExecutorError::ShapeMismatch {
                node_id,
                expected: expected_shape.to_vec(),
                actual: entry.shape().to_vec(),
            });
        }

        if entry.dtype() != expected_dtype {
            return Err(ExecutorError::DTypeMismatch {
                node_id,
                expected: expected_dtype.clone(),
                actual: entry.dtype().clone(),
            });
        }

        entry
            .buffer
            .as_any()
            .downcast_ref::<Buffer<R, E>>()
            .cloned()
            .ok_or(ExecutorError::DTypeMismatch {
                node_id,
                expected: expected_dtype.clone(),
                actual: entry.dtype().clone(),
            })
    }

    pub fn get_entry(&self, node_id: NodeId) -> Option<CacheEntry> {
        let cache = self.cache.read().unwrap();
        cache.get(&node_id).cloned()
    }

    pub fn get_shape(&self, node_id: NodeId) -> Option<Vec<usize>> {
        let cache = self.cache.read().unwrap();
        cache.get(&node_id).map(|e| e.shape.clone())
    }

    pub fn get_dtype(&self, node_id: NodeId) -> Option<DType> {
        let cache = self.cache.read().unwrap();
        cache.get(&node_id).map(|e| e.dtype.clone())
    }

    pub fn validate_shape(&self, node_id: NodeId, expected: &[usize]) -> ExecutorResult<()> {
        let cache = self.cache.read().unwrap();
        let entry = cache
            .get(&node_id)
            .ok_or(ExecutorError::CacheMiss { node_id })?;

        if entry.shape() != expected {
            return Err(ExecutorError::ShapeMismatch {
                node_id,
                expected: expected.to_vec(),
                actual: entry.shape().to_vec(),
            });
        }
        Ok(())
    }

    pub fn validate_dtype(&self, node_id: NodeId, expected: &DType) -> ExecutorResult<()> {
        let cache = self.cache.read().unwrap();
        let entry = cache
            .get(&node_id)
            .ok_or(ExecutorError::CacheMiss { node_id })?;

        if entry.dtype() != expected {
            return Err(ExecutorError::DTypeMismatch {
                node_id,
                expected: expected.clone(),
                actual: entry.dtype().clone(),
            });
        }
        Ok(())
    }

    pub fn contains(&self, node_id: NodeId) -> bool {
        let cache = self.cache.read().unwrap();
        cache.contains_key(&node_id)
    }

    pub fn remove(&self, node_id: NodeId) -> bool {
        let mut cache = self.cache.write().unwrap();
        cache.remove(&node_id).is_some()
    }

    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    pub fn len(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn retain<F>(&self, mut predicate: F)
    where
        F: FnMut(&NodeId, &CacheEntry) -> bool,
    {
        let mut cache = self.cache.write().unwrap();
        cache.retain(|k, v| predicate(k, v));
    }
}
