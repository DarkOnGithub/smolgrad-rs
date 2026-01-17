use crate::core::backend::buffer::buffer::Buffer;
use crate::core::backend::buffer::id::BufferId;
use cubecl::prelude::*;
use std::any::Any;

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
        self.shape()
    }

    fn strides(&self) -> &[usize] {
        self.strides()
    }

    fn offset(&self) -> usize {
        self.offset()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn storage_id(&self) -> BufferId {
        self.storage_id()
    }

    fn is_contiguous(&self) -> bool {
        self.is_contiguous()
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
    pub(crate) inner: Box<dyn DynBufferOps<R>>,
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
