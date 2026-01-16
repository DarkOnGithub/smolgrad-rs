use cubecl::prelude::*;
use std::sync::Arc;
use crate::core::backend::buffer::buffer::Buffer;

/// A reference-counted backend context that wraps the ComputeClient.
pub struct Backend<R: Runtime> {
    pub(crate) client: Arc<ComputeClient<R>>,
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
