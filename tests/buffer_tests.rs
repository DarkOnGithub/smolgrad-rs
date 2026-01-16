//! Unit tests for the Buffer architecture
//!
//! These tests verify:
//! 1. Storage and View separation
//! 2. Zero-copy slicing with offset support
//! 3. Backend context management
//! 4. Type-erased DynBuffer for heterogeneous collections
//! 5. View operations (reshape, transpose, slice, permute, squeeze, unsqueeze)
//! 6. Contiguity checking

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use smolgrad_rs::core::backend::{
    AllocationStrategy, Backend, Buffer, BufferId, BufferPool, BufferView, DynBuffer, Storage,
};

type TestRuntime = WgpuRuntime;

/// Helper to create a test client
fn get_test_client() -> ComputeClient<TestRuntime> {
    let device = WgpuDevice::default();
    WgpuRuntime::client(&device)
}

// =============================================================================
// BufferView Unit Tests (no GPU required)
// =============================================================================

mod buffer_view_tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        let strides = BufferView::compute_contiguous_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);

        let strides = BufferView::compute_contiguous_strides(&[5]);
        assert_eq!(strides, vec![1]);

        let strides = BufferView::compute_contiguous_strides(&[]);
        assert_eq!(strides, Vec::<usize>::new());
    }

    #[test]
    fn test_contiguous_view_creation() {
        let view = BufferView::contiguous(vec![2, 3, 4]);
        assert_eq!(view.shape, vec![2, 3, 4]);
        assert_eq!(view.strides, vec![12, 4, 1]);
        assert_eq!(view.offset, 0);
        assert_eq!(view.len, 24);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_is_contiguous() {
        let contiguous = BufferView::contiguous(vec![2, 3]);
        assert!(contiguous.is_contiguous());

        // Non-contiguous (transposed)
        let transposed = BufferView::new(vec![3, 2], vec![1, 3], 0);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_is_fortran_contiguous() {
        // Row-major is not Fortran contiguous
        let row_major = BufferView::contiguous(vec![2, 3]);
        assert!(!row_major.is_fortran_contiguous());

        // Column-major (Fortran) contiguous
        let col_major = BufferView::new(vec![2, 3], vec![1, 2], 0);
        assert!(col_major.is_fortran_contiguous());
    }

    #[test]
    fn test_reshape_valid() {
        let view = BufferView::contiguous(vec![2, 3, 4]);
        let reshaped = view.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.shape, vec![6, 4]);
        assert_eq!(reshaped.len, 24);
        assert_eq!(reshaped.offset, 0);
        assert!(reshaped.is_contiguous());
    }

    #[test]
    fn test_reshape_invalid_size() {
        let view = BufferView::contiguous(vec![2, 3, 4]);
        assert!(view.reshape(vec![5, 5]).is_none()); // 25 != 24
    }

    #[test]
    fn test_reshape_non_contiguous_fails() {
        let transposed = BufferView::new(vec![3, 2], vec![1, 3], 0);
        assert!(transposed.reshape(vec![6]).is_none());
    }

    #[test]
    fn test_transpose() {
        let view = BufferView::contiguous(vec![2, 3]);
        let transposed = view.transpose().unwrap();
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.strides, vec![1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_transpose_3d() {
        let view = BufferView::contiguous(vec![2, 3, 4]);
        let transposed = view.transpose().unwrap();
        assert_eq!(transposed.shape, vec![2, 4, 3]);
        assert_eq!(transposed.strides, vec![12, 1, 4]);
    }

    #[test]
    fn test_transpose_1d_fails() {
        let view = BufferView::contiguous(vec![6]);
        assert!(view.transpose().is_none());
    }

    #[test]
    fn test_permute() {
        let view = BufferView::contiguous(vec![2, 3, 4]);
        let permuted = view.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape, vec![4, 2, 3]);
        assert_eq!(permuted.strides, vec![1, 12, 4]);
    }

    #[test]
    fn test_permute_invalid_axes() {
        let view = BufferView::contiguous(vec![2, 3, 4]);
        assert!(view.permute(&[0, 1]).is_none()); // Wrong length
        assert!(view.permute(&[0, 1, 3]).is_none()); // Out of bounds
        assert!(view.permute(&[0, 1, 1]).is_none()); // Duplicate
    }

    #[test]
    fn test_slice_basic() {
        let view = BufferView::contiguous(vec![10]);
        let sliced = view.slice(0, 2, 5, 1).unwrap();
        assert_eq!(sliced.shape, vec![3]);
        assert_eq!(sliced.offset, 2);
        assert_eq!(sliced.len, 3);
    }

    #[test]
    fn test_slice_with_step() {
        let view = BufferView::contiguous(vec![10]);
        let sliced = view.slice(0, 0, 10, 2).unwrap();
        assert_eq!(sliced.shape, vec![5]);
        assert_eq!(sliced.strides, vec![2]);
        assert_eq!(sliced.offset, 0);
    }

    #[test]
    fn test_slice_2d() {
        let view = BufferView::contiguous(vec![4, 6]);
        // Slice rows 1..3
        let sliced = view.slice(0, 1, 3, 1).unwrap();
        assert_eq!(sliced.shape, vec![2, 6]);
        assert_eq!(sliced.offset, 6); // 1 * stride[0] = 1 * 6
        assert_eq!(sliced.strides, vec![6, 1]);
    }

    #[test]
    fn test_slice_invalid() {
        let view = BufferView::contiguous(vec![10]);
        assert!(view.slice(0, 5, 3, 1).is_none()); // start >= end
        assert!(view.slice(0, 0, 11, 1).is_none()); // end > dim
        assert!(view.slice(1, 0, 5, 1).is_none()); // dim out of bounds
        assert!(view.slice(0, 0, 5, 0).is_none()); // step == 0
    }

    #[test]
    fn test_squeeze() {
        let view = BufferView::contiguous(vec![1, 3, 1, 4]);
        let squeezed = view.squeeze(None);
        assert_eq!(squeezed.shape, vec![3, 4]);
    }

    #[test]
    fn test_squeeze_specific_dim() {
        let view = BufferView::contiguous(vec![1, 3, 1, 4]);
        let squeezed = view.squeeze(Some(0));
        assert_eq!(squeezed.shape, vec![3, 1, 4]);
    }

    #[test]
    fn test_squeeze_non_singleton_noop() {
        let view = BufferView::contiguous(vec![1, 3, 1, 4]);
        let squeezed = view.squeeze(Some(1)); // dim 1 has size 3, not 1
        assert_eq!(squeezed.shape, vec![1, 3, 1, 4]);
    }

    #[test]
    fn test_unsqueeze() {
        let view = BufferView::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape, vec![1, 3, 4]);

        let unsqueezed = view.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape, vec![3, 1, 4]);

        let unsqueezed = view.unsqueeze(2).unwrap();
        assert_eq!(unsqueezed.shape, vec![3, 4, 1]);
    }

    #[test]
    fn test_unsqueeze_invalid() {
        let view = BufferView::contiguous(vec![3, 4]);
        assert!(view.unsqueeze(3).is_none()); // dim > ndim
    }

    #[test]
    fn test_ndim() {
        assert_eq!(BufferView::contiguous(vec![2, 3, 4]).ndim(), 3);
        assert_eq!(BufferView::contiguous(vec![5]).ndim(), 1);
        assert_eq!(BufferView::contiguous(vec![]).ndim(), 0);
    }
}

// =============================================================================
// BufferId Tests
// =============================================================================

mod buffer_id_tests {
    use super::*;

    #[test]
    fn test_buffer_id_unique() {
        let id1 = BufferId::new();
        let id2 = BufferId::new();
        let id3 = BufferId::new();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_buffer_id_default() {
        let id1 = BufferId::default();
        let id2 = BufferId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_buffer_id_clone_eq() {
        let id = BufferId::new();
        let cloned = id;
        assert_eq!(id, cloned);
    }
}

// =============================================================================
// GPU-dependent Tests (require runtime)
// =============================================================================

mod storage_tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        let client = get_test_client();
        let storage = Storage::<TestRuntime>::empty(&client, 1024);
        assert_eq!(storage.size_bytes(), 1024);
    }

    #[test]
    fn test_storage_from_bytes() {
        let client = get_test_client();
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let storage = Storage::<TestRuntime>::from_bytes(&client, &data);
        assert_eq!(storage.size_bytes(), 8);
    }

    #[test]
    fn test_storage_id_unique() {
        let client = get_test_client();
        let s1 = Storage::<TestRuntime>::empty(&client, 64);
        let s2 = Storage::<TestRuntime>::empty(&client, 64);
        assert_ne!(s1.id(), s2.id());
    }
}

mod buffer_tests {
    use super::*;

    #[test]
    fn test_buffer_from_data() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 3]);

        assert_eq!(buffer.shape(), &[2, 3]);
        assert_eq!(buffer.strides(), &[3, 1]);
        assert_eq!(buffer.len(), 6);
        assert_eq!(buffer.ndim(), 2);
        assert!(buffer.is_contiguous());
    }

    #[test]
    fn test_buffer_empty() {
        let client = get_test_client();
        let buffer = Buffer::<TestRuntime, f32>::empty(&client, vec![4, 4]);

        assert_eq!(buffer.shape(), &[4, 4]);
        assert_eq!(buffer.len(), 16);
    }

    #[test]
    fn test_buffer_zeros() {
        let client = get_test_client();
        let buffer = Buffer::<TestRuntime, f32>::zeros(&client, vec![3, 3]);

        let data = buffer.to_data(&client);
        assert_eq!(data.len(), 9);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_buffer_to_data_roundtrip() {
        let client = get_test_client();
        let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &original, vec![2, 3]);

        let retrieved = buffer.to_data(&client);
        assert_eq!(original, retrieved);
    }

    #[test]
    fn test_buffer_clone_shares_storage() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer1 = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![4]);
        let buffer2 = buffer1.clone();

        assert!(buffer1.shares_storage_with(&buffer2));
        assert_eq!(buffer1.storage_id(), buffer2.storage_id());
    }

    #[test]
    fn test_buffer_is_storage_unique() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![4]);

        assert!(buffer.is_storage_unique());

        let _clone = buffer.clone();
        assert!(!buffer.is_storage_unique());
    }

    #[test]
    fn test_buffer_reshape_zero_copy() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 3]);
        let reshaped = buffer.reshape(vec![3, 2]).unwrap();

        // Should share storage
        assert!(buffer.shares_storage_with(&reshaped));
        assert_eq!(reshaped.shape(), &[3, 2]);

        // Data should be the same
        let original_data = buffer.to_data(&client);
        let reshaped_data = reshaped.to_data(&client);
        assert_eq!(original_data, reshaped_data);
    }

    #[test]
    fn test_buffer_transpose_zero_copy() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 3]);
        let transposed = buffer.transpose().unwrap();

        // Should share storage
        assert!(buffer.shares_storage_with(&transposed));
        assert_eq!(transposed.shape(), &[3, 2]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_buffer_slice_zero_copy() {
        let client = get_test_client();
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![4, 5]);

        // Slice rows 1..3
        let sliced = buffer.slice(0, 1, 3).unwrap();
        assert!(buffer.shares_storage_with(&sliced));
        assert_eq!(sliced.shape(), &[2, 5]);
        assert_eq!(sliced.offset(), 5); // Row 1 starts at offset 5

        // Verify sliced data
        let sliced_data = sliced.to_data(&client);
        assert_eq!(
            sliced_data,
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        );
    }

    #[test]
    fn test_buffer_permute() {
        let client = get_test_client();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 3, 4]);
        let permuted = buffer.permute(&[2, 0, 1]).unwrap();

        assert!(buffer.shares_storage_with(&permuted));
        assert_eq!(permuted.shape(), &[4, 2, 3]);
    }

    #[test]
    fn test_buffer_squeeze_unsqueeze() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![1, 2, 3, 1]);

        let squeezed = buffer.squeeze(None);
        assert_eq!(squeezed.shape(), &[2, 3]);
        assert!(buffer.shares_storage_with(&squeezed));

        let unsqueezed = squeezed.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 2, 3]);
    }

    #[test]
    fn test_buffer_contiguous_copy() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 3]);
        let transposed = buffer.transpose().unwrap();

        assert!(!transposed.is_contiguous());

        let contiguous = transposed.contiguous(&client);
        assert!(contiguous.is_contiguous());
        assert!(!buffer.shares_storage_with(&contiguous)); // New storage
    }

    #[test]
    fn test_buffer_contiguous_with_offset() {
        let client = get_test_client();
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![6]);

        // Slice to get [1.0, 2.0, 3.0]
        let sliced = buffer.slice(0, 1, 4).unwrap();
        assert_eq!(sliced.view().offset, 1);
        assert_eq!(sliced.len(), 3);

        let contiguous = sliced.contiguous(&client);
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.view().offset, 0);
        assert_eq!(contiguous.len(), 3);

        let result_data = contiguous.to_data(&client);
        assert_eq!(result_data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_buffer_can_modify_inplace() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![4]);

        assert!(buffer.can_modify_inplace());

        let _clone = buffer.clone();
        assert!(!buffer.can_modify_inplace()); // Multiple references

        let transposed = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 2])
            .transpose()
            .unwrap();
        assert!(!transposed.can_modify_inplace()); // Not contiguous
    }
}

// =============================================================================
// Backend Context Tests
// =============================================================================

mod backend_tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let client = get_test_client();
        let backend = Backend::new(client);
        // Just verify it compiles and doesn't panic
        let _ = backend.client();
    }

    #[test]
    fn test_backend_clone() {
        let client = get_test_client();
        let backend1 = Backend::new(client);
        let backend2 = backend1.clone();
        // Both should work
        let _ = backend1.client();
        let _ = backend2.client();
    }

    #[test]
    fn test_buffer_with_backend() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = Buffer::<TestRuntime, f32>::from_data_with_backend(&backend, &data, vec![4]);

        assert!(buffer.backend().is_some());

        // Use auto methods
        let retrieved = buffer.to_data_auto().unwrap();
        assert_eq!(data, retrieved);
    }

    #[test]
    fn test_buffer_set_backend() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut buffer = Buffer::<TestRuntime, f32>::from_data(backend.client(), &data, vec![4]);

        assert!(buffer.backend().is_none());

        buffer.set_backend(backend);
        assert!(buffer.backend().is_some());
    }

    #[test]
    fn test_buffer_zeros_with_backend() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let buffer = Buffer::<TestRuntime, f32>::zeros_with_backend(&backend, vec![3, 3]);
        let data = buffer.to_data_auto().unwrap();
        assert_eq!(data.len(), 9);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_buffer_contiguous_auto() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer =
            Buffer::<TestRuntime, f32>::from_data_with_backend(&backend, &data, vec![2, 3]);
        let transposed = buffer.transpose().unwrap();

        let contiguous = transposed.contiguous_auto().unwrap();
        assert!(contiguous.is_contiguous());
        assert!(contiguous.backend().is_some());
    }

    // =========================================================================
    // Tests for new Backend buffer creation methods
    // =========================================================================

    #[test]
    fn test_backend_from_data() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = backend.from_data(&data, vec![2, 3]);

        assert_eq!(buffer.shape(), &[2, 3]);
        assert!(buffer.backend().is_some());

        let retrieved = buffer.to_data_auto().unwrap();
        assert_eq!(data, retrieved);
    }

    #[test]
    fn test_backend_empty() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let buffer: Buffer<TestRuntime, f32> = backend.empty(vec![3, 4]);

        assert_eq!(buffer.shape(), &[3, 4]);
        assert_eq!(buffer.len(), 12);
        assert!(buffer.backend().is_some());
        assert!(buffer.is_contiguous());
    }

    #[test]
    fn test_backend_zeros() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let buffer: Buffer<TestRuntime, f32> = backend.zeros(vec![2, 3]);

        assert_eq!(buffer.shape(), &[2, 3]);
        assert!(buffer.backend().is_some());

        let data = buffer.to_data_auto().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_backend_ones() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let buffer: Buffer<TestRuntime, f32> = backend.ones(vec![2, 3]);

        assert_eq!(buffer.shape(), &[2, 3]);
        assert!(buffer.backend().is_some());

        let data = buffer.to_data_auto().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_backend_full() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let buffer: Buffer<TestRuntime, f32> = backend.full(vec![2, 3], 42.0);

        assert_eq!(buffer.shape(), &[2, 3]);
        assert!(buffer.backend().is_some());

        let data = buffer.to_data_auto().unwrap();
        assert!(data.iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_backend_chained_operations() {
        // Test that backend methods integrate well with view operations
        let client = get_test_client();
        let backend = Backend::new(client);

        let buffer: Buffer<TestRuntime, f32> =
            backend.from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Chain operations - backend should be preserved
        let transposed = buffer.transpose().unwrap();
        assert!(transposed.backend().is_some());

        let reshaped = transposed
            .contiguous_auto()
            .unwrap()
            .reshape(vec![6])
            .unwrap();
        assert!(reshaped.backend().is_some());
    }
}

// =============================================================================
// DynBuffer (Type Erasure) Tests
// =============================================================================

mod dyn_buffer_tests {
    use super::*;

    #[test]
    fn test_dyn_buffer_creation() {
        let client = get_test_client();
        let f32_buf = Buffer::<TestRuntime, f32>::zeros(&client, vec![2, 3]);
        let dyn_buf = DynBuffer::new(f32_buf);

        assert_eq!(dyn_buf.shape(), &[2, 3]);
        assert_eq!(dyn_buf.len(), 6);
        assert!(dyn_buf.is_contiguous());
    }

    #[test]
    fn test_dyn_buffer_heterogeneous_collection() {
        let client = get_test_client();

        let f32_buf = Buffer::<TestRuntime, f32>::zeros(&client, vec![2, 3]);
        let i32_buf = Buffer::<TestRuntime, i32>::zeros(&client, vec![4, 5]);
        let u32_buf = Buffer::<TestRuntime, u32>::zeros(&client, vec![6]);

        let collection: Vec<DynBuffer<TestRuntime>> = vec![
            DynBuffer::new(f32_buf),
            DynBuffer::new(i32_buf),
            DynBuffer::new(u32_buf),
        ];

        assert_eq!(collection.len(), 3);
        assert_eq!(collection[0].shape(), &[2, 3]);
        assert_eq!(collection[1].shape(), &[4, 5]);
        assert_eq!(collection[2].shape(), &[6]);
    }

    #[test]
    fn test_dyn_buffer_element_type_name() {
        let client = get_test_client();

        let f32_buf = Buffer::<TestRuntime, f32>::zeros(&client, vec![2, 3]);
        let dyn_buf = DynBuffer::new(f32_buf);

        assert!(dyn_buf.element_type_name().contains("f32"));
    }

    #[test]
    fn test_dyn_buffer_downcast() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let f32_buf = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![3]);
        let dyn_buf = DynBuffer::new(f32_buf);

        // Successful downcast
        let concrete = dyn_buf.downcast_ref::<f32>().unwrap();
        let retrieved = concrete.to_data(&client);
        assert_eq!(retrieved, data);

        // Failed downcast (wrong type)
        assert!(dyn_buf.downcast_ref::<i32>().is_none());
    }

    #[test]
    fn test_dyn_buffer_downcast_mut() {
        let client = get_test_client();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let f32_buf = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![3]);
        let mut dyn_buf = DynBuffer::new(f32_buf);

        let concrete = dyn_buf.downcast_mut::<f32>().unwrap();
        // Just verify we can get a mutable reference
        let _ = concrete.shape();
    }

    #[test]
    fn test_dyn_buffer_debug() {
        let client = get_test_client();
        let f32_buf = Buffer::<TestRuntime, f32>::zeros(&client, vec![2, 3]);
        let dyn_buf = DynBuffer::new(f32_buf);

        let debug_str = format!("{:?}", dyn_buf);
        assert!(debug_str.contains("DynBuffer"));
        assert!(debug_str.contains("shape"));
    }

    #[test]
    fn test_dyn_buffer_ops_trait() {
        let client = get_test_client();
        let f32_buf = Buffer::<TestRuntime, f32>::zeros(&client, vec![2, 3]);
        let dyn_buf = DynBuffer::new(f32_buf);

        // Test all DynBufferOps methods
        assert_eq!(dyn_buf.shape(), &[2, 3]);
        assert_eq!(dyn_buf.strides(), &[3, 1]);
        assert_eq!(dyn_buf.offset(), 0);
        assert_eq!(dyn_buf.len(), 6);
        assert!(!dyn_buf.is_empty());
        assert_eq!(dyn_buf.ndim(), 2);
        assert!(dyn_buf.is_contiguous());
        assert_eq!(dyn_buf.element_size(), std::mem::size_of::<f32>());
    }
}

// =============================================================================
// Buffer Pool Tests
// =============================================================================

mod buffer_pool_tests {
    use super::*;

    #[test]
    fn test_buffer_pool_exact() {
        let client = get_test_client();
        let mut pool = BufferPool::from_client(client, AllocationStrategy::Exact);

        let handle = pool.get_or_alloc(1024);
        pool.return_buffer(handle, 1024);

        // Should reuse the returned buffer
        let _handle2 = pool.get_or_alloc(1024);
    }

    #[test]
    fn test_buffer_pool_padded() {
        let client = get_test_client();
        let mut pool =
            BufferPool::from_client(client, AllocationStrategy::Padded { alignment: 256 });

        let handle = pool.get_or_alloc(100);
        pool.return_buffer(handle, 100);

        // Should reuse (100 rounds up to 256)
        let _handle2 = pool.get_or_alloc(100);
    }

    #[test]
    fn test_buffer_pool_pooled() {
        let client = get_test_client();
        let mut pool = BufferPool::from_client(client, AllocationStrategy::Pooled);

        let handle = pool.get_or_alloc(100);
        pool.return_buffer(handle, 100);

        // Should reuse (100 rounds up to 128)
        let _handle2 = pool.get_or_alloc(100);
    }

    #[test]
    fn test_buffer_pool_clear() {
        let client = get_test_client();
        let mut pool = BufferPool::from_client(client, AllocationStrategy::Exact);

        let handle = pool.get_or_alloc(1024);
        pool.return_buffer(handle, 1024);
        pool.clear();

        // After clear, should allocate new (but this is hard to test directly)
        let _handle2 = pool.get_or_alloc(1024);
    }

    #[test]
    fn test_buffer_pool_get_backend() {
        let client = get_test_client();
        let backend = Backend::new(client);
        let pool = BufferPool::new(backend.clone(), AllocationStrategy::Exact);

        // Verify we can get the backend from the pool
        let pool_backend = pool.backend();
        let _ = pool_backend.client();
    }

    #[test]
    fn test_buffer_pool_get_or_alloc_buffer() {
        let client = get_test_client();
        let backend = Backend::new(client);
        let mut pool = BufferPool::new(backend, AllocationStrategy::Exact);

        // Allocate a typed buffer with the pool's backend attached
        let buffer: Buffer<TestRuntime, f32> = pool.get_or_alloc_buffer(vec![3, 4]);

        assert_eq!(buffer.shape(), &[3, 4]);
        assert_eq!(buffer.len(), 12);
        assert!(buffer.backend().is_some());
        assert!(buffer.is_contiguous());
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_multiple_views_same_storage() {
        let client = get_test_client();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let original = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 3, 4]);

        // Create multiple views
        let reshaped = original.reshape(vec![6, 4]).unwrap();
        let transposed = original.transpose().unwrap();
        let sliced = original.slice(0, 0, 1).unwrap();

        // All should share storage
        assert!(original.shares_storage_with(&reshaped));
        assert!(original.shares_storage_with(&transposed));
        assert!(original.shares_storage_with(&sliced));

        // But have different shapes
        assert_eq!(original.shape(), &[2, 3, 4]);
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(transposed.shape(), &[2, 4, 3]);
        assert_eq!(sliced.shape(), &[1, 3, 4]);
    }

    #[test]
    fn test_chained_view_operations() {
        let client = get_test_client();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let buffer = Buffer::<TestRuntime, f32>::from_data(&client, &data, vec![2, 3, 4]);

        // Chain: transpose -> slice -> unsqueeze
        let result = buffer
            .transpose()
            .unwrap()
            .slice(0, 0, 2)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        // Should still share storage
        assert!(buffer.shares_storage_with(&result));
    }

    #[test]
    fn test_view_preserves_backend() {
        let client = get_test_client();
        let backend = Backend::new(client);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer =
            Buffer::<TestRuntime, f32>::from_data_with_backend(&backend, &data, vec![2, 3]);

        let transposed = buffer.transpose().unwrap();
        let reshaped = buffer.reshape(vec![6]).unwrap();
        let sliced = buffer.slice(0, 0, 1).unwrap();

        // All views should have backend
        assert!(transposed.backend().is_some());
        assert!(reshaped.backend().is_some());
        assert!(sliced.backend().is_some());
    }
}
