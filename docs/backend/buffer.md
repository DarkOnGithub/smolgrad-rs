# Backend Buffer

This section documents the buffer subsystem in `src/core/backend/buffer/`. It provides
GPU-backed storage, view metadata, pooling, and kernel helpers used across the backend.
The design separates physical allocation (storage) from logical layout (view), so many
buffer transformations are zero-copy and only adjust view metadata.

## High-level model

- `Storage` owns the raw GPU allocation (a `Handle`) and a stable `BufferId`.
- `BufferView` describes a logical tensor view (shape, strides, offset, length).
- `Buffer` combines `Storage` and `BufferView` with an optional `Backend` context.
- Optional pooling uses `BufferPool`/`ManagedBufferPool` to recycle allocations.
- Lightweight kernels perform contiguous materialization and device-to-device copies.

A `Buffer` is a typed, reference-counted view over shared storage. Cloning a `Buffer`
shares the same `Storage` but can carry distinct views (reshape, slice, broadcast, etc.).
A `BufferView` only contains metadata, so view changes are cheap while the underlying
allocation stays shared until a copy is required.

## Files and responsibilities

### `buffer.rs`

Defines the main `Buffer<R, E>` type and its operations.

Core fields:
- `storage: Arc<Storage<R>>` is the shared GPU allocation.
- `view: BufferView` describes shape, strides, offset, and length.
- `backend: Option<Backend<R>>` stores an optional context for auto methods.

Construction:
- `new` builds a contiguous view over an existing `Storage`.
- `from_parts` assembles a buffer from storage, view, and backend.
- `from_data` allocates storage and uploads host bytes.
- `empty`, `zeros`, `filled` allocate storage and initialize in host memory.
- `_with_backend` variants attach a `Backend` for auto methods.

View logic (zero-copy):
- `reshape` validates total element count and contiguity, then adjusts shape/strides.
- `transpose`, `permute` swap or reorder axes by swapping strides and shape.
- `slice` and `slice_with_step` update shape, strides, and offset; no data copy.
- `squeeze` and `unsqueeze` remove/add size-1 dimensions by editing metadata.
- `broadcast_to` returns a view where broadcasted dimensions have stride 0.

Data access:
- `to_data` reads the entire storage and then gathers elements based on strides.
- `to_data_auto` uses the attached backend if present.
- `gather_elements` walks the stride tree to collect the logical view.

Contiguity and copying:
- `is_contiguous` and `is_fortran_contiguous` inspect the stride pattern.
- `contiguous` returns a contiguous buffer. It avoids a copy if already contiguous
  with offset 0, otherwise it launches a kernel to pack the view or falls back to
  host round-trip for complex non-contiguous views.
- `copy_from` performs GPU-to-GPU copy when both buffers are contiguous and equal
  length; offsets are passed to the kernel.

Kernel interop:
- `as_tensor_handle_ref` and `as_tensor_arg` package the storage and view into
  CubeCL kernel arguments, encoding strides and shape for non-contiguous access.

Safety helpers:
- `is_storage_unique` detects exclusive ownership of the allocation.
- `can_modify_inplace` requires unique storage and contiguity to allow mutations.

### `storage.rs`

Defines the physical allocation and recycling hooks.

- `Storage` holds the `Handle`, `size_bytes`, and a unique `BufferId`.
- `Storage::empty` and `Storage::from_bytes` create allocations through the
  `ComputeClient`.
- `Storage::into_handle` consumes the storage and yields the raw handle.
- `PooledStorage` wraps a handle plus a weak `HandleRecycler`; on drop it returns
  the handle to the pool if the recycler is still alive.
- `HandleRecycler` abstracts recycling so pools can be swapped easily.

### `view.rs`

Defines the logical view metadata used by every buffer.

Layout:
- `shape` is the logical tensor dimensions.
- `strides` map indices to linear offsets in storage.
- `offset` is the base linear offset into the storage.
- `len` is the total logical element count.

Stride helpers:
- `compute_contiguous_strides` builds row-major (C) strides.
- `compute_fortran_strides` builds column-major (Fortran) strides.
- `compute_aligned_strides` and `compute_aligned_storage_size` support padding
  the innermost stride for alignment without changing logical shape.

View transforms:
- `reshape` requires contiguity and element count preservation.
- `transpose` swaps the last two axes by swapping both shape and stride entries.
- `permute` applies arbitrary axis ordering after validating permutation input.
- `slice` adjusts shape, strides, and offset; the step multiplies the stride.
- `squeeze` and `unsqueeze` remove or insert size-1 dimensions with adjusted
  strides to preserve the logical mapping.

Broadcasting:
- `broadcast_to` extends the view to a target shape by adding leading dimensions
  and setting stride 0 where broadcasting occurs.
- `broadcast_shape` computes the combined output shape from two inputs.

### `backend.rs`

A reference-counted wrapper around `ComputeClient`.

- `Backend` stores an `Arc<ComputeClient<R>>`.
- Provides convenient constructors (`from_data`, `empty`, `zeros`, `ones`, `full`)
  that return buffers already attached to the backend for `*_auto` methods.

### `pool.rs`

Allocation strategies and pooling for reusing GPU handles.

- `AllocationStrategy::Exact` allocates exactly the requested size.
- `AllocationStrategy::Padded` rounds up allocations to an alignment boundary.
- `AllocationStrategy::Pooled` uses power-of-two buckets to maximize reuse.

Pool types:
- `BufferPool` is a simple pool of available handles by size bucket.
- `ManagedBufferPool` wraps `BufferPool` in an `Arc<Mutex<_>>` and implements
  `HandleRecycler` so dropped buffers return their handles automatically.
- `PooledBuffer` is a view wrapper for pooled allocations and mirrors many of
  the read/shape helpers of `Buffer`.

### `dynamic.rs`

Type-erased buffers for heterogeneous collections.

- `DynBufferOps` exposes a buffer-like API without knowing the element type.
- `DynBuffer` stores a boxed trait object and supports downcasting back to a
  concrete `Buffer<R, E>`.
- Useful for generic containers or logging without templating on element type.

### `kernels.rs`

Lightweight kernels for buffer copies.

- `contiguous_kernel` copies a view with an offset into a contiguous output.
- `copy_kernel` copies between two contiguous buffers with explicit offsets.

## Lifecycle of a buffer

1. Allocate storage (directly or through a pool).
2. Create a `BufferView` describing the logical shape and layout.
3. Wrap both in a `Buffer` or `PooledBuffer` and optionally attach a `Backend`.
4. Apply view transforms for reshape/slice/broadcast without copying.
5. Materialize a contiguous buffer when kernels require dense layout.
6. Drop the buffer; if pooled, the handle returns to the pool automatically.

## Common usage patterns

### Create and read back

```rust
let buffer = Buffer::<R, f32>::from_data(client, &data, vec![2, 3]);
let host = buffer.to_data(client);
```

### Create a view without copy

```rust
let buffer = Buffer::<R, f32>::from_data(client, &data, vec![2, 3]);
let sliced = buffer.slice(1, 0, 2).expect("valid slice");
```

### Attach backend context for auto readback

```rust
let backend = Backend::new(client.clone());
let buffer = backend.zeros::<f32>(vec![4, 4]);
let host = buffer.to_data_auto().expect("backend attached");
```

### Ensure contiguity for kernels

```rust
let contiguous = non_contiguous.contiguous(client);
```

### Pool-backed allocations

```rust
let pool = ManagedBufferPool::from_client(client.clone(), AllocationStrategy::Pooled);
let pooled = pool.alloc::<f32>(vec![1024]);
```

## Notes on contiguity and broadcasting

- A view can be non-contiguous but still zero-copy; the view metadata remaps
  indices to linear storage offsets.
- `to_data` gathers elements when contiguity is not available, preserving the
  logical order implied by strides.
- Broadcasting is implemented by setting strides to zero on broadcasted dims.
  Kernels can treat these strides as repeated reads of the same element.
- `broadcast_metadata_with` returns the output shape and per-input strides for
  binary kernels so they can iterate without explicit expansion.

## Debugging and diagnostics

- `storage_id` lets you check whether two buffers share storage.
- `shares_storage_with` compares underlying `Arc<Storage>` pointers.
- `is_storage_unique` is a quick guard before attempting in-place operations.
- `PoolStats` exposes handle and byte counts for pool introspection.
