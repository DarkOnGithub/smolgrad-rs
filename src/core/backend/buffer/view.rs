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

    /// Calculate column-major (Fortran-contiguous) strides from shape
    pub fn compute_fortran_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in 1..shape.len() {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
        strides
    }

    /// Calculate row-major strides with memory alignment
    /// Alignment is applied to the innermost dimension's stride in bytes
    pub fn compute_aligned_strides(
        shape: &[usize],
        element_size: usize,
        alignment: usize,
    ) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; shape.len()];

        let last_dim = shape.len() - 1;
        let inner_bytes = shape[last_dim] * element_size;
        let aligned_bytes = (inner_bytes + alignment - 1) / alignment * alignment;
        let aligned_inner_elements = aligned_bytes / element_size;

        if shape.len() > 1 {
            strides[last_dim - 1] = aligned_inner_elements;
            for i in (0..last_dim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        strides
    }

    /// Calculate the total storage size in elements needed for aligned strides
    pub fn compute_aligned_storage_size(
        shape: &[usize],
        element_size: usize,
        alignment: usize,
    ) -> usize {
        if shape.is_empty() {
            return 0;
        }
        let strides = Self::compute_aligned_strides(shape, element_size, alignment);
        if shape.len() == 1 {
            shape[0]
        } else {
            strides[0] * shape[0]
        }
    }

    /// Create a Fortran-contiguous (column-major) view from shape
    pub fn fortran_contiguous(shape: Vec<usize>) -> Self {
        let strides = Self::compute_fortran_strides(&shape);
        let len = shape.iter().product();
        BufferView {
            shape,
            strides,
            offset: 0,
            len,
        }
    }

    /// Create a view with aligned strides for GPU performance
    pub fn aligned(shape: Vec<usize>, element_size: usize, alignment: usize) -> Self {
        let strides = Self::compute_aligned_strides(&shape, element_size, alignment);
        let len = shape.iter().product();
        BufferView {
            shape,
            strides,
            offset: 0,
            len,
        }
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

    /// Broadcast this view to a target shape
    /// Returns a new view with strides set to 0 for broadcasted dimensions
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Option<Self> {
        if target_shape.len() < self.ndim() {
            return None;
        }

        let rank_diff = target_shape.len() - self.ndim();
        let mut new_shape = vec![1; rank_diff];
        new_shape.extend_from_slice(&self.shape);
        let mut new_strides = vec![0; rank_diff];
        new_strides.extend_from_slice(&self.strides);

        for i in 0..target_shape.len() {
            if new_shape[i] == target_shape[i] {
            } else if new_shape[i] == 1 {
                new_shape[i] = target_shape[i];
                new_strides[i] = 0;
            } else {
                return None;
            }
        }

        let new_len = new_shape.iter().product();
        Some(BufferView {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            len: new_len,
        })
    }

    /// Check if this view can be broadcast to the target shape
    pub fn is_broadcastable_to(&self, target_shape: &[usize]) -> bool {
        self.broadcast_to(target_shape).is_some()
    }

    /// Compute the broadcast shape for two views
    pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Option<Vec<usize>> {
        let max_rank = shape_a.len().max(shape_b.len());
        let mut result = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let a = if i < shape_a.len() {
                shape_a[shape_a.len() - 1 - i]
            } else {
                1
            };
            let b = if i < shape_b.len() {
                shape_b[shape_b.len() - 1 - i]
            } else {
                1
            };

            if a == b {
                result.push(a);
            } else if a == 1 {
                result.push(b);
            } else if b == 1 {
                result.push(a);
            } else {
                return None;
            }
        }

        result.reverse();
        Some(result)
    }
}

/// Metadata for broadcasting operations in kernels
#[derive(Debug, Clone)]
pub struct BroadcastMetadata {
    pub output_shape: Vec<usize>,
    /// Strides for input A (0 means broadcast)
    pub strides_a: Vec<usize>,
    /// Strides for input B (0 means broadcast)
    pub strides_b: Vec<usize>,
    /// Whether input A needs broadcasting
    pub a_broadcasts: bool,
    /// Whether input B needs broadcasting
    pub b_broadcasts: bool,
}

impl BroadcastMetadata {
    pub fn compute(view_a: &BufferView, view_b: &BufferView) -> Option<Self> {
        let output_shape = BufferView::broadcast_shape(&view_a.shape, &view_b.shape)?;

        let broadcasted_a = view_a.broadcast_to(&output_shape)?;
        let broadcasted_b = view_b.broadcast_to(&output_shape)?;

        let a_broadcasts = broadcasted_a.strides.iter().any(|&s| s == 0);
        let b_broadcasts = broadcasted_b.strides.iter().any(|&s| s == 0);

        Some(BroadcastMetadata {
            output_shape,
            strides_a: broadcasted_a.strides,
            strides_b: broadcasted_b.strides,
            a_broadcasts,
            b_broadcasts,
        })
    }

    /// Total number of elements in the output
    pub fn output_len(&self) -> usize {
        self.output_shape.iter().product()
    }
}
