use dynamic_expressions::XColumns;
use dynamic_expressions::utils::ZipEq;
use fastrand::Rng;
use ndarray::{ArcArray2, Array1, Array2};
use num_traits::Float;

use crate::random::usize_range;
use crate::units::Units;

#[derive(Copy, Clone, Debug)]
pub struct TaggedDataset<'a, T: Float> {
    data: &'a Dataset<T>,
    /// `data.x`, scanned once for non-finite values; the borrow keeps the scan current.
    x_columns: XColumns<'a, T>,
    pub baseline_loss: Option<T>,
    /// Row count of the *full* dataset this tag refers to. Equal to `data.n_rows` when the tag
    /// wraps the full dataset; when it wraps a batch buffer, this is the underlying full size.
    /// Mirrors SymbolicRegression.jl's `SubDataset` / `BasicDataset` distinction.
    pub full_n_rows: usize,
}

impl<'a, T: Float> TaggedDataset<'a, T> {
    pub fn new(data: &'a Dataset<T>, baseline_loss: Option<T>) -> Self {
        Self::for_batch(data, baseline_loss, data.n_rows)
    }
    pub fn for_batch(batch: &'a Dataset<T>, baseline_loss: Option<T>, full_n_rows: usize) -> Self {
        Self {
            data: batch,
            x_columns: XColumns::checked(batch.x.view()),
            baseline_loss,
            full_n_rows,
        }
    }
    pub fn data(&self) -> &'a Dataset<T> {
        self.data
    }
    /// The feature columns to evaluate expressions on.
    pub fn x_columns(&self) -> XColumns<'a, T> {
        self.x_columns
    }
    /// Matches SymbolicRegression.jl `dataset_fraction`: ratio of `data.n_rows` to the full size.
    pub fn dataset_fraction(&self) -> f64 {
        if self.full_n_rows == 0 {
            1.0
        } else {
            (self.data.n_rows as f64) / (self.full_n_rows as f64)
        }
    }
}

impl<'a, T: Float> std::ops::Deref for TaggedDataset<'a, T> {
    type Target = Dataset<T>;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

#[derive(Clone, Debug)]
pub struct Dataset<T: Float> {
    /// Column-major contiguous data with shape `(n_features, n_rows)` for vectorization over rows,
    /// shared by the datasets of a multi-output search.
    pub x: ArcArray2<T>,
    /// Target vector with length `n_rows`.
    pub y: Array1<T>,
    pub n_features: usize,
    pub n_rows: usize,
    pub weights: Option<Array1<T>>,
    pub variable_names: Vec<String>,
    /// Weighted mean of `y` (or unweighted mean when no weights).
    pub avg_y: T,
    /// SymbolicRegression.jl's `X_units` and `y_units`; see [`Self::with_units`].
    pub units: Option<Units>,
}

impl<T: Float> Dataset<T> {
    fn build_dataset(
        x: ArcArray2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
        avg_y: Option<T>,
    ) -> Self {
        let y = contiguous(y);
        let weights = weights.map(contiguous);
        let (n_features, n_rows) = x.dim();
        assert!(
            y.len() == n_rows,
            "Number of samples in `x` (x.ncols() = {n_rows}) does not match the length of `y` ({}). `x` should have shape (n_features, n_rows).",
            y.len()
        );
        if let Some(w) = &weights {
            assert!(
                w.len() == n_rows,
                "Number of samples in `x` (x.ncols() = {n_rows}) does not match the length of `weights` ({}).",
                w.len()
            );
        }

        let avg_y = avg_y.unwrap_or_else(|| {
            Self::compute_avg_y(y.as_slice().unwrap(), weights.as_ref().map(|w| w.as_slice().unwrap()))
        });

        Self {
            x,
            y,
            n_features,
            n_rows,
            weights,
            variable_names,
            avg_y,
            units: None,
        }
    }

    pub fn new(x: Array2<T>, y: Array1<T>) -> Self {
        Self::build_dataset(shared(x), y, None, Vec::new(), None)
    }

    pub fn with_weights_and_names(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
    ) -> Self {
        Self::build_dataset(shared(x), y, weights, variable_names, None)
    }

    /// One dataset per row of `y`, whose shape is `(n_outputs, n_rows)`, as SymbolicRegression.jl
    /// splits a matrix target for a multi-output search: the datasets share `x` and
    /// `variable_names`, and row `j` of `weights` weights output `j`.
    ///
    /// ```
    /// use ndarray::{Array2, array};
    /// use symbolic_regression::Dataset;
    ///
    /// let x = Array2::<f64>::zeros((1, 3));
    /// let datasets = Dataset::multi_output(x, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], None, Vec::new());
    /// assert_eq!(datasets[1].avg_y, 5.0);
    /// ```
    pub fn multi_output(
        x: Array2<T>,
        y: Array2<T>,
        weights: Option<Array2<T>>,
        variable_names: Vec<String>,
    ) -> Vec<Self> {
        if let Some(w) = &weights {
            assert_eq!(w.dim(), y.dim(), "`weights` must have the shape of `y`");
        }
        let x = shared(x);
        y.rows()
            .into_iter()
            .enumerate()
            .map(|(j, yj)| {
                let wj = weights.as_ref().map(|w| w.row(j).to_owned());
                Self::build_dataset(x.clone(), yj.to_owned(), wj, variable_names.clone(), None)
            })
            .collect()
    }

    /// Attaches units to the features and target, written as PySR and SymbolicRegression.jl take
    /// them (`"kg"`, `"m/s^2"`, `"Constants.G"`). The search then adds
    /// `Options::dimensional_constraint_penalty` to the loss of each dimensionally inconsistent
    /// expression. Target units alone make every feature dimensionless.
    ///
    /// ```
    /// use ndarray::{Array1, Array2};
    /// use symbolic_regression::Dataset;
    ///
    /// let dataset = Dataset::<f64>::new(Array2::zeros((2, 3)), Array1::zeros(3))
    ///     .with_units(Some(&["m", "s"]), Some("m/s"))
    ///     .unwrap();
    /// assert_eq!(dataset.units.unwrap().x[0].label(), "[m]");
    /// ```
    pub fn with_units<S: AsRef<str>>(mut self, x_units: Option<&[S]>, y_units: Option<&str>) -> Result<Self, String> {
        self.units = Units::parse(self.n_features, x_units, y_units)?;
        Ok(self)
    }

    pub fn y_slice(&self) -> &[T] {
        self.y.as_slice().expect("y is contiguous")
    }

    pub fn weights_slice(&self) -> Option<&[T]> {
        self.weights
            .as_ref()
            .map(|w| w.as_slice().expect("weights are contiguous"))
    }

    pub fn compute_avg_y(y: &[T], weights: Option<&[T]>) -> T {
        if y.is_empty() {
            return T::zero();
        }
        match weights {
            None => {
                let n = T::from(y.len()).unwrap();
                y.iter().copied().fold(T::zero(), |a, b| a + b) / n
            }
            Some(w) => {
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                y.iter()
                    .copied()
                    .zip_eq(w.iter().copied())
                    .map(|(yi, wi)| yi * wi)
                    .fold(T::zero(), |a, b| a + b)
                    / sum_w
            }
        }
    }

    pub fn make_batch_buffer(full: &Dataset<T>, batch_size: usize) -> Dataset<T> {
        if full.n_rows == 0 {
            panic!("Cannot batch from an empty dataset (n_rows = 0).");
        }
        let batch_size = batch_size.max(1);
        let x = Array2::<T>::zeros((full.n_features, batch_size));
        let y = Array1::<T>::zeros(batch_size);
        let weights = full.weights.as_ref().map(|_| Array1::<T>::zeros(batch_size));
        let batch = Self::build_dataset(
            x.into_shared(),
            y,
            weights,
            full.variable_names.clone(),
            Some(full.avg_y),
        );
        Dataset {
            units: full.units.clone(),
            ..batch
        }
    }

    pub fn resample_from(&mut self, full: &Dataset<T>, rng: &mut Rng) {
        if full.n_rows == 0 {
            panic!("Cannot batch from an empty dataset (n_rows = 0).");
        }
        assert_eq!(self.n_features, full.n_features);
        assert_eq!(self.x.dim().0, self.n_features);
        assert_eq!(self.x.dim().1, self.n_rows);
        assert_eq!(self.y.len(), self.n_rows);
        if let Some(w) = &self.weights {
            assert_eq!(w.len(), self.n_rows);
            assert!(full.weights.is_some());
        } else {
            assert!(full.weights.is_none());
        }

        let mut x = self.x.view_mut();
        for (dst_col, src_idx) in (0..self.n_rows).map(|i| (i, usize_range(rng, 0..full.n_rows))) {
            x.column_mut(dst_col).assign(&full.x.column(src_idx));
            self.y[dst_col] = full.y[src_idx];
            if let (Some(dst), Some(src)) = (self.weights.as_mut(), full.weights.as_ref()) {
                dst[dst_col] = src[src_idx];
            }
        }
    }
}

/// `x`, copied into standard layout only if it is not in it already, as a shared array.
fn shared<T: Clone>(x: Array2<T>) -> ArcArray2<T> {
    if x.is_standard_layout() {
        x.into_shared()
    } else {
        x.as_standard_layout().into_owned().into_shared()
    }
}

/// `a`, copied into a contiguous array only if it is not one already, so it has a slice view.
fn contiguous<T: Clone>(a: Array1<T>) -> Array1<T> {
    if a.is_standard_layout() {
        a
    } else {
        a.as_standard_layout().into_owned()
    }
}
