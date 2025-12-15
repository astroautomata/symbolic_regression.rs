use num_traits::Float;
use std::sync::Arc;

pub trait LossFn<T: Float>: Send + Sync {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T;
    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]);
}

pub type LossObject<T> = Arc<dyn LossFn<T> + Send + Sync>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LossKind {
    Mse,
    Mae,
    Rmse,
    Huber { delta: f64 },
}

impl LossKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mse" => Some(Self::Mse),
            "mae" => Some(Self::Mae),
            "rmse" => Some(Self::Rmse),
            "huber" => Some(Self::Huber { delta: 1.0 }),
            _ => None,
        }
    }
}

pub fn make_loss<T: Float>(kind: LossKind) -> LossObject<T> {
    match kind {
        LossKind::Mse => mse::<T>(),
        LossKind::Mae => mae::<T>(),
        LossKind::Rmse => rmse::<T>(),
        LossKind::Huber { delta } => huber::<T>(delta),
    }
}

#[derive(Clone, Debug, Default)]
pub struct Mse;

impl<T: Float> LossFn<T> for Mse {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        assert_eq!(yhat.len(), y.len());
        match w {
            None => {
                let n = T::from(y.len()).unwrap();
                yhat.iter()
                    .copied()
                    .zip(y.iter().copied())
                    .map(|(a, b)| {
                        let r = a - b;
                        r * r
                    })
                    .fold(T::zero(), |acc, v| acc + v)
                    / n
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    return T::zero();
                }
                yhat.iter()
                    .copied()
                    .zip(y.iter().copied())
                    .zip(w.iter().copied())
                    .map(|((a, b), wi)| {
                        let r = a - b;
                        wi * r * r
                    })
                    .fold(T::zero(), |acc, v| acc + v)
                    / sum_w
            }
        }
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        assert_eq!(yhat.len(), y.len());
        assert_eq!(out.len(), y.len());
        match w {
            None => {
                let inv = T::from(y.len()).unwrap().recip();
                let scale = T::from(2.0).unwrap() * inv;
                for ((o, a), b) in out.iter_mut().zip(yhat.iter()).zip(y.iter()) {
                    *o = scale * (*a - *b);
                }
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    out.fill(T::zero());
                    return;
                }
                let scale = T::from(2.0).unwrap() / sum_w;
                for (((o, a), b), wi) in out.iter_mut().zip(yhat.iter()).zip(y.iter()).zip(w.iter())
                {
                    *o = scale * (*wi) * (*a - *b);
                }
            }
        }
    }
}

pub fn mse<T: Float>() -> LossObject<T> {
    Arc::new(Mse)
}

#[derive(Clone, Debug, Default)]
pub struct Mae;

impl<T: Float> LossFn<T> for Mae {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        assert_eq!(yhat.len(), y.len());
        match w {
            None => {
                let n = T::from(y.len()).unwrap();
                yhat.iter()
                    .copied()
                    .zip(y.iter().copied())
                    .map(|(a, b)| (a - b).abs())
                    .fold(T::zero(), |acc, v| acc + v)
                    / n
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    return T::zero();
                }
                yhat.iter()
                    .copied()
                    .zip(y.iter().copied())
                    .zip(w.iter().copied())
                    .map(|((a, b), wi)| wi * (a - b).abs())
                    .fold(T::zero(), |acc, v| acc + v)
                    / sum_w
            }
        }
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        assert_eq!(yhat.len(), y.len());
        assert_eq!(out.len(), y.len());
        match w {
            None => {
                let scale = T::from(y.len()).unwrap().recip();
                for ((o, a), b) in out.iter_mut().zip(yhat.iter()).zip(y.iter()) {
                    let r = *a - *b;
                    *o = if r > T::zero() {
                        scale
                    } else if r < T::zero() {
                        -scale
                    } else {
                        T::zero()
                    };
                }
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    out.fill(T::zero());
                    return;
                }
                let inv = sum_w.recip();
                for (((o, a), b), wi) in out.iter_mut().zip(yhat.iter()).zip(y.iter()).zip(w.iter())
                {
                    let r = *a - *b;
                    let s = if r > T::zero() {
                        T::one()
                    } else if r < T::zero() {
                        -T::one()
                    } else {
                        T::zero()
                    };
                    *o = (*wi) * s * inv;
                }
            }
        }
    }
}

pub fn mae<T: Float>() -> LossObject<T> {
    Arc::new(Mae)
}

#[derive(Clone, Debug, Default)]
pub struct Rmse;

impl<T: Float> LossFn<T> for Rmse {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        mse::<T>().loss(yhat, y, w).sqrt()
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        assert_eq!(yhat.len(), y.len());
        assert_eq!(out.len(), y.len());

        let m = mse::<T>().loss(yhat, y, w);
        let rmse = m.sqrt();
        if rmse == T::zero() || !rmse.is_finite() {
            out.fill(T::zero());
            return;
        }

        match w {
            None => {
                let inv = T::from(y.len()).unwrap().recip();
                let scale = inv / rmse;
                for ((o, a), b) in out.iter_mut().zip(yhat.iter()).zip(y.iter()) {
                    *o = scale * (*a - *b);
                }
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    out.fill(T::zero());
                    return;
                }
                let scale = sum_w.recip() / rmse;
                for (((o, a), b), wi) in out.iter_mut().zip(yhat.iter()).zip(y.iter()).zip(w.iter())
                {
                    *o = scale * (*wi) * (*a - *b);
                }
            }
        }
    }
}

pub fn rmse<T: Float>() -> LossObject<T> {
    Arc::new(Rmse)
}

#[derive(Clone, Debug)]
pub struct Huber {
    pub delta: f64,
}

impl<T: Float> LossFn<T> for Huber {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        assert_eq!(yhat.len(), y.len());
        let delta = T::from(self.delta).unwrap_or_else(T::one);
        let half = T::from(0.5).unwrap();
        match w {
            None => {
                let n = T::from(y.len()).unwrap();
                yhat.iter()
                    .copied()
                    .zip(y.iter().copied())
                    .map(|(a, b)| {
                        let r = a - b;
                        let ar = r.abs();
                        if ar <= delta {
                            half * r * r
                        } else {
                            delta * (ar - half * delta)
                        }
                    })
                    .fold(T::zero(), |acc, v| acc + v)
                    / n
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    return T::zero();
                }
                yhat.iter()
                    .copied()
                    .zip(y.iter().copied())
                    .zip(w.iter().copied())
                    .map(|((a, b), wi)| {
                        let r = a - b;
                        let ar = r.abs();
                        let l = if ar <= delta {
                            half * r * r
                        } else {
                            delta * (ar - half * delta)
                        };
                        wi * l
                    })
                    .fold(T::zero(), |acc, v| acc + v)
                    / sum_w
            }
        }
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        assert_eq!(yhat.len(), y.len());
        assert_eq!(out.len(), y.len());

        let delta = T::from(self.delta).unwrap_or_else(T::one);
        match w {
            None => {
                let scale = T::from(y.len()).unwrap().recip();
                for ((o, a), b) in out.iter_mut().zip(yhat.iter()).zip(y.iter()) {
                    let r = *a - *b;
                    let ar = r.abs();
                    let g = if ar <= delta {
                        r
                    } else if r > T::zero() {
                        delta
                    } else if r < T::zero() {
                        -delta
                    } else {
                        T::zero()
                    };
                    *o = scale * g;
                }
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    out.fill(T::zero());
                    return;
                }
                let inv = sum_w.recip();
                for (((o, a), b), wi) in out.iter_mut().zip(yhat.iter()).zip(y.iter()).zip(w.iter())
                {
                    let r = *a - *b;
                    let ar = r.abs();
                    let g = if ar <= delta {
                        r
                    } else if r > T::zero() {
                        delta
                    } else if r < T::zero() {
                        -delta
                    } else {
                        T::zero()
                    };
                    *o = (*wi) * g * inv;
                }
            }
        }
    }
}

pub fn huber<T: Float>(delta: f64) -> LossObject<T> {
    Arc::new(Huber { delta })
}
