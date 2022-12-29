use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub enum Array1Recycle {
    Empty,
    Recycle(Array1<usize>),
}

impl Array1Recycle {
    pub fn get(self, required_dim: usize) -> Array1<usize> {
        match self {
            Self::Empty => Array1::<usize>::zeros(required_dim),
            Self::Recycle(current) => {
                if current.dim() != required_dim {
                    Array1::<usize>::zeros(required_dim)
                } else {
                    current
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        if let Self::Empty = self {
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
pub enum Array2Recycle<T> {
    Empty,
    Recycle(Array2<T>),
}

impl<T> Array2Recycle<T> where T: num::Zero + Clone {
    pub fn get(self, required_dim: (usize, usize)) -> Array2<T> {
        match self {
            Self::Empty => Array2::<T>::zeros(required_dim),
            Self::Recycle(current) => {
                if current.dim() != required_dim {
                    Array2::<T>::zeros(required_dim)
                } else {
                    current
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        if let Self::Empty = self {
            true
        } else {
            false
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::Array1Recycle;

    #[test]
    fn test_reuse() {
        let mut r = Array1Recycle::Empty;

        r = Array1Recycle::Recycle(r.get(1000));
        {
            if let Array1Recycle::Recycle(rr) = &r {
                assert_eq!(rr.dim(), 1000);
            } else {
                assert!(false, "reuse still empty")
            }
        }

        r = Array1Recycle::Recycle(r.get(1000));
        r = Array1Recycle::Recycle(r.get(2000));
    }
}
