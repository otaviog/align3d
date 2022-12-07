use ndarray::prelude::*;

use std::cmp::Ordering;
use std::rc::Rc;

enum KdNode {
    Leaf {
        points: Array2<f32>,
        indices: Vec<usize>,
    },
    NonLeaf {
        middle_value: f32,
        left: Rc<KdNode>,
        right: Rc<KdNode>,
    },
}

pub struct KdTree {
    root: Rc<KdNode>,
}

impl KdTree {
    pub fn new(points: &Array2<f32>) -> Self {
        // Recursive creation.
        fn rec(points: &Array2<f32>, mut indices: Vec<usize>, depth: usize) -> KdNode {
            // Stop recursion if this should be a leaf node.
            if indices.len() < 10 {
                return KdNode::Leaf {
                    points: points.select(ndarray::Axis(0), &indices),
                    indices: indices,
                };
            }

            let k = depth % points.shape()[1];
            indices.sort_by(|idx1, idx2| {
                let a = points[[*idx1, k]];
                let b = points[[*idx2, k]];
                a.partial_cmp(&b).unwrap()
            });
            let mid = indices.len() / 2;

            KdNode::NonLeaf {
                middle_value: points[[indices[mid], k]],
                left: Rc::new(rec(
                    points,
                    indices[0..mid].iter().copied().collect(),
                    depth + 1,
                )),
                right: Rc::new(rec(
                    points,
                    indices[mid..].iter().copied().collect(),
                    depth + 1,
                )),
            }
        }

        let indices = Vec::from_iter(0..points.shape()[0]);
        KdTree {
            root: Rc::new(rec(points, indices, 0)),
        }
    }

    pub fn nearest(&self, queries: &Array2<f32>) -> Array1<usize> {
        let queries_shape = queries.shape();
        let point_dim = queries_shape[1];
        let mut nearest = Array1::from_elem((queries_shape[0],), 0);

        for (point_idx, point) in queries.rows().into_iter().enumerate() {
            let mut curr_node = &self.root;
            let mut depth = 0;

            loop {
                match curr_node.as_ref() {
                    KdNode::NonLeaf {
                        middle_value: mid,
                        left,
                        right,
                    } => {
                        let v = point[depth % point_dim];
                        curr_node = if v < *mid { &left } else { &right };
                        depth += 1;
                    }
                    KdNode::Leaf {
                        points: leaf_points,
                        indices: leaf_indices,
                    } => {
                        let diff = leaf_points - &point;
                        let v = diff
                            .rows()
                            .into_iter()
                            .map(|x| x.dot(&x).sqrt())
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .map(|(idx, _)| idx);

                        nearest[point_idx] = leaf_indices[v.unwrap_or(0)];
                        break;
                    }
                }
            }
        }

        nearest
    }
}

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    #[test]
    fn test_nearest() {
        let points = array![[1., 2., 3.], [2., 3., 4.], [5., 6., 7.], [8., 9., 1.]];
        let tree = super::KdTree::new(&points);

        let queries = array![
            [8., 9.1, 1.3],
            [5.1, 6.4, 7.],
            [1.5, 2.1, 3.3],
            [2.2, 3.1, 4.2]
        ];

        let found = tree.nearest(&queries);
        println!("{}", found);

        assert_eq!(found, array![3, 2, 0, 1]);
    }

    #[test]
    fn test_big_search() {
        let ordered_points =
            Array::from_shape_vec((500, 3), (0..500 * 3).map(|x| x as f32).collect()).unwrap();

        let mut random_indices = (0..500).collect::<Vec<usize>>();
        let seed: [u8; 32] = [5; 32];
        random_indices.shuffle(&mut SmallRng::from_seed(seed));
        let mut randomized_points = ordered_points.clone();
        for i in 0..500 as usize {
            randomized_points
                .slice_mut(s![random_indices[i], ..])
                .assign(&ordered_points.slice(s![i, ..]).view());
        }

        let tree = super::KdTree::new(&randomized_points);

        let found_indices = tree.nearest(&ordered_points);
        assert_eq!(Array::from_vec(random_indices), found_indices);
    }
}
