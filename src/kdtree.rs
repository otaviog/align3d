use nalgebra::Vector3;
use ndarray::prelude::*;

enum Node {
    Leaf {
        points: Array1<Vector3<f32>>,
        indices: Vec<usize>,
    },
    NonLeaf {
        middle_value: f32,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// KdTree for fast nearest neighbor search.
pub struct R3dTree {
    root: Box<Node>,
}

impl R3dTree {
    /// Create a new KdTree from a set of points.
    /// The points are stored in a 2D array, where each row is a point.
    ///
    /// # Arguments
    ///
    /// * points - 2D array of points.
    pub fn new(points: &ArrayView1<Vector3<f32>>) -> Self {
        // Recursive creation.
        fn rec(points: &ArrayView1<Vector3<f32>>, mut indices: Vec<usize>, depth: usize) -> Node {
            // Stop recursion if this should be a leaf node.
            if indices.len() <= 16 {
                return Node::Leaf {
                    points: points.select(ndarray::Axis(0), &indices),
                    indices,
                };
            }

            let k = depth % 3;
            indices.sort_by(|idx1, idx2| {
                let a = points[*idx1][k];
                let b = points[*idx2][k];
                a.partial_cmp(&b).unwrap()
            });

            let mid = indices.len() / 2;
            Node::NonLeaf {
                middle_value: points[indices[mid]][k],
                left: Box::new(rec(points, indices[0..mid].to_vec(), depth + 1)),
                right: Box::new(rec(points, indices[mid..].to_vec(), depth + 1)),
            }
        }

        let indices = Vec::from_iter(0..points.shape()[0]);
        Self {
            root: Box::new(rec(points, indices, 0)),
        }
    }

    /// Find the nearest neighbor to a query point. This version is for 3D points only.
    ///
    /// # Arguments
    ///
    /// * point - The query point.
    ///
    /// # Returns
    ///
    /// A tuple containing the index of the nearest neighbor and the distance to it.
    pub fn nearest(&self, point: &Vector3<f32>) -> (usize, f32) {
        let mut curr_node = &self.root;
        let mut current_dim = 0;

        loop {
            match curr_node.as_ref() {
                Node::NonLeaf {
                    middle_value: mid,
                    left,
                    right,
                } => {
                    curr_node = if point[current_dim] < *mid {
                        left
                    } else {
                        right
                    };
                    current_dim = (current_dim + 1) % 3;
                }
                Node::Leaf {
                    points: leaf_points,
                    indices,
                } => {
                    let mut min_dist = f32::MAX;
                    let mut min_idx = 0;
                    for (idx, leaf_point) in leaf_points.iter().enumerate() {
                        let leaf_point = Vector3::new(leaf_point[0], leaf_point[1], leaf_point[2]);
                        let dist = (point - leaf_point).norm_squared();
                        if dist < min_dist {
                            min_dist = dist;
                            min_idx = idx;
                        }
                    }
                    return (indices[min_idx], min_dist);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::kdtree::R3dTree;
    use crate::unit_test::access::UnflattenVector3;
    use nalgebra::Vector3;
    use ndarray::prelude::*;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    #[test]
    fn should_find_nearest_points() {
        let points = array![[1., 2., 3.], [2., 3., 4.], [5., 6., 7.], [8., 9., 1.]]
            .unflatten_vector3()
            .unwrap();
        let tree = R3dTree::new(&points.view());

        let queries = array![
            [8., 9.1, 1.3],
            [5.1, 6.4, 7.],
            [1.5, 2.1, 3.3],
            [2.2, 3.1, 4.2]
        ];

        for (query, expected) in queries.outer_iter().zip(&[3, 2, 0, 1]) {
            let query = Vector3::new(query[0], query[1], query[2]);
            let (idx, _) = tree.nearest(&query);
            assert_eq!(idx, *expected);
        }
    }

    #[test]
    fn should_find_nearest_points_big() {
        let ordered_points =
            Array::from_shape_vec((500, 3), (0..500 * 3).map(|x| x as f32).collect()).unwrap();

        let (random_indices, randomized_points) = {
            let mut random_indices = (0..500).collect::<Vec<usize>>();
            let seed: [u8; 32] = [5; 32];
            random_indices.shuffle(&mut SmallRng::from_seed(seed));

            let mut randomized_points = ordered_points.clone();
            for (i, rand_index) in random_indices.iter().enumerate().take(500_usize) {
                randomized_points
                    .slice_mut(s![*rand_index, ..])
                    .assign(&ordered_points.slice(s![i, ..]).view());
            }
            (
                random_indices,
                randomized_points.unflatten_vector3().unwrap(),
            )
        };

        let tree = R3dTree::new(&randomized_points.view());

        for (query, expected) in ordered_points.outer_iter().zip(random_indices.iter()) {
            let query = Vector3::new(query[0], query[1], query[2]);
            let (idx, _) = tree.nearest(&query);
            assert_eq!(idx, *expected);
        }
    }

    #[test]
    fn bench_nearest() {
        const N: usize = 500_000;
        let ordered_points =
            Array::from_shape_vec((N, 3), (0..N * 3).map(|x| x as f32).collect()).unwrap();

        let randomized_points = {
            let mut random_indices = (0..N).collect::<Vec<usize>>();
            let seed: [u8; 32] = [5; 32];
            random_indices.shuffle(&mut SmallRng::from_seed(seed));

            let mut randomized_points = ordered_points.clone();
            for (i, rand_index) in random_indices.iter().enumerate().take(N) {
                randomized_points
                    .slice_mut(s![*rand_index, ..])
                    .assign(&ordered_points.slice(s![i, ..]).view());
            }
            randomized_points
                .slice_move(s![0..5000, ..])
                .unflatten_vector3()
                .unwrap()
        };

        let tree = R3dTree::new(&randomized_points.view());

        let mut sum_millis = 0;
        const M: usize = 10;
        for _ in 0..M {
            let start = Instant::now();
            for point in ordered_points.outer_iter() {
                let point = Vector3::new(point[0], point[1], point[2]);
                tree.nearest(&point);
            }
            sum_millis += start.elapsed().as_millis();
        }

        println!("Mean time: {}", sum_millis as f64 / M as f64);
    }
}
