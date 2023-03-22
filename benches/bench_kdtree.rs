use std::time::Instant;

use align3d::kdtree::KdTree;
use align3d::Array1Recycle;
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{s, Array};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use pprof::criterion::{PProfProfiler, Output};

fn kdtree_benchmark(c: &mut Criterion) {
    const N: usize = 500000;

    let ordered_points =
        Array::from_shape_vec((N, 3), (0..N * 3).map(|x| x as f32).collect()).unwrap();

    let randomized_points = {
        let mut random_indices = (0..N).collect::<Vec<usize>>();
        let seed: [u8; 32] = [5; 32];
        random_indices.shuffle(&mut SmallRng::from_seed(seed));

        let mut randomized_points = ordered_points.clone();
        for i in 0..N as usize {
            randomized_points
                .slice_mut(s![random_indices[i], ..])
                .assign(&ordered_points.slice(s![i, ..]).view());
        }
        randomized_points
    };

    c.bench_function("kdtree creation", |b| {
        b.iter(|| KdTree::new(&randomized_points.view()));
    });

    c.bench_function("kdtree search", |b| {
        let tree = KdTree::new(&randomized_points.view());
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                tree.nearest::<3>(&ordered_points.view(), Array1Recycle::Empty);
            }
            start.elapsed()
        });
    });
}

criterion_group! {
    name = benches;
    //targets = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = kdtree_benchmark
}

criterion_main!(benches);
