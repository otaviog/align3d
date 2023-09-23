use std::time::Instant;

use align3d::kdtree::R3dTree;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::Vector3;
use ndarray::Array1;
use pprof::criterion::{Output, PProfProfiler};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn kdtree_benchmark(c: &mut Criterion) {
    const N: usize = 500000;
    const SEED: u64 = 10;
    let mut rng = StdRng::seed_from_u64(SEED);

    let db_points = Array1::from_shape_fn(N, |_| {
        Vector3::new(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
        )
    });

    c.bench_function("R3dTree creation", |b| {
        b.iter(|| R3dTree::new(&db_points.view()));
    });

    c.bench_function("R3dTree search", |b| {
        let tree = R3dTree::new(&db_points.view());
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                let rand_vector = Vector3::new(
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                );
                tree.nearest(&rand_vector);
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
