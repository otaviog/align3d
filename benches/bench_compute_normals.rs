use align3d::io::dataset::{RgbdDataset, SlamTbDataset};
use align3d::range_image::RangeImage;
use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn compute_normals_benchmark(c: &mut Criterion) {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let item = dataset.get(0).unwrap();

    c.bench_function("compute_normals", |b| {
        let mut image = RangeImage::from_rgbd_frame(&item);
        b.iter(|| {
            image.compute_normals();
        });
    });
}

criterion_group! {
    name = benches;
    //targets = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = compute_normals_benchmark
}

criterion_main!(benches);
