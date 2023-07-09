use align3d::bilateral::BilateralFilter;
use align3d::io::dataset::{RgbdDataset, SlamTbDataset};
use align3d::range_image::RangeImageBuilder;
use align3d::surfel::{SurfelFusion, SurfelFusionParameters, SurfelModel};
use align3d::viz::Manager;
use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn integrate_benchmark(c: &mut Criterion) {
    const NUM_ITER: usize = 4;
    let manager = Manager::default();
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

    let rimage_builder = RangeImageBuilder::default()
        .with_normals(true)
        .with_bilateral_filter(Some(BilateralFilter::default()));
    let mut rimages = Vec::new();
    let mut cameras = Vec::new();
    for i in 0..NUM_ITER {
        let frame = dataset.get(i).unwrap();
        let camera = frame.get_pinhole_camera().unwrap();

        let range_image = rimage_builder.build(frame)[0].clone();

        rimages.push(range_image);
        cameras.push(camera);
    }

    c.bench_function("Fusion integration", |b| {
        let mut model = SurfelModel::new(&manager.memory_allocator, 1_600_000);

        let mut surfel_fusion = SurfelFusion::new(640, 480, 4, SurfelFusionParameters::default());
        b.iter(|| {
            for (range_image, camera) in rimages.iter().zip(cameras.iter()) {
                surfel_fusion.integrate(&mut model, range_image, camera);
            }
        });
    });
}

criterion_group! {
    name = benches;

    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = integration_benchmark
}

criterion_main!(benches);
