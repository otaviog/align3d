use align3d::icp::{IcpParams, ImageIcp};
use align3d::io::dataset::{RgbdDataset, SlamTbDataset};
use align3d::range_image::RangeImage;
use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn image_icp_benchmark(c: &mut Criterion) {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let item = dataset.get(0).unwrap();

    let image0 = {
        let mut image = RangeImage::from_rgbd_frame(&item);
        image.compute_intensity();
        image.compute_intensity_map();
        image.compute_normals();
        image
    };

    let item = dataset.get(5).unwrap();
    let image1 = {
        let mut image = RangeImage::from_rgbd_frame(&item);
        image.compute_intensity();
        image.compute_intensity_map();
        image.compute_normals();
        image
    };

    let icp = ImageIcp::new(
        IcpParams {
            max_iterations: 10,
            ..Default::default()
        },
        &image0,
    );

    c.bench_function("icp align", |b| {
        b.iter(|| {
            icp.align(&image1);
        });
    });
}

criterion_group! {
    name = benches;
    //targets = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = image_icp_benchmark
}

criterion_main!(benches);
