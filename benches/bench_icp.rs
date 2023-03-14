use std::time::Instant;

use align3d::icp::{ICPParams, ICP};
use align3d::imagepointcloud::ImagePointCloud;
use align3d::io::core::RGBDDataset;
use align3d::pointcloud::PointCloud;

use align3d::io::slamtb::SlamTbDataset;
use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn unordered_icp_benchmark(c: &mut Criterion) {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let item = dataset.get_item(0).unwrap();

    let pcl0: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(item.0, item.1);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let item = dataset.get_item(5).unwrap();
    let pcl1: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(item.0, item.1);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let icp = ICP::new(
        ICPParams {
            max_iterations: 10,
            weight: 0.01,
        },
        &pcl0,
    );

    c.bench_function("icp align", |b| {
        b.iter(|| {
            icp.align(&pcl1);
        });
    });
}

criterion_group! {
    name = benches;
    //targets = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = unordered_icp_benchmark
}

criterion_main!(benches);
