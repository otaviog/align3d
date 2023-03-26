use align3d::icp::{IcpParams, Icp};
use align3d::range_image::RangeImage;
use align3d::io::core::RgbdDataset;
use align3d::pointcloud::PointCloud;

use align3d::io::slamtb_dataset::SlamTbDataset;
use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn icp_benchmark(c: &mut Criterion) {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let item = dataset.get_item(0).unwrap();

    let pcl0: PointCloud = {
        let mut pcl = RangeImage::from_rgbd_frame(&item);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let item = dataset.get_item(5).unwrap();
    let pcl1: PointCloud = {
        let mut pcl = RangeImage::from_rgbd_frame(&item);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let icp = Icp::new(
        IcpParams {
            max_iterations: 10,
            ..Default::default()
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
    targets = icp_benchmark
}

criterion_main!(benches);
