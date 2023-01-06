use align3d::{bilateral::{BilateralFilter, BilateralGrid}, Array2Recycle};

use criterion::{criterion_group, criterion_main, Criterion};
use nshare::ToNdarray2;

fn criterion_benchmark(c: &mut Criterion) {
    let bloei_luma16 = {
        let mut image = image::io::Reader::open("tests/data/images/bloei.jpg")
            .unwrap()
            .decode()
            .unwrap()
            .into_luma16()
            .into_ndarray2();

        image.iter_mut().for_each(|v| {
            *v /= std::u16::MAX / 5000;
        });
        image
    };
    c.bench_function("bilateral grid creation", |b| {
        b.iter(|| {
            BilateralGrid::from_image(&bloei_luma16, 4.5, 30.0);
        });
    });
    c.bench_function("bilateral slice operation", |b| {
        let mut grid = BilateralGrid::from_image(&bloei_luma16, 4.5, 30.0);
        let mut dst_image = bloei_luma16.clone();
        grid.normalize();
        b.iter(|| {
            grid.slice(&bloei_luma16, &mut dst_image);
        });
    });
    c.bench_function("bilateral filter", |b| {
        b.iter(|| {
            BilateralFilter::<u16>::default().filter(&bloei_luma16, Array2Recycle::Empty);
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
