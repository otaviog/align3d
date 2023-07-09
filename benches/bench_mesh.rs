use align3d::io::read_off;
use align3d::mesh::compute_normals;
use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let geometry = read_off("tests/data/teapot.off").unwrap();
    c.bench_function("compute_normals", |b| {
        b.iter(|| {
            compute_normals(
                &geometry.points.view(),
                &geometry.faces.as_ref().unwrap().view(),
            )
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
