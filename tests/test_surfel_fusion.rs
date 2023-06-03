use align3d::{
    bilateral::BilateralFilter,
    io::dataset::{RgbdDataset, SlamTbDataset},
    range_image::RangeImageBuilder,
    surfel::{self, RimageSurfelBuilder, SurfelFusion, SurfelModel},
    viz::{GeoViewer, Manager},
};

fn main() {
    let mut camera = FullCamera::from_simple_intrinsic(
        525.0,
        525.0,
        319.5,
        239.5,
        TransformBuilder::default()
            .axis_angle(
                UnitVector3::new_normalize(Vector3::new(4.0, 1.0, 0.0)),
                35.0_f32.to_radians(),
            )
            .build(),
        640,
        480,
    );

    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let ribuilder = RangeImageBuilder::default()
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .with_normals(true)
        .pyramid_levels(1);
    let rgbd_frame = dataset.get(0).unwrap();
    let ri_frame = ribuilder.build(rgbd_frame)[0];

    let manager = Manager::default();
    let model = SurfelModel::new(&manager.memory_allocator, 1000);
    let surfel_builder = RimageSurfelBuilder::new(rgbd_frame.get_pinhole_camera());
    let mut model_writer = model.write().unwrap();
    for surfel in surfel_builder.build(&ri_frame) {
        model.write().unwrap().add(surfel);
    }
    drop(model_writer);

    let mut geo_viewer = GeoViewer::from_manager(manager);
    geo_viewer.add(model.into_node());

    geo_viewer.run();
}
