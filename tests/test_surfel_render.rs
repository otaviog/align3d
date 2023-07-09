use align3d::{
    bilateral::BilateralFilter,
    io::dataset::{RgbdDataset, SlamTbDataset},
    range_image::RangeImageBuilder,
    surfel::{SurfelBuilder, SurfelModel},
    viz::{node::MakeNode, GeoViewer, Manager},
};

fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let ribuilder = RangeImageBuilder::default()
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .with_normals(true)
        .pyramid_levels(1);
    let rgbd_frame = dataset.get(0).unwrap();
    let intrinsics = rgbd_frame.camera.clone();
    let ri_frame = ribuilder.build(rgbd_frame);

    let mut manager = Manager::default();
    let model = SurfelModel::new(&manager.memory_allocator, 500_000);

    let surfel_builder = SurfelBuilder::new(&intrinsics);
    let mut gpu_guard = model.lock_gpu();
    let mut writer = gpu_guard.get_writer();
    for (i, surfel) in surfel_builder
        .from_range_image(&ri_frame[0])
        .iter()
        .enumerate()
    {
        writer.update(i, surfel);
    }
    drop(writer);
    drop(gpu_guard);

    let node = model.vk_data.make_node(&mut manager);

    let mut geo_viewer = GeoViewer::from_manager(manager);
    geo_viewer.add_node(node);

    geo_viewer.run();
}
