use std::sync::{Arc, Mutex};

use align3d::{
    bilateral::BilateralFilter,
    io::dataset::{RgbdDataset, SlamTbDataset},
    range_image::RangeImageBuilder,
    surfel::{RimageSurfelBuilder, SurfelModel},
    viz::{GeoViewer, Manager, node::MakeNode},
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
    let model = Arc::new(Mutex::new(SurfelModel::new(&manager.memory_allocator, 500_000)));
    
    let surfel_builder = RimageSurfelBuilder::new(&intrinsics);
    let mut model_lock = model.lock().unwrap();
    let mut model_writer = model_lock.write().unwrap();
     
    for surfel in surfel_builder.build_from_rimage(&ri_frame[0]) {
        model_writer.add(&surfel);
    }
    drop(model_writer);
    drop(model_lock);
    
    let node = model.make_node(&mut manager);

    let mut geo_viewer = GeoViewer::from_manager(manager);
    geo_viewer.add_node(node);

    geo_viewer.run();
}
