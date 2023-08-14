use std::thread;

use align3d::{
    bilateral::BilateralFilter,
    io::dataset::{RgbdDataset, SlamTbDataset},
    range_image::RangeImageBuilder,
    surfel::{SurfelFusion, SurfelFusionParameters, SurfelModel},
    viz::{node::MakeNode, GeoViewer, Manager},
};

fn main() {
    let mut manager = Manager::default();
    let mut model = SurfelModel::new(&manager.memory_allocator, 4_000_000);

    let render_model = model.vk_data.clone();
    let node = render_model.make_node(&mut manager);
    let mut geo_viewer = GeoViewer::from_manager(manager);

    let fusion_thread = thread::spawn(move || {
        let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();

        let (camera_intrinsics, _) = dataset.camera(0);
        let ribuilder = RangeImageBuilder::default()
            .with_bilateral_filter(Some(BilateralFilter::default()))
            .with_normals(true)
            .pyramid_levels(1);

        let mut fusion = SurfelFusion::new(
            camera_intrinsics.width,
            camera_intrinsics.height,
            4,
            SurfelFusionParameters::default(),
        );

        for i in 0..dataset.len() {
            let rgbd_frame = dataset.get(i).unwrap();
            let pinhole_camera = rgbd_frame.get_pinhole_camera().unwrap();
            let ri_frame = ribuilder.build(rgbd_frame);

            let summary = fusion.integrate(&mut model, &ri_frame[0], &pinhole_camera);
            println!("Frame {i} fused {:?}", summary);
        }
    });

    geo_viewer.add_node(node);
    geo_viewer.run();
    fusion_thread.join().unwrap();
}
