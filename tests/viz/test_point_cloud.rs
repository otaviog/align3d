use align3d::viz::{
    geometry::{VkPointCloud, VkPointCloudNode},
    Manager, Window,
};

mod data;
use data::sample_rgbd_pointcloud;

fn main() {
    let mut manager = Manager::default();
    let pointcloud =
        VkPointCloud::from_pointcloud(&manager.memory_allocator, &sample_rgbd_pointcloud());

    Window::create(&mut manager, VkPointCloudNode::new(pointcloud)).show();
}
