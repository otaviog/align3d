use align3d::viz::{
    geometry::{VkPointCloud, VkPointCloudNode},
    Manager, Window,
};

mod test_data;
use test_data::{sample_teapot_pointcloud, sample_rgbd_pointcloud};

fn main() {
    let mut manager = Manager::default();
    let pointcloud =
        VkPointCloud::from_pointcloud(&manager.memory_allocator, &sample_rgbd_pointcloud());

    Window::create(&mut manager, VkPointCloudNode::new(pointcloud)).show();
}
