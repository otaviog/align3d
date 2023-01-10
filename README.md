# Align3D

A crate for aligning 3D data and other stuff related with point clouds and SLAM. All in Rust.


## Learning Rust

* [ ] `impl` and `dyn` differance 
* [ ] use `impl Foo + ?Sized`
* [ ] How to make a macro `vertex_impl!`
* [ ] Learn vulkan
* [ ] Better builder
* [ ] Option and borrowing (vkMesh)

## TODO:

* ICP
    * [ ] Optimization
    * [ ] Optimization with color
    * [ ] Discard by normal or color difference
    * [ ] Pyramid Downsampling
* RGBD ICP
    * [ ] Pyramid Downsampling
    * [ ] Optimization
    * [ ] Discard by normal or color difference
    * [ ] Optimization with color
* TSDF 
    * [ ] integration
    * [ ] to_mesh
    * [ ] visualization
* SLAM
    * [ ] Trajectory aggregation
    * [ ] Associate PCL with features from sparse descriptors
    * [ ] Submap generation
    * [ ] Pose graph optimization