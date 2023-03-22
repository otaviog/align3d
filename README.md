# Align3D

A crate for aligning 3D data and other stuff related with point clouds and SLAM. All in Rust.

## How to use

```rust
let rgbd_dataset = R::load();

fn process_frame(frame: RgbdFrame) {
    Bilateral()
}
let RangeImage
let source = MsRangeImage::from(rgbd_dataset.get(0).unwrap().pyramid(3)).with_normals().with_luma().with_
let target = RangeImage::from(rgbd_dataset.get(0).unwrap().pyramid(3));

```


## Learning Rust

* [ ] `impl` and `dyn` differance 
* [ ] use `impl Foo + ?Sized`
* [ ] How to make a macro `vertex_impl!`
* [ ] Learn vulkan
* [ ] Better builder
* [x] Option and borrowing (vkMesh)
* [ ] Why from and into don't compile sometimes
* [ ] When make a builder

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
