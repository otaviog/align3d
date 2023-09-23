pub(crate) mod access;

mod datasets;
pub(crate) use datasets::{sample_rgbd_dataset1, sample_rgbd_frame_dataset1, TestRgbdFrameDataset};
mod geometries;
pub(crate) use geometries::sample_teapot_geometry;
mod images;
pub(crate) use images::{bloei_luma16, bloei_luma8, bloei_rgb};
mod point_clouds;
pub(crate) use point_clouds::{sample_pcl_ds1, sample_teapot_pointcloud, TestPclDataset};
mod range_images;
pub(crate) use range_images::{sample_range_img_ds1, sample_range_img_ds2, TestRangeImageDataset};
