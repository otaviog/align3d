// Todo delete depth.rs
// mod depth;
// pub use depth::resize_depth_image;

mod rgb;
pub use rgb::{scale_down_rgb8, IntoArray3, IntoImageRgb8};

mod luma;
pub use luma::{rgb_to_luma, rgb_to_luma_u8, IntoLumaArray, IntoLumaImage};

mod rgbd_image;
pub use rgbd_image::{RgbdFrame, RgbdImage};
