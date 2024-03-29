mod icp_params;
pub use icp_params::{IcpParams, MsIcpParams};
mod cost_function;
mod pcl_icp;
pub use pcl_icp::Icp;
mod image_icp;
pub use image_icp::ImageIcp;
pub mod multiscale;
