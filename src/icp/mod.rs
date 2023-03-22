mod icp_params;
pub use icp_params::{IcpParams, MsIcpParams};
mod cost_function;
mod icp;
pub use icp::Icp;
mod image_icp;
pub use image_icp::ImageIcp;
pub mod multiscale;
