mod core;
pub use self::core::{DatasetError, RgbdDataset, SubsetDataset};

mod indoor_lidar;
pub use indoor_lidar::IndoorLidarDataset;

mod slamtb;
#[doc(hidden)]
pub use slamtb::SlamTbDataset;

mod tum;
pub use tum::TumRgbdDataset;
