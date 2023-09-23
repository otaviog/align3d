use rstest::fixture;

use crate::{
    bilateral::BilateralFilter,
    image::RgbdFrame,
    io::dataset::{DatasetError, RgbdDataset, SlamTbDataset},
};

#[fixture]
pub fn sample_rgbd_dataset1() -> impl RgbdDataset {
    SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()
}

pub struct TestRgbdFrameDataset {
    dataset: Box<dyn RgbdDataset>,
}

impl TestRgbdFrameDataset {
    pub fn get_item(&self, index: usize) -> Result<RgbdFrame, DatasetError> {
        let (cam, mut rgbd_image, transform) = self.dataset.get(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth)
        };
        Ok(RgbdFrame::new(cam, rgbd_image, transform))
    }
}

#[fixture]
pub fn sample_rgbd_frame_dataset1() -> TestRgbdFrameDataset {
    TestRgbdFrameDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()),
    }
}
