use rstest::fixture;

use crate::{
    image::RgbdFrame,
    io::dataset::{DatasetError, RgbdDataset, SlamTbDataset}, bilateral::BilateralFilter, Array2Recycle,
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
        let (cam, mut rgbd_image) = self.dataset.get(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth, Array2Recycle::Empty)
        };
        Ok(RgbdFrame::new(cam, rgbd_image))
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[fixture]
pub fn sample_rgbd_frame_dataset1() -> TestRgbdFrameDataset {
    TestRgbdFrameDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()),
    }
}
