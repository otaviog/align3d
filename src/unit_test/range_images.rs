use rstest::fixture;

use crate::{
    bilateral::BilateralFilter,
    io::dataset::{DatasetError, RgbdDataset, SlamTbDataset},
    range_image::RangeImage,
    Array2Recycle,
};

pub struct TestRangeImageDataset {
    dataset: Box<dyn RgbdDataset>,
}

impl TestRangeImageDataset {
    pub fn get(&self, index: usize) -> Result<RangeImage, DatasetError> {
        let (cam, mut rgbd_image) = self.dataset.get(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth, Array2Recycle::Empty)
        };
        let mut range_img = RangeImage::from_rgbd_image(&cam, &rgbd_image);
        range_img.compute_normals();
        range_img.compute_intensity();
        Ok(range_img)
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[fixture]
pub fn sample_range_img_ds1() -> TestRangeImageDataset {
    TestRangeImageDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()),
    }
}

#[fixture]
pub fn sample_range_img_ds2() -> TestRangeImageDataset {
    TestRangeImageDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample2").unwrap()),
    }
}
