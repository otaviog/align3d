use rstest::fixture;

use crate::{
    bilateral::BilateralFilter,
    io::dataset::{DatasetError, RgbdDataset, SlamTbDataset},
    range_image::RangeImage,
    transform::Transform,
};

pub struct TestRangeImageDataset {
    dataset: Box<dyn RgbdDataset>,
}

impl TestRangeImageDataset {
    pub fn get(&self, index: usize) -> Result<RangeImage, DatasetError> {
        let (cam, mut rgbd_image, _) = self.dataset.get(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth)
        };
        let mut range_img = RangeImage::from_rgbd_image(&cam, &rgbd_image);
        range_img.compute_normals();
        range_img.compute_intensity();
        range_img.compute_intensity_map();
        Ok(range_img)
    }

    pub fn get_ground_truth(&self, source_index: usize, target_index: usize) -> Transform {
        self.dataset
            .trajectory()
            .unwrap()
            .get_relative_transform(source_index, target_index)
            .unwrap()
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
