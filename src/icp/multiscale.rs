use super::{IcpParams, ImageIcp};
use crate::{intensity_map::IntensityMap, range_image::RangeImage, transform::Transform};
use itertools::izip;

pub struct MultiscaleAlign<'pyramid_lt> {
    target_pyramid: &'pyramid_lt mut Vec<RangeImage>,
    params: IcpParams,
}

impl<'pyramid_lt> MultiscaleAlign<'pyramid_lt> {
    pub fn new(target_pyramid: &'pyramid_lt mut Vec<RangeImage>, params: IcpParams) -> Self {
        Self {
            target_pyramid,
            params,
        }
    }

    pub fn align(&mut self, source_pyramid: &Vec<RangeImage>) -> Transform {
        let mut optim_transform = Transform::eye();

        for (target, source) in izip!(self.target_pyramid.iter_mut(), source_pyramid.iter()).rev() {
            let mut icp = ImageIcp::new(self.params.clone(), target);
            icp.initial_transform = optim_transform;
            optim_transform = icp.align(source);
        }

        optim_transform
    }
}

#[cfg(test)]
mod tests {
    use itertools::izip;
    use rstest::rstest;

    use crate::{
        range_image::RangeImage,
        unit_test::{sample_rgbd_frame_dataset1, TestRgbdFrameDataset},
    };

    #[rstest]
    fn test_align(sample_rgbd_frame_dataset1: TestRgbdFrameDataset) {
        let mut target =
            RangeImage::from_pyramid(&sample_rgbd_frame_dataset1.get_item(0).unwrap().pyramid(3));

        let mut source =
            RangeImage::from_pyramid(&sample_rgbd_frame_dataset1.get_item(1).unwrap().pyramid(3));

        for (target, source) in izip!(target.iter_mut(), source.iter_mut()) {
            target.compute_intensity();
            target.compute_normals();
            source.compute_intensity();
            source.compute_normals();
        }

        let mut align = super::MultiscaleAlign {
            target_pyramid: &mut target,
            params: crate::icp::icp_params::IcpParams {
                max_iterations: 5,
                weight: 0.05,
            },
        };

        let _ = align.align(&source);
    }
}
