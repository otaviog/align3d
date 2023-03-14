use super::{ICPParams, ImageICP};
use crate::{
    camera::Camera, imagepointcloud::ImagePointCloud, intensity_map::IntensityMap,
    transform::Transform,
};
use itertools::izip;

pub struct MultiscaleAlign<'pyramid_lt> {
    target_pyramid: &'pyramid_lt Vec<ImagePointCloud>,
    cameras: Vec<Camera>,
    params: ICPParams,
}

impl<'pyramid_lt> MultiscaleAlign<'pyramid_lt> {
    pub fn new(
        target_pyramid: &'pyramid_lt Vec<ImagePointCloud>,
        cameras: Vec<Camera>,
        params: ICPParams,
    ) -> Self {
        Self {
            target_pyramid,
            cameras,
            params,
        }
    }
    pub fn align(&self, source_pyramid: &Vec<ImagePointCloud>) -> Transform {
        let mut optim_transform = Transform::eye();

        for (camera, target, source) in izip!(
            self.cameras.iter(),
            self.target_pyramid.iter(),
            source_pyramid.iter()
        ).rev() {
            let intensity_map = IntensityMap::from_rgb_image(target.colors.as_ref().unwrap());
            let mut icp = ImageICP::new(self.params.clone(), camera, target, &intensity_map);
            icp.initial_transform = optim_transform;
            optim_transform = icp.align(source);
            
        }

        optim_transform
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use itertools::izip;

    use crate::{
        imagepointcloud::ImagePointCloud,
        unit_test::{sample_rgbd_frame_dataset1, TestRGBDFrameDataset},
    };

    #[rstest]
    fn test_align(sample_rgbd_frame_dataset1: TestRGBDFrameDataset) {
        let (cameras, mut target) = {
            let target = sample_rgbd_frame_dataset1.get_item(0).unwrap().pyramid(3);
            let cameras = target.iter().map(|x| x.camera.clone()).collect();

            (cameras, ImagePointCloud::from_pyramid(&target))
        };

        let mut source = ImagePointCloud::from_pyramid(
            &sample_rgbd_frame_dataset1.get_item(1).unwrap().pyramid(3),
        );

        for (target, source) in izip!(target.iter_mut(), source.iter_mut()) {
            target.compute_intensity();
            target.compute_normals();
            source.compute_intensity();
            source.compute_normals();
        }

        let align = super::MultiscaleAlign {
            target_pyramid: &target,
            cameras: cameras,
            params: crate::icp::icp_params::ICPParams {
                max_iterations: 5,
                weight: 0.05,
            },
        };

        let _ = align.align(&source);
    }
}
