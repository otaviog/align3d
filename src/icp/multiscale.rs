use super::{ImageIcp, MsIcpParams};
use crate::{error::Error, range_image::RangeImage, transform::Transform};
use itertools::izip;

/// Multiscale interface for ICP algorithms.
/// TODO: Make it generic for point cloud ICP.
pub struct MultiscaleAlign<'pyramid_lt> {
    target_pyramid: &'pyramid_lt mut Vec<RangeImage>,
    params: MsIcpParams,
}

impl<'pyramid_lt> MultiscaleAlign<'pyramid_lt> {
    /// Creates a new multiscale ICP instance.
    /// The number of levels in the target pyramid and the number of ICP parameters must be equal.
    ///
    /// # Arguments
    ///
    /// * target_pyramid: The target point cloud pyramid.
    /// * params: The ICP parameters for each pyramid level.
    ///
    /// # Returns
    ///
    /// * Ok(MultiscaleAlign)
    /// * Err(Error(InvalidParameter)) if the number of levels in the target pyramid and the number 
    ///   of ICP parameters are equal.
    pub fn new(
        target_pyramid: &'pyramid_lt mut Vec<RangeImage>,
        params: MsIcpParams,
    ) -> Result<Self, Error> {
        if params.len() != target_pyramid.len() {
            return Err(Error::invalid_parameter(
                "The number of range images pyramid levels and ICP parameters must be equal.",
            ));
        }

        Ok(Self {
            target_pyramid,
            params,
        })
    }

    /// Aligns the source point cloud to the target point cloud.
    /// 
    /// # Arguments
    /// 
    /// * source_pyramid: The source point cloud pyramid.
    /// 
    /// # Returns
    /// 
    /// * The optimized transform.
    pub fn align(&mut self, source_pyramid: &Vec<RangeImage>) -> Transform {
        let mut optim_transform = Transform::eye();

        for (params, mut target, source) in izip!(
            self.params.iter(),
            self.target_pyramid.iter_mut(),
            source_pyramid.iter()
        )
        .rev()
        {
            let mut icp = ImageIcp::new(params.clone(), &mut target);
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
        icp::{IcpParams, MsIcpParams},
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
            params: MsIcpParams::repeat(3, &IcpParams::default()),
        };

        let _ = align.align(&source);
    }
}
