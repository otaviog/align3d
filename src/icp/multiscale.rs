use super::{ImageIcp, MsIcpParams};
use crate::{error::Error, range_image::RangeImage, transform::Transform};
use itertools::izip;

/// Multiscale interface for ICP algorithms.
/// TODO: Make it generic for point cloud ICP.
pub struct MultiscaleAlign<'pyramid_lt> {
    params: MsIcpParams,
    target_pyramid: &'pyramid_lt Vec<RangeImage>,
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
        params: MsIcpParams,
        target_pyramid: &'pyramid_lt Vec<RangeImage>
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
    pub fn align(&self, source_pyramid: &[RangeImage]) -> Transform {
        let mut optim_transform = Transform::eye();

        for (params, target, source) in izip!(
            self.params.iter(),
            self.target_pyramid.iter(),
            source_pyramid.iter()
        )
        .rev()
        {
            let mut icp = ImageIcp::new(*params, target);
            icp.initial_transform = optim_transform;
            optim_transform = icp.align(source);
        }

        optim_transform
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{
        bilateral::BilateralFilter,
        icp::{IcpParams, MsIcpParams},
        range_image::RangeImageBuilder,
        unit_test::{sample_rgbd_frame_dataset1, TestRgbdFrameDataset},
    };

    #[rstest]
    fn test_align(sample_rgbd_frame_dataset1: TestRgbdFrameDataset) {
        let ri_builder = RangeImageBuilder::default()
            .with_bilateral_filter(Some(BilateralFilter::default()))
            .with_intensity(true)
            .with_normals(true);
        let target = ri_builder.build(sample_rgbd_frame_dataset1.get_item(0).unwrap());
        let source = ri_builder.build(sample_rgbd_frame_dataset1.get_item(4).unwrap());

        let align = super::MultiscaleAlign {
            target_pyramid: &target,
            params: MsIcpParams::repeat(3, &IcpParams::default()),
        };
        // Just test that it doesn't crash. Use integration tests for more thorough testing.
        let _ = align.align(&source);
    }
}
