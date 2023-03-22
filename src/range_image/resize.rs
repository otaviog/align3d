use nalgebra::Vector3;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

fn get_neighborhood_mean_point(
    src_v: usize,
    src_u: usize,
    src_mask: &ArrayView2<u8>,
    src_points: &ArrayView3<f32>,
) -> Option<Vector3<f32>> {
    let local_points = {
        let mut local_points = Vec::<Vector3<f32>>::new();
        for i in 0..2 {
            for j in 0..2 {
                let (i, j) = (src_v + i, src_u + j);
                if src_mask[[i, j]] == 1 {
                    let point = Vector3::new(
                        src_points[(i, j, 0)],
                        src_points[(i, j, 1)],
                        src_points[(i, j, 2)],
                    );
                    local_points.push(point);
                }
            }
        }
        local_points
    };
    if local_points.is_empty() {
        // dst_mask[(dst_row, dst_col)] = 0;
        return None;
    }
    let mean_point =
        local_points.iter().fold(Vector3::zeros(), |x1, x2| x1 + x2) / local_points.len() as f32;
    let mut min_dist = f32::MAX;
    let mut nearest_point = Vector3::zeros();
    for point in local_points.iter() {
        let dist = (*point - mean_point).norm_squared();
        if dist < min_dist {
            min_dist = dist;
            nearest_point = *point;
        }
    }

    Some(nearest_point)
}

pub fn resize_range_points(
    src_points: &ArrayView3<f32>,
    src_mask: &ArrayView2<u8>,
    dst_width: usize,
    dst_height: usize,
) -> (Array3<f32>, Array2<u8>) {
    // Todo: Make mure rustacean: use iterators, etc.

    let mut dst_points = Array3::zeros((dst_height, dst_width, 3));
    let mut dst_mask = Array2::zeros((dst_height, dst_width));
    let (src_height, src_width) = (src_points.shape()[0], src_points.shape()[1]);

    let height_ratio = src_height as f32 / dst_height as f32;
    let width_ratio = src_width as f32 / dst_width as f32;

    for dst_v in 0..dst_height {
        let src_v = (dst_v as f32 * height_ratio) as usize;
        for dst_u in 0..dst_width {
            let src_u = (dst_u as f32 * width_ratio) as usize;
            let nearest_point =
                match get_neighborhood_mean_point(src_v, src_u, src_mask, src_points) {
                    Some(value) => value,
                    None => continue,
                };

            dst_mask[(dst_v, dst_u)] = 1;

            // dst_points.slice_mut(s![dst_v, dst_u, ..]).assign(&nearest_point.into_ndarray2());
            dst_points[(dst_v, dst_u, 0)] = nearest_point[0];
            dst_points[(dst_v, dst_u, 1)] = nearest_point[1];
            dst_points[(dst_v, dst_u, 2)] = nearest_point[2];
        }
    }

    (dst_points, dst_mask)
}

pub fn resize_range_normals(
    src_normals: &ArrayView3<f32>,
    src_mask: &ArrayView2<u8>,
    dst_width: usize,
    dst_height: usize,
) -> Array3<f32> {
    // Todo: Make mure rustacean: use iterators, etc.

    let mut dst_points = Array3::zeros((dst_height, dst_width, 3));
    let (src_height, src_width) = (src_normals.shape()[0], src_normals.shape()[1]);

    let height_ratio = src_height as f32 / dst_height as f32;
    let width_ratio = src_width as f32 / dst_width as f32;

    for dst_v in 0..dst_height {
        let src_v = (dst_v as f32 * height_ratio) as usize;
        for dst_u in 0..dst_width {
            let src_u = (dst_u as f32 * width_ratio) as usize;
            let nearest_point =
                match get_neighborhood_mean_point(src_v, src_u, src_mask, src_normals) {
                    Some(value) => value,
                    None => continue,
                };

            dst_points[(dst_v, dst_u, 0)] = nearest_point[0];
            dst_points[(dst_v, dst_u, 1)] = nearest_point[1];
            dst_points[(dst_v, dst_u, 2)] = nearest_point[2];
        }
    }

    dst_points
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{
        io::{core::RgbdDataset, write_ply, GeometryBuilder},
        range_image::RangeImage,
        unit_test::sample_rgbd_dataset1,
    };

    use super::{resize_range_normals, resize_range_points};

    #[rstest]
    pub fn verify_downsample(sample_rgbd_dataset1: impl RgbdDataset) {
        let frame = sample_rgbd_dataset1.get_item(0).unwrap();
        let mut ri = RangeImage::from_rgbd_frame(&frame);
        ri.compute_normals();

        let (width, height) = (320, 240);
        let (points, _) = resize_range_points(&ri.points.view(), &ri.mask.view(), width, height);
        let normals = resize_range_normals(
            &ri.normals.as_ref().unwrap().view(),
            &ri.mask.view(),
            width,
            height,
        );

        write_ply(
            "tests/outputs/downsample_points.ply",
            &GeometryBuilder::new(points.into_shape((width * height, 3)).unwrap())
                .with_normals(normals.into_shape((width * height, 3)).unwrap())
                .build(),
        )
        .expect("Error while writing the results");
    }
}
