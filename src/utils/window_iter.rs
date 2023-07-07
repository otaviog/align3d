use ndarray::ArrayView2;

pub fn window<'a, T>(
    arr: ArrayView2<'a, T>,
    u: usize,
    v: usize,
    n: usize,
) -> impl Iterator<Item = T> + 'a
where
    T: Copy,
{
    let n = n / 2;
    let start = -(n as isize);
    let end = n as isize;
    let width = arr.shape()[1] as isize;
    let height = arr.shape()[0] as isize;

    (start..=end)
        .flat_map(move |du| (start..=end).map(move |dv| (du, dv)))
        .filter_map(move |(du, dv)| {
            let x = (u as isize) + du;
            let y = (v as isize) + dv;
            if width > x && x >= 0 && height > y && y >= 0 {
                Some(arr[(y as usize, x as usize)])
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn test_window() {
        use super::*;
        use ndarray::Array2;
        #[rustfmt::skip]
        let arr = Array2::from_shape_vec(
            (5, 5),
            {
            vec![
                0, 0, 0, 0, 0,
                0, 1, 2, 3, 0,
                0, 4, 5, 6, 0,
                0, 7, 8, 9, 0,
                0, 0, 0, 0, 0,
            ]},
        )
        .unwrap();

        let iter = window(arr.view(), 2, 2, 3);
        let vec: Vec<_> = iter.collect();
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let iter = window(arr.view(), 2, 0, 3);
        let vec: Vec<_> = iter.collect();
        assert_eq!(vec, vec![0, 1, 0, 4, 0, 7]);
    }
}
