pub struct MsRangeImage {
    pub images: Vec<RangeImage>,
}

impl MsRangeImage {
    pub fn new(images: Vec<RangeImage>) -> Self {
        Self { images }
    }
}

impl Index<usize> for MsRangeImage {
    type Output = RangeImage;

    fn index(&self, index: usize) -> &Self::Output {
        &self.images[index]
    }
}
