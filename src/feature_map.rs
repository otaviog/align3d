use ndarray::Array2;

struct IntensityMap {
    pub map: Array2<f32>
}


impl IntensityMap {
    pub fn from_luma_image(image: Array2<u8>) -> Self {
        let map = Array2::zeros((height + 2, width + 2));
        IntensityMap { map:  map }
    }
    pub fn get(&self, u: f32, v: f32) -> f32 {
        let ui = u  as u32 + 1;
        let vi = v  as u32 + 1;

        let u_ratio = u - ui as f32;
        let v_ratio = v - vi as f32;
        
        let val00 = self.map[![vi, ui]];
        let val10 = self.map[![vi, ui + 1]];
        let val11 = self.map[![vi + 1, ui + 1]];
        let val01 = self.map[![vi + 1, ui]];

        let u0_interp = val00 * (1 - u_ratio) + val10 * u_ratio;
        let u1_interp = val01 * (1 - u_ratio) + val11 * u_ratio;
        let val = u0_interp * (1 - v_ratio) + u1_interp * v_ratio;
    }
}