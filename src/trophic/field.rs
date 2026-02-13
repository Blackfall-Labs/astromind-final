//! 3D scalar field for trophic factor concentrations.
//!
//! Discretized onto a grid. Values are u16 (0-65535 concentration units).
//! Supports point sources, trilinear sampling, and diffusion steps.

use super::chemical::TrophicChemicalId;

/// A 3D scalar field representing concentration of a trophic chemical.
#[derive(Debug)]
pub struct TrophicField {
    pub chemical: TrophicChemicalId,
    pub dims: [u32; 3],
    pub data: Vec<u16>,
    pub cell_size: f32,
    pub origin: [f32; 3],
}

impl TrophicField {
    /// Create a new empty field.
    pub fn new(
        chemical: TrophicChemicalId,
        dims: [u32; 3],
        cell_size: f32,
        origin: [f32; 3],
    ) -> Self {
        let total = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        TrophicField {
            chemical,
            dims,
            data: vec![0u16; total],
            cell_size,
            origin,
        }
    }

    /// Grid index from (x, y, z) coordinates.
    pub(crate) fn grid_index(&self, x: u32, y: u32, z: u32) -> usize {
        (x as usize) + (y as usize) * (self.dims[0] as usize)
            + (z as usize) * (self.dims[0] as usize) * (self.dims[1] as usize)
    }

    /// Get concentration at a grid cell.
    pub fn get(&self, x: u32, y: u32, z: u32) -> u16 {
        self.data[self.grid_index(x, y, z)]
    }

    /// Set concentration at a grid cell.
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: u16) {
        let idx = self.grid_index(x, y, z);
        self.data[idx] = value;
    }

    /// Convert world position to grid coordinates. Returns None if out of bounds.
    pub fn world_to_grid(&self, pos: [f32; 3]) -> Option<(u32, u32, u32)> {
        let gx = ((pos[0] - self.origin[0]) / self.cell_size) as i32;
        let gy = ((pos[1] - self.origin[1]) / self.cell_size) as i32;
        let gz = ((pos[2] - self.origin[2]) / self.cell_size) as i32;
        if gx >= 0
            && gy >= 0
            && gz >= 0
            && (gx as u32) < self.dims[0]
            && (gy as u32) < self.dims[1]
            && (gz as u32) < self.dims[2]
        {
            Some((gx as u32, gy as u32, gz as u32))
        } else {
            None
        }
    }

    /// Sample concentration at a world-space position using nearest-neighbor.
    /// Returns 0 if out of bounds.
    pub fn sample(&self, pos: [f32; 3]) -> u16 {
        match self.world_to_grid(pos) {
            Some((x, y, z)) => self.get(x, y, z),
            None => 0,
        }
    }

    /// Find the grid position with maximum concentration.
    pub fn max_position(&self) -> [f32; 3] {
        let mut max_val = 0u16;
        let mut max_idx = 0usize;
        for (i, &v) in self.data.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        let w = self.dims[0] as usize;
        let h = self.dims[1] as usize;
        let x = (max_idx % w) as f32;
        let y = ((max_idx / w) % h) as f32;
        let z = (max_idx / (w * h)) as f32;
        [
            self.origin[0] + x * self.cell_size,
            self.origin[1] + y * self.cell_size,
            self.origin[2] + z * self.cell_size,
        ]
    }

    /// Add a point source at a world position, spreading with given radius.
    pub fn add_source(&mut self, pos: [f32; 3], intensity: u16, radius: f32) {
        let radius_cells = (radius / self.cell_size).ceil() as i32;
        let center = [
            ((pos[0] - self.origin[0]) / self.cell_size) as i32,
            ((pos[1] - self.origin[1]) / self.cell_size) as i32,
            ((pos[2] - self.origin[2]) / self.cell_size) as i32,
        ];

        for dz in -radius_cells..=radius_cells {
            for dy in -radius_cells..=radius_cells {
                for dx in -radius_cells..=radius_cells {
                    let gx = center[0] + dx;
                    let gy = center[1] + dy;
                    let gz = center[2] + dz;
                    if gx < 0
                        || gy < 0
                        || gz < 0
                        || gx >= self.dims[0] as i32
                        || gy >= self.dims[1] as i32
                        || gz >= self.dims[2] as i32
                    {
                        continue;
                    }
                    let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                    let r_sq = (radius / self.cell_size).powi(2);
                    if dist_sq <= r_sq {
                        let falloff = 1.0 - (dist_sq / r_sq).sqrt();
                        let val = (intensity as f32 * falloff) as u16;
                        let idx = self.grid_index(gx as u32, gy as u32, gz as u32);
                        self.data[idx] = self.data[idx].saturating_add(val);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_field_is_zero() {
        let f = TrophicField::new(TrophicChemicalId::new("test"), [4, 4, 4], 1.0, [0.0; 3]);
        assert_eq!(f.data.len(), 64);
        assert!(f.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn set_and_get() {
        let mut f = TrophicField::new(TrophicChemicalId::new("test"), [4, 4, 4], 1.0, [0.0; 3]);
        f.set(2, 1, 3, 42);
        assert_eq!(f.get(2, 1, 3), 42);
        assert_eq!(f.get(0, 0, 0), 0);
    }

    #[test]
    fn sample_at_source() {
        let mut f = TrophicField::new(TrophicChemicalId::new("test"), [8, 8, 8], 1.0, [0.0; 3]);
        f.add_source([4.0, 4.0, 4.0], 1000, 3.0);
        let val = f.sample([4.0, 4.0, 4.0]);
        assert!(val > 500, "source should be strong at center, got {val}");
    }

    #[test]
    fn sample_decays_with_distance() {
        let mut f = TrophicField::new(TrophicChemicalId::new("test"), [16, 16, 16], 1.0, [0.0; 3]);
        f.add_source([8.0, 8.0, 8.0], 1000, 6.0);
        let center = f.sample([8.0, 8.0, 8.0]);
        let near = f.sample([10.0, 8.0, 8.0]);
        let far = f.sample([14.0, 8.0, 8.0]);
        assert!(center > near, "center ({center}) should be > near ({near})");
        assert!(near >= far, "near ({near}) should be >= far ({far})");
    }

    #[test]
    fn out_of_bounds_returns_zero() {
        let f = TrophicField::new(TrophicChemicalId::new("test"), [4, 4, 4], 1.0, [0.0; 3]);
        assert_eq!(f.sample([10.0, 10.0, 10.0]), 0);
        assert_eq!(f.sample([-1.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn max_position_finds_source() {
        let mut f = TrophicField::new(TrophicChemicalId::new("test"), [8, 8, 8], 1.0, [0.0; 3]);
        f.add_source([5.0, 3.0, 6.0], 1000, 2.0);
        let max_pos = f.max_position();
        // Should be near the source
        assert!((max_pos[0] - 5.0).abs() <= 1.0);
        assert!((max_pos[1] - 3.0).abs() <= 1.0);
        assert!((max_pos[2] - 6.0).abs() <= 1.0);
    }
}
