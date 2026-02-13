//! Diffusion kernel for trophic fields.
//!
//! Runs a 3D box-blur approximation of Gaussian diffusion with decay.

use super::field::TrophicField;

/// Parameters for diffusion (genome-modulated).
pub struct DiffusionParams {
    /// Blur radius in grid cells.
    pub radius: u32,
    /// Decay factor per step (0.0 = no decay, 1.0 = full decay).
    pub decay: f32,
    /// Maximum concentration (saturation).
    pub saturation: u16,
}

/// Run one diffusion step on a trophic field.
///
/// Uses a separable box blur (3-pass: X, Y, Z) as a fast approximation
/// of Gaussian diffusion. Applies decay after blurring.
pub fn diffuse_step(field: &mut TrophicField, params: &DiffusionParams) {
    let [w, h, d] = field.dims;
    let r = params.radius as i32;

    // Temporary buffer for separable blur
    let mut temp = field.data.clone();

    // X-pass
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dx in -r..=r {
                    let nx = x as i32 + dx;
                    if nx >= 0 && nx < w as i32 {
                        let idx = field.grid_index(nx as u32, y, z);
                        sum += field.data[idx] as u32;
                        count += 1;
                    }
                }
                let idx = field.grid_index(x, y, z);
                temp[idx] = (sum / count.max(1)) as u16;
            }
        }
    }

    // Y-pass
    let src = temp.clone();
    for z in 0..d {
        for x in 0..w {
            for y in 0..h {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in -r..=r {
                    let ny = y as i32 + dy;
                    if ny >= 0 && ny < h as i32 {
                        let idx = field.grid_index(x, ny as u32, z);
                        sum += src[idx] as u32;
                        count += 1;
                    }
                }
                let idx = field.grid_index(x, y, z);
                temp[idx] = (sum / count.max(1)) as u16;
            }
        }
    }

    // Z-pass
    let src = temp.clone();
    for y in 0..h {
        for x in 0..w {
            for z in 0..d {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dz in -r..=r {
                    let nz = z as i32 + dz;
                    if nz >= 0 && nz < d as i32 {
                        let idx = field.grid_index(x, y, nz as u32);
                        sum += src[idx] as u32;
                        count += 1;
                    }
                }
                let idx = field.grid_index(x, y, z);
                temp[idx] = (sum / count.max(1)) as u16;
            }
        }
    }

    // Apply decay and saturation
    for (dst, &src_val) in field.data.iter_mut().zip(temp.iter()) {
        let decayed = (src_val as f32 * (1.0 - params.decay)) as u16;
        *dst = decayed.min(params.saturation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trophic::chemical::TrophicChemicalId;

    #[test]
    fn diffusion_spreads_source() {
        let mut field =
            TrophicField::new(TrophicChemicalId::new("test"), [8, 8, 8], 1.0, [0.0; 3]);
        // Place a point source at center
        field.set(4, 4, 4, 1000);

        let params = DiffusionParams {
            radius: 1,
            decay: 0.0,
            saturation: 65535,
        };

        diffuse_step(&mut field, &params);

        // Center should have decreased (averaged with neighbors)
        let center = field.get(4, 4, 4);
        // Neighbors should have increased (got some of center's value)
        let neighbor = field.get(5, 4, 4);

        assert!(center < 1000, "center should decrease after diffusion");
        assert!(neighbor > 0, "neighbor should receive some concentration");
    }

    #[test]
    fn decay_reduces_total() {
        // Use a uniform field so blur doesn't redistribute mass — only decay acts
        let mut field =
            TrophicField::new(TrophicChemicalId::new("test"), [4, 4, 4], 1.0, [0.0; 3]);
        for v in field.data.iter_mut() {
            *v = 1000;
        }

        let total_before: u64 = field.data.iter().map(|&v| v as u64).sum();

        let params = DiffusionParams {
            radius: 1,
            decay: 0.1,
            saturation: 65535,
        };
        diffuse_step(&mut field, &params);

        let total_after: u64 = field.data.iter().map(|&v| v as u64).sum();
        assert!(
            total_after < total_before,
            "total should decrease with decay: {total_before} -> {total_after}"
        );
    }

    #[test]
    fn saturation_caps_values() {
        let mut field =
            TrophicField::new(TrophicChemicalId::new("test"), [4, 4, 4], 1.0, [0.0; 3]);
        // Fill everything high
        for v in field.data.iter_mut() {
            *v = 60000;
        }

        let params = DiffusionParams {
            radius: 1,
            decay: 0.0,
            saturation: 50000,
        };
        diffuse_step(&mut field, &params);

        assert!(
            field.data.iter().all(|&v| v <= 50000),
            "all values should be <= saturation"
        );
    }
}
