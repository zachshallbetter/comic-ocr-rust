//! The degradations scanning introduces, applied to a clean render.
//!
//! A model trained only on pristine renders learns a distribution real pages
//! never occupy. These are the four that dominate the corpus: JPEG ringing
//! around high-contrast glyph edges, slight rotation from imperfect page
//! placement, focus blur, and sensor noise.

use image::{DynamicImage, GrayImage, ImageFormat, Luma};
use imageproc::filter::gaussian_blur_f32;
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct DegradeSpec {
    /// JPEG quality, 1-100. Below ~60 ringing becomes clearly visible.
    pub jpeg_quality: Option<u8>,
    /// Rotation in degrees. Real scans sit within about +/-1.5.
    pub rotate_deg: f32,
    /// Gaussian sigma. Above ~1.2 small kana start to close up.
    pub blur_sigma: f32,
    /// Standard deviation of additive Gaussian noise, in grey levels.
    pub noise_sd: f32,
}

impl Default for DegradeSpec {
    fn default() -> Self {
        Self {
            jpeg_quality: Some(75),
            rotate_deg: 0.0,
            blur_sigma: 0.0,
            noise_sd: 0.0,
        }
    }
}

/// How badly the page was scanned. Named rather than numeric because the
/// ranges are an empirical claim about real scans, and a caller passing raw
/// numbers cannot be corrected when that claim turns out wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanQuality {
    /// A clean digital release: mild recompression, near-square placement.
    Typical,
    /// A hand-scanned or multiply-recompressed copy -- the tail of the corpus
    /// where a reader actually earns its accuracy.
    Poor,
}

impl DegradeSpec {
    /// A plausible draw for a scanned page. Deterministic given `rng`, so a
    /// generation run can be reproduced from its seed.
    pub fn sample<R: Rng>(rng: &mut R) -> Self {
        Self::sample_at(rng, ScanQuality::Typical)
    }

    /// Measured 2026-08-22 by `examples/ablate_degradation`: on the eight-label
    /// probe, NOTHING in either range moves the reference model -- pristine,
    /// jpeg 30, rotation 3 deg and noise 20 all tie at 1.25%, and that 1.25% is a
    /// single hallucinated punctuation mark rather than a reading error. Only
    /// blur >= 3.0 produces genuine confusion (女川 for 立川).
    ///
    /// So these ranges are NOT yet validated against evidence; the probe is
    /// saturated and cannot tell a good range from a bad one. They are a
    /// starting position drawn from what scans plausibly do, and the honest
    /// status is unverified.
    pub fn sample_at<R: Rng>(rng: &mut R, quality: ScanQuality) -> Self {
        match quality {
            ScanQuality::Typical => Self {
                jpeg_quality: Some(rng.gen_range(55..=92)),
                rotate_deg: rng.gen_range(-1.5..=1.5),
                blur_sigma: rng.gen_range(0.0..=0.9),
                noise_sd: rng.gen_range(0.0..=6.0),
            },
            ScanQuality::Poor => Self {
                jpeg_quality: Some(rng.gen_range(22..=55)),
                rotate_deg: rng.gen_range(-4.0..=4.0),
                blur_sigma: rng.gen_range(0.8..=2.2),
                noise_sd: rng.gen_range(6.0..=22.0),
            },
        }
    }
}

/// Apply the degradations in the order a scanner actually imposes them:
/// optical blur and rotation happen in the lens and on the platen, sensor noise
/// is added at capture, and JPEG is applied last when the file is written.
pub fn apply<R: Rng>(
    img: &GrayImage,
    spec: &DegradeSpec,
    rng: &mut R,
) -> Result<GrayImage, String> {
    let mut out = img.clone();

    if spec.blur_sigma > 0.0 {
        out = gaussian_blur_f32(&out, spec.blur_sigma);
    }

    if spec.rotate_deg.abs() > f32::EPSILON {
        // Fill with the ground colour, not black: a black corner is a feature
        // the model would happily learn and never see on a real balloon crop.
        let ground = estimate_ground(&out);
        out = rotate_about_center(
            &out,
            spec.rotate_deg.to_radians(),
            Interpolation::Bilinear,
            Luma([ground]),
        );
    }

    if spec.noise_sd > 0.0 {
        for px in out.pixels_mut() {
            let n: f32 = rng.gen_range(-1.0..=1.0) * spec.noise_sd;
            px.0[0] = (px.0[0] as f32 + n).clamp(0.0, 255.0) as u8;
        }
    }

    if let Some(q) = spec.jpeg_quality {
        out = jpeg_roundtrip(&out, q)?;
    }

    Ok(out)
}

/// The modal bright value, which for a balloon crop is the interior.
fn estimate_ground(img: &GrayImage) -> u8 {
    let mut hist = [0u32; 256];
    for px in img.pixels() {
        hist[px.0[0] as usize] += 1;
    }
    hist.iter()
        .enumerate()
        .skip(128) // ground is the bright mode; ink is the dark one
        .max_by_key(|(_, n)| **n)
        .map(|(v, _)| v as u8)
        .unwrap_or(255)
}

/// Encode to JPEG at `quality` and decode back, so the crop carries real
/// ringing rather than a simulation of it.
fn jpeg_roundtrip(img: &GrayImage, quality: u8) -> Result<GrayImage, String> {
    let mut buf = Vec::new();
    DynamicImage::ImageLuma8(img.clone())
        .write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
            &mut std::io::Cursor::new(&mut buf),
            quality.clamp(1, 100),
        ))
        .map_err(|e| format!("jpeg encode: {e}"))?;
    let decoded = image::load_from_memory_with_format(&buf, ImageFormat::Jpeg)
        .map_err(|e| format!("jpeg decode: {e}"))?;
    Ok(decoded.to_luma8())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn plain() -> GrayImage {
        let mut img = GrayImage::from_pixel(64, 64, Luma([250]));
        for y in 20..44 {
            for x in 20..44 {
                img.put_pixel(x, y, Luma([20]));
            }
        }
        img
    }

    #[test]
    fn jpeg_changes_pixels_without_destroying_the_shape() {
        let img = plain();
        let out = jpeg_roundtrip(&img, 60).expect("roundtrip");
        assert_eq!(out.dimensions(), img.dimensions());
        assert!(
            img.pixels().zip(out.pixels()).any(|(a, b)| a != b),
            "quality 60 must actually perturb pixels"
        );
        // The dark square must survive as dark.
        assert!(out.get_pixel(32, 32).0[0] < 80);
    }

    #[test]
    fn rotation_fills_with_ground_not_black() {
        let img = plain();
        let spec = DegradeSpec {
            jpeg_quality: None,
            rotate_deg: 10.0,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(1);
        let out = apply(&img, &spec, &mut rng).expect("apply");
        // A corner is now outside the original frame; it must be light.
        assert!(
            out.get_pixel(0, 0).0[0] > 200,
            "corner was {} -- black fill would teach a false feature",
            out.get_pixel(0, 0).0[0]
        );
    }

    #[test]
    fn ground_estimate_finds_the_bright_mode() {
        assert_eq!(estimate_ground(&plain()), 250);
    }

    #[test]
    fn same_seed_gives_the_same_degradation() {
        let img = plain();
        let a = {
            let mut r = StdRng::seed_from_u64(7);
            let s = DegradeSpec::sample(&mut r);
            apply(&img, &s, &mut r).unwrap()
        };
        let b = {
            let mut r = StdRng::seed_from_u64(7);
            let s = DegradeSpec::sample(&mut r);
            apply(&img, &s, &mut r).unwrap()
        };
        assert_eq!(
            a.into_raw(),
            b.into_raw(),
            "a run must be reproducible from its seed"
        );
    }

    #[test]
    fn noise_perturbs_but_stays_in_range() {
        let img = plain();
        let spec = DegradeSpec {
            jpeg_quality: None,
            noise_sd: 20.0,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(3);
        let out = apply(&img, &spec, &mut rng).expect("apply");
        assert!(img.pixels().zip(out.pixels()).any(|(a, b)| a != b));
    }
}
