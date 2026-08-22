//! Render text into a balloon-shaped crop whose label is known by construction.
//!
//! This is stage 1 of `docs/TRAINING_PATH.md`: the label does not come from a
//! detector or a transcriber, so the composed-confidence rule does not apply and
//! `confidence` is 1.0 because nothing inferred it.
//!
//! CJK layout here is deliberately em-square: every glyph is centred in a cell
//! of `font_px`, advancing down a column (vertical) or across a line
//! (horizontal). Real manga lettering is close to this for the kana and kanji
//! that dominate balloon text, and it keeps the geometry predictable enough that
//! a crop's bounds are exactly what the label describes.

use ab_glyph::{Font, FontVec, PxScale, point};
use image::{GrayImage, Luma};

/// Which way the text runs. Manga balloons are overwhelmingly `VerticalRl`;
/// sound effects, signage and Western comics are `HorizontalTb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Columns run top-to-bottom, successive columns right-to-left.
    VerticalRl,
    /// Lines run left-to-right, successive lines top-to-bottom.
    HorizontalTb,
}

#[derive(Debug, Clone)]
pub struct RenderSpec {
    pub text: String,
    pub direction: Direction,
    /// Em size in pixels. Balloon text in a 1200px-wide scan is typically 20-40.
    pub font_px: f32,
    /// White space between the text block and the crop edge.
    pub padding_px: u32,
    /// Glyphs per column (vertical) or per line (horizontal) before wrapping.
    pub cells_per_run: usize,
    /// 0 = black ink. Real scans rarely reach 0; 20-40 is common.
    pub ink: u8,
    /// 255 = white ground. Balloon interiors are usually 240-255.
    pub ground: u8,
}

impl Default for RenderSpec {
    fn default() -> Self {
        Self {
            text: String::new(),
            direction: Direction::VerticalRl,
            font_px: 28.0,
            padding_px: 8,
            cells_per_run: 8,
            ink: 24,
            ground: 250,
        }
    }
}

/// A font loaded once and reused. `.ttc` collections need an index; a plain
/// `.ttf`/`.otf` uses 0.
pub struct SynthFont {
    inner: FontVec,
}

impl SynthFont {
    pub fn from_path(path: &str, index: u32) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let inner = FontVec::try_from_vec_and_index(bytes, index)
            .map_err(|e| format!("{path} (index {index}): {e}"))?;
        Ok(Self { inner })
    }

    /// True when the font can actually draw this character. A font missing the
    /// glyph renders the notdef box, which would train the model on a label the
    /// image does not contain -- the exact fabrication this whole path exists to
    /// avoid.
    pub fn covers(&self, c: char) -> bool {
        self.inner.glyph_id(c).0 != 0
    }

    /// Every character the font cannot draw, in order of first appearance.
    pub fn uncovered(&self, text: &str) -> Vec<char> {
        let mut seen = Vec::new();
        for c in text.chars() {
            if !c.is_whitespace() && !self.covers(c) && !seen.contains(&c) {
                seen.push(c);
            }
        }
        seen
    }
}

/// Render `spec` into a grayscale crop.
///
/// Fails rather than silently drawing notdef boxes when the font cannot cover
/// the text: a crop whose pixels disagree with its label is worse than no crop.
pub fn render(spec: &RenderSpec, font: &SynthFont) -> Result<GrayImage, String> {
    let chars: Vec<char> = spec.text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return Err("refusing to render empty text".into());
    }
    let missing = font.uncovered(&spec.text);
    if !missing.is_empty() {
        return Err(format!(
            "font cannot draw {missing:?} -- rendering would produce notdef boxes \
             that disagree with the label"
        ));
    }
    if spec.cells_per_run == 0 {
        return Err("cells_per_run must be at least 1".into());
    }

    let cell = spec.font_px.ceil() as u32;
    let runs = chars.len().div_ceil(spec.cells_per_run);
    let in_run = chars.len().min(spec.cells_per_run) as u32;
    let pad = spec.padding_px;

    let (w, h) = match spec.direction {
        Direction::VerticalRl => (runs as u32 * cell + 2 * pad, in_run * cell + 2 * pad),
        Direction::HorizontalTb => (in_run * cell + 2 * pad, runs as u32 * cell + 2 * pad),
    };

    let mut img = GrayImage::from_pixel(w, h, Luma([spec.ground]));
    let scale = PxScale::from(spec.font_px);
    let scaled = font.inner.as_scaled(scale);
    use ab_glyph::ScaleFont;
    let ascent = scaled.ascent();

    for (i, &c) in chars.iter().enumerate() {
        let run = i / spec.cells_per_run;
        let slot = (i % spec.cells_per_run) as u32;

        // Vertical columns advance right-to-left, so the first column sits at
        // the RIGHT edge -- reversing this is the classic way to render manga
        // that reads backwards.
        let (cx, cy) = match spec.direction {
            Direction::VerticalRl => (w - pad - (run as u32 + 1) * cell, pad + slot * cell),
            Direction::HorizontalTb => (pad + slot * cell, pad + run as u32 * cell),
        };

        let glyph = font
            .inner
            .glyph_id(c)
            .with_scale_and_position(scale, point(cx as f32, cy as f32 + ascent));
        let Some(outlined) = font.inner.outline_glyph(glyph) else {
            continue; // whitespace-like glyph with no outline
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, coverage| {
            let px = bounds.min.x as i32 + gx as i32;
            let py = bounds.min.y as i32 + gy as i32;
            if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                return;
            }
            let dst = img.get_pixel_mut(px as u32, py as u32);
            let blended = spec.ground as f32 * (1.0 - coverage) + spec.ink as f32 * coverage;
            // Keep the darkest value: overlapping outlines must not lighten ink.
            dst.0[0] = dst.0[0].min(blended.round() as u8);
        });
    }

    Ok(img)
}

/// Fraction of pixels darker than the midpoint between ink and ground.
/// Used by tests to assert something was actually drawn.
pub fn ink_coverage(img: &GrayImage, spec: &RenderSpec) -> f32 {
    let threshold = (spec.ink as u16 + spec.ground as u16) / 2;
    let dark = img.pixels().filter(|p| (p.0[0] as u16) < threshold).count();
    dark as f32 / (img.width() * img.height()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const HIRAGINO: &str = "/System/Library/Fonts/Hiragino Sans GB.ttc";

    fn font() -> Option<SynthFont> {
        SynthFont::from_path(HIRAGINO, 0).ok()
    }

    #[test]
    fn vertical_text_draws_ink_and_is_taller_than_wide() {
        let Some(f) = font() else { return }; // font is a host asset, not vendored
        let spec = RenderSpec {
            text: "立川で見た".into(),
            direction: Direction::VerticalRl,
            cells_per_run: 8,
            ..Default::default()
        };
        let img = render(&spec, &f).expect("render");
        assert!(
            img.height() > img.width(),
            "one vertical column should be tall"
        );
        let cov = ink_coverage(&img, &spec);
        assert!(cov > 0.02, "expected visible ink, got coverage {cov}");
    }

    #[test]
    fn horizontal_text_is_wider_than_tall() {
        let Some(f) = font() else { return };
        let spec = RenderSpec {
            text: "HELLO".into(),
            direction: Direction::HorizontalTb,
            cells_per_run: 8,
            ..Default::default()
        };
        let img = render(&spec, &f).expect("render");
        assert!(img.width() > img.height());
    }

    #[test]
    fn a_character_the_font_cannot_draw_is_refused_not_boxed() {
        let Some(f) = font() else { return };
        // U+E000 is a private-use codepoint no general font maps.
        let spec = RenderSpec {
            text: "ab\u{E000}".into(),
            ..Default::default()
        };
        let err = render(&spec, &f).expect_err("must refuse");
        assert!(err.contains("notdef"), "unexpected error: {err}");
    }

    #[test]
    fn empty_text_is_refused() {
        let Some(f) = font() else { return };
        let spec = RenderSpec {
            text: "   ".into(),
            ..Default::default()
        };
        assert!(render(&spec, &f).is_err());
    }

    #[test]
    fn wrapping_adds_a_column_rather_than_overflowing() {
        let Some(f) = font() else { return };
        let narrow = RenderSpec {
            text: "あいうえおかきくけこ".into(),
            cells_per_run: 5,
            ..Default::default()
        };
        let wide = RenderSpec {
            cells_per_run: 10,
            ..narrow.clone()
        };
        let a = render(&narrow, &f).expect("render");
        let b = render(&wide, &f).expect("render");
        assert!(
            a.width() > b.width(),
            "5-per-column should need more columns"
        );
        assert!(a.height() < b.height());
    }
}
