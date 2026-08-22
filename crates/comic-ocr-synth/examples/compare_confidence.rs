//! Paired: the same text, once as a real scanned crop and once synthesised.
//!
//! `verify_realism` measures CER and is saturated -- the model reads eight
//! labels essentially perfectly from pristine through jpeg 30, so CER cannot
//! distinguish good synthetic data from excellent.
//!
//! Confidence has the dynamic range CER lacks: on real crops this model
//! separates 2.78% CER above 0.60 confidence from 77.59% below it. Pairing each
//! synthetic crop against the REAL crop of the same text controls for the text
//! itself, so a gap is attributable to the rendering.
//!
//! What a gap would mean. Synthetic confidence far ABOVE real says the crops are
//! too easy -- a model trained on them meets real pages under-prepared, and the
//! composed-confidence weighting in TRAINING_EXPORT would over-trust them.
//! Far BELOW says they are out of distribution and would teach the wrong thing.
//!
//!   COMIC_OCR_ONNX_DIR=... cargo run -p comic-ocr-synth --example compare_confidence

use comic_ocr_core::types::OcrEngine as _;
use comic_ocr_synth::degrade::{DegradeSpec, ScanQuality, apply};
use comic_ocr_synth::render::{Direction, RenderSpec, SynthFont, render};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let root = std::env::var("COMIC_OCR_CORPUS")
        .unwrap_or_else(|_| "/Users/zachshallbetter/Projects/comic-ocr-rust".into());
    let font_path = std::env::var("COMIC_OCR_SYNTH_FONT")
        .unwrap_or_else(|_| "/System/Library/Fonts/Hiragino Sans GB.ttc".into());
    let Ok(font) = SynthFont::from_path(&font_path, 0) else {
        eprintln!("no font at {font_path}");
        std::process::exit(2);
    };
    let engine = comic_ocr_ort::OrtEngine::new("compare");
    if engine.generator.is_none() {
        eprintln!("no generator; set COMIC_OCR_ONNX_DIR");
        std::process::exit(2);
    }

    let raw = std::fs::read_to_string(format!("{root}/tests/data/benchmark_results.json"))
        .expect("benchmark_results.json");
    let rows: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("valid json");

    println!(
        "{:<20} {:>6} {:>6} {:>6} {:>6} {:>7}  px x run",
        "label", "real", "clean", "typ", "poor", "delta"
    );
    println!("{}", "-".repeat(88));

    let (mut real_sum, mut synth_sum, mut n, mut skipped) = (0.0f32, 0.0f32, 0usize, 0usize);
    let mut deltas: Vec<f32> = Vec::new();
    let (mut t_sum, mut p_sum) = (0.0f32, 0.0f32);

    for row in rows.iter().filter(|r| r["label_kind"] == "crop") {
        let (Some(name), Some(text)) = (row["filename"].as_str(), row["expected_text"].as_str())
        else {
            continue;
        };
        let Ok(real_img) = image::open(format!("{root}/tests/data/images/{name}")) else {
            continue;
        };
        let Ok(real) = engine.predict(&real_img) else {
            continue;
        };

        // Match the real crop's scale. Assuming ONE column is wrong for anything
        // but short text: an 11-character balloon is typeset in several columns,
        // so height/chars underestimates the glyph size by roughly the column
        // count and produces renders far smaller than the crop they are paired
        // against.
        //
        // For C columns of N characters in a W x H crop:
        //     H ~= ceil(N/C) * font_px      W ~= C * font_px
        // Eliminating font_px gives C ~= sqrt(N * W / H).
        let chars = text.chars().filter(|c| !c.is_whitespace()).count().max(1);
        let (w, h) = (real_img.width() as f32, real_img.height() as f32);

        // Direction is not a constant. A crop wider than it is tall holds
        // horizontal text -- forcing VerticalRl on it packs 11 characters into
        // 10 one-character columns, which is nothing like the page and reads
        // worse than the too-small render it was meant to fix.
        let direction = if w > h {
            Direction::HorizontalTb
        } else {
            Direction::VerticalRl
        };

        // For R runs of N characters in a W x H crop, where a "run" is a column
        // (vertical) or a line (horizontal):
        //     vertical:   H ~= ceil(N/R) * px,  W ~= R * px
        //     horizontal: W ~= ceil(N/R) * px,  H ~= R * px
        // Eliminating px gives R ~= sqrt(N * across / along).
        let (along, across) = match direction {
            Direction::VerticalRl => (h, w),
            Direction::HorizontalTb => (w, h),
        };
        let runs = ((chars as f32 * across / along).sqrt()).round().max(1.0);
        let per_run = (chars as f32 / runs).ceil().max(1.0);
        let font_px = (along / per_run).clamp(12.0, 64.0);

        let spec = RenderSpec {
            text: text.to_string(),
            direction,
            font_px,
            cells_per_run: per_run as usize,
            ..Default::default()
        };
        let Ok(synth_img) = render(&spec, &font) else {
            skipped += 1;
            continue;
        };
        let mut rng = StdRng::seed_from_u64(20260822 + n as u64);
        let typ = apply(
            &synth_img,
            &DegradeSpec::sample_at(&mut rng, ScanQuality::Typical),
            &mut rng,
        )
        .unwrap_or_else(|_| synth_img.clone());
        let poor = apply(
            &synth_img,
            &DegradeSpec::sample_at(&mut rng, ScanQuality::Poor),
            &mut rng,
        )
        .unwrap_or_else(|_| synth_img.clone());
        let Ok(synth) = engine.predict(&image::DynamicImage::ImageLuma8(synth_img)) else {
            continue;
        };
        let Ok(synth_t) = engine.predict(&image::DynamicImage::ImageLuma8(typ)) else {
            continue;
        };
        let Ok(synth_p) = engine.predict(&image::DynamicImage::ImageLuma8(poor)) else {
            continue;
        };
        t_sum += synth_t.confidence;
        p_sum += synth_p.confidence;

        let d = synth.confidence - real.confidence;
        println!(
            "{:<20} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>+7.3}  {}{:>3.0}x{:<2.0}",
            text.chars().take(11).collect::<String>(),
            real.confidence,
            synth.confidence,
            synth_t.confidence,
            synth_p.confidence,
            d,
            if direction == Direction::VerticalRl {
                "V"
            } else {
                "H"
            },
            font_px,
            runs
        );
        deltas.push(d);
        real_sum += real.confidence;
        synth_sum += synth.confidence;
        n += 1;
    }

    if n == 0 {
        eprintln!("no paired crops");
        std::process::exit(1);
    }
    println!("{}", "-".repeat(88));
    let (rm, sm) = (real_sum / n as f32, synth_sum / n as f32);
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = deltas[deltas.len() / 2];
    println!(
        "mean over {n} paired crops   real {rm:.3}   synthetic {sm:.3}   delta {:+.3}",
        sm - rm
    );
    // One pathological real crop (confidence 0.208, well under the 0.60 line)
    // pairs against a trivially clean synthetic render and contributes +0.79 on
    // its own. The median is the honest summary of the typical pair.
    println!(
        "median delta {median:+.3}   worst {:+.3}   best {:+.3}",
        deltas[0],
        deltas[deltas.len() - 1]
    );
    // Does degrading the synthetic crop bring it back toward real difficulty?
    // CER could not answer this -- it was saturated. Confidence can.
    println!(
        "synthetic mean confidence:  clean {sm:.3}   typical-scan {:.3}   poor-scan {:.3}   (real {rm:.3})",
        t_sum / n as f32,
        p_sum / n as f32
    );
    if skipped > 0 {
        println!("{skipped} skipped: font could not cover the text (not counted either way)");
    }
    println!(
        "\nThe 0.60 threshold is where this model's real-crop CER goes from 2.78% to 77.59%.\n\
         Synthetic sitting far above real means the crops are easier than the pages\n\
         they are meant to prepare a model for."
    );
}
