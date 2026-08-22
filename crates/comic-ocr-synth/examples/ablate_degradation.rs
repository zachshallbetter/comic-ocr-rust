//! Which degradation moves the reference model's reading, and in which direction?
//!
//! `verify_realism` showed something counterintuitive: heavily degraded crops
//! read BETTER than clean ones. This isolates one parameter at a time to find
//! out why, because "degradation helps" is a conclusion that should not be
//! believed on an aggregate.
//!
//!   COMIC_OCR_ONNX_DIR=... cargo run -p comic-ocr-synth --example ablate_degradation

use comic_ocr_core::types::OcrEngine as _;
use comic_ocr_synth::degrade::{DegradeSpec, apply};
use comic_ocr_synth::render::{Direction, RenderSpec, SynthFont, render};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn cer(expected: &str, actual: &str) -> f64 {
    let a: Vec<char> = expected.chars().collect();
    let b: Vec<char> = actual.chars().collect();
    if a.is_empty() {
        return if b.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as f64 / a.len() as f64
}

const LABELS: &[&str] = &[
    "立川で見た",
    "なんちてキャー",
    "やったわさすがあたし",
    "そうだね",
    "ちょっとまって",
    "しかもなんか前より",
    "ウソでしょ",
    "また迷路だし",
];

fn main() {
    let font_path = std::env::var("COMIC_OCR_SYNTH_FONT")
        .unwrap_or_else(|_| "/System/Library/Fonts/Hiragino Sans GB.ttc".into());
    let Ok(font) = SynthFont::from_path(&font_path, 0) else {
        eprintln!("no font at {font_path}");
        std::process::exit(2);
    };
    let engine = comic_ocr_ort::OrtEngine::new("ablate");
    if engine.generator.is_none() {
        eprintln!("no generator; set COMIC_OCR_ONNX_DIR");
        std::process::exit(2);
    }

    // One variable at a time, everything else off.
    let arms: Vec<(String, DegradeSpec)> = vec![
        (
            "none".into(),
            DegradeSpec {
                jpeg_quality: None,
                ..Default::default()
            },
        ),
        (
            "blur 0.5".into(),
            DegradeSpec {
                jpeg_quality: None,
                blur_sigma: 0.5,
                ..Default::default()
            },
        ),
        (
            "blur 1.0".into(),
            DegradeSpec {
                jpeg_quality: None,
                blur_sigma: 1.0,
                ..Default::default()
            },
        ),
        (
            "blur 1.5".into(),
            DegradeSpec {
                jpeg_quality: None,
                blur_sigma: 1.5,
                ..Default::default()
            },
        ),
        (
            "blur 2.0".into(),
            DegradeSpec {
                jpeg_quality: None,
                blur_sigma: 2.0,
                ..Default::default()
            },
        ),
        (
            "blur 3.0".into(),
            DegradeSpec {
                jpeg_quality: None,
                blur_sigma: 3.0,
                ..Default::default()
            },
        ),
        (
            "jpeg 30".into(),
            DegradeSpec {
                jpeg_quality: Some(30),
                ..Default::default()
            },
        ),
        (
            "jpeg 75".into(),
            DegradeSpec {
                jpeg_quality: Some(75),
                ..Default::default()
            },
        ),
        (
            "rotate 3".into(),
            DegradeSpec {
                jpeg_quality: None,
                rotate_deg: 3.0,
                ..Default::default()
            },
        ),
        (
            "noise 20".into(),
            DegradeSpec {
                jpeg_quality: None,
                noise_sd: 20.0,
                ..Default::default()
            },
        ),
    ];

    println!("{:<12} {:>9}   errors", "arm", "mean CER");
    println!("{}", "-".repeat(64));

    for (name, spec) in &arms {
        let mut total = 0.0;
        let mut n = 0;
        let mut errs: Vec<String> = Vec::new();
        for label in LABELS {
            let rspec = RenderSpec {
                text: (*label).into(),
                direction: Direction::VerticalRl,
                font_px: 32.0,
                cells_per_run: 12,
                ..Default::default()
            };
            let Ok(clean) = render(&rspec, &font) else {
                continue;
            };
            // Fixed seed per arm so noise is the same draw across labels.
            let mut rng = StdRng::seed_from_u64(20260822);
            let Ok(img) = apply(&clean, spec, &mut rng) else {
                continue;
            };
            let Ok(out) = engine.predict(&image::DynamicImage::ImageLuma8(img)) else {
                continue;
            };
            let c = cer(label, &out.text);
            if c > 0.0 {
                errs.push(format!("{label}->{}", out.text));
            }
            total += c;
            n += 1;
        }
        if n > 0 {
            println!(
                "{name:<12} {:>8.2}%   {}",
                100.0 * total / n as f64,
                errs.join("  ")
            );
        }
    }

    println!(
        "\nResult 2026-08-22: every arm ties at 1.25% except blur 2.0 (0.00%) and\n\
         blur 3.0 (4.58%). That 1.25% is ONE label's hallucinated trailing punctuation,\n\
         which blur 2.0 happens to suppress -- n=1 on a single quirk, not evidence that\n\
         blur corrects toward the real distribution. Only blur >=3.0 produces genuine\n\
         character confusion.\n\n\
         The real finding is that this instrument is SATURATED: eight labels the model\n\
         reads essentially perfectly from pristine through jpeg 30, rotation 3 deg and\n\
         noise 20. It can confirm the crops are legible and cannot distinguish good\n\
         synthetic data from excellent. Harder material or far more of it is needed\n\
         before any degradation range can be tuned on evidence."
    );
}
