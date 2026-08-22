//! Does the reference reader recognise our synthetic crops as manga text?
//!
//! An example rather than a test, for the same reason `eval_corpus` is: it needs
//! a model directory and ~554 MB of graphs, and a test that silently skips
//! without them is indistinguishable from one that passes.
//!
//!   COMIC_OCR_ONNX_DIR=models/onnx cargo run -p comic-ocr-synth --example verify_realism
//!
//! What the number means. The reference model was trained on real manga crops
//! and knows nothing about this generator. If it reads synthetic crops at a CER
//! near its own performance on real ones, the crops occupy roughly the right
//! distribution. A high CER says the synthetic data is unrealistic -- which is
//! the failure this example exists to catch BEFORE 100k pairs are generated from
//! it and a model is trained on a distribution real pages never occupy.
//!
//! It is deliberately not a pass/fail gate. It is a measurement.

use comic_ocr_core::types::OcrEngine as _;
use comic_ocr_synth::degrade::{DegradeSpec, ScanQuality, apply};
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

/// Real balloon text, so the character distribution is not invented.
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
        eprintln!("no font at {font_path}; set COMIC_OCR_SYNTH_FONT");
        std::process::exit(2);
    };

    let engine = comic_ocr_ort::OrtEngine::new("synth-realism");
    if engine.generator.is_none() {
        eprintln!(
            "no generator loaded; set COMIC_OCR_ONNX_DIR to a directory with the three graphs and vocab.txt"
        );
        std::process::exit(2);
    }

    let mut rng = StdRng::seed_from_u64(20260821);
    let mut clean_total = 0.0;
    let mut degraded_total = 0.0;
    let mut poor_total = 0.0;
    let mut n = 0usize;

    println!(
        "{:<24} {:>8} {:>8}  {:<24} prediction",
        "label", "clean", "degraded", "degradation applied"
    );
    println!("{}", "-".repeat(96));

    for label in LABELS {
        let spec = RenderSpec {
            text: (*label).into(),
            direction: Direction::VerticalRl,
            font_px: 32.0,
            cells_per_run: 12,
            ..Default::default()
        };
        let Ok(clean) = render(&spec, &font) else {
            eprintln!("skip (font coverage): {label}");
            continue;
        };
        let dspec = DegradeSpec::sample_at(&mut rng, ScanQuality::Typical);
        let degraded = apply(&clean, &dspec, &mut rng).expect("degrade");
        let pspec = DegradeSpec::sample_at(&mut rng, ScanQuality::Poor);
        let poor = apply(&clean, &pspec, &mut rng).expect("degrade");

        let Ok(c) = engine.predict(&image::DynamicImage::ImageLuma8(clean)) else {
            eprintln!("predict failed (clean): {label}");
            continue;
        };
        let Ok(d) = engine.predict(&image::DynamicImage::ImageLuma8(degraded)) else {
            eprintln!("predict failed (degraded): {label}");
            continue;
        };
        let Ok(p) = engine.predict(&image::DynamicImage::ImageLuma8(poor)) else {
            eprintln!("predict failed (poor): {label}");
            continue;
        };
        let (cc, dd, pp) = (
            cer(label, &c.text),
            cer(label, &d.text),
            cer(label, &p.text),
        );
        poor_total += pp;
        clean_total += cc;
        degraded_total += dd;
        n += 1;

        // Report how much the degradation actually perturbed the image, so a
        // no-op degradation cannot masquerade as robustness.
        println!(
            "{label:<22} {:>6.1}% {:>6.1}% {:>6.1}%   poor: q{:<3} r{:+.1} b{:.1} n{:.0}  {}",
            100.0 * cc,
            100.0 * dd,
            100.0 * pp,
            pspec.jpeg_quality.unwrap_or(0),
            pspec.rotate_deg,
            pspec.blur_sigma,
            pspec.noise_sd,
            p.text
        );
    }

    if n == 0 {
        eprintln!("nothing rendered");
        std::process::exit(1);
    }
    println!("{}", "-".repeat(96));
    println!(
        "mean CER over {n} crops   clean {:.2}%   typical {:.2}%   poor {:.2}%",
        100.0 * clean_total / n as f64,
        100.0 * degraded_total / n as f64,
        100.0 * poor_total / n as f64
    );
    println!(
        "\nReference: this model reads REAL crops at 2.78% CER above 0.60 confidence\n\
         and 77.59% below it. A synthetic CER in the former range means the crops\n\
         look like manga to a model that has never seen this generator."
    );
}
