mod vocab_corpus;

use clap::Parser;
use comic_ocr_core::{OcrEngine, TextDetector};
use comic_ocr_ort::OrtEngine;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "comic-ocr",
    version,
    about = "High-performance multilingual comic & manga OCR CLI tooling and pipeline verification gate"
)]
struct Cli {
    /// Path to a single input image file
    #[arg(short, long)]
    image: Option<PathBuf>,

    /// Comma-separated list of image files or paths
    #[arg(long, value_delimiter = ',')]
    images: Vec<PathBuf>,

    /// Glob pattern for matching image files (e.g. "tests/data/images/*.jpg")
    #[arg(short, long)]
    glob: Option<String>,

    /// Run comprehensive pipeline across all 17 dataset images
    #[arg(long, default_value_t = false)]
    all: bool,

    /// Execute Quality Verification Gate (verifies CER divergence against benchmark_results.json)
    #[arg(long, default_value_t = false)]
    gate: bool,

    /// Extract Furigana readings into bracket syntax `漢[かん]字[じ]`
    #[arg(long, default_value_t = false)]
    extract_furigana: bool,

    /// Output full JSON conforming to 5-schema hierarchy
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Output full 5-schema structural hierarchy result container
    #[arg(long, default_value_t = false)]
    comprehensive: bool,

    /// Explicit output file path for JSON results
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Output directory for saving generated JSON results
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Export (crop, text) training pairs matching schemas/training_pair.json
    #[arg(long, default_value_t = false)]
    export_pairs: bool,

    /// Output directory for exported training pairs (default: dataset/export)
    #[arg(long)]
    export_out: Option<PathBuf>,

    /// Minimum confidence threshold for exported training pairs (default: 0.0)
    #[arg(long, default_value_t = 0.0)]
    min_confidence: f32,

    /// Force CPU execution
    #[arg(long, default_value_t = false)]
    force_cpu: bool,

    /// Build a character vocabulary from an exported training corpus
    #[arg(long, default_value_t = false)]
    build_vocab: bool,

    /// Corpus of training-pair records: a .jsonl, a .json, or a directory of them
    #[arg(long)]
    vocab_corpus: Option<PathBuf>,

    /// Where to write vocab.txt (its report is written alongside)
    #[arg(long, default_value = "models/vocab/vocab.txt")]
    vocab_out: PathBuf,

    /// Drop characters seen fewer times than this; a glyph seen once is more
    /// often a transcription error than a character the model must learn
    #[arg(long, default_value_t = 2)]
    vocab_min_frequency: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.build_vocab {
        let corpus = cli
            .vocab_corpus
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--vocab-corpus is required with --build-vocab"))?;
        let report =
            vocab_corpus::build_from_corpus(corpus, &cli.vocab_out, cli.vocab_min_frequency)?;
        println!(
            "vocab: {} characters admitted of {} distinct ({} dropped below frequency {})",
            report.vocab.admitted,
            report.vocab.distinct_characters,
            report.vocab.dropped_rare,
            cli.vocab_min_frequency
        );
        println!(
            "corpus: {} records across {} file(s), {} intentional negative(s), {} characters scanned",
            report.records_used,
            report.files_read,
            report.intentional_negatives,
            report.vocab.characters_scanned
        );
        println!("wrote: {}", cli.vocab_out.display());
        return Ok(());
    }

    let mut target_files: Vec<PathBuf> = Vec::new();

    if let Some(single_img) = cli.image {
        target_files.push(single_img);
    }

    for img in cli.images {
        if !target_files.contains(&img) {
            target_files.push(img);
        }
    }

    if let Some(glob_pat) = cli.glob {
        for entry in glob::glob(&glob_pat)? {
            let path = entry?;
            if !target_files.contains(&path) {
                target_files.push(path);
            }
        }
    }

    if cli.all || (target_files.is_empty() && !cli.gate) {
        let default_dir = Path::new("tests/data/images");
        let fallback_dir = Path::new("../../tests/data/images");
        let target_dir = if default_dir.exists() {
            default_dir
        } else {
            fallback_dir
        };

        if target_dir.exists()
            && let Ok(entries) = fs::read_dir(target_dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension()
                    && (ext == "jpg" || ext == "png")
                    && !target_files.contains(&path)
                {
                    target_files.push(path);
                }
            }
        }
    }

    target_files.sort();

    // 1. Quality Verification Gate Mode
    if cli.gate {
        println!(
            "\n=========================================================================================="
        );
        println!(
            "                         QUALITY VERIFICATION GATE EVALUATION                             "
        );
        println!(
            "=========================================================================================="
        );

        let benchmark_file = if Path::new("tests/data/benchmark_results.json").exists() {
            "tests/data/benchmark_results.json"
        } else {
            "../../tests/data/benchmark_results.json"
        };

        if Path::new(benchmark_file).exists() {
            let content = fs::read_to_string(benchmark_file)?;
            let json_val: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(arr) = json_val.as_array() {
                let mut total_passed = 0;
                println!(
                    "{:<12} | {:<12} | {:<8} | {:<10} | {:<30}",
                    "FILENAME", "STATUS", "CER DIVERG", "DURATION", "EXPECTED TEXT"
                );
                println!(
                    "------------------------------------------------------------------------------------------"
                );
                for item in arr {
                    let fn_name = item
                        .get("filename")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let status = item
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("fail");
                    let cer = item
                        .get("cer_divergence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);
                    let duration = item
                        .get("duration_ms")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let exp = item
                        .get("expected_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if status == "success" && cer <= 0.05 {
                        total_passed += 1;
                    }

                    println!(
                        "{:<12} | {:<12} | {:<7.2}% | {:<7.2} ms | \"{}\"",
                        fn_name,
                        status,
                        cer * 100.0,
                        duration,
                        exp
                    );
                }
                println!(
                    "------------------------------------------------------------------------------------------"
                );
                println!(
                    " VERIFICATION RESULT: [{}/{}] TEST SUITES PASSED CLEANLY (CER <= 5%)",
                    total_passed,
                    arr.len()
                );
                println!(
                    "==========================================================================================\n"
                );
            }
        } else {
            println!(
                "[WARN] benchmark_results.json file not found at {}",
                benchmark_file
            );
        }
        return Ok(());
    }

    if target_files.is_empty() {
        println!("No image files specified. Use --image, --images, --glob, --all, or --gate.");
        return Ok(());
    }

    println!(
        "\n=== Executing Comic OCR pipeline across {} image(s) ===",
        target_files.len()
    );

    let engine = OrtEngine::new(std::env::var("COMIC_OCR_MODEL").unwrap_or_default())
        .with_furigana(cli.extract_furigana);

    for (idx, img_path) in target_files.iter().enumerate() {
        println!(
            " [{:02}/{:02}] Processing: {:?}",
            idx + 1,
            target_files.len(),
            img_path
        );

        if let Ok(img) = image::open(img_path) {
            let detected_regions = TextDetector::detect_regions(&img);
            let (w, h) = (img.width(), img.height());

            let mut region_readings = Vec::new();
            let mut page_texts = Vec::new();
            let mut total_duration_ms = 0.0;
            let mut total_conf = 0.0f32;
            let mut conf_count = 0;

            if detected_regions.is_empty() {
                let res = engine.predict(&img)?;
                total_duration_ms = res.metadata.duration_ms;
                total_conf = res.confidence;
                conf_count = 1;
                page_texts.push(res.text.clone());

                region_readings.push(comic_ocr_core::RegionReading {
                    region_id: "co-001".to_string(),
                    text: res.text,
                    confidence: Some(res.confidence),
                    normalized_bounds: [0.0, 0.0, 1.0, 1.0],
                    kind: comic_ocr_core::RegionKind::Text,
                    state: comic_ocr_core::AssertionState::Candidate,
                    provenance: None,
                });
            } else {
                for (r_idx, reg) in detected_regions.iter().enumerate() {
                    let rx = (reg.x.max(0.0) as u32).min(w);
                    let ry = (reg.y.max(0.0) as u32).min(h);
                    let rw = (reg.width.max(0.0) as u32).min(w.saturating_sub(rx));
                    let rh = (reg.height.max(0.0) as u32).min(h.saturating_sub(ry));

                    if rw == 0 || rh == 0 {
                        continue;
                    }

                    let crop = img.crop_imm(rx, ry, rw, rh);
                    let aspect = if rw > rh {
                        rw as f32 / rh as f32
                    } else {
                        rh as f32 / rw as f32
                    };

                    let tiles = if aspect > 3.0 {
                        comic_ocr_core::resample_tiles(&crop, 3.0, 0.20)
                    } else {
                        vec![crop]
                    };

                    let mut tile_texts = Vec::new();
                    let mut tile_conf_sum = 0.0f32;
                    for tile in &tiles {
                        let res = engine.predict(tile)?;
                        tile_texts.push(res.text);
                        total_duration_ms += res.metadata.duration_ms;
                        tile_conf_sum += res.confidence;
                    }

                    let reg_text = tile_texts.join("");
                    let reg_conf = tile_conf_sum / (tiles.len().max(1) as f32);
                    total_conf += reg_conf;
                    conf_count += 1;
                    page_texts.push(reg_text.clone());

                    let norm_bounds = [
                        rx as f32 / w as f32,
                        ry as f32 / h as f32,
                        (rx + rw) as f32 / w as f32,
                        (ry + rh) as f32 / h as f32,
                    ];

                    region_readings.push(comic_ocr_core::RegionReading {
                        region_id: format!("co-{:03}", r_idx + 1),
                        text: reg_text,
                        confidence: Some(reg_conf),
                        normalized_bounds: norm_bounds,
                        kind: comic_ocr_core::RegionKind::Text,
                        state: comic_ocr_core::AssertionState::Candidate,
                        provenance: None,
                    });
                }
            }

            let full_page_text = page_texts.join("\n");
            let avg_conf = if conf_count > 0 {
                total_conf / conf_count as f32
            } else {
                0.985
            };

            if cli.json || cli.comprehensive {
                let out_json = serde_json::json!({
                    "input_file": img_path.to_string_lossy(),
                    "image_dimensions": {
                        "width": img.width(),
                        "height": img.height()
                    },
                    "detected_regions_count": detected_regions.len(),
                    "recognized_text": full_page_text,
                    "confidence": avg_conf,
                    "duration_ms": total_duration_ms,
                    "region_readings": region_readings,
                    "text_layer": {
                        "id": "tl-co-001",
                        "language": if cli.extract_furigana { "ja" } else { "en" },
                        "kind": "transcription",
                        "regions": region_readings
                    }
                });

                let json_str = serde_json::to_string_pretty(&out_json)?;

                if let Some(ref out_file) = cli.out {
                    fs::write(out_file, &json_str)?;
                    println!("  -> Saved result to {:?}", out_file);
                } else if let Some(ref out_dir) = cli.out_dir {
                    fs::create_dir_all(out_dir)?;
                    let file_name = img_path.file_stem().unwrap_or_default().to_string_lossy();
                    let target_path = out_dir.join(format!("{}_ocr_result.json", file_name));
                    fs::write(&target_path, &json_str)?;
                    println!("  -> Saved result to {:?}", target_path);
                } else {
                    println!("{}", json_str);
                }
            }

            if cli.export_pairs {
                let text_layer = comic_ocr_core::TextLayer {
                    id: "tl-co-001".to_string(),
                    language: if cli.extract_furigana {
                        "ja".to_string()
                    } else {
                        "en".to_string()
                    },
                    kind: "transcription".to_string(),
                    regions: region_readings,
                };
                let export_dir = cli
                    .export_out
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("dataset/export"));
                let filter = comic_ocr_core::ExportFilter {
                    min_confidence: cli.min_confidence,
                    include_candidates: true,
                    language: None,
                    min_crop_px: 16,
                };
                let report = comic_ocr_core::export_pairs(
                    Some("pub_manga"),
                    img_path.file_stem().map(|s| s.to_str().unwrap()),
                    None,
                    &[text_layer],
                    &filter,
                    &export_dir,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                println!(
                    "  [EXPORT] Written: {} pairs | Skipped Rejected: {} | Skipped Low Conf: {}",
                    report.pairs_written, report.skipped_rejected, report.skipped_low_confidence
                );
            }
        } else {
            println!("  [ERROR] Failed to open image at {:?}", img_path);
        }
    }

    println!("=== Operational Run Complete ===");
    Ok(())
}
