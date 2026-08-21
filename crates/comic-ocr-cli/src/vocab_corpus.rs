//! Build `vocab.txt` from an exported training corpus.
//!
//! [`comic_ocr_core::vocab_build::build_character_vocab`] has existed, tested,
//! since the vocabulary gap was written down — and until now nothing called it.
//! Seven unit tests over a function that had never read a real transcription is
//! the same shape as a green suite over a component that returns nothing: the
//! tests prove the arithmetic, not that the corpus is reachable or that the
//! output is usable.
//!
//! This module is that caller. It reads the records the platform's export
//! writes — [`schemas/training_pair.json`](../../../schemas/training_pair.json)
//! — rather than inventing a private corpus format, so whatever the platform
//! compiler produces feeds this directly with no adapter in between.
//!
//! ## What it refuses
//!
//! The vocabulary is a structural commitment: ids are positions, so every
//! checkpoint trained against a vocabulary is invalidated by a different one.
//! That makes silence the expensive failure mode, and three cases error rather
//! than produce a file:
//!
//! - **No records found.** Writing a vocabulary containing only the five
//!   special tokens is a valid-looking file that teaches nothing, and it would
//!   be indistinguishable from a corpus that legitimately held no text.
//! - **No characters admitted.** Same reasoning one level down: records were
//!   read, and the frequency floor rejected everything.
//! - **A `rejected` label.** `training_pair.json` excludes that state at the
//!   boundary, so encountering one means the input violates the export
//!   contract. Counting it and carrying on would fold human-invalidated
//!   readings into the vocabulary — the exact thing the state exists to stop.

use anyhow::{Context, bail};
use comic_ocr_core::vocab_build::{VocabReport, build_character_vocab};
use std::fs;
use std::path::{Path, PathBuf};

/// What the corpus scan found, kept alongside the vocabulary itself.
///
/// A vocabulary that shrank because the export was partial looks identical to
/// one built from a small corpus. Recording the inputs is what tells them
/// apart later, when the checkpoint is what's in hand and the corpus is gone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusReport {
    /// Files that parsed as training-pair records.
    pub files_read: usize,
    /// Records whose label contributed to the scan.
    pub records_used: usize,
    /// Records carrying `empty_is_intentional` — deliberate hard negatives,
    /// which have no characters to contribute and are not a defect.
    pub intentional_negatives: usize,
    /// The character-level result from `comic-ocr-core`.
    pub vocab: VocabReport,
}

/// One training-pair label, reduced to what the vocabulary needs.
struct Label {
    text: String,
    intentional_negative: bool,
}

/// Read every training-pair record under `path`.
///
/// Accepts a single `.jsonl` file (one record per line), a single `.json`
/// record, or a directory scanned recursively for both — the export writes one
/// record per pair, and a caller should not have to know which layout it got.
fn read_labels(path: &Path) -> anyhow::Result<(Vec<Label>, usize)> {
    let mut labels = Vec::new();
    let mut files_read = 0usize;

    if path.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .with_context(|| format!("reading corpus directory {}", path.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        // Sorted so a rebuild on an unchanged corpus reads in the same order.
        // The vocabulary is order-independent by construction, but the report's
        // counts should not depend on filesystem enumeration order either.
        entries.sort();
        for entry in entries {
            let (mut nested, count) = read_labels(&entry)?;
            labels.append(&mut nested);
            files_read += count;
        }
        return Ok((labels, files_read));
    }

    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if extension != "json" && extension != "jsonl" {
        return Ok((labels, 0));
    }

    let raw = fs::read_to_string(path)
        .with_context(|| format!("reading corpus file {}", path.display()))?;

    if extension == "jsonl" {
        for (index, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("parsing {} line {}", path.display(), index + 1))?;
            labels.push(label_from_record(&value, path, Some(index + 1))?);
        }
    } else {
        let value: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        labels.push(label_from_record(&value, path, None)?);
    }

    Ok((labels, 1))
}

/// Pull the label out of one record, enforcing the export contract's exclusion.
fn label_from_record(
    value: &serde_json::Value,
    path: &Path,
    line: Option<usize>,
) -> anyhow::Result<Label> {
    let where_ = match line {
        Some(n) => format!("{} line {}", path.display(), n),
        None => path.display().to_string(),
    };

    let state = value
        .get("provenance")
        .and_then(|p| p.get("state"))
        .and_then(|s| s.as_str());

    if state == Some("rejected") {
        bail!(
            "{where_} carries provenance.state='rejected'. training_pair.json excludes that \
             state at the export boundary, so this corpus violates the contract — a \
             human-invalidated reading must not reach the vocabulary. Fix the export rather \
             than filtering here."
        );
    }

    let text = value
        .get("text")
        .and_then(|t| t.as_str())
        .ok_or_else(|| anyhow::anyhow!("{where_} has no string `text` member"))?
        .to_string();

    let intentional_negative = value
        .get("empty_is_intentional")
        .and_then(|f| f.as_bool())
        .unwrap_or(false);

    Ok(Label {
        text,
        intentional_negative,
    })
}

/// Build the vocabulary and write it, with its report, next to each other.
///
/// Returns the report rather than printing it, so a caller can render it and a
/// test can assert on it.
pub fn build_from_corpus(
    corpus: &Path,
    out: &Path,
    min_frequency: usize,
) -> anyhow::Result<CorpusReport> {
    if !corpus.exists() {
        bail!("corpus path {} does not exist", corpus.display());
    }

    let (labels, files_read) = read_labels(corpus)?;

    if labels.is_empty() {
        bail!(
            "no training-pair records found under {}. Refusing to write a vocabulary of only \
             the five special tokens: that file would look valid, load cleanly, and decode \
             every real character as [UNK].",
            corpus.display()
        );
    }

    let intentional_negatives = labels.iter().filter(|l| l.intentional_negative).count();
    let (body, vocab) =
        build_character_vocab(labels.iter().map(|l| l.text.as_str()), min_frequency);

    if vocab.admitted == 0 {
        bail!(
            "{} records scanned and no character met the frequency floor of {}. Lower \
             --vocab-min-frequency or widen the corpus; writing this file would produce a \
             model that cannot spell.",
            labels.len(),
            min_frequency
        );
    }

    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    fs::write(out, &body).with_context(|| format!("writing {}", out.display()))?;

    let report = CorpusReport {
        files_read,
        records_used: labels.len(),
        intentional_negatives,
        vocab,
    };

    let report_path = out.with_file_name(format!(
        "{}.report.json",
        out.file_name().and_then(|n| n.to_str()).unwrap_or("vocab")
    ));
    let report_json = serde_json::json!({
        "min_frequency": min_frequency,
        "corpus": corpus.display().to_string(),
        "files_read": report.files_read,
        "records_used": report.records_used,
        "intentional_negatives": report.intentional_negatives,
        "distinct_characters": report.vocab.distinct_characters,
        "admitted": report.vocab.admitted,
        "dropped_rare": report.vocab.dropped_rare,
        "characters_scanned": report.vocab.characters_scanned,
        "empty_lines": report.vocab.empty_lines,
    });
    fs::write(
        &report_path,
        format!("{}\n", serde_json::to_string_pretty(&report_json)?),
    )
    .with_context(|| format!("writing {}", report_path.display()))?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temp directory that cleans up, without adding a dev-dependency for it.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!("comic-ocr-vocab-{tag}"));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp dir");
            TempDir(path)
        }
        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn pair(text: &str, state: &str) -> String {
        serde_json::json!({
            "crop": "crops/x.png",
            "text": text,
            "language": "ja",
            "direction": "ttb",
            "source": "own-corpus",
            "provenance": { "state": state, "confidence": 1.0 }
        })
        .to_string()
    }

    #[test]
    fn builds_a_vocabulary_from_exported_pairs() {
        let dir = TempDir::new("builds");
        let corpus = dir.join("pairs.jsonl");
        fs::write(
            &corpus,
            format!(
                "{}\n{}\n",
                pair("こんにちは", "verified"),
                pair("こんばんは", "accepted")
            ),
        )
        .unwrap();

        let out = dir.join("vocab.txt");
        let report = build_from_corpus(&corpus, &out, 1).expect("build");

        assert_eq!(report.records_used, 2);
        let body = fs::read_to_string(&out).unwrap();
        // Special tokens lead, in order, so id 0 is [PAD].
        assert!(body.starts_with("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"));
        assert!(body.contains('こ'));
        assert!(body.contains('ば'));
        // The report lands beside the vocabulary, not somewhere a later reader
        // has to go looking for.
        assert!(dir.join("vocab.txt.report.json").exists());
    }

    #[test]
    fn the_same_corpus_produces_a_byte_identical_vocabulary() {
        let dir = TempDir::new("determinism");
        let corpus = dir.join("pairs.jsonl");
        fs::write(
            &corpus,
            format!(
                "{}\n{}\n",
                pair("世界だれ", "verified"),
                pair("HELLO", "verified")
            ),
        )
        .unwrap();

        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        build_from_corpus(&corpus, &a, 1).unwrap();
        build_from_corpus(&corpus, &b, 1).unwrap();

        // Ids are positions. A vocabulary that reorders between builds silently
        // invalidates every checkpoint trained against the previous one, and
        // both files would look fine.
        assert_eq!(
            fs::read_to_string(&a).unwrap(),
            fs::read_to_string(&b).unwrap()
        );
    }

    #[test]
    fn an_empty_corpus_is_an_error_not_a_special_tokens_only_file() {
        let dir = TempDir::new("empty");
        let corpus = dir.join("empty");
        fs::create_dir_all(&corpus).unwrap();

        let out = dir.join("vocab.txt");
        let err = build_from_corpus(&corpus, &out, 1).expect_err("must refuse");
        assert!(
            err.to_string().contains("no training-pair records"),
            "unexpected error: {err}"
        );
        // The decisive part: nothing was written. A file here would load
        // cleanly and decode every real character as [UNK].
        assert!(!out.exists());
    }

    #[test]
    fn a_rejected_label_fails_the_build_rather_than_being_filtered() {
        let dir = TempDir::new("rejected");
        let corpus = dir.join("pairs.jsonl");
        fs::write(
            &corpus,
            format!(
                "{}\n{}\n",
                pair("よい", "verified"),
                pair("わるい", "rejected")
            ),
        )
        .unwrap();

        let out = dir.join("vocab.txt");
        let err = build_from_corpus(&corpus, &out, 1).expect_err("must refuse");
        assert!(
            err.to_string().contains("rejected"),
            "unexpected error: {err}"
        );
        assert!(!out.exists());
    }

    #[test]
    fn a_floor_that_admits_nothing_is_an_error() {
        let dir = TempDir::new("floor");
        let corpus = dir.join("pairs.jsonl");
        fs::write(&corpus, format!("{}\n", pair("あい", "verified"))).unwrap();

        let out = dir.join("vocab.txt");
        let err = build_from_corpus(&corpus, &out, 99).expect_err("must refuse");
        assert!(
            err.to_string().contains("frequency floor"),
            "unexpected error: {err}"
        );
        assert!(!out.exists());
    }

    #[test]
    fn the_written_file_loads_in_the_tokenizer_that_will_consume_it() {
        // The gap this whole module exists to close is "the model can run and
        // its output cannot be read". A vocabulary that builds but does not
        // load leaves that gap exactly where it was, so the round-trip is
        // asserted against the file on disk rather than the in-memory body —
        // including the space character, which English lettering puts in the
        // corpus and which a line-oriented parser is the most likely to drop.
        let dir = TempDir::new("loads");
        let corpus = dir.join("pairs.jsonl");
        fs::write(
            &corpus,
            format!(
                "{}\n{}\n",
                pair("WHAT WAS THAT", "verified"),
                pair("こんにちは", "verified")
            ),
        )
        .unwrap();

        let out = dir.join("vocab.txt");
        build_from_corpus(&corpus, &out, 1).unwrap();

        let vocab = comic_ocr_core::tokenizer::WordPieceVocab::from_file(&out)
            .expect("a vocabulary this command writes must load from disk");

        // [PAD] must be id 0: a padded position decoding to a real character is
        // the difference between a short reading and one with garbage appended.
        assert_eq!(vocab.id_of("[PAD]"), Some(0));
        for ch in "こんにちは".chars() {
            assert!(
                vocab.id_of(&ch.to_string()).is_some(),
                "admitted character {ch:?} has no id after a disk round-trip"
            );
        }
        assert!(
            vocab.id_of(" ").is_some(),
            "the space seen in English lettering was written but did not survive loading"
        );
    }

    #[test]
    fn intentional_negatives_are_counted_not_treated_as_missing_labels() {
        let dir = TempDir::new("negatives");
        let corpus = dir.join("pairs.jsonl");
        let negative = serde_json::json!({
            "crop": "crops/blank.png",
            "text": "",
            "empty_is_intentional": true,
            "language": "ja",
            "direction": "ttb",
            "source": "own-corpus",
            "provenance": { "state": "verified", "confidence": 1.0 }
        })
        .to_string();
        fs::write(
            &corpus,
            format!("{}\n{}\n", pair("ねこ", "verified"), negative),
        )
        .unwrap();

        let out = dir.join("vocab.txt");
        let report = build_from_corpus(&corpus, &out, 1).unwrap();

        assert_eq!(report.records_used, 2);
        assert_eq!(report.intentional_negatives, 1);
        // The blank contributes no characters but is not an error: a reviewer
        // confirmed the region carries no text, which is a real training signal.
        assert_eq!(report.vocab.empty_lines, 1);
    }
}
