# The path to weights that are ours

Status: normative. Written 2026-08-20, against measured counts.

How this repository gets a model it owns, given that the only weights currently
on disk are a converted copy of someone else's.

---

## The distinction the whole plan rests on

**Architecture is not weights.**

ViT/DeiT plus a BERT-family decoder is a public architecture. Using it derives
from nobody and encumbers nothing. What carries licence and lineage is the
**learned parameters**.

So `kha-white/manga-ocr-base` is a problem only if it is an *ancestor*. It is not
a problem as a *benchmark*.

That is the resolution adopted here:

| Relationship | Verdict |
| --- | --- |
| initialise our weights from it | **no** — that makes our model a derivative |
| ship it, or anything descended from it | **no** |
| run it to produce a baseline number | **yes** |
| distil from its output as a teacher | owner's call, not assumed |

`scripts/export_onnx.py` does `import manga_ocr; m = manga_ocr.MangaOcr()` and
exports that checkpoint. The 554 MB in `models/onnx/` is therefore a **reference
model**, and every claim measured against it — including the 29.41% CER the
benchmark reports — describes *its* quality, not ours.

`models/` is gitignored, so nothing ships today. The constraint is on what we
build *from*, not on what sits in a working tree.

---

## Three stages

### 1. Synthetic pretraining — teaches the model to read glyphs

The `source` enum in
[`schemas/training_pair.json`](../schemas/training_pair.json) already carries
`synthetic`; this is that path, and it is how OCR models are actually made.

Render text into crops directly: manga fonts, vertical and horizontal setting,
furigana, screentone and halftone grounds, balloon interiors, and the
degradations scanning introduces — JPEG artefacts, slight rotation, blur, bleed.

The label is known **by construction**. There is no detector and no transcriber
in the chain, so the composed-confidence rule does not apply: `confidence` is
1.0 because nothing inferred it.

**Status 2026-08-21: the generator exists** (`crates/comic-ocr-synth`), and the
crops have been checked against the one question that matters — does a model
that has never seen this generator read them as manga?

Measured with `cargo run -p comic-ocr-synth --example verify_realism`, eight
real balloon labels rendered vertically and read by the reference model:

| | mean CER |
| --- | --- |
| synthetic, clean | **1.25%** |
| synthetic, degraded (JPEG 63-89, rotation to 0.9 deg, blur to 0.81, noise to 5.7) | **1.25%** |
| *reference model on real crops above 0.60 confidence* | *2.78%* |

Seven of eight were exact; the single error was a hallucinated trailing `、`.

The crops occupy the right distribution: a model trained on real manga reads
them without being told they are synthetic.

**What the degradation numbers do NOT show.** I first read "degradation did not
move the CER" as ranges being too conservative, widened them, and found *heavier*
degradation scored **better** (0.00%) than clean. The tempting story — that
clean synthetic renders are unrealistically sharp and blur corrects toward real
scans — is not supported. `examples/ablate_degradation` isolates each parameter:

| arm | mean CER |
| --- | --- |
| none, blur 0.5–1.5, jpeg 30, jpeg 75, rotate 3°, noise 20 | 1.25% |
| blur 2.0 | 0.00% |
| blur 3.0 | 4.58% (女川 for 立川, 迷惑 for 迷路) |

Everything ties. The 1.25% is **one label's hallucinated trailing `、`**, which
blur 2.0 happens to suppress — n=1 on a single quirk. Only blur ≥3.0 causes
genuine character confusion.

### A paired instrument, because CER could not discriminate

`examples/compare_confidence` renders each of the 13 crop-labelled benchmark
entries as synthetic text and compares the model's **confidence** against the
real crop of the same string, at matched scale. Pairing controls for the text,
so a gap is attributable to the rendering. Confidence has the range CER lacks —
it is the axis on which real crops separate 2.78% CER from 77.59%.

Measured 2026-08-22 over 11 pairs (2 skipped: the font could not cover the text,
and the renderer refused rather than drawing notdef boxes):

| | |
| --- | --- |
| mean real confidence | 0.869 |
| mean synthetic confidence | 0.929 |
| **median delta** | **+0.023** |
| worst / best delta | −0.206 / +0.791 |

The median is the honest number. The +0.060 mean is dragged by one pair where
the real crop scores 0.208 — far below the 0.60 line, so it is a genuinely
unreadable scan — against a trivially clean synthetic render.

**What it caught that CER did not**, and what fixing it revealed:

```
第30話重苦しい闇の奥  ->  第30回国苦しい国の姿で静かに比較す   (0.655)
LINK!私達7人の力     ->  LINK!私学人の女子ガノンの学の場      (0.662)
```

Two pairs came out *worse* than their real counterparts, both at the smallest
rendered size. Diagnosing that took two attempts, and the first was wrong:

| render model | worst delta | median |
| --- | --- | --- |
| one column, direction forced vertical | −0.206 | +0.023 |
| column count estimated, direction still forced | **−0.339** | +0.036 |
| column count estimated **and direction inferred** | **+0.002** | +0.036 |

Estimating the column count was the obvious fix — an 11-character balloon is not
one column, so `height / chars` underestimates glyph size — and on its own it
made the result **worse**. The actual defect was forcing `VerticalRl` on every
crop. `LINK!私達7人の力` sits in a crop wider than it is tall; it is *horizontal*
text, and packing it into ten one-character columns produced something nothing
like the page. Inferring direction from the crop's aspect ratio took that pair
from 0.529 to 0.933, and every crop now reads its label.

**The result is a cleaner and less comfortable signal.** With the renderer
fixed, **all 11 pairs sit above their real counterparts** — worst delta +0.002,
mean +0.114. There is no longer any pair where synthetic is harder than real.
The generator produces uniformly easier crops than the pages it is meant to
prepare a model for, and the composed-confidence weighting in
[`TRAINING_EXPORT.md`](TRAINING_EXPORT.md) would over-trust them.

### Degradation is not what closes the gap

With a sensitive instrument available, the obvious next move was to tune the
`ScanQuality` ranges until synthetic difficulty matched real. Measured
2026-08-22, it does not work:

| | mean confidence |
| --- | --- |
| synthetic, clean | 0.983 |
| synthetic, typical scan | 0.967 |
| synthetic, **poor** scan (JPEG 22–55, ±4°, blur to 2.2, noise to 22) | 0.957 |
| **real crops** | **0.869** |

Aggressive scan degradation closes **0.026 of a 0.114 gap — about a fifth** —
and it does so unevenly: two crops drop sharply (1.000 → 0.849, 0.906 → 0.753)
while most barely move. Pushing the ranges further would degrade those two into
noise long before the average reached real difficulty.

**So what makes a real crop hard is not scan quality.** The remaining gap has to
live in what is still unmodelled:

- **Typography.** These renders use a clean system sans (Hiragino). Real manga is
  hand-lettered or set in display faces with different weights, proportions and
  stroke contrast.
- **Crop context.** Real boxes clip balloon borders, tails, and slivers of
  adjacent text; these renders sit on flat ground with clean margins.
- **Ground.** Real balloon interiors carry paper grain, tone bleed from adjacent
  panels, and uneven ink — not a uniform value with additive noise.

That is the next increment, and it is a redirection: **do not tune the
degradation ranges further.** They are doing what they can.

Separately, hallucinated continuation survives every rendering fix — `…で静かに
呼吸づ`, `…でガンの塔の結`, `…人達に!!` all appear after a correctly-read label.
That is decoder behaviour rather than a rendering flaw, and it is the same
pattern under investigation in Infinite-Verse#837.

**The CER probe, by contrast, is saturated.** Eight labels the model
reads essentially perfectly from pristine through jpeg 30, 3° rotation and
noise 20. It can confirm the crops are legible; it cannot distinguish good
synthetic data from excellent, and no degradation range can be tuned on it.
The `ScanQuality` ranges are therefore a starting position drawn from what scans
plausibly do, and their honest status is **unverified**. Harder material, or far
more of it, is the next increment — not wider ranges.

Two properties make this the load-bearing stage:

- **It is unlimited and free.** 100k pairs is compute, not a data-collection
  project.
- **It is unencumbered.** No provider terms, no corpus agreement, no third
  party's output. Weights trained here are ours without qualification.

Initialise from a permissively licensed general vision encoder — an
ImageNet-pretrained ViT under Apache or MIT. It knows about edges and shapes and
nothing about manga; everything domain-specific is learned from data we own.

### 2. Fine-tune on the real corpus — teaches it to read *these* pages

Measured 2026-08-20 on staging:

| | |
| --- | --- |
| transcriptions available | **3,004** |
| text regions detected | **3,372** |
| coverage | 89% of detected text is already transcribed |
| publications ingested | 7 of 167 available |

3,004 pairs cannot train a VisionEncoderDecoder from scratch. They are entirely
adequate to *adapt* a synthetically-pretrained model to real pages — actual paper
texture, actual balloon crops, actual typesetting, actual scan quality.

Weighted by composed confidence per
[`TRAINING_EXPORT.md`](TRAINING_EXPORT.md), and this is where the corpus does its
work.

### 3. The flywheel — the model becomes an independent reader

Once stage 2 produces a reader, ours and the third party fail *differently*.
Their disagreements concentrate where one of them is wrong, which is a far better
review queue than random sampling. Corrections raise confidence, retraining
sharpens, and the disagreement set shrinks while becoming more informative.

This is the loop the platform's `The Correction Flywheel` describes. It cannot
start turning until stage 2 exists.

---

## The corpus is effectively unbounded

`_references/Publications/` holds **167 archives, 12 GB**, and grows. Seven
publications are ingested — **4% of what is already on disk**, before anything
new is added.

Each ingested volume yields on the order of 150–190 pages, and a page carries
roughly 11 text regions. So the ceiling on real training pairs is not the
corpus; it is ingestion throughput and transcription cost.

That changes stage 2's character. It is not "we have 3,004 pairs and must make
them count" — it is "3,004 is the first 4%, and the constraint is how fast pages
move through detection and OCR."

### ISBN is the alignment key, and it is not working

Identity that survives re-ingestion is what lets a growing corpus be *organised*
rather than merely large: series grouping, volume ordering, language, publisher,
and the categorical metadata that makes a training split meaningful (train on
these series, hold out those) rather than arbitrary.

The machinery exists — `barcode.rs` decodes EAN-13 from a classified cover
(#540), `normalize_isbn` validates ISBN-10/13, and both feed
`publication_edition.identifiers`.

**It yields nothing on real content.** Measured 2026-08-20, all 7 live
publications:

```
Declared Edition        ["urn:ipub:sha256-52a33d5c6"]
Naruto (cm)             ["urn:ipub:sha256-f750bde93"]
Sakura Trick            ["urn:ipub:sha256-233a3986c"]
Real                    ["urn:ipub:sha256-d0c75fe21"]
Summertime Rendering    ["urn:ipub:sha256-3c89e5ee6"]
LoGH Vol. 1             ["urn:uuid:…", "urn:ipub:sha256-b12b9e636"]
LoGH Vol. 2             ["urn:uuid:…", "urn:ipub:sha256-082837754"]
```

Every identifier is a content digest. The two EPUBs add a publisher UUID. **Not
one ISBN**, across seven publications.

Getting this working at the onset matters more the larger the corpus grows: a
volume ingested without an identifier is a volume that cannot be aligned to its
series, its siblings, or its language later without re-deriving everything.
Fixing it at 7 publications is cheap; fixing it at 167 is a migration.

---

## Two gaps that block every path

### No vocabulary was exported

`Generator::from_dir` needs a `WordPieceVocab` to turn token ids into text.
`models/onnx/` holds three graphs and **no `vocab.txt`**, so the model can run and
its output cannot be read.

This is an opportunity rather than a chore. **Build the vocabulary from our own
corpus**: a character vocabulary derived from the 3,004 transcriptions is ours,
is sized to what we actually encounter, and removes the last structural tie to
someone else's 6,144-token vocabulary.

### The third party's terms are unresolved

Stage 2 distils from Gemini output. Most provider terms restrict using output to
train a competing model, and that is not resolvable from inside this repository —
it is a real constraint, not a technicality.

It is also precisely why stage 1 leads. A model that already reads glyphs needs
far less of anyone else's output to specialise, and synthetic data carries no
such encumbrance.

---

## Order of work

1. **Vocabulary** — emit `vocab.txt` on export, and build a corpus-derived
   character vocabulary. Needed by every path; blocks the wiring.
2. **Wire the generation loop** (Infinite-Verse#842) — against the reference
   graphs, to prove the loop end to end and produce the baseline number. Not to
   ship them.
3. **Synthetic generator** — the largest single determinant of whether we ever
   hold our own weights, and nothing blocks it: not attestation, not the
   provider's terms, not the platform.
4. **Train**, and measure against a **held-out human-labelled set** — the one
   thing neither synthetic data nor cross-model agreement can substitute for.
5. **ISBN at ingest**, before the corpus grows past the point where retrofitting
   identity is a migration.

## What is honestly still unknown

- Whether the reference checkpoint may be used as a distillation teacher. Owner's
  call; not assumed either way here.
- Whether Gemini's terms permit stage 2 at all.
- How much synthetic data closes the gap before real pages are needed. Unknown
  until stage 1 runs; the literature says "a lot, and less than you fear."
