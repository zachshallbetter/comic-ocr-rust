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

### The vocabulary — partly closed, 2026-08-21

`Generator::from_dir` needs a `WordPieceVocab` to turn token ids into text. Two
things have changed since this was written, and the remaining gap is narrower
and in a different place than recorded here.

**The reference vocabulary was exported.** `models/onnx/vocab.txt` exists
(24 KB, alongside the three graphs). The reference model can be read end to end.
Note what it is: that file is the reference checkpoint's own 6,144-token
vocabulary, so it belongs to running a baseline, not to anything we train.

**The corpus-derived builder now has a caller.** `build_character_vocab` was
correct and tested from the start and nothing invoked it — seven unit tests over
a function that had never read a real transcription. `comic-ocr --build-vocab`
now consumes `schemas/training_pair.json` records and writes `vocab.txt` with a
provenance report, refusing to emit a file when the corpus is empty, when no
character clears the frequency floor, or when a `rejected` label appears.

**What is still missing is the corpus, not the builder.** The 3,004
transcriptions sit in CAS behind `RESOURCE_SIGNING_SECRET`; `asset_chunks`
carries digests but no bytes. Extraction belongs to the platform compiler that
holds that secret — Infinite-Verse#855 — not to a workaround here. The consumer
is ready and has nothing to consume.

A character vocabulary derived from our own corpus remains the goal: it is
sized to what we actually encounter and removes the last structural tie to
someone else's vocabulary.

### The third party's terms are unresolved

Stage 2 distils from Gemini output. Most provider terms restrict using output to
train a competing model, and that is not resolvable from inside this repository —
it is a real constraint, not a technicality.

It is also precisely why stage 1 leads. A model that already reads glyphs needs
far less of anyone else's output to specialise, and synthetic data carries no
such encumbrance.

---

## Order of work

1. **Vocabulary** — ~~emit `vocab.txt` on export~~ (done) and ~~build a
   corpus-derived character vocabulary~~ (builder wired; blocked on the corpus,
   Infinite-Verse#855). No longer blocks the wiring.
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
