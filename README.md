# Military Drone Detection and Origin Analysis

An end-to-end computer-vision project that builds audited datasets, trains two
Faster R-CNN models, benchmarks false positives on held-out hard negatives, and
serves results through a CLI and Streamlit app.

The system deliberately separates two questions:

1. **Detection:** Is there a drone, and where is it?
2. **Attribution:** If a drone was localized, which country designed or
   manufactured that airframe?

The second answer is a ranked hypothesis, not an ownership or operator claim.

**Start here:** [run the demo](#run-the-demo), [review the results](#measured-result),
[rebuild the datasets](#rebuild-and-validate), or [run the tests](#tests).

## Measured result

All numbers below are from held-out test data, never the training split.

| Model / operating point | Result |
|---|---:|
| Binary baseline mAP@0.5 | 0.2070 |
| Binary baseline FPR at score 0.5 | 89.8% (1,017/1,133) |
| Final binary mAP@0.5 | **0.9503** |
| Final binary mAP@0.5:0.95 | **0.6972** |
| Final binary FPR / recall at score 0.5 | **7.82% / 100%** (92/1,177 negatives) |
| Bird-flock FPR at score 0.5 | **2.27%** (1/44; before mining: 22/44) |
| Conservative point, score 0.9 | **1.78% FPR, 96.76% recall** |
| Binary detector speed, RTX 3050 | 60.2 ms/image, 16.60 FPS |
| Country-aware mAP@0.5 | 0.1864 |
| Country top-1, given IoU >= 0.5 | **33.6%** (162/482 boxes) |
| Country top-3, given IoU >= 0.5 | **59.1%** (285/482 boxes) |
| Country-model speed, RTX 3050 | 47.2 ms/image, 21.18 FPS |

The hard-negative retrains were necessary. The first reduced false alarms at
score 0.5 from 89.8% to 10.1%. A later bird-flock audit exposed 22 false alerts
in 44 unseen flock images; focused mining reduced that to 1/44 while preserving
100% image-level recall at the demo's 0.5 threshold.

The country model is useful for narrowing an analyst's search, but not for an
autonomous claim. It localizes drones class-agnostically at 0.5334 mAP@0.5, then
loses 55.8% of that mAP when the country label must also be correct. Similar
airframes, weak web labels, and long-tail classes are the main limitations.

Detailed evidence is in:

- `reports/binary_v3_baseline_corrected_test.txt`
- `reports/binary_v3_retrained_test.txt`
- `reports/binary_v5_birds_test.txt`
- `reports/binary_v5_on_original_test.txt`
- `reports/binary_v5_cross_domain_country_test.txt`
- `reports/country_v3_test.txt`
- `reports/country_v3_diagnosis.txt`

## Architecture

```text
input image
    |
    +--> binary Faster R-CNN (ResNet50-FPN) --> drone boxes and alert scores
    |
    +--> 18-class Faster R-CNN -------------> country boxes and scores
              |
              +-- pair to binary boxes by IoU
                  and return top-k origin candidates
```

Keeping the stages separate prevents uncertain attribution from suppressing a
good binary alert. Each model can also be retrained or calibrated independently.

## Project layout

| Path | Purpose |
|---|---|
| `app.py`, `predict.py` | Streamlit and command-line inference |
| `models/` | Faster R-CNN construction and two-stage inference |
| `data/build/` | Source collection, auto-labelling, deduplication, and assembly |
| `data/dataset.py`, `data/transforms.py` | Training data loading and augmentation |
| `train.py`, `evaluate.py` | Training and held-out evaluation |
| `tools/` | Diagnostics and optional analysis commands |
| `tests/` | Focused regression checks |
| `reports/` | Benchmark outputs and [evidence index](reports/README.md) |

Large datasets, model checkpoints, raw provenance records, and run outputs are
local artifacts excluded from the code-only GitHub release. Keep
`dataset_raw/` with your local datasets when reproducing the experiments.

## Datasets

### Binary dataset

`drone_dataset_binary_v4_birds` contains **22,784 images**:

- 20,016 train, 1,325 validation, 1,443 test
- 11,663 drone-positive and 11,121 negative images
- 11,879 boxes
- 684 duplicates removed
- hard negatives include birds, 222 licensed Wikimedia bird-flock images,
  aircraft, helicopters, bombs, and real deployment false alarms

After score/area quality filters, the test loader evaluates 1,424 images: 247
positive and 1,177 negative. This filtered count is expected and is printed by
the evaluator.

### Country dataset

`drone_dataset_country_v3` contains **15,118 images**:

- 12,092 train, 1,513 validation, 1,513 test
- 4,219 positive and 10,899 negative images
- 18 countries: China, France, Germany, India, Iran, Israel, Italy, Pakistan,
  Poland, Russia, Serbia, South Africa, South Korea, Turkey, UAE, UK, USA, and
  Ukraine
- 440 duplicates removed; no cross-label conflict groups retained

Country labels describe airframe design/manufacturing origin. Query provenance
supplies the weak country label; Grounding DINO supplies proposed box geometry.
The loader drops uncertain or oversized boxes, so the country test benchmark
uses 1,476 of the 1,513 files.

An independent perceptual-duplicate audit found zero train/validation/test
leakage pairs in the final country dataset. Dataset validators also report zero
structural, box, and label errors.

## Data provenance and licensing

- **UAV_SMID:** 13,928 real images, CC BY 4.0, DOI `10.17632/3k3hjc7rkt.1`.
- **Synthetic military-drone data:** 9,000 train-only images, CC0.
- **AeroScan:** Apache 2.0 source; 1,567 relevant positive images and 2,792
  imported UAV boxes.
- **Web collection:** 3,238 unique records, of which 522 have explicit Wikimedia
  Commons licensing. For 2,716 source-page records, redistribution terms were
  not established.

Accordingly, the assembled web dataset is a local research artifact and should
not be redistributed as a blanket-licensed dataset. Keep each image's manifest
and source URL when reviewing or replacing it.

## Setup

Windows PowerShell example:

```powershell
uv venv
uv pip install "torch>=2.6.0" "torchvision>=0.21.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last command should print a CUDA build and `True` before GPU training.

## Run the demo

```powershell
# One image; the demo uses score threshold 0.50 to favor recall.
.venv\Scripts\python.exe predict.py --image path\to\image.jpg --output result.jpg --json result.json

# Interactive application.
.venv\Scripts\streamlit.exe run app.py
```

Expected checkpoints:

- `weights/binary_v5_birds_focused/best_model.pth`
- `weights/country_v3/best_model.pth`

Checkpoints are local artifacts and are not included in Git. Supply these files
before starting the app or CLI on a fresh clone.

The UI calls country score shares *relative score shares*. They are normalized
ranking aids, not calibrated probabilities.

## Rebuild and validate

```powershell
# Binary data from downloaded sources and mined deployment incidents.
.venv\Scripts\python.exe -m data.build.collect_bird_flocks --target 250
.venv\Scripts\python.exe -m data.build.build_binary_dataset --out drone_dataset_binary_v4_birds

# Country collection, weak geometry labels, and final assembly.
.venv\Scripts\python.exe -m data.build.scrape
.venv\Scripts\python.exe -m data.build.autolabel --batch-size 1 --device cuda
.venv\Scripts\python.exe -m data.build.build_country_dataset --out drone_dataset_country_v3

# Structure, taxonomy, boxes, duplicates, and split leakage.
.venv\Scripts\python.exe -m data.validate_dataset --root drone_dataset_binary_v4_birds --check-leakage
.venv\Scripts\python.exe -m data.validate_dataset --root drone_dataset_country_v3 --check-leakage
```

The builders are deterministic at seed 1337. Synthetic positives remain
train-only, and duplicates are grouped before splitting.

The binary builder needs the extracted `dataset_sources/UAV_SMID/dataset` and
`dataset_sources/military_synthetic/dataset` folders. The country builder also
uses `dataset_sources/aeroscan/dataset`, `dataset_raw/`, and the historical
`drone_dataset_binary_v3_dedup2` negatives. Source ZIP files are optional once
their extracted folders exist. Historical deployment frames in `detections/`
are needed only to repeat the original hard-negative mining step.

## Train

The following commands reproduce the current training configuration. A 4 GB GPU
requires batch size 1 for the 18-class model.

```powershell
# Focused binary hard-negative adaptation from the previous binary checkpoint.
.venv\Scripts\python.exe train.py --root drone_dataset_binary_v4_birds --binary --resume weights/binary_v4_birds/best_model.pth --reset-best --epochs 7 --batch-size 2 --adverse-prob 0 --balance-negative-categories --focus-negative-category bird_flock_commons --focus-negative-mass 0.35 --samples-per-epoch 4000 --weights-dir weights/binary_v5_birds_focused --no-tensorboard --step-checkpoint-every 100 --skip-val-loss

# Country detector, warm-starting shared detector features.
.venv\Scripts\python.exe train.py --root drone_dataset_country_v3 --warm-start weights/checkpoint_epoch15.pth --epochs 2 --batch-size 1 --min-size 320 --max-size 512 --adverse-prob 0 --balance-classes 0.5 --weights-dir weights/country_v3 --no-tensorboard --step-checkpoint-every 100 --skip-val-loss
```

Training uses mixed precision, gradient clipping, resumable checkpoints, and a
square-root inverse-frequency sampler for country imbalance. Very large images
are decoded at reduced resolution before transforms to avoid host-memory spikes;
boxes are scaled to match.

## Benchmark

```powershell
.venv\Scripts\python.exe evaluate.py --root drone_dataset_binary_v4_birds --weights weights/binary_v5_birds_focused/best_model.pth --binary --split test --fpr-thresh 0.5 --out reports/binary_v5_birds_test.txt

.venv\Scripts\python.exe evaluate.py --root drone_dataset_country_v3 --weights weights/country_v3/best_model.pth --split test --fpr-thresh 0.5 --out reports/country_v3_test.txt

.venv\Scripts\python.exe -m tools.diagnose_taxonomy --root drone_dataset_country_v3 --weights weights/country_v3/best_model.pth --split test --out reports/country_v3_diagnosis.txt
```

`evaluate.py` reports COCO-style box AP, image-level precision/recall/specificity,
false-positive Wilson intervals, hard-negative categories, throughput, and an
operating-point sweep. `diagnose_taxonomy.py` separates localization error from
country-classification error.

## Tests

The small data checks use Python's standard-library test runner:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_transforms tests.test_quarantine
```

`tests.test_resume` exercises checkpoint restoration with a model and takes
longer to run.

## Tested examples

- The final detector found all 247 drone-positive images in the main held-out
  test loader at threshold 0.50.
- It falsely alerted on 1 of 44 unseen bird-flock images; the pre-mining model
  alerted on 22 of the same 44.
- On the independently assembled country-domain test set, binary mAP@0.5
  improved from 0.280 to 0.360 and image recall from 47.8% to 51.7%.

These cases demonstrate why threshold calibration, domain-shift tests, and hard-
negative evaluation matter more than showing only successful positive images.

## Main limitations and next work

- Country labels are weak labels derived from search provenance and need expert
  review before safety-critical use.
- Several country classes are small; per-class percentages such as Germany's
  are unstable because the localized test denominator is tiny.
- The benchmark measures still images, not range in metres. Physical range
  depends on optics, sensor resolution, target size, atmosphere, and camera
  geometry.
- Next improvements should prioritize human-reviewed country labels, more
  thin-class examples, probability calibration, temporal tracking for video,
  and evaluation on a separately collected deployment domain.

See `INTERVIEW_GUIDE.md` for concise explanations of the design and metrics.
