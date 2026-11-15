# Interview Guide

## Thirty-second explanation

“I built an end-to-end military-drone vision system with two Faster R-CNN
models. The first model detects any drone; the second ranks 18 possible design
countries. I assembled and audited a 22,784-image binary dataset and a
15,118-image country dataset, grouped duplicates before splitting, and included
11,121 hard negatives. The original detector had an 89.8% false-positive rate
at threshold 0.5. After hard-negative retraining it reached 0.950 mAP@0.5,
7.82% FPR, and 100% recall at 0.5. A focused bird-flock pass cut unseen flock
false positives from 22/44 to 1/44. Country attribution is much harder—33.6% top-1 and 59.1%
top-3—so the application presents a shortlist, not a factual nationality.”

## System flow

```text
image
  -> binary Faster R-CNN -> drone boxes, alert score, NMS
  -> country Faster R-CNN -> 18-class boxes
  -> match country boxes to accepted binary boxes using IoU
  -> show top-k design-origin candidates
```

The binary model controls whether an alert exists. The country stage cannot
erase a valid alert and is allowed to return no attribution.

## Numbers to remember

| Item | Number |
|---|---:|
| Binary dataset | 22,784 images |
| Country dataset | 15,118 images |
| Hard-negative images | 11,121 |
| Countries | 18 |
| Binary baseline mAP@0.5 | 0.2070 |
| Binary baseline FPR at 0.5 | 89.8% |
| Final binary mAP@0.5 | 0.9503 |
| Final binary mAP@0.5:0.95 | 0.6972 |
| Final FPR / recall at 0.5 | 7.82% / 100% |
| Unseen bird-flock FPR at 0.5 | 2.27% (1/44) |
| Conservative FPR / recall at 0.90 | 1.78% / 96.76% |
| Country mAP@0.5 | 0.1864 |
| Country top-1 / top-3 after localization | 33.6% / 59.1% |
| Binary speed on RTX 3050 | 16.60 FPS |
| Country-model speed on RTX 3050 | 21.18 FPS |

Do not say that the complete two-stage pipeline runs at 16.60 or 21.18 FPS.
Those timings are per model; end-to-end latency also includes both passes,
matching, decoding, and visualization.

## Why two models?

Detection and attribution have different difficulty and risk. A silhouette may
be enough to decide “drone,” but not enough to distinguish similar airframes
from different countries. Separating the objectives gives three advantages:

- uncertain attribution cannot suppress a binary alert
- each stage can use its own resolution and operating threshold
- either stage can be retrained or replaced independently

The country model runs on the full image and proposes its own boxes. The
pipeline pairs those boxes to binary detections at IoU >= 0.3, keeps the best
score per country, and returns the top candidates. The displayed relative score
share is a ranking aid, not a calibrated probability.

## Why Faster R-CNN?

Faster R-CNN is a two-stage detector: a Region Proposal Network proposes likely
objects, then an ROI head refines the box and class. It is a sensible interview
baseline for small or sparse objects because proposal quality and final
classification are explicit and easy to diagnose. ResNet50-FPN adds multi-scale
features, which helps when a drone occupies few pixels.

A one-stage detector such as YOLO would likely be faster and is a good next
deployment comparison. I did not claim Faster R-CNN is universally superior;
I chose it because this project prioritized detection quality, false-positive
analysis, and interpretable error decomposition on available hardware.

## How the datasets were built

### Binary dataset

- UAV_SMID contributes real positives and distractors.
- Only `drone` boxes are foreground.
- birds, aircraft, helicopters, and bombs are retained with empty annotations
  as hard negatives
- 222 licensed Wikimedia bird-flock images were added; 44 remained test-only
- 9,000 CC0 synthetic positives are train-only
- previous deployment false alarms are mined as additional negatives
- perceptual duplicates are grouped before deterministic splitting

The final binary dataset has 22,784 files: 11,663 positives, 11,121 negatives,
and 11,879 boxes. Quality filtering leaves 1,424 usable held-out test images.

### Country dataset

- AeroScan contributes licensed UAV boxes.
- Search queries identify candidate airframes and country provenance.
- Grounding DINO proposes weak bounding boxes.
- uncertain, oversized, malformed, and known ambiguous annotations are filtered
- the same hard negatives are included
- classes with fewer than 20 examples would be removed; all 18 retained classes
  passed that build threshold

The final country dataset has 15,118 files: 4,219 positives and 10,899 negatives.
It covers China, France, Germany, India, Iran, Israel, Italy, Pakistan, Poland,
Russia, Serbia, South Africa, South Korea, Turkey, UAE, UK, USA, and Ukraine.

The label means *airframe design/manufacturing origin*. It does not mean current
operator, launch country, ownership, or location.

## How leakage was prevented

Exact and perceptually similar images can make a random image-level split look
artificially strong. The builders therefore hash/group duplicates before split
assignment. Synthetic images are train-only. A final independent audit found
zero near-duplicate pairs across train/validation/test for the country dataset,
and validators found zero structural or annotation issues in each final split.

## Why hard negatives mattered

The old model learned that “small object in the sky” often meant drone. On 1,133
held-out negatives it fired 1,017 times at threshold 0.5. Retraining with birds,
aircraft, helicopters, bombs, and real deployment incidents taught explicit
decision boundaries around confusing objects. A second audit added 222 licensed
flock images and held 44 out. Focused mining reduced flock errors from 22 to 1;
on the expanded test set, false positives are 92/1,177 at threshold 0.5.

This is a stronger demonstration than testing only easy drone images because
real alert systems usually see far more non-drones than drones.

## Why threshold 0.50 in the demo?

A model score is not the same thing as an operating policy. On held-out test
images:

| Threshold | Precision | Recall | Specificity | False positives |
|---:|---:|---:|---:|---:|
| 0.50 | 72.86% | 100.00% | 92.18% | 92 |
| 0.75 | 82.77% | 99.19% | 95.67% | 51 |
| 0.90 | 91.92% | 96.76% | 98.22% | 21 |

The demo uses 0.50 to prioritize recall and make the fixed operating policy
explicit; 0.90 is a reasonable conservative alternative. A customer would
choose the final point
from the cost of missed detections versus false alarms, not from accuracy alone.

## What do the metrics mean?

- **IoU:** overlap between predicted and ground-truth boxes. IoU 0.5 means the
  intersection is at least half the union.
- **AP@0.5:** area under the precision-recall curve when a box is correct at IoU
  >= 0.5.
- **mAP@0.5:0.95:** AP averaged over stricter IoU thresholds; it penalizes loose
  localization.
- **Recall:** fraction of positive images that produce an accepted detection.
- **Specificity:** fraction of negative images with no accepted detection.
- **FPR:** `1 - specificity`, or negative images that trigger at least once.
- **Wilson interval:** uncertainty interval for a measured proportion; the binary
  FPR at threshold 0.5 is 7.82%, with a 95% interval of 6.4–9.5%.

Box mAP and image-level FPR answer different questions. A detector can localize
positives well but still be unusable if it repeatedly alarms on negative images.

## Why is country performance lower?

The diagnosis separates localization from taxonomy:

- class-agnostic mAP@0.5: 0.5334
- country-aware mAP@0.5 on positive images: 0.2355
- 55.8% relative mAP loss comes from requiring the correct country
- top-1 is 33.6%, top-3 is 59.1%, versus 5.6% uniform-chance top-1

The model learned signal above chance, but several airframes share layouts,
countries import or copy designs, web query labels are noisy, and the class
distribution is long-tailed. For example, USA and Turkey have much deeper
coverage than Germany or Pakistan. That is why the product uses a top-three
analyst shortlist.

## How was class imbalance handled?

The country loader uses a weighted sampler with weight
`(1 / class_count) ** 0.5`. Square-root balancing increases rare-class exposure
without fully oversampling every rare image to the frequency of the largest
class, which would make overfitting likely. Negative images are treated as one
sampling group.

## Training details

- architecture: torchvision Faster R-CNN, ResNet50-FPN
- binary resize: short side 480, long side at most 800
- country resize: short side 320, long side at most 512
- optimizer: SGD with momentum and weight decay
- mixed precision: enabled
- gradient clipping: max norm 1.0
- checkpoints: epoch, optimizer, scheduler, scaler, RNG, and in-epoch resume state
- country warm start: shared detector weights plus semantically matched old
  country-head rows; four countries were newly initialized

Country training used batch size 1 because the available RTX 3050 has 4 GB VRAM.
The reduced country resolution was a measured hardware constraint, not an
architectural change. Huge source images are decoded with OpenCV's reduced modes
before resizing to avoid Windows host-memory failures, with boxes rescaled from
annotation dimensions.

## What false-positive test did you run?

A previously captured false-alarm frame scored 0.599 at threshold 0.10 after
retraining, demonstrating that the confusing visual pattern remained. At the
calibrated threshold 0.90 it was rejected. The benchmark is stronger evidence:
only 14 of 1,133 held-out negatives fired at that threshold.

The category breakdown at score 0.5 was 4.2% for birds, 12.4% for aircraft,
14.8% for helicopters, 8.3% for bombs, and 9.5% for deployment incidents.

## Can this estimate detection range?

No—not from this still-image dataset. Pixel-scale performance can be measured,
but physical range requires camera focal length, sensor pitch/resolution, target
dimensions, atmospheric conditions, and calibrated field testing. Claiming a
distance in metres from these benchmarks would be unsupported.

## What would you improve next?

1. Have domain experts review country labels and boxes, prioritizing thin classes.
2. Collect a new deployment-domain test set from cameras not represented in
   training.
3. Calibrate binary and country scores with temperature scaling or isotonic
   regression on a dedicated calibration split.
4. Compare Faster R-CNN with a modern one-stage detector at matched resolution
   and report accuracy/latency/memory tradeoffs.
5. Add temporal tracking and multi-frame voting for video to suppress transient
   false alarms.
6. Version manifests and model artifacts with DVC or an object store.

## Honest limitations

- Most country labels are weak labels, not expert-verified intelligence.
- Web imagery contains domain and search-engine bias.
- Some classes have tiny held-out denominators, so their AP can look unstable.
- Country score shares are not calibrated probabilities.
- The evaluation is image-based and does not establish physical range.
- Web images without explicit licenses must not be redistributed as though the
  whole dataset had a single permissive license.

An interview-quality answer acknowledges these limits and explains the next
measurement needed; it does not hide uncertainty behind a single accuracy value.
