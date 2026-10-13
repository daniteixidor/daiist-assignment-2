# Assignment 2 Report

- **Name**: Daniel Teixidor
- **Student ID**: 19971
- **Email**: dteixidor.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

**Decision.** On a construction site, fixed cameras take a snapshot of each work
zone every 5 minutes. For each snapshot the system decides one of three things:
pass (nothing happens), review (the site safety officer looks at the frame), or
fail (the supervisor gets an alert and walks over to the worker without a helmet).
Today that decision is made by a supervisor walking the site, so most violations
are never seen.

**Who acts.** The site safety officer (reviews) and the zone supervisor (interventions).

**Unit and positive class.** One snapshot. Positive = the snapshot contains at
least one (a worker's head without a helmet).

**Why the two errors cost different amounts.**

| Outcome | € | Reasoning |
|---|---|---|
| False negative (missed violation) | 60 | The worker stays unprotected until at least the next snapshot. That exposure carries a small chance of a head injury (medical, lost days, liability, insurance premium) plus a chance of a labour-inspection fine. Spread over a 5-minute window, I put the expected cost at ~€60. |
| False positive (false alarm) | 8 | A supervisor walks over for nothing: ~10 min at ~€45/h fully loaded, plus a little lost credibility for the alerts. |
| True positive | 8 | The same walk-over, but it prevents the €60. |
| True negative | 0 | Nothing happens. |
| Human review | 0.50 | ~30 s of the safety officer's time per frame. |

A miss costs **7.5× a false alarm**, so recall on `head` matters more than precision.
The break-even point for a binary alert is P(violation | score) ≈ 8 / (60 + 8) ≈ 12%:
it is worth sending the supervisor even when we are far from sure.

**Base rate.** I assume **2%** of snapshots on a normally run site show a
violation. The test split has **19.8%** (140 of 706 images), ten times more, so every
cost figure below is reweighted to 2% rather than read off the raw test counts.

**Volume.** 20 cameras × 12 snapshots/h × 10 h = **2,400 snapshots/day**.

## Dataset

- Roboflow Universe: **workspace `joseph-nelson`, project `hard-hat-workers`, version 5**
  (`raw_HeadHelmetClasses`): https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/5
- Licence: public domain. Used as published, not forked.
- One image = one photo of a work scene with one or more workers. Classes: `head`
  (no helmet) and `helmet`. Version 5 drops the original `person` class.
- Why it fits: it labels exactly the thing that triggers the decision (a head
  without a helmet) and contrasts it with the compliant case (a helmet), in real
  site photos.

## What the dataset's composition implies

| Split | Images | Images with a `head` | `head` instances | `helmet` instances |
|---|---|---|---|---|
| train | 4,916 | 911 (18.5%) | 4,612 | 13,919 |
| valid | 1,413 | 256 (18.1%) | 1,339 | 3,913 |
| test | 706 | 140 (19.8%) | 726 | 1,915 |

Object size: the median `head` box covers **0.60%** of the image (helmet 0.83%), and
**72% of heads are smaller than 1% of the image**.

What this means:
- **`head` is the minority class** (25% of instances), and it's the one carrying the
  business decision. Even so, 726 test instances in 140 images are enough for its
  per-class numbers to be meaningful. Image-level recall rests on only 140 positive
  images, so ±3 points of noise is plausible.
- **Tiny objects.** Most heads are small, distant figures. Small objects are where
  detectors fail first, which is why mAP50-95 (0.68) is far below AP50 (0.97): the
  model finds the heads but its boxes are loose.
- **The test split isn't CCTV.** These are mostly ground-level photos taken in decent
  light. A fixed, elevated, wide-angle camera running in rain or dusk will make heads
  even smaller, so real-world recall will be lower than reported here.
- **The base rate is unrealistic** (19.8% vs my assumed 2%), which is why the costing
  reweights by class rather than using raw counts.

## Finetuning and baselines

- Model: **yolo11n, object detection**, finetuned from COCO weights. Detection
  rather than classification, because the decision depends on *whether any head in
  the frame* is bare, and frames often contain several workers. I chose nano because
  it's cheap enough to run on 20 camera streams.
- Settings: 640 px, batch 16, 40 epochs (early-stopping patience 10, which never
  triggered), AdamW chosen automatically, seed 0. Trained on **Google Colab, Tesla T4
  GPU, 51 minutes**. Validation mAP50 plateaued around epoch 30 (0.970), so more
  epochs would add little.

Detection metrics, same test split:

| Model | AP50 | mAP50-95 |
|---|---|---|
| **Finetuned yolo11n** | **0.972** | **0.680** |
| COCO yolo11n zero-shot | 0 | 0 |
| Trivial: predict nothing | 0 | 0 |

The COCO model has no `head` or `helmet` class, so it scores 0 by construction. To
give it a fair chance at the actual *decision*, I also scored it at image level,
using its `person` confidence as a proxy ("flag any frame where a person is confident"):

| Image-level decision | AUC | Cheapest threshold | Recall | False-alarm rate | € / 1,000 snapshots |
|---|---|---|---|---|---|
| **Finetuned** | **0.978** | 0.54 | **88.6%** | 1.2% | **376** |
| COCO zero-shot (person proxy) | 0.677 | 0.95 | 1.4% | 0.2% | 1,199 |
| Predict nothing | 0.500 | n/a | 0% | 0% | 1,200 |

COCO zero-shot is effectively no better than doing nothing. Seeing people isn't the
same as seeing missing helmets, and the cheapest thing it can do is almost never
alert. The finetuned model cuts the cost by 69%.

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| **head (positive)** | 0.942 | 0.930 | 0.962 | 0.676 |
| helmet | 0.953 | 0.939 | 0.983 | 0.684 |

The headline overall AP50 of 0.972 is **flattered by `helmet`**, the majority and
easier class. The class that matters, `head`, is the weaker one (0.962).

Box-level confusion matrix (test split, `artifacts/confusion_matrix.png`):

| | true head | true helmet |
|---|---|---|
| predicted head | 622 | 174 |
| predicted helmet | **102** | 1,730 |
| missed (background) | 2 | 11 |

- **The main failure is calling a bare head a helmet: 102 of 726 (14%).** That's
  the expensive error (a missed violation). The model almost never misses a head
  completely (2 of 726). It sees the person and gets the helmet wrong, probably on
  caps, hoods, light hair and very small faces.
- **The reverse error: 174 of 1,915 helmets (9%) called a bare head.** These cause
  false alarms, which cost €8 each, not €60.
- The background row of the matrix (not shown) counts thousands of low-confidence
  extra boxes, because it's computed at the validator's very low confidence cut-off.
  The cost analysis below uses a swept threshold instead.

## Cost matrix and operating point

Costs and base rate are as in section 1. Expected cost per snapshot at threshold *t*:

`cost = p·[FNR·60 + TPR·8] + (1−p)·[FPR·8]`, with p = 0.02,

where TPR/FPR are measured on the 140 positive and 566 negative test images
separately, then reweighted to p.

I swept *t* from 0.01 to 0.99 (`artifacts/cost_curve.png`). The **cheapest
threshold is t = 0.54**, at **€376 per 1,000 snapshots**, with 88.6% of violations
caught and 1.2% false alarms.

Sensitivity (assumptions that would change the answer):
- **The FN cost (€60) is the assumption I'm least sure of.** It's an expected value
  of rare, expensive events. A larger FN/FP ratio pushes the threshold down (more
  alerts), and a smaller one pushes it up.
- **Base rate.** At t = 0.54 the model beats doing nothing for any base rate above
  **~0.2%**. Above **~7%** violations, having a person check every snapshot becomes
  cheaper than the model alone, because missed violations dominate. Neither break-even
  is close to my 2%, so the conclusion is robust.

## What a person still has to check

Two thresholds, both chosen by grid search over every (t_low, t_high) pair. I took
the cheapest pair whose review load fits one safety officer (6 productive h/day ÷
30 s = **720 reviews/day**):

- **t_low = 0.10, t_high = 0.86**: below 0.10 pass, 0.10–0.86 review, above 0.86 alert.
- Share sent to review: **6.5%** → **157 snapshots/day** → **1.3 h/day** of review.
- **Realistic? Yes.** It's under a quarter of one person's capacity, so the safety
  officer can do it alongside their other duties. The capacity limit didn't bind: the
  unconstrained optimum is the same band.

| Option | € per 1,000 snapshots | € per day (2,400) | Comment |
|---|---|---|---|
| Do nothing | 1,200 | 2,880 | Every violation missed |
| Person checks every snapshot | 660 | 1,584 | Needs 20 h/day = 3.3 people, which isn't feasible with 1 |
| Model decides alone (t = 0.54) | 376 | 902 | Misses 11% of violations |
| **Model + review band** | **245** | **587** | **Cheapest, and fits one reviewer** |

The review band saves about **€2,300/day vs doing nothing** and about **€315/day vs the
model alone**. The extra €0.50 reviews catch violations the model was unsure about.

## Limitations & next steps

- **The test images aren't CCTV frames.** Next step: record a week of the site's own
  cameras, label a few hundred frames, and re-measure `head` recall before going live.
- **Head-vs-helmet confusion (14% of bare heads).** Add more training examples of
  caps, hoods and distant workers, and try yolo11s or a larger image size (960 px)
  for the small heads.
- **Snapshots aren't independent.** The same violation appears in consecutive frames,
  so a miss now is often caught 5 minutes later. Tracking a worker across frames
  would turn this into "time to catch a violation", which is the real KPI and would
  likely lower the effective FN cost.
- **The FN cost is an estimate.** With the site's accident and inspection history I'd
  replace it with an actual figure.
- **I assumed reviewers are always right.** In reality they'll miss a few, which would
  slightly raise the cost of the review band.

## Generative AI use disclosure

I used Claude (Anthropic) to write the code (`train.py`, `app.py`, `costing.py`,
`config.py`) and to help draft and format this report. I checked the problem
framing, cost figures, thresholds and conclusions, and I can explain each of them.
