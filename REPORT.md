# Assignment 2 Report

- **Name**: Daniel Teixidor
- **Student ID**: TODO
- **Email**: TODO
- **Group**: TODO [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

> Lines marked **[FILL]** come from `artifacts/` after the real training run.

## Business problem, and the cost of being wrong

**Decision.** On a construction site, fixed cameras take a snapshot of each work
zone every 5 minutes. For each snapshot the system decides one of three things:
*pass* (nothing happens), *review* (the site safety officer looks at the frame), or
*fail* (the supervisor gets an alert and walks over to the worker without a helmet).
Today that decision is made by a supervisor walking the site, which means most
violations go unseen.

**Who acts.** The site safety officer (reviews) and the zone supervisor (interventions).

**Unit and positive class.** One snapshot. Positive = the snapshot contains at
least one `head` (a worker's head without a helmet).

**Why the two errors cost different amounts.**

| Outcome | € | Reasoning |
|---|---|---|
| False negative (missed violation) | 60 | The worker stays unprotected until at least the next snapshot. That exposure carries a small chance of a head injury (medical, lost days, liability, insurance premium) plus a chance of a labour-inspection fine. Spread over a 5-minute window, I put this at ~€60. |
| False positive (false alarm) | 8 | A supervisor walks over for nothing: ~10 min at ~€45/h fully loaded, plus a little lost credibility for the alerts. |
| True positive | 8 | The same walk-over, but it's money well spent because it prevents the €60. |
| True negative | 0 | Nothing happens. |
| Human review | 0.50 | ~30 s of the safety officer's time per frame. |

A miss costs about **7.5× a false alarm**. So I care more about recall on `head`
than on precision, and the threshold should sit lower than 0.5.

**Base rate.** I assume **2%** of snapshots on a normally run site show a
violation. The test split has a far higher share **[FILL: test_base_rate_positive_images]**,
so every cost figure is reweighted to 2% rather than read off the raw test counts.

**Volume.** 20 cameras × 12 snapshots/h × 10 h = **2,400 snapshots/day**.

## Dataset

- Roboflow Universe: **workspace `joseph-nelson`, project `hard-hat-workers`, version 5**
  (`raw_HeadHelmetClasses`): https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/5
- Licence: public domain. Used as published, not forked.
- One image = one photo of a work scene with one or more workers. Classes: `head`
  (no helmet) and `helmet`. The original `person` class was dropped in this version.
- Why it fits the problem: it labels exactly the thing that triggers the decision
  (a head without a helmet) and contrasts it with the compliant case (a helmet),
  in real site photos.

## What the dataset's composition implies

**[FILL from `artifacts/dataset_stats.json`]**

| Split | Images | `head` instances | `helmet` instances |
|---|---|---|---|
| train | | | |
| valid | | | |
| test | | | |

Object size: median `head` box = **[FILL]%** of the image, and **[FILL]%** of heads are
smaller than 1% of the image.

What this means:
- **[FILL]** Which class is the minority. If it's `head`, that's the class carrying the
  business decision, and its recall is the number to trust least.
- Small objects: helmets and heads are tiny in wide shots, so the model will be weakest
  on distant workers. Those are the cases a site camera sees most often.
- The test split is close-up, well-lit photos, while a fixed CCTV camera is wide-angle,
  higher up, and runs in rain and dusk. Expect real-world recall to be lower than the test number.
- The test split's violation rate is much higher than 2%, which is why the costing is reweighted.

## Finetuning and baselines

- Model: **yolo11n, object detection**, finetuned from COCO weights. I chose detection
  because the decision depends on *whether any head in the frame* is bare, and a frame
  often contains several workers. I chose nano because it's cheap enough to run on
  20 camera streams.
- Settings: 640 px, batch 16, up to 40 epochs, early stopping patience 10, seed 0.
  Trained on **[FILL: Google Colab T4 GPU]**, which took **[FILL]** min.

Detection metrics, same test split:

| Model | AP50 | mAP50-95 |
|---|---|---|
| Finetuned yolo11n | [FILL] | [FILL] |
| COCO yolo11n zero-shot | 0 | 0 |
| Trivial: predict nothing | 0 | 0 |

The COCO model has no `head` or `helmet` class, so it can't produce either label. To
give it a fair chance at the actual *decision*, I also scored it at image level, using
its `person` confidence as a proxy ("flag any frame with a person"):

| Image-level (decision) | AUC | Recall @ cheapest t | False-alarm rate | € / 1,000 |
|---|---|---|---|---|
| Finetuned | [FILL] | [FILL] | [FILL] | [FILL] |
| COCO zero-shot (person proxy) | [FILL] | [FILL] | [FILL] | [FILL] |
| Predict nothing | 0.5 | 0 | 0 | [FILL] (= 2% × €60 × 1,000 = €1,200) |

## Per-class evaluation and failure modes

**[FILL from `artifacts/metrics.json`]**

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| head (positive) | | | | |
| helmet | | | | |

Confusion matrix: `artifacts/confusion_matrix.png`.

- Headline class: **`head`**, because it's the violation.
- Failure modes to check in the confusion matrix: `head` predicted as `background`
  (missed bare heads, often small or distant), `helmet` vs `head` confusion (caps,
  hoods and hair colour can look like a helmet), and background boxes.
  **[FILL with what you actually see.]**

## Cost matrix and operating point

Costs and base rate are as in section 1. Cost per snapshot at threshold *t*:

`cost = p·[FNR·60 + TPR·8] + (1−p)·[FPR·8]`, with p = 0.02.

I swept *t* from 0.01 to 0.99 (`artifacts/cost_curve.png`). The cheapest point is
**t = [FILL]**, at **€[FILL] per 1,000 snapshots**.

Sensitivity: the threshold depends mainly on the ratio of FN cost to FP cost. If a
missed violation were worth only ~€8 (as cheap as a false alarm), the optimum would
move up and the system would barely alert. If the base rate drops below **[FILL: try
it with the dashboard slider]**, doing nothing becomes cheaper than running the model.
**The FN cost is the assumption I'm least sure of.** It's an expected value of rare,
expensive events.

## What a person still has to check

Two thresholds, both chosen by grid search over all (t_low, t_high) pairs, using the
cheapest pair whose review load fits one safety officer
(6 productive h/day ÷ 30 s = **720 reviews/day**, i.e. 30% of 2,400 snapshots):

- t_low = **[FILL]**, t_high = **[FILL]**
- Share sent to review: **[FILL]%** → **[FILL] snapshots/day** → **[FILL] h/day** of review.
- Realistic? **[FILL]**

| Option | € per 1,000 snapshots | € per day (2,400) |
|---|---|---|
| Do nothing | 1,200 | 2,880 |
| Person checks every snapshot | 660 (but needs 20 h/day = 3.3 people) | 1,584 |
| Model decides alone | [FILL] | [FILL] |
| Model + review band | [FILL] | [FILL] |

## Limitations & next steps

- The test images aren't CCTV frames. Next step: collect a week of the site's own
  footage, label a sample, and re-check recall on `head` before going live.
- Snapshots aren't independent: the same violation shows up in consecutive frames.
  Tracking a worker across frames would turn this into "time to catch a violation",
  which is the real KPI.
- The FN cost is an estimate. With the site's own accident and inspection history I'd
  replace it with an actual figure.
- I assumed reviewers are always right. In reality they'll miss a few, which would
  slightly raise the cost of the review band.

## Generative AI use disclosure

I used Claude (Anthropic) to write the code (`train.py`, `app.py`, `costing.py`,
`config.py`) and to draft the structure of this report. I chose and checked the
problem framing, cost figures and thresholds, and I can explain each of them.
