# Architecture

A live-demo app: hold a Java-Dex card in front of a webcam, an image classifier
names it. Spring Boot serves a small capture page and REST API; JobRunr runs the
pipeline as background jobs, observable in its dashboard.

```
webcam ──▶ /api/captures/{id}      (CaptureController)
             │
             ▼
   data/captures/{cardId}/
             │  1. prepare        (CardDataPreparationJob)
             ▼
   data/dataset/{train,valid}/{cardId}/
             │  2. train          (CardTrainingJob)
             ▼
   data/model/  (classes.txt, head.jd, model.properties)
             │  3. evaluate       (CardEvaluationJob)
             ▼
   data/model/evaluation.json ──▶ GET /api/evaluation
             │
             ▼  inference
   webcam ──▶ /api/predict        (CardPredictionService + CardClassifier)
```

Stages pass files under `data/` only — the paths all come from
`JavaDexApplicationConfiguration` (root package). The ML engine is DJL with PyTorch; the
pretrained ResNet18 and the engine itself are cached in `~/.djl.ai` (shared
across projects, downloaded on first use).

## Jobs — `org.javadex.jobs`

Background jobs, enqueued by `api.TrainingController` (`POST /api/jobs/prepare`
`/train` `/evaluate`), each failing with a clear message when its input stage
hasn't run yet. All log through `JobRunrDashboardLogger` and report progress via
`JobContext.progressBar`.

- **`CardDataPreparationJob.prepare`** — raw webcam captures become a training
  dataset. Keeps only directories that match a card in the catalog, needs at
  least 3 captures per card, splits 80/20 into train/valid (always at least 1
  validation image), center-crops each image to a square and resizes it to
  256×256 JPEG. Rebuilds `data/dataset/` from scratch on every run.
- **`CardTrainingJob.train(epochs, batchSize, …)`** — transfer learning. Loads
  the ImageNet-pretrained ResNet18 from the DJL model zoo, freezes it, appends a
  trainable linear head sized by the class count. Adam (lr 0.001) +
  softmax cross-entropy, defaults 60 epochs / batch 8 (overridable per request).
  The train set gets light augmentation (random resized crop); the valid set
  plain resize. Persists `classes.txt`, the head weights as `head.jd`, and
  metadata (`accuracy`, `loss`, `imageSize`, `epochs`, `backbone`) as
  `model.properties`. The backbone itself is never saved — it is identical zoo
  weight, re-fetched on every load.
- **`CardEvaluationJob.evaluate`** — reloads the trained model, classifies every
  validation image, tallies a confusion matrix and writes `evaluation.json`,
  logging the overall accuracy.
- **`EvaluationReport`** — the record that serializes a `ConfusionMatrix` to
  `evaluation.json` via Jackson: `overallAccuracy`, `perClass` (class, accuracy),
  `confusion` (row per expected class, column per predicted).

## ML — `org.javadex.ml`

- **`CardClassifier`** — the trained model, reloaded for inference. Fetches the
  same frozen ResNet18 from the zoo cache, reads `head.jd`, and applies the head
  math in plain Java (`logit = bias + weight · features`). Also owns
  `BACKBONE_LAYERS = "18"`, the single ResNet depth the whole app uses.
  AutoCloseable (holds a zoo model + predictor).
- **`LogitsTranslator`** — DJL `Translator` that feeds raw pixels through the
  frozen backbone and returns its output vector as features, skipping the zoo's
  ImageNet post-processing. Its `centerSquare` crops webcam frames to a square
  first, because frames are wide while training images are square — a plain
  resize would stretch the frame and shift the input distribution.
- **`ConfusionMatrix`** — tallies expected-vs-actual classifications per class;
  exposes overall accuracy, per-class accuracy and the raw count matrix.
- **`CenterSquare`** — the centered largest-square crop for a width x height,
  one place. Dataset preparation (`BufferedImage`) and inference (`DJL Image`)
  both crop by it, so training and webcam frames can never drift apart.
- **`CardPredictionService`** — the inference entry point behind `/api/predict`:
  softmax over the head logits, returns the best label and top 3. Caches a
  `CardClassifier` and, keyed on the `classes.txt` timestamp, swaps in a
  freshly trained model without an application restart.

## Supporting packages

- **`api`** — `CaptureController` (lists cards, stores webcam frames per card
  id), `PredictionController` (classify a frame), `TrainingController`
  (enqueues the three jobs, serves `evaluation.json`).
- **`catalog`** — `CardCatalog`/`Card`: loads `cards/cards.json`, the source of
  valid card ids and titles.
- **`tools`** — `SyntheticCaptureGenerator`: writes fake captures so the whole
  pipeline can be exercised without a camera.
