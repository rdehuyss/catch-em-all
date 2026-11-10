# Java-Dex Card Classifier Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working Spring Boot app that collects webcam pictures of the 20 Java-Dex cards, trains an image classifier on them via JobRunr background jobs (prepare → train → evaluate, each with dashboard progress), and recognizes a card in a live webcam picture.

**Architecture:** Browser capture page (getUserMedia) POSTs frames to the server; server stores them under `data/captures/{cardId}/`. Three separately-triggered JobRunr jobs move the data through `data/dataset/{train,valid}/{cardId}/`, train a new head on a pretrained ResNet18 (transfer learning, DJL + PyTorch engine), and evaluate per-class accuracy plus a confusion matrix into `data/model/evaluation.json`. A prediction endpoint and page classify live webcam frames against the saved model.

**Tech Stack:** Spring Boot 4 (webmvc), JobRunr spring-boot-4-starter 9.0.3 (H2 file StorageProvider), DJL 0.40.0 BOM + PyTorch engine + pytorch-model-zoo (ResNet18, ImageNet pretrained), vanilla JS frontend.

**Spec:** `docs/superpowers/specs/2026-11-10-card-classifier-design.md` (this plan + conversation is the design record; the repo README describes the cards)

## Global Constraints

- No git commits — the human commits at the end.
- Java 25, Spring Boot 4.1.2, JobRunr 9.0.3, DJL 0.40.0 (already in `build.gradle` — keep versions).
- Transfer learning from a pretrained model, never training a CNN from scratch (rewrites the existing `CardTrainingJob`).
- JobRunr storage: H2 file database; JobRunr dashboard enabled.
- Prepare, train, evaluate are separately triggerable jobs (three buttons), each reporting progress into the JobRunr dashboard via injected `JobContext`.
- ~10 captures per card expected during the stream; defaults sized for that.
- Model: ResNet18 backbone (frozen), new trainable classification head, input 224×224.
- Data lives under `data/` (add to `.gitignore`); card metadata read from `cards/cards.json`.
- All model metadata (classes list, image size, accuracy) persisted next to the model so prediction always agrees with training.

## Review Focus

1. **Too few captures:** fewer than ~3 pictures for a card (or a class folder missing entirely) should fail prepare/train with a clear message, not produce a silently useless model or a 500 stack trace. Test: `CardDataPreparationJobTest#rejectsClassWithTooFewImages`.
2. **Predicting before a model exists:** hitting `/api/predict` (or the predict page) before any training completed should return a clear 409/"no model trained yet", not a file-not-found stack trace. Test: `PredictionControllerTest#predictWithoutModelIs409`.
3. **Camera-ready but wrong picture:** the live predict page must show confidence and top-3 candidates (not just one label), so a wrong guess is visible and nameable on stream. Test: `PredictionResponse` includes `top3` list; manually checked in the page.
4. **Class order drift between train and predict:** predictions rely on `classes.txt` order; if train/valid class sets differ, training must abort (existing behavior, keep). Test: `CardTrainingJobTest#rejectsMismatchedClassSets`.
5. **H2/JobRunr restart:** after app restart with H2 file storage, the dashboard must still show past jobs (that is the point of choosing H2 over in-memory). Verified manually in the end-to-end run.

---

### Task 1: Build fixes, H2 + JobRunr dashboard, data directory

**Files:**
- Modify: `build.gradle`
- Modify: `src/main/resources/application.properties`
- Modify: `.gitignore`
- Create: `src/main/java/org/jobrunr/democatch/config/AppPaths.java`

**Interfaces:**
- Produces: `AppPaths` record with `captures()`, `datasetTrain()`, `datasetValid()`, `model()` `Path` accessors, all under the `app.data-dir` property (default `data`); a `@Bean AppPaths` so controllers/jobs inject it.

- [ ] **Step 1: Fix `build.gradle`** — remove the duplicated `runtimeOnly` pytorch lines; add `runtimeOnly("ai.djl.pytorch:pytorch-model-zoo")`, `runtimeOnly("com.h2database:h2")`, `implementation("org.springframework.boot:spring-boot-starter-thymeleaf")` is NOT needed (static pages only) — skip it.
- [ ] **Step 2: `application.properties`** — `spring.datasource.url=jdbc:h2:file:./data/jobrunr-db;AUTO_SERVER=TRUE` plus H2 driver/username; confirm `org.jobrunr.dashboard.enabled=true` and `org.jobrunr.database.skip-create=false`; add `app.data-dir=data`.
- [ ] **Step 3: `.gitignore`** — append `data/`.
- [ ] **Step 4: Verify** — `./gradlew bootRun` starts, `http://localhost:8080/dashboard` renders the JobRunr dashboard, an `H2` file appears under `data/`. Stop the app.
- [ ] **Step 5: Context test** — `DemoCatchApplicationTests` (existing `contextLoads`) passes: `./gradlew test --tests DemoCatchApplicationTests`.

### Task 2: Spike — reshape ResNet18 with the PyTorch engine

**Files:**
- Create: `src/test/java/org/jobrunr/democatch/ml/ResnetSpikeTest.java` (throwaway, stays as the executable proof)

**Interfaces:**
- Produces: the verified recipe used verbatim in Task 5 — how to obtain a 512-dim feature block from a pretrained ResNet18 (`ai.djl.pytorch:pytorch-model-zoo`, filter `backbone=18`) and append a new head for 20 classes, plus the exact `Criteria` needed to reload the saved model in Task 7.

- [ ] **Step 1: Write the spike** — load the zoo model via `Criteria` (`Application.CV.IMAGE_CLASSIFICATION`, artifact `resnet`, engine PyTorch, filter `backbone=18`); print `model.getBlock()`'s class and children; attempt `removeLastBlock()` + `batchFlattenBlock()` + new `Linear(20)`; run one 224×224 forward pass and assert the output `Shape` is `[1, 20]`.
- [ ] **Step 2: Run it** — `./gradlew test --tests ResnetSpikeTest` (first run downloads the model, ~100 MB).
- [ ] **Step 3: Decide the recipe.** If the zoo block is directly reshapable, that is the recipe. If the PyTorch zoo block cannot be reshaped (traced `PtSymbolBlock`), the fallback recipe is: keep the whole pretrained block frozen (`freezeParameters(true)`) as a feature extractor and train only a head on top of its output; record which recipe worked in a comment at the top of `ResnetSpikeTest`.
- [ ] **Step 4: Verify the fallback also loads for inference** — confirm a `Criteria` with `optModelPath(data/model)` reloads a `Model.save()`d model and predicts, since Tasks 6–7 depend on save/reload symmetry.

### Task 3: Card catalog + capture page

**Files:**
- Create: `src/main/java/org/jobrunr/democatch/catalog/CardCatalog.java`
- Create: `src/main/java/org/jobrunr/democatch/catalog/Card.java`
- Create: `src/main/java/org/jobrunr/democatch/web/CaptureController.java`
- Create: `src/main/resources/static/index.html` (capture page)
- Create: `src/main/resources/static/capture.js`
- Create: `src/main/resources/static/capture.css`
- Test: `src/test/java/org/jobrunr/democatch/catalog/CardCatalogTest.java`

**Interfaces:**
- Produces: `CardCatalog` with `List<Card> all()` and `Optional<Card> byId(String id)`; `Card` = `record Card(String id, int n, String title, String type)`. HTTP: `GET /api/cards` → JSON array of cards; `POST /api/captures/{cardId}` with a JPEG `multipart/form-data` field `image` → saves `data/captures/{cardId}/<timestamp>.jpg`, returns `201 {count: <n>}`.
- The capture page shows the webcam preview, a card picker (20 thumbnails from `/cards/png/`, which must be added as a static resource location), a capture button, and the saved count per card.

- [ ] **Step 1: Failing test for `CardCatalog`** — parses `cards/cards.json` into 20 `Card`s; `byId("jd01")` has title "Duke"; unknown id returns empty.
- [ ] **Step 2: Run** — `./gradlew test --tests CardCatalogTest`; expect FAIL (class missing).
- [ ] **Step 3: Implement `Card` + `CardCatalog`** — parse `cards/cards.json` with Jackson from the file-system path (configurable `app.cards-dir`, default `cards`).
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Serve the cards as static assets** — add `spring.web.resources.static-locations` including `file:cards/` so `/cards/png/jd01.png` resolves; verify in browser.
- [ ] **Step 6: `CaptureController`** — `GET /api/cards`, `POST /api/captures/{cardId}` (404 for unknown id; saves under `AppPaths.captures()/cardId`); return capture count by listing the folder.
- [ ] **Step 7: Capture page** — `index.html` + `capture.js`: `getUserMedia` preview, 20-card picker, "capture" posts a canvas JPEG frame; per-card saved-count from the POST response; "no model / start with capturing" hint.
- [ ] **Step 8: Verify manually** — allow camera in browser, capture a frame, see file appear under `data/captures/jd01/` and count increment.

### Task 4: Prepare job — dataset split + resize

**Files:**
- Create: `src/main/java/org/jobrunr/democatch/jobs/CardDataPreparationJob.java`
- Test: `src/test/java/org/jobrunr/democatch/jobs/CardDataPreparationJobTest.java`

**Interfaces:**
- Consumes: `AppPaths.captures()`, `AppPaths.datasetTrain()`, `AppPaths.datasetValid()`; `CardCatalog` for class existence.
- Produces: `@Job(name="Prepare card images")` method `prepare(JobContext ctx)` — wipes and rebuilds `data/dataset/{train,valid}/{cardId}/` from `data/captures/`, 80/20 split per class (min 3 captures per class, else throws), images resized so the longest side is 256 px (center-cropped square) and re-encoded as JPEG. JobContext progress 0→100 in 10% steps per class.

- [ ] **Step 1: Failing tests** — split logic with a fake folder of 10 tiny PNGs: expect 8 train / 2 valid; 2 captures → `IllegalStateException` naming the card id; unknown folder under captures is ignored.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** — pure `split(List<Path>)` helper (testable, deterministic) + DJL-free resize via `ImageIO`/`Thumbnails`-style Java2D scaling; progress via injected `JobContext` (verify the 0–100 percent API against the JobRunr 9 javadoc before writing it).
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Wire the trigger** — `POST /api/jobs/prepare` on the existing `TrainingController`-to-be (create `web/TrainingController.java` here) enqueues via `JobScheduler`; returns the job id.
- [ ] **Step 6: Verify in dashboard** — run prepare against a few synthetic captures; watch the job progress on the dashboard and `data/dataset/` appear correctly.

### Task 5: Train job — transfer learning with dashboard progress

**Files:**
- Modify: `src/main/java/org/jobrunr/democatch/jobs/CardTrainingJob.java` (full rewrite)
- Test: `src/test/java/org/jobrunr/democatch/jobs/CardTrainingJobTest.java` (only the cheap, deterministic parts are unit-tested; the model itself is proven by the Task 8 end-to-end run)

**Interfaces:**
- Consumes: `AppPaths.datasetTrain()/datasetValid()/model()`; the Task 2 recipe (Criteria + block reshape + head).
- Produces: `@Job(name="Train Java-Dex card classifier", retries=0)` method `train(JobContext ctx)` with defaults: epochs 5, batch size 8, image size 224, learning rate 0.001 (adam). Writes `data/model/java-dex` (model), `data/model/classes.txt` (class order), and model properties (`imageSize`, `accuracy`, `loss`). Progress: 0–5% dataset prep, 5–90% epochs (per-epoch via a `TrainingListener`), 90–100% save. Logged through `JobRunrDashboardLogger` so DJL output shows in the dashboard.

- [ ] **Step 1: Failing test** — training aborts when `dataset/valid` class list differs from `dataset/train` (reuse the existing behavior); aborts with a clear message when fewer than 2 classes exist.
- [ ] **Step 2: Run** — expect FAIL (rewrite drops the old behavior unless kept).
- [ ] **Step 3: Rewrite `CardTrainingJob`** per the Task 2 recipe: pretrained ResNet18 backbone + new `Linear(20)` head, `EasyTrain.fit`, adam/softmax-cross-entropy, epoch-count progress into `JobContext`, save model + `classes.txt` + metadata. Delete the old from-scratch CNN (`createNetwork`). The training dataset pipeline applies light augmentation per epoch (random resized crop of the 224×224 input, slight color/brightness jitter) so ~10 captures per card multiply into more effective training samples; the validation pipeline uses plain resize only.
- [ ] **Step 4: Run tests** — expect PASS.
- [ ] **Step 5: Wire the trigger** — `POST /api/jobs/train` (params: epochs, batchSize — optional, defaults above).
- [ ] **Step 6: Verify** — train on the Task 4 synthetic dataset (20 classes × 8 images, CPU); confirm the dashboard shows per-epoch progress and logs, and `data/model/` contains model + `classes.txt`.

### Task 6: Evaluate job — accuracy + confusion matrix

**Files:**
- Create: `src/main/java/org/jobrunr/democatch/jobs/CardEvaluationJob.java`
- Create: `src/main/java/org/jobrunr/democatch/ml/ConfusionMatrix.java`
- Create: `src/main/resources/static/evaluation.html` (+ small `evaluation.js`)
- Test: `src/test/java/org/jobrunr/democatch/ml/ConfusionMatrixTest.java`

**Interfaces:**
- Consumes: `data/model/` (Task 5), `data/dataset/valid/` (Task 4), `AppPaths.model()`.
- Produces: `@Job(name="Evaluate card classifier")` method `evaluate(JobContext ctx)` — classifies every valid image, computes overall + per-class accuracy and a 20×20 confusion matrix, writes `data/model/evaluation.json` (`{overallAccuracy, perClass: [{class, accuracy, sampleCount}], confusion: [[...]]}`). `ConfusionMatrix` = pure class: `record(int expected, int actual)`, `toMatrix()`, `accuracyPerRow()` — unit-tested. `evaluation.html` renders the JSON (fetch `/api/evaluation` served by the evaluation job's controller sibling or a small read endpoint on `TrainingController`).

- [ ] **Step 1: Failing `ConfusionMatrix` test** — 3×3 hand-computed case: correct diagonal counts, per-row accuracy, unseen class → NaN/0 documented choice.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement `ConfusionMatrix`** — run test, expect PASS.
- [ ] **Step 4: Implement `CardEvaluationJob`** — DJL `Predictor<Image, Classification>` over the valid folder (per-image progress via `JobContext`), writes `evaluation.json`.
- [ ] **Step 5: Wire the trigger + read endpoint** — `POST /api/jobs/evaluate`, `GET /api/evaluation` (404 until `evaluation.json` exists).
- [ ] **Step 6: Verify** — run evaluate on the Task 5 model; open `evaluation.html`, confirm numbers appear and the confusion matrix is 20×20. Expect near-100% on synthetic data.

### Task 7: Prediction endpoint + live predict page

**Files:**
- Create: `src/main/java/org/jobrunr/democatch/ml/CardPredictionService.java`
- Create: `src/main/java/org/jobrunr/democatch/web/PredictionController.java`
- Create: `src/main/resources/static/predict.html` (+ `predict.js`)
- Test: `src/test/java/org/jobrunr/democatch/web/PredictionControllerTest.java`

**Interfaces:**
- Consumes: `data/model/` via a `Criteria` reload (Task 2 verified); `classes.txt`.
- Produces: `CardPredictionService.predict(byte[] jpeg)` → `record Prediction(String label, double probability, List<Ranking> top3)` with `record Ranking(String label, double probability)`. HTTP: `POST /api/predict` (multipart `image`) → `200 Prediction` JSON; `409 {error:"no model trained yet"}` when `data/model/java-dex` is absent. `predict.html` mirrors the capture page's webcam code, POSTs the current frame on a button click, shows the winning card with its image, plus top-3 probabilities.

- [ ] **Step 1: Failing controller test** — no model dir → 409; with a model dir stubbed by the Task 8 generator, a POST returns a `Prediction` with a non-empty `label` and 3 ranked entries.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement service + controller** — one lazily-created `Predictor` per model directory (rebuild if `classes.txt` mtime changes, so a retrain is picked up without restart).
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: `predict.html` page** — webcam preview + "What card is this?" button + result panel (winner image/title, top-3 bars).
- [ ] **Step 6: Verify manually** — hold up a card (or upload a PNG), see a named card and confidence; wrong guesses show as low-confidence top-3.

### Task 8: Synthetic capture generator + full end-to-end verification

**Files:**
- Create: `src/main/java/org/jobrunr/democatch/tools/SyntheticCaptureGenerator.java`
- Modify: `build.gradle` (add a `generateSyntheticCaptures` JavaExec task)
- Test: runs of the whole pipeline (manual verification, no unit test)

**Interfaces:**
- Consumes: `AppPaths.captures()`; `cards/png/*.png` (697×1075 originals — pixel size is correct for 300 dpi; no fix needed).
- Produces: 10 "captures" per card = card image pasted onto a random solid/gradient background with random scale (70–95%), rotation (±6°), and slight brightness shift; deterministic seed for reproducibility.

- [ ] **Step 1: Implement the generator** (Java2D, `main()`; `--cards-dir`, `--out-dir`, `--per-card` args).
- [ ] **Step 2: Generate** — `./gradlew generateSyntheticCaptures`; confirm 20 folders × 10 JPEGs under `data/captures/`.
- [ ] **Step 3: End-to-end run** — boot the app: prepare → train → evaluate through the dashboard UI, then predict `cards/png/jd07.png` via the predict page and confirm it names Gradle (jd07) with high confidence, and that a held-out *synthetic* capture from `dataset/valid` also names its card.
- [ ] **Step 4: Restart check** — restart the app; confirm the dashboard still lists the earlier jobs (H2 persistence) and `/api/predict` still works (model reload).
- [ ] **Step 5: Clean run for the stream** — delete `data/`, confirm the app starts with an empty dashboard and the capture page guides the flow: capture → prepare → train → evaluate → predict.
