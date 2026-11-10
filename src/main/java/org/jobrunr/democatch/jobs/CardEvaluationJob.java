package org.jobrunr.democatch.jobs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.jobrunr.democatch.config.AppPaths;
import org.jobrunr.democatch.ml.CardClassifier;
import org.jobrunr.democatch.ml.ConfusionMatrix;
import org.jobrunr.jobs.annotations.Job;
import org.jobrunr.jobs.context.JobContext;
import org.jobrunr.jobs.context.JobRunrDashboardLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;

@Component
public class CardEvaluationJob {

    private static final Logger LOGGER = new JobRunrDashboardLogger(
            LoggerFactory.getLogger(CardEvaluationJob.class));

    private final AppPaths paths;

    public CardEvaluationJob(AppPaths paths) {
        this.paths = paths;
    }

    @Job(name = "Evaluate card classifier", retries = 0)
    public void evaluate(JobContext jobContext) throws Exception {
        Path modelDir = requireModel();
        List<Path> imagePaths = validImages();
        var progressBar = jobContext.progressBar(imagePaths.size());

        List<String> classes = Files.readAllLines(modelDir.resolve("classes.txt"));
        List<ConfusionMatrix.Record> records = new ArrayList<>();

        LOGGER.info("Evaluating on {} validation images", imagePaths.size());
        try (CardClassifier classifier = CardClassifier.load(modelDir)) {
            for (Path image : imagePaths) {
                String expected = image.getParent().getFileName().toString();
                String actual = CardClassifier.bestLabel(
                        classifier.classes(),
                        classifier.logits(ImageFactory.getInstance().fromFile(image)));
                records.add(new ConfusionMatrix.Record(expected, actual));
                progressBar.incrementSucceeded();
            }
        }

        ConfusionMatrix matrix = new ConfusionMatrix(classes, records);
        writeReport(matrix);
        LOGGER.info("Overall accuracy: {}", matrix.overallAccuracy());
    }

    Path requireModel() {
        Path modelDir = paths.model();
        if (!Files.isDirectory(modelDir)) {
            throw new IllegalStateException(
                    "No trained model found in " + modelDir.toAbsolutePath() + " — run train first");
        }
        if (!Files.exists(modelDir.resolve("head.jd"))) {
            throw new IllegalStateException(
                    "No trained model found in " + modelDir.toAbsolutePath() + " — run train first");
        }
        return modelDir;
    }

    private List<Path> validImages() throws IOException {
        Path validDir = paths.datasetValid();
        if (!Files.isDirectory(validDir)) {
            throw new IllegalStateException("No validation dataset at " + validDir.toAbsolutePath() + " — run prepare first");
        }
        List<Path> result = new ArrayList<>();
        try (var dirs = Files.list(validDir)) {
            dirs.filter(Files::isDirectory).sorted().forEach(dir -> {
                try {
                    result.addAll(CardDataPreparationJob.sortedImages(dir));
                } catch (IOException e) {
                    throw new IllegalStateException("Cannot read " + dir, e);
                }
            });
        }
        if (result.isEmpty()) {
            throw new IllegalStateException("Validation dataset is empty — run prepare first");
        }
        return result;
    }

    private void writeReport(ConfusionMatrix matrix) throws IOException {
        StringBuilder json = new StringBuilder();
        json.append("{\n  \"overallAccuracy\": ").append(matrix.overallAccuracy()).append(",\n");
        json.append("  \"perClass\": [\n");
        var perClass = matrix.perClassAccuracy();
        List<String> classEntries = matrix.classes().stream()
                .map(c -> "    {\"class\": \"%s\", \"accuracy\": %s}".formatted(c, perClass.get(c)))
                .toList();
        json.append(String.join(",\n", classEntries)).append("\n  ],\n");
        json.append("  \"confusion\": [\n");
        int[][] m = matrix.matrix();
        for (int i = 0; i < m.length; i++) {
            StringBuilder row = new StringBuilder("    [");
            for (int j = 0; j < m.length; j++) {
                row.append(m[i][j]);
                if (j < m.length - 1) {
                    row.append(", ");
                }
            }
            row.append("]");
            json.append(row);
            if (i < m.length - 1) {
                json.append(",");
            }
            json.append("\n");
        }
        json.append("  ]\n}\n");

        Path report = paths.model().resolve("evaluation.json");
        Files.writeString(report, json.toString());
        LOGGER.info("Evaluation report written to {}", report);
    }
}
