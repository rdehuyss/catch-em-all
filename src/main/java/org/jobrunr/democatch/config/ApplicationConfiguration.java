package org.jobrunr.democatch.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.nio.file.Path;

@Component
public class ApplicationConfiguration {

    private final Path dataDir;

    public ApplicationConfiguration(@Value("${app.data-dir:data}") String dataDir) {
        this.dataDir = Path.of(dataDir);
    }

    public Path capturesPath() {
        return dataDir.resolve("captures");
    }

    public Path trainingDatasetPath() {
        return dataDir.resolve("dataset").resolve("train");
    }

    public Path validationDatasetPath() {
        return dataDir.resolve("dataset").resolve("valid");
    }

    public Path model() {
        return dataDir.resolve("model");
    }
}
