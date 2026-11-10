package org.jobrunr.democatch.config;

import java.nio.file.Path;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class AppPaths {

    private final Path dataDir;

    public AppPaths(@Value("${app.data-dir:data}") String dataDir) {
        this.dataDir = Path.of(dataDir);
    }

    public Path root() {
        return dataDir;
    }

    public Path captures() {
        return dataDir.resolve("captures");
    }

    public Path datasetTrain() {
        return dataDir.resolve("dataset").resolve("train");
    }

    public Path datasetValid() {
        return dataDir.resolve("dataset").resolve("valid");
    }

    public Path model() {
        return dataDir.resolve("model");
    }
}
