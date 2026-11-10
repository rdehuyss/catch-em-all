package org.jobrunr.democatch.web;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;

import org.jobrunr.democatch.jobs.CardDataPreparationJob;
import org.jobrunr.democatch.jobs.CardEvaluationJob;
import org.jobrunr.democatch.jobs.CardTrainingJob;
import org.jobrunr.scheduling.JobScheduler;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/jobs")
public class TrainingController {

    private final JobScheduler jobScheduler;
    private final CardDataPreparationJob preparationJob;
    private final CardTrainingJob trainingJob;
    private final CardEvaluationJob evaluationJob;
    private final org.jobrunr.democatch.config.AppPaths paths;

    public TrainingController(JobScheduler jobScheduler, CardDataPreparationJob preparationJob,
                              CardTrainingJob trainingJob, CardEvaluationJob evaluationJob,
                              org.jobrunr.democatch.config.AppPaths paths) {
        this.jobScheduler = jobScheduler;
        this.preparationJob = preparationJob;
        this.trainingJob = trainingJob;
        this.evaluationJob = evaluationJob;
        this.paths = paths;
    }

    @PostMapping("/prepare")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public Map<String, String> prepare() {
        var id = jobScheduler.enqueue(() -> preparationJob.prepare(org.jobrunr.jobs.context.JobContext.Null));
        return Map.of("jobId", id.toString());
    }

    @PostMapping("/evaluate")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public Map<String, String> evaluate() {
        var id = jobScheduler.enqueue(() -> evaluationJob.evaluate(org.jobrunr.jobs.context.JobContext.Null));
        return Map.of("jobId", id.toString());
    }

    @GetMapping("/evaluation")
    public org.springframework.http.ResponseEntity<String> evaluation() throws java.io.IOException {
        Path report = paths.model().resolve("evaluation.json");
        if (!Files.exists(report)) {
            return org.springframework.http.ResponseEntity.notFound().build();
        }
        return org.springframework.http.ResponseEntity.ok()
                .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                .body(Files.readString(report));
    }

    @PostMapping("/train")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public Map<String, String> train(Integer epochs, Integer batchSize) {
        var id = jobScheduler.enqueue(() ->
                trainingJob.train(org.jobrunr.jobs.context.JobContext.Null, epochs, batchSize));
        return Map.of("jobId", id.toString());
    }
}
