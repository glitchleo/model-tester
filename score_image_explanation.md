# score_image.py Explanation

## One-sentence version

`score_image.py` is the coordinator. It does not contain the deepfake models themselves; it chooses the right model wrapper scripts, runs them, reads their scores, and builds one final result.

## Main idea

The project has several model-specific runners under `models/`. Each runner knows how to use one model: AltFreezing, EFFORT, F3Net, RECCE, SelfBlendedImages, or UCF. The coordinator gives each runner the same input media and expects the same simple output format back:

- `SCORE:<number>` means the model produced a fakeness score.
- `UNAVAILABLE:<reason>` means the model cannot run because a file, dependency, or runtime requirement is missing.
- `DETAIL_JSON:{...}` gives optional evidence, such as sampled frames or video metadata.

Because every model runner speaks this small output format, the main script can treat different models in the same way.

## Execution flow

1. `parse_args()` reads the command-line options.
2. `infer_media_kind()` decides whether a single input is an image or a video.
3. `collect_video_paths()` handles video folders and `.txt` or `.lst` video lists.
4. `selected_model_names()` decides which models should run.
5. `model_unavailable_reason()` checks local files and dependencies before running models.
6. `build_command()` creates the exact command for each model wrapper.
7. `run_model()` starts the wrapper with `subprocess.run()` and parses its output.
8. `build_summary()` combines successful model scores into one final score.
9. `format_table()`, `format_final_report()`, `format_batch_report()`, and `result_payload()` print either human-readable output, batch output, or JSON.

## How models are connected

`MODEL_SPECS` is the central registry. Each model has:

- a short internal name
- a display label
- an image runner path
- a video runner path
- a default note explaining what the runner did

Adding a new model means adding a new entry there, adding availability checks, and making sure the runner prints `SCORE`, `UNAVAILABLE`, and optional `DETAIL_JSON`.

## What the final score means

Each model returns a score between 0 and 1, where higher means more suspicious. If more than one model succeeds, the script uses the simple average of the successful scores. Models that are skipped, unavailable, or errored are reported, but they are not included in the average.

## How to explain it in class

This script is an orchestration layer. The individual AI models are separate programs, and this file standardizes how they are called. It checks which models are usable, sends the input to each one, collects their results, averages the successful scores, and prints a clear report. That keeps the model code separate from the application logic.
