# model-tester

Private workspace for testing pretrained deepfake-detection models from one command.

Git should track source code, wrappers, setup checks, and documentation only. Model weights, datasets, sample media, cached detections, and generated outputs stay local and are ignored.

## Fresh Laptop Setup

Use Python 3.11 on Windows. From a fresh clone:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r .\requirements.txt
python .\check_setup.py
```

If `python` does not point at the repo venv, either activate it again or run commands with:

```powershell
.\.venv\Scripts\python.exe
```

## Local Files To Copy

After cloning on another machine, the only manual work should be copying weights and optional sample media into the ignored folders below.

```text
model-tester/
|-- auxillary/
|   |-- mobilenet0.25_Final.pth
|   `-- mobilenet_224_model_best_gdconv_external.pth
|-- models/
|   |-- altfreezing/
|   |   `-- checkpoints/
|   |       `-- model.pth
|   |-- selfblendedimages/
|   |   `-- weights/
|   |       `-- FFc23_extracted/
|   |       OR FFc23_repacked.tar
|   `-- f3net/
|       `-- weights/
|           |-- xception-b5690688.pth
|           `-- your_trained_f3net_detector.pth
|-- data/
|   `-- samples/
|       `-- your_test_video_or_image.mp4
`-- outputs/
```

Required for the current default workflow:

- AltFreezing checkpoint: `models/altfreezing/checkpoints/model.pth`
- AltFreezing auxiliary weights:
  - `auxillary/mobilenet0.25_Final.pth`
  - `auxillary/mobilenet_224_model_best_gdconv_external.pth`
- SelfBlendedImages checkpoint: either `models/selfblendedimages/weights/FFc23_repacked.tar` or the extracted folder `models/selfblendedimages/weights/FFc23_extracted/`

Optional:

- F3Net backbone: `models/f3net/weights/xception-b5690688.pth`
- F3Net trained detector checkpoint: any additional `.pth` in `models/f3net/weights/`

F3Net is not part of the default `both` run because the current local setup only has the Xception backbone. Add a trained detector checkpoint before expecting F3Net scores.

## Verify Setup

Run the setup check after copying weights:

```powershell
python .\check_setup.py
```

Expected status:

- `AltFreezing`: OK when CUDA, `model.pth`, and both auxiliary weights are present.
- `SelfBlendedImages`: OK when its FF-c23 checkpoint exists.
- `F3Net`: WARN until a trained detector checkpoint is added.

Run a quick image smoke test:

```powershell
python .\check_setup.py --smoke --model selfblendedimages
```

Run a small AltFreezing video smoke test:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --video-frames 8 --json
```

## Scoring Commands

Default video scoring runs the two ready models: AltFreezing and SelfBlendedImages.

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --video-frames 48
```

SelfBlendedImages only, adaptive frame scan:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model selfblendedimages --video-frame-mode adaptive --video-frames 48 --video-frame-interval 20 --video-refine-window 8 --json
```

SelfBlendedImages all-frame scan:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model selfblendedimages --video-frame-mode all --json
```

AltFreezing only:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --video-frames 400 --json
```

Image scoring:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model selfblendedimages
python .\score_image.py .\data\samples\your_image.jpg --model altfreezing
```

F3Net, only after adding a trained detector checkpoint:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model f3net --f3net-checkpoint .\models\f3net\weights\your_f3net_detector.pth
python .\score_image.py .\data\samples\your_video.mp4 --model f3net --f3net-checkpoint .\models\f3net\weights\your_f3net_detector.pth --video-frames 32
```

## Output Format

Normal text output ends with a final summary:

- each successful model score
- final score as a simple mean of successful model scores
- video metadata, including total frame count when available
- one reasoning example per successful model

With `--json`, stdout remains valid JSON:

```json
{
  "summary": {},
  "results": []
}
```

The readable final explanation is also printed to stderr so you can see it in the console while another program safely parses stdout JSON.

Each successful result includes:

- `score`
- `details.video.total_frames`
- `details.top_frames`
- `details.suspicious_windows`
- `human_explanation`

## Video Frame Modes

SelfBlendedImages is image-based, so the video wrapper selects frames, detects faces, scores face crops, and averages per-frame scores.

- `--video-frame-mode adaptive`: coarse scan first, then refine around higher-scoring timeframes.
- `--video-frame-mode uniform`: evenly spaced frame sampling.
- `--video-frame-mode all`: every decoded frame.

Useful SelfBlendedImages controls:

- `--video-frames`: maximum frames to analyze in limited modes.
- `--video-frame-interval`: coarse interval for adaptive mode.
- `--video-refine-window`: number of neighboring frames to inspect around hotspots.
- `--video-batch-size`: face-crop inference batch size; lower this if CUDA memory is tight.
- `--selfblendedimages-face-max-size`: cap RetinaFace detector size; lower this if CUDA memory is tight.

AltFreezing is already video-based. For videos, `--video-frames` is passed as AltFreezing's `--max-frame`.

## Model Layout

- `models/altfreezing/`: wrapper plus upstream `ZhendongWang6/AltFreezing` source in `repo/`.
- `models/selfblendedimages/`: wrapper plus upstream `mapooon/SelfBlendedImages` source in `repo/`.
- `models/f3net/`: wrapper plus upstream `yyk-wew/F3Net` source in `repo/`.
- `data/samples/`: local test images/videos, ignored by git.
- `outputs/`: generated predictions/logs, ignored by git.

## Before Commit

Run:

```powershell
git status --short
python .\check_setup.py
```

Make sure no large local artifacts are staged:

- no files under `models/*/weights/`
- no files under `models/*/checkpoints/`
- no `.pth` files under `auxillary/`
- no sample videos/images under `data/samples/`
- no generated files under `outputs/`

The `.gitkeep` and README files in those folders should remain tracked so the folder structure exists after cloning.
