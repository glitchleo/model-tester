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
|   |-- f3net/
|   |   `-- weights/
|   |       |-- xception-b5690688.pth
|   |       `-- f3net_best.pth
|   `-- ucf/
|       `-- ucf_best.pth
|-- data/
|   `-- samples/
|       `-- your_test_video_or_image.mp4
`-- outputs/
```

Required for the previous multi-model workflow:

- AltFreezing checkpoint: `models/altfreezing/checkpoints/model.pth`
- AltFreezing auxiliary weights:
  - `auxillary/mobilenet0.25_Final.pth`
  - `auxillary/mobilenet_224_model_best_gdconv_external.pth`
- SelfBlendedImages checkpoint: either `models/selfblendedimages/weights/FFc23_repacked.tar` or the extracted folder `models/selfblendedimages/weights/FFc23_extracted/`

F3Net:

- F3Net backbone: `models/f3net/weights/xception-b5690688.pth`
- F3Net detector checkpoint: `models/f3net/weights/f3net_best.pth`, or another detector `.pth` passed with `--f3net-checkpoint`

The included `f3net_best.pth` is a FAD-only 256px detector checkpoint. The runner detects that format and uses its `backbone.*` classifier weights directly. The Xception backbone file is still used for upstream-style F3Net checkpoints that need ImageNet Xception initialization.

UCF:

- UCF detector checkpoint: `models/ucf/ucf_best.pth`, or another checkpoint passed with `--ucf-checkpoint`

UCF uses the DeepfakeBench Xception-style backbone at 256px resolution. The local wrapper reuses the tracked F3Net Xception source for the compatible backbone implementation.

## Verify Setup

Run the setup check after copying weights:

```powershell
python .\check_setup.py
```

Expected status:

- `AltFreezing`: OK when CUDA, `model.pth`, and both auxiliary weights are present.
- `SelfBlendedImages`: OK when its FF-c23 checkpoint exists.
- `F3Net`: OK when `xception-b5690688.pth` and `f3net_best.pth` exist.
- `UCF`: OK when `models/ucf/ucf_best.pth` exists.

Run a F3Net-only or UCF-only setup check and smoke test:

```powershell
python .\check_setup.py --model f3net --smoke
python .\check_setup.py --model ucf --smoke
```

Run a quick image smoke test:

```powershell
python .\check_setup.py --smoke --model selfblendedimages
```

Run a small AltFreezing video smoke test:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --video-frames 8 --json
```

## Scoring Commands

Default scoring auto-detects available local models. In this workspace that currently means F3Net and UCF, because AltFreezing and SelfBlendedImages only have placeholder weight folders.

```powershell
python .\score_image.py .\data\samples\your_image.jpg --json
python .\score_image.py .\data\samples\your_image.jpg --model available --json
python .\score_image.py .\data\samples\your_video.mp4 --json
```

Use `--model all` only when you intentionally want to try every configured model and see which ones are missing weights or dependencies.

For videos, the default is a quick 8-frame test across every available model. Increase the frame budget when you want more coverage:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --video-preset quick --json
python .\score_image.py .\data\samples\your_video.mp4 --video-preset balanced --json
python .\score_image.py .\data\samples\your_video.mp4 --video-preset thorough --json
python .\score_image.py .\data\samples\your_video.mp4 --video-frames 48 --json
```

The final report includes the models that ran, any locally unavailable models that were skipped, the combined score, video metadata, peak sampled frames, and a short explanation of what the score means.

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

F3Net image/video scoring:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model f3net --json
python .\score_image.py .\data\samples\your_video.mp4 --model f3net --video-frames 32 --json
```

If more than one F3Net detector checkpoint is present, select one explicitly with `--f3net-checkpoint`.

UCF image/video scoring:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model ucf --json
python .\score_image.py .\data\samples\your_video.mp4 --model ucf --video-frames 32 --json
```

If you want a non-default UCF checkpoint, pass it with `--ucf-checkpoint`.

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
- `models/ucf/`: wrapper plus local UCF checkpoint; uses the tracked F3Net Xception source.
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
- no `.pth` or `.pt` files directly under `models/*/`
- no `.pth` files under `auxillary/`
- no sample videos/images under `data/samples/`
- no generated files under `outputs/`

The `.gitkeep` and README files in those folders should remain tracked so the folder structure exists after cloning.
