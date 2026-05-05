# SelfBlendedImages

Wrapper for the upstream SelfBlendedImages detector.

- Upstream repo: https://github.com/mapooon/SelfBlendedImages
- Upstream source lives in `repo/SelfBlendedImages-master/`
- Pretrained checkpoint files live in `weights/`

## Required Local Weights

Copy one of these after cloning:

```text
models/selfblendedimages/weights/FFc23_repacked.tar
```

or:

```text
models/selfblendedimages/weights/FFc23_extracted/
```

If only `FFc23_extracted/` exists, the wrapper will rebuild `weights/FFc23_repacked.tar` automatically.

The weight files are ignored by git.

## Verify

From the repo root:

```powershell
python .\check_setup.py --model selfblendedimages
```

## Run

Image:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model selfblendedimages
```

Video, adaptive scan:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model selfblendedimages --video-frame-mode adaptive --video-frames 48 --video-frame-interval 20 --video-refine-window 8 --json
```

Video, every frame:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model selfblendedimages --video-frame-mode all --json
```

Direct wrapper:

```powershell
python .\models\selfblendedimages\run_video.py --video .\data\samples\your_video.mp4 --frame-mode adaptive --frames 48 --coarse-interval 20 --refine-window 8
```

## How Video Scoring Works

SelfBlendedImages is image-based. The video wrapper:

1. Selects frames using `adaptive`, `uniform`, or `all` mode.
2. Detects faces in selected frames.
3. Scores face crops in small batches to reduce CUDA memory spikes.
4. Uses the highest face score as each frame's suspiciousness score.
5. Averages scored frame values for the final video score.

The JSON output includes total frame count, sampled frames, top suspicious frames, suspicious windows, and `human_explanation`.

Useful memory controls:

- `--video-batch-size 1`
- `--selfblendedimages-face-max-size 640`
- `--cpu`
