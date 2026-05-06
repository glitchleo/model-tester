# UCF

Wrapper for the UCF detector from DeepfakeBench.

## Required Local Checkpoint

```text
models/ucf/ucf_best.pth
```

The checkpoint is ignored by git. The wrapper uses the local F3Net repo's Xception implementation because UCF uses the same Xception backbone shape.

## Verify

From the repo root:

```powershell
python .\check_setup.py --model ucf --smoke
```

## Run

Image:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model ucf --json
```

Video:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model ucf --video-frames 32 --json
```

UCF is image-based, so video scoring samples frames and averages the frame scores.
