# F3Net

Wrapper for the upstream F3Net detector.

- Upstream repo: https://github.com/yyk-wew/F3Net
- Upstream source lives in `repo/F3Net-main/`
- Weight files live in `weights/`

## Required Local Weights

The current repo setup expects:

```text
models/f3net/weights/xception-b5690688.pth
models/f3net/weights/f3net_best.pth
```

`f3net_best.pth` is the detector checkpoint used for scoring. The wrapper auto-selects it when no explicit checkpoint is passed. `xception-b5690688.pth` is kept for upstream-style F3Net checkpoints that initialize from the ImageNet Xception backbone.

To use a different detector checkpoint, copy it into `weights/`, for example:

```text
models/f3net/weights/your_f3net_detector.pth
```

All `.pth` files in `weights/` are ignored by git.

## Verify

From the repo root:

```powershell
python .\check_setup.py --model f3net --smoke
```

Expected F3Net result:

- `F3Net Xception backbone`: OK when `xception-b5690688.pth` exists
- `F3Net detector checkpoint`: OK when `f3net_best.pth` exists
- smoke test prints a F3Net score

## Run

Image:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model f3net --json
```

Video:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model f3net --video-frames 32 --json
```

F3Net is image-based, so the video wrapper samples frames and averages frame scores.

If you have multiple detector checkpoints in `weights/`, choose one explicitly:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model f3net --f3net-checkpoint .\models\f3net\weights\your_f3net_detector.pth
```
