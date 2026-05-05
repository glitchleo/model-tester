# F3Net

Wrapper for the upstream F3Net detector.

- Upstream repo: https://github.com/yyk-wew/F3Net
- Upstream source lives in `repo/F3Net-main/`
- Weight files live in `weights/`

## Required Local Weights

The current repo setup has only the pretrained Xception backbone:

```text
models/f3net/weights/xception-b5690688.pth
```

That file is not enough to score deepfakes by itself. To use F3Net, also copy a trained F3Net detector checkpoint into `weights/`, for example:

```text
models/f3net/weights/your_f3net_detector.pth
```

All `.pth` files in `weights/` are ignored by git.

## Verify

From the repo root:

```powershell
python .\check_setup.py --model f3net
```

Expected result right now:

- backbone OK if `xception-b5690688.pth` exists
- detector checkpoint WARN until you add a trained detector checkpoint

## Run

Image:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model f3net --f3net-checkpoint .\models\f3net\weights\your_f3net_detector.pth
```

Video:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model f3net --f3net-checkpoint .\models\f3net\weights\your_f3net_detector.pth --video-frames 32
```

F3Net is image-based, so the video wrapper samples frames and averages frame scores.
