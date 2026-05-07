# RECCE

Local wrapper for RECCE / DeepfakeBench scoring.

## Required Local Files

```text
models/recce/
|-- run_image.py
|-- run_video.py
|-- weights/
|   `-- recce_best.pth
|-- deepfakebench/
|   `-- recce_detector.py
`-- repo/
    `-- RECCE/
```

The local runner uses the upstream `repo/RECCE/model/network/Recce.py` model class, disables runtime Xception weight downloads, strips the DeepfakeBench `model.` checkpoint prefix, and loads `weights/recce_best.pth`.

## Commands

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model recce --json
python .\score_image.py .\data\samples\your_video.mp4 --model recce --video-frames 8 --json
```

Large weights stay under `weights/` and are ignored by git.
