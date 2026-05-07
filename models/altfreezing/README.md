# AltFreezing

Wrapper for the upstream AltFreezing video model.

- Upstream repo: https://github.com/ZhendongWang6/AltFreezing
- Upstream source lives in `repo/AltFreezing-main/`
- Inference checkpoint lives in `checkpoints/`
- Auxiliary face/landmark weights live in the repo-root `auxillary/` folder

## Required Local Weights

These files are ignored by git and must be copied after cloning:

```text
models/altfreezing/checkpoints/model.pth
auxillary/mobilenet0.25_Final.pth
auxillary/mobilenet_224_model_best_gdconv_external.pth
```

The `model.pth` file is the AltFreezing classifier checkpoint. The two `auxillary/*.pth` files are used by the bundled face detector and landmark detector.

## Verify

From the repo root:

```powershell
python .\check_setup.py --model altfreezing
```

The AltFreezing rows should be OK, and CUDA must be available.

## Run

Video:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --json
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --altfreezing-max-frame 400 --json
```

Direct wrapper:

```powershell
python .\models\altfreezing\run_video.py --video .\data\samples\your_video.mp4 --max-frame 400 --skip-output
```

Image proxy:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model altfreezing
```

AltFreezing is a video model. The image wrapper duplicates the input image into a short temporary clip so you can get a proxy image score with one command.

## Notes

- `--video-frames` in `score_image.py` is for frame-based video wrappers and is ignored by AltFreezing.
- `--altfreezing-max-frame` controls AltFreezing's native leading-frame cap. It defaults to the upstream demo value of 400.
- Top-level scoring skips the annotated AltFreezing output video by default. Add `--write-altfreezing-output` to write it under `outputs/altfreezing/`.
- The wrapper emits `DETAIL_JSON` with top suspicious frames, suspicious windows, total video frames, fps, resolution, and a human-readable explanation.
- PyTorch 2.6+ changed `torch.load` defaults. The local wrapper and patched upstream load helpers explicitly use compatibility loading for AltFreezing checkpoints and cached detection files.
