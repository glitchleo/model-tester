# model-tester

Private workspace for testing three pretrained deepfake-detection models separately in the same repo.

Git tracks the project structure and source code only. Model weights, datasets, sample media, and generated outputs are ignored by `.gitignore`.

## Current layout

```text
model-tester/
|-- models/
|   |-- f3net/
|   |   |-- repo/
|   |   `-- weights/
|   |       `-- xception-b5690688.pth
|   |-- altfreezing/
|   |   |-- repo/
|   |   `-- checkpoints/
|   |       `-- model.pth
|   `-- selfblendedimages/
|       |-- repo/
|       `-- weights/
|           `-- FFc23_extracted/
|-- data/
|   |-- datasets/
|   `-- samples/
`-- outputs/
```

## Model mapping

- `models/f3net/`: for code from `yyk-wew/F3Net`. The `xception-b5690688.pth` file fits this model because F3Net expects a pretrained Xception backbone.
- `models/altfreezing/`: for code from `ZhendongWang6/AltFreezing`. The `model.pth` file fits here because AltFreezing expects its inference checkpoint inside `checkpoints/`.
- `models/selfblendedimages/`: for code from `mapooon/SelfBlendedImages`. The `FFc23_extracted` folder appears to be the extracted contents of the FF-c23 checkpoint rather than the original checkpoint file.

## How to use this structure

1. Put each upstream repository inside its matching `models/<name>/repo/` folder.
2. Keep shared datasets in `data/datasets/`.
3. Keep a few manual test images or videos in `data/samples/`.
4. Save predictions, logs, and comparison outputs in `outputs/`.

## Important note for SelfBlendedImages

The upstream repository expects a weight file inside `weights/` such as `FFraw.tar` or the FF-c23 checkpoint, not an extracted folder. If inference fails later, the safest fix is to re-download the original FF-c23 checkpoint file and place it in `models/selfblendedimages/weights/`.
