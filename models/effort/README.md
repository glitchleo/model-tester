# EFFORT

Local wrapper for EFFORT / CLIP-L14 deepfake scoring.

## Required Local Files

```text
models/effort/
|-- run_image.py
|-- run_video.py
|-- effort_detector.py
|-- weights/
|   `-- effort_clip_L14_trainOn_FaceForensic.pth
|-- pretrained/
|   `-- clip-vit-large-patch14/
|       |-- config.json
|       |-- model.safetensors
|       `-- preprocessor_config.json
|-- landmarks/
|   `-- shape_predictor_81_face_landmarks.dat
`-- repo/
    `-- Effort-AIGI-Detection/
```

The local runner instantiates the CLIP-L14 vision backbone from `config.json`, replaces the self-attention linears with EFFORT residual layers, and loads `weights/effort_clip_L14_trainOn_FaceForensic.pth`.

## Commands

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model effort --json
python .\score_image.py .\data\samples\your_video.mp4 --model effort --video-frames 8 --json
```

The copied upstream repository is kept under `repo/Effort-AIGI-Detection/` for reference. Large weights, CLIP files, and landmark assets stay local and ignored by git.
