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
|   |-- effort/
|   |   |-- weights/
|   |   |   `-- effort_clip_L14_trainOn_FaceForensic.pth
|   |   |-- pretrained/
|   |   |   `-- clip-vit-large-patch14/
|   |   |       |-- config.json
|   |   |       |-- model.safetensors
|   |   |       `-- preprocessor_config.json
|   |   `-- landmarks/
|   |       `-- shape_predictor_81_face_landmarks.dat
|   |-- selfblendedimages/
|   |   `-- weights/
|   |       `-- FFc23_extracted/
|   |       OR FFc23_repacked.tar
|   |-- f3net/
|   |   `-- weights/
|   |       |-- xception-b5690688.pth
|   |       `-- f3net_best.pth
|   |-- recce/
|   |   `-- weights/
|   |       `-- recce_best.pth
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

EFFORT:

- EFFORT detector checkpoint: `models/effort/weights/effort_clip_L14_trainOn_FaceForensic.pth`
- CLIP-L14 local config/model files: `models/effort/pretrained/clip-vit-large-patch14/`
- Optional landmark predictor: `models/effort/landmarks/shape_predictor_81_face_landmarks.dat`

F3Net:

- F3Net backbone: `models/f3net/weights/xception-b5690688.pth`
- F3Net detector checkpoint: `models/f3net/weights/f3net_best.pth`, or another detector `.pth` passed with `--f3net-checkpoint`

The included `f3net_best.pth` is a FAD-only 256px detector checkpoint. The runner detects that format and uses its `backbone.*` classifier weights directly. The Xception backbone file is still used for upstream-style F3Net checkpoints that need ImageNet Xception initialization.

RECCE:

- RECCE detector checkpoint: `models/recce/weights/recce_best.pth`, or another checkpoint passed with `--recce-checkpoint`

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
- `EFFORT`: OK when its checkpoint, CLIP config, and source repo are present.
- `SelfBlendedImages`: OK when its FF-c23 checkpoint exists.
- `F3Net`: OK when `xception-b5690688.pth` and `f3net_best.pth` exist.
- `RECCE`: OK when `models/recce/weights/recce_best.pth` and the RECCE source repo exist.
- `UCF`: OK when `models/ucf/ucf_best.pth` exists.

Run a F3Net-only or UCF-only setup check and smoke test:

```powershell
python .\check_setup.py --model f3net --smoke
python .\check_setup.py --model recce --smoke
python .\check_setup.py --model ucf --smoke
python .\check_setup.py --model effort --smoke
```

Run a quick image smoke test:

```powershell
python .\check_setup.py --smoke --model selfblendedimages
```

Run an AltFreezing video smoke test:

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --json
```

## Scoring Commands

Default scoring auto-detects available local models based on the weights, source repos, and runtime dependencies present in this workspace.

```powershell
python .\score_image.py .\data\samples\your_image.jpg --json
python .\score_image.py .\data\samples\your_image.jpg --model available --json
python .\score_image.py .\data\samples\your_video.mp4 --json
```

Use `--model all` only when you intentionally want to try every configured model and see which ones are missing weights or dependencies.
AltFreezing is video-only and is not selected for image/photo inputs.

For frame-based video wrappers, the default is a quick 8-frame test. Increase the frame budget when you want more coverage. AltFreezing is video-based, so it keeps its native 400-frame cap unless you pass `--altfreezing-max-frame`.

```powershell
python .\score_image.py .\data\samples\your_video.mp4 --video-preset quick --json
python .\score_image.py .\data\samples\your_video.mp4 --video-preset balanced --json
python .\score_image.py .\data\samples\your_video.mp4 --video-preset thorough --json
python .\score_image.py .\data\samples\your_video.mp4 --video-frames 48 --json
```

Frame-based video wrappers use fixed interval sampling from the first decoded frame. For example, `--video-frames 10` on a 200-frame video checks internal frame indices `0,20,40,60,80,100,120,140,160,180`.

The final report includes the models that ran, skipped models, the combined score, video metadata, peak sampled frames, and a short explanation of what the score means.

## Standard Prediction CLI

Use `predict.py` when you want each model to behave like a small black box: input media in, standardized JSON out.

```powershell
python .\predict.py --model f3net --input .\data\samples\your_video.mp4 --pretty
python .\predict.py --model sbi --input .\data\samples\your_image.jpg --pretty
```

The per-model folders also include tiny shims, so this works when testing one model in isolation:

```powershell
cd .\models\f3net
python .\predict.py --input ..\..\data\samples\your_video.mp4 --pretty
```

Successful output uses the same structure for every model:

```json
{
  "model_name": "F3Net",
  "model_id": "f3net",
  "input_type": "video",
  "fake_score": 0.73,
  "real_score": 0.27,
  "status": "suspicious",
  "processing_time": 4.2
}
```

If you run `--model available` or `--model all`, the CLI returns a combined summary plus one standardized result per model.

## Backend API

The FastAPI backend processes uploads immediately, which keeps the thesis demo simple while still giving the app stable API boundaries.

Start it locally:

```powershell
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

## React Frontend

The browser UI lives in `app/web` as a Vite React app with normal CSS.

For local frontend development, start the combined dev launcher:

```powershell
cd .\app\web
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`. This starts FastAPI on `http://127.0.0.1:8000` if it is not already running, waits for `/health`, then starts Vite.

If you already have the backend running in another terminal and only want the frontend:

```powershell
cd .\app\web
npm run dev:ui
```

For the backend to serve the React app directly:

```powershell
cd .\app\web
npm install
npm run build
cd ..\..
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Then open the built web UI on the PC at `http://127.0.0.1:8000/`. On another device on the same network, use the PC's IPv4 address, for example `http://192.168.1.25:8000/`.

Endpoints:

```text
POST /analyze-image
POST /analyze-video
GET  /result/{id}
GET  /models
GET  /health
```

Example upload:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze-video" `
  -F "file=@data/samples/your_video.mp4" `
  -F "model=f3net" `
  -F "video_frames=8"
```

The response includes an `id`; fetch it later with:

```powershell
curl.exe "http://127.0.0.1:8000/result/YOUR_RESULT_ID"
```

Check what the backend can actually run in the current environment:

```powershell
curl.exe "http://127.0.0.1:8000/models"
curl.exe "http://127.0.0.1:8000/models?input_type=video"
```

Uploaded files are saved under `outputs/uploads/`, and result JSON is saved under `outputs/api_results/`.

## Docker Backend

The first Docker target is one backend container. The image builds the React frontend first, then serves the compiled assets from FastAPI. The build context excludes large local weights and sample media; `docker-compose.yml` mounts `models/`, `auxillary/`, `data/`, and `outputs/` at runtime so your local checkpoints are still visible inside the container.

```powershell
docker compose build
docker compose up backend
```

Then open the site on the PC:

```text
http://127.0.0.1:8000/
```

To open it from a laptop on the same Wi-Fi/LAN:

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.IPAddress -notlike "169.254*" } |
  Select-Object InterfaceAlias, IPAddress
```

Use the PC address in the laptop browser, for example:

```text
http://192.168.1.25:8000/
```

If the laptop cannot connect, allow inbound TCP traffic for port `8000` in Windows Defender Firewall or change `MODEL_TESTER_PORT` before starting Compose:

```powershell
$env:MODEL_TESTER_PORT = "8080"
docker compose up --build backend
```

The backend allows browser requests from other machines by default with `MODEL_TESTER_ALLOWED_ORIGINS=*`. For a known frontend origin, lock it down like this:

```powershell
$env:MODEL_TESTER_ALLOWED_ORIGINS = "http://192.168.1.40:5173,http://localhost:5173"
docker compose up --build backend
```

By default the Dockerfile installs CPU PyTorch. That is enough for CPU-capable wrappers, but AltFreezing is expected to report unavailable without CUDA. On a Windows PC with Docker Desktop, WSL2, an NVIDIA driver, and GPU support enabled, use the GPU overlay:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build backend
```

That overlay installs the CUDA PyTorch wheel index and asks Docker to expose all GPUs to the backend container.

Batch video table:

```powershell
python .\score_image.py .\data\samples --model all --video-frames 10 --altfreezing-max-frame 400
python .\score_image.py .\data\samples --model available --video-frames 10 --recursive
python .\score_image.py .\my_video_list.txt --model all --video-frames 10
```

When the input path is a folder, or a `.txt`/`.lst` file with one video path per line, output is a clean tab-separated table:

```text
video	total_frames	final_score	status	altfreezing	effort	f3net	recce	selfblendedimages	ucf
2.mp4	153	0.533414	suspicious	0.289679	0.505050	0.659942	0.597777	0.581584	0.566454
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
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --json
python .\score_image.py .\data\samples\your_video.mp4 --model altfreezing --altfreezing-max-frame 400 --json
```

Image scoring:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model selfblendedimages
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

EFFORT image/video scoring:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model effort --json
python .\score_image.py .\data\samples\your_video.mp4 --model effort --video-frames 8 --json
```

If you want non-default EFFORT assets, pass them with `--effort-checkpoint` and `--effort-clip-model`.

RECCE image/video scoring:

```powershell
python .\score_image.py .\data\samples\your_image.jpg --model recce --json
python .\score_image.py .\data\samples\your_video.mp4 --model recce --video-frames 8 --json
```

If you want a non-default RECCE checkpoint, pass it with `--recce-checkpoint`.

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

- `--video-frame-mode uniform`: fixed interval sampling from the first decoded frame.
- `--video-frame-mode adaptive`: coarse scan first, then refine around higher-scoring timeframes.
- `--video-frame-mode all`: every decoded frame.

Useful SelfBlendedImages controls:

- `--video-frames`: maximum frames to analyze in limited modes.
- `--video-frame-interval`: coarse interval for adaptive mode.
- `--video-refine-window`: number of neighboring frames to inspect around hotspots.
- `--video-batch-size`: face-crop inference batch size; lower this if CUDA memory is tight.
- `--selfblendedimages-face-max-size`: cap RetinaFace detector size; lower this if CUDA memory is tight.

AltFreezing is already video-based and is not selected for image/photo inputs. For videos, `--video-frames` is ignored by AltFreezing so frame-based model testing can use the chosen frame count without changing AltFreezing behavior. Use `--altfreezing-max-frame` only when you intentionally want to change AltFreezing's native leading-frame cap.

## Model Layout

- `models/altfreezing/`: wrapper plus upstream `ZhendongWang6/AltFreezing` source in `repo/`.
- `models/effort/`: wrapper plus upstream `YZY-stack/Effort-AIGI-Detection` source in `repo/`; large checkpoint and CLIP assets live under ignored local folders.
- `models/selfblendedimages/`: wrapper plus upstream `mapooon/SelfBlendedImages` source in `repo/`.
- `models/f3net/`: wrapper plus upstream `yyk-wew/F3Net` source in `repo/`.
- `models/recce/`: wrapper plus upstream RECCE source in `repo/`; DeepfakeBench detector reference in `deepfakebench/`.
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
- no files under `models/*/pretrained/`
- no files under `models/*/landmarks/`
- no `.pth` or `.pt` files directly under `models/*/`
- no `.safetensors` files directly under `models/*/`
- no `.pth` files under `auxillary/`
- no sample videos/images under `data/samples/`
- no generated files under `outputs/`

The `.gitkeep` and README files in those folders should remain tracked so the folder structure exists after cloning.
