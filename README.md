# Deepfake Detection Model Tester

A unified workspace for testing multiple pretrained deepfake-detection models from a single command. Supports image and video inputs, a FastAPI backend, and a React web UI.

**Models included:** AltFreezing · EFFORT · F3Net · RECCE · SelfBlendedImages · UCF

> **Notice:** This repository contains the source code, wrappers, and UI. Pretrained model weights are excluded from Git due to their size. You must download and place the weights in the correct folders before running the tool.

---

## 1. Cloning & Environment Setup

### Requirements
- **Python 3.11**
- **Node.js 18+** (for the web UI)
- **Git**

### Clone the repository
```powershell
git clone https://github.com/YOUR_USERNAME/model-tester.git
cd model-tester
```

### Install Python Dependencies
We highly recommend using a virtual environment (`.venv`) to avoid conflicts.

```powershell
# 1. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install PyTorch with CUDA (GPU) support
# If you don't have an NVIDIA GPU, this will still install PyTorch but run on CPU.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install all other project requirements
python -m pip install -r .\requirements.txt
```

### Install Node.js Frontend Dependencies
```powershell
cd .\app\web
npm install
cd ..\..
```

---

## 2. Model Weights Setup

Before you can run the system, you must download the pretrained weights for each model and place them in the correct directories. Below is the list of expected file locations. 

*(Note: Depending on where you acquired this project, you may have been provided a ZIP file or a shared drive link containing these exact files. If you are setting this up from scratch, you will need to acquire the original weights from each respective author's repository.)*

| Model | Weight file required | Target path in this repo |
|---|---|---|
| **AltFreezing** | `model.pth` | `models/altfreezing/checkpoints/model.pth` |
| **AltFreezing (aux)** | `mobilenet0.25_Final.pth` | `auxillary/mobilenet0.25_Final.pth` |
| **AltFreezing (aux)** | `mobilenet_224_model_best_gdconv_external.pth` | `auxillary/mobilenet_224_model_best_gdconv_external.pth` |
| **EFFORT** | `effort_clip_L14_trainOn_FaceForensic.pth` | `models/effort/weights/effort_clip_L14_trainOn_FaceForensic.pth` |
| **EFFORT (CLIP)** | `model.safetensors` & `config.json` | `models/effort/pretrained/clip-vit-large-patch14/` |
| **EFFORT (landmarks)** | `shape_predictor_81_face_landmarks.dat` | `models/effort/landmarks/shape_predictor_81_face_landmarks.dat` |
| **F3Net** | `f3net_best.pth` | `models/f3net/weights/f3net_best.pth` |
| **F3Net (Xception)** | `xception-b5690688.pth` | `models/f3net/weights/xception-b5690688.pth` |
| **RECCE** | `recce_best.pth` | `models/recce/weights/recce_best.pth` |
| **SelfBlendedImages**| `FFc23_repacked.tar` | `models/selfblendedimages/weights/FFc23_repacked.tar` |
| **UCF** | `ucf_best.pth` | `models/ucf/ucf_best.pth` |

### Verify Your Setup
Once you have placed the weights in their respective folders, run the verification script to ensure everything is correct:

```powershell
python .\check_setup.py
```
If you see **OK** for all models, you are ready to proceed!

---

## 3. Running the Application

### Option A: Running with Docker (Easiest)
If you have Docker installed, you can skip the manual setup and let Docker handle everything (provided you have already placed the weights in their folders).

```powershell
# For CPU only
docker-compose up --build

# For NVIDIA GPU support
docker-compose -f docker-compose.gpu.yml up --build
```
Access the UI at `http://127.0.0.1:5173/` and the API at `http://127.0.0.1:8000/docs`.

### Option B: Running Locally

Open two PowerShell terminals in the project root:

```powershell
# Terminal 1 — Backend API
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend Web UI
cd .\app\web
npm run dev
```

Open `http://127.0.0.1:5173/` in your browser. The UI lets you upload an image or video and get a combined deepfake score from all available models.

---

## 4. Command-Line Usage

You can also use the CLI to score videos or images in bulk.

### Score a single file
```powershell
# Auto-detect all available models
python .\score_image.py .\data\samples\deepfake\id0_id16_0001.mp4 --json

# Run a specific model
python .\score_image.py .\data\samples\deepfake\id0_id16_0001.mp4 --model f3net --json
```

### Batch scoring (entire folder → TSV table)
```powershell
python .\score_image.py .\data\samples\deepfake --model available --recursive > deepfake_results.tsv
```

For more CLI options, run:
```powershell
python .\score_image.py --help
```
