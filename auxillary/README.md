# AltFreezing auxiliary weights

This folder is for the face detector and landmark auxiliary weights used by AltFreezing.

These files are intentionally ignored by git:

- `mobilenet0.25_Final.pth`
- `mobilenet_224_model_best_gdconv_external.pth`

After cloning on a new machine, copy those two files into this folder before running AltFreezing.

Run this from the repo root to verify:

```powershell
python .\check_setup.py --model altfreezing
```
