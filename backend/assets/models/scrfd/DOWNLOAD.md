# ScrFD Model Download Instructions

This directory should contain the ScrFD 2.5G face detection model file: `scrfd_2.5g_bnkps.onnx`

**Model Details:**
- Name: ScrFD 2.5G (with keypoints)
- Size: ~2.5-3 MB
- Input: 640x640
- License: MIT (free for commercial use)
- Source: InsightFace

---

## Method 1: Direct Download via Browser (Recommended)

1. **Open this link in your browser:**
   ```
   https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx
   ```
   
   **Alternative links (if above fails):**
   - https://huggingface.co/Linaqruf/face-detectors/resolve/main/scrfd_2.5g_bnkps.onnx
   - https://github.com/linghu8812/scrfd/releases (check releases)

2. **Verify file size:**
   - The downloaded file should be approximately **2.5-3 MB**
   - If it's only a few KB, it's an LFS pointer (not the real file)

3. **On your local machine, verify it's ONNX:**
   ```bash
   file scrfd_2.5g_bnkps.onnx
   ```
   Expected output: `ONNX Model` or `data`

4. **Copy to server:**
   ```bash
   scp scrfd_2.5g_bnkps.onnx root@178.72.132.235:/opt/youtube-shorts-generator/backend/assets/models/scrfd/
   ```

---

## Method 2: Python Script on Server

If direct download works on the server, create a script `download_scrfd.py`:

```python
#!/usr/bin/env python3
import urllib.request
import os

url = "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx"
output = "scrfd_2.5g_bnkps.onnx"

print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, output)

size = os.path.getsize(output)
print(f"Downloaded: {output} ({size / 1024 / 1024:.2f} MB)")

if size < 1000000:  # Less than 1 MB
    print("WARNING: File is too small, might be an LFS pointer!")
else:
    print("Success! File size looks correct.")
```

Run on server:
```bash
cd /opt/youtube-shorts-generator/backend/assets/models/scrfd/
python3 download_scrfd.py
```

---

## Method 3: Using wget with Headers

```bash
cd /opt/youtube-shorts-generator/backend/assets/models/scrfd/
wget --header="Accept: application/octet-stream" \
     -O scrfd_2.5g_bnkps.onnx \
     https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx

# Verify
ls -lh scrfd_2.5g_bnkps.onnx
file scrfd_2.5g_bnkps.onnx
```

---

## Verification

After download, verify the file:

```bash
cd /opt/youtube-shorts-generator/backend/assets/models/scrfd/
ls -lh scrfd_2.5g_bnkps.onnx
file scrfd_2.5g_bnkps.onnx
sha256sum scrfd_2.5g_bnkps.onnx
```

**Expected:**
- Size: 2.5-3 MB (2,600,000 - 3,000,000 bytes)
- Type: `data` or `ONNX model`
- SHA256: (will vary by version, but consistent hash indicates valid file)

---

## Troubleshooting

**If file is ~1-5 KB:**
- It's a Git LFS pointer, not the real file
- Use **Method 1 (browser)** instead

**If all methods fail:**
- Contact the user to manually download and scp the file
- Alternative: Use a different model (MediaPipe, RetinaFace)

