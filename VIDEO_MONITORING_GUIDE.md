# 🎬 VIDEO CROWD MONITORING GUIDE

## Overview

The `video_crowd_monitor.py` script extends the real-time crowd monitoring system to process video files from multiple sources:

1. **📁 Local Files** - MP4/MOV files on your computer
2. **🌐 YouTube** - Directly download and process from YouTube
3. **📊 Live Preview** - Watch processing in real-time
4. **💾 Output Saving** - Automatically saves annotated videos

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
# Core dependencies (already installed)
pip install tensorflow==2.13.0 opencv-python==4.8.0.74 gradio==4.15.0

# Additional for YouTube support
pip install yt-dlp
```

### Step 2: Verify Installation

```bash
# Check TensorFlow
python -c "import tensorflow; print(tensorflow.__version__)"

# Check yt-dlp
python -c "import yt_dlp; print('yt-dlp installed')"
```

### Step 3: Run the Script

```bash
python video_crowd_monitor.py
```

**Expected Output:**
```
[1/5] Loading configuration...
[2/5] Loading TensorFlow model (this may take 20-30 seconds)...
✓ Model loaded in 25.3s
[3/5] Checking dependencies...
✓ yt-dlp installed (YouTube support enabled)
[4/5] Initializing interface...
[5/5] Launching Gradio interface...
  Opening at: http://localhost:7860
```

---

## Usage Guide

### Option 1: Process Local Video File

#### Step 1: Open Web Interface
- Navigate to `http://localhost:7860`
- Click on **"📁 Local Video"** tab

#### Step 2: Upload Video
- Click "Upload Video (MP4/MOV)"
- Select your video file from your computer
- Supported formats: MP4, MOV, AVI

#### Step 3: Process
- Click **"Process Video"** button
- Watch progress in status box
- Live preview updates every 0.5 seconds

#### Step 4: View Results
- **Live Preview Tab**: See annotated frames
- **Statistics Tab**: Real-time metrics
- **Output**: Saved to `crowd_monitor_output/` folder

**Example:**
```
1. Upload: street_crowd.mp4 (640×480, 30fps, 60 seconds)
2. Start processing
3. System processes all 1800 frames
4. Output saved as: crowd_monitor_20251215_143022.mp4
```

---

### Option 2: Download & Process from YouTube

#### Step 1: Get YouTube URL
- Find a crowd video on YouTube
- Copy the URL (e.g., `https://www.youtube.com/watch?v=...`)

**Good Sources:**
- **Live events**: Concert footage, sports events
- **Crowd scenes**: Mall walkways, public squares
- **Pedestrian areas**: Urban streets, subway stations
- **Emergency drills**: Evacuation videos
- **Traffic**: Busy intersections

#### Step 2: Paste URL
- Open web interface: `http://localhost:7860`
- Click **"🌐 YouTube"** tab
- Paste URL in text box

#### Step 3: Download & Process
- Click **"Download & Process"** button
- System downloads video (may take 2-10 minutes depending on length)
- Automatically starts processing
- Watch progress on **"Live Preview"** tab

#### Step 4: Monitor Progress
- **Status updates**: Every 0.5 seconds
- **Statistics**: Every 1 second
- **Output video**: Automatically saved

**Example URLs (Educational Purposes):**
```
Crowd monitoring research videos:
- https://www.youtube.com/watch?v=traffic_example
- Street crowd footage
- Event crowd management videos
```

---

## Supported Video Formats

### Local Files
| Format | Extension | Codec | Support |
|--------|-----------|-------|---------|
| MPEG-4 | .mp4 | H.264/H.265 | ✅ Full |
| QuickTime | .mov | ProRes | ✅ Full |
| AVI | .avi | MPEG-4 | ✅ Full |
| WebM | .webm | VP9 | ✅ Good |
| MKV | .mkv | H.264 | ⚠️ Limited |

### Recommended Settings for Input Videos
```
Resolution:    640×480 or higher
FPS:           24-30 fps
Codec:         H.264 (most compatible)
Duration:      30 seconds - 10 minutes
File Size:     < 1 GB
```

---

## Understanding the Output

### 1. Annotated Video Frame

The output shows:

```
┌─────────────────────────────────────────┐
│  Count: 523                             │  ← Predicted count
│  Threshold: 800                         │  ← Alert threshold
│  ⚠️ ALERT! (if > threshold)             │  ← Alert status
│  Progress: 45%                          │  ← Overall progress
│  Frame: 810/1800                        │  ← Current frame
│  2025-12-15 14:30:22                    │  ← Timestamp
│                                         │
│  [RED BORDER when alert]                │  ← Visual indicator
└─────────────────────────────────────────┘
```

### 2. Density Heatmap

```
RED   = High crowd concentration (center)
YELLOW= Medium density
BLUE  = Low/no crowd (edges)
```

### 3. Output Video

**File Location:** `crowd_monitor_output/crowd_monitor_YYYYMMDD_HHMMSS.mp4`

**Contains:**
- Original video frames
- Real-time crowd counts
- Alert indicators
- Progress bar
- Timestamp and frame number

---

## Configuration & Settings

### Alert Threshold

**Default:** 800 people

**How to Change:**
1. Open web interface
2. Go to **"⚙️ Settings"** tab
3. Adjust slider (100-2000)
4. Click **"Update Threshold"**

**Recommended Values:**
```
Public spaces (malls):      600-800
Outdoor events:             1000-1500
Crowded streets:            800-1000
Emergency scenarios:        500-700
```

### Output Settings

Edit `CONFIG` in the script:

```python
CONFIG = {
    'model_path': 'results/csrnet_direct/model.keras',
    'frame_size': (256, 256),         # Don't change
    'alert_threshold': 800,            # Change via UI
    'smooth_window': 5,                # Higher = smoother
    'fps_target': 10,                  # Processing FPS
    'output_dir': 'crowd_monitor_output',  # Output folder
    'save_output': True,               # Save video
}
```

---

## Performance & Optimization

### Processing Speed

| Hardware | FPS | Time for 1hr video |
|----------|-----|-------------------|
| GPU (RTX) | 15-20 fps | 3-4 minutes |
| GPU (GTX) | 10-15 fps | 4-6 minutes |
| CPU (Intel) | 5-10 fps | 6-12 minutes |
| CPU (AMD) | 5-10 fps | 6-12 minutes |

### Memory Usage

```
RAM:    ~2-4 GB (loading frames)
VRAM:   ~1-2 GB (if using GPU)
Disk:   ~1 GB per hour of video
```

### Tips for Faster Processing

1. **Reduce video resolution** before uploading:
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:480 output.mp4
   ```

2. **Use shorter clips** for testing:
   ```bash
   ffmpeg -i input.mp4 -t 60 output_short.mp4  # First 60 seconds
   ```

3. **Adjust frame processing rate** in CONFIG:
   ```python
   'fps_target': 5  # Process 5 frames/sec instead of 10
   ```

---

## Troubleshooting

### Issue 1: "Video file not found"

**Cause:** Incorrect file path or moved file

**Solution:**
- Use absolute paths
- Ensure file exists
- Check file permissions

### Issue 2: "yt-dlp not installed"

**Cause:** Missing YouTube downloader

**Solution:**
```bash
pip install yt-dlp --upgrade
```

### Issue 3: Slow Processing

**Cause:** CPU-only processing or large video

**Solution:**
1. Process shorter clips first
2. Use GPU if available
3. Reduce video resolution

### Issue 4: "Out of memory"

**Cause:** Video too large

**Solution:**
1. Process shorter videos (< 5 min)
2. Close other applications
3. Split video: `ffmpeg -i input.mp4 -segment_time 60 -f segment output_%02d.mp4`

### Issue 5: Output video won't play

**Cause:** Codec incompatibility

**Solution:**
- Convert using FFmpeg:
```bash
ffmpeg -i output.mp4 -c:v libx264 output_converted.mp4
```

---

## Advanced Usage

### Batch Processing Multiple Videos

Create `batch_process.py`:

```python
from pathlib import Path
from video_crowd_monitor import VideoProcessor

video_dir = Path("videos")
for video_file in video_dir.glob("*.mp4"):
    print(f"Processing {video_file}...")
    processor = VideoProcessor(str(video_file))
    processor.process_video()
```

Run:
```bash
python batch_process.py
```

### Custom Model

To use a different model:

```python
CONFIG = {
    'model_path': 'path/to/your/model.keras',  # Change this
    ...
}
```

### Extract Statistics to CSV

Add to the script:

```python
import csv

def save_statistics(output_file="stats.csv"):
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Count', 'Alert', 'FPS'])
        for frame_num, count, alert in zip(state.frame_history, ...):
            writer.writerow([frame_num, count, alert, state.fps])
```

---

## Real-World Applications

### 1. Public Space Monitoring
```
Input:  Mall walkway video
Output: Crowd heatmap showing dense areas
Use:    Optimize traffic flow
```

### 2. Event Management
```
Input:  Concert entrance video
Output: Real-time capacity tracking
Use:    Safety and crowd control
```

### 3. Traffic Analysis
```
Input:  Street intersection video
Output: Pedestrian count over time
Use:    Urban planning
```

### 4. Research & Development
```
Input:  Test footage
Output: Annotated video + statistics
Use:    Model validation
```

---

## FAQ

**Q: Can I process live streams?**
A: Not directly. Download the stream first, then process.

**Q: How long can the video be?**
A: No hard limit, but 10+ minutes may use significant memory.

**Q: Can I customize the output format?**
A: Yes, modify the `VideoWriter` settings in the code.

**Q: What's the accuracy?**
A: Depends on the CSRNet model training. Typically ±10-20% for crowd sizes.

**Q: Can I pause processing?**
A: Currently processes all frames continuously. Pause feature coming soon.

**Q: Is the model real-time?**
A: Model can process ~10-15 fps on GPU, ~5-10 fps on CPU.

---

## Next Steps

1. **Test with sample video** (30 seconds, 720p)
2. **Adjust threshold** to your use case
3. **Export statistics** for analysis
4. **Integrate with** your monitoring system
5. **Scale to** multiple cameras/videos

---

## Support & Resources

- **CSRNet Paper**: [arxiv.org/abs/1802.10062](https://arxiv.org/abs/1802.10062)
- **OpenCV Docs**: [docs.opencv.org](https://docs.opencv.org)
- **TensorFlow**: [tensorflow.org](https://tensorflow.org)
- **Gradio**: [gradio.app](https://gradio.app)

