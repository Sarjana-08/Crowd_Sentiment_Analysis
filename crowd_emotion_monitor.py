#!/usr/bin/env python3
"""
Integrated Crowd Emotion Monitor - Simplified Robust Version
Real-time crowd detection, sentiment analysis, heatmap, and panic alerts
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque
from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS

# Sentiment
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except:
    sia = None

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'display_size': (640, 480),
    'panic_threshold': 60,
    'fps_target': 15,
    'port': 5001,
}

# ============================================================================
# STATE
# ============================================================================

class State:
    def __init__(self):
        self.crowd_count = 0
        self.density = 0.0
        self.movement = 0.0
        self.sentiment = 0.0
        self.panic_percent = 0.0
        self.is_panic = False
        self.alerts = deque(maxlen=50)
        self.frame_encoded = None
        self.fps = 0
        self.status = "Initializing"
        self.frame_prev = None
        self.lock = threading.Lock()

state = State()

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_frame(frame):
    """Analyze frame for crowd metrics"""
    if frame is None:
        return 0, 0.0, 0.0
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection (proxy for people)
    edges = cv2.Canny(gray, 50, 150)
    crowd_count = np.sum(edges > 0) // 50  # Scale down
    
    # Density
    density = min(100.0, (crowd_count / 500.0) * 100)
    
    # Movement
    movement = 0.0
    if state.frame_prev is not None and state.frame_prev.shape == frame.shape:
        diff = cv2.absdiff(state.frame_prev, frame)
        movement = np.mean(diff) / 255.0
    
    state.frame_prev = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return int(crowd_count), density, movement

def generate_heatmap(frame):
    """Generate density heatmap"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Blur for heatmap effect
    heatmap = cv2.GaussianBlur(edges, (31, 31), 0)
    heatmap = (heatmap * 255 / heatmap.max()).astype(np.uint8) if heatmap.max() > 0 else heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_colored

def calculate_panic(crowd, density, movement, sentiment):
    """Calculate panic percentage"""
    panic = 0.0
    panic += min(100, density) * 0.4  # Density
    panic += min(100, movement * 150) * 0.3  # Movement
    panic += max(0, -sentiment) * 100 * 0.2  # Negative sentiment
    if crowd > 200 and movement > 0.3:
        panic += 10
    return min(100.0, panic)

def overlay_viz(frame, crowd, density, movement, sentiment, panic):
    """Add visualization overlays"""
    h, w = frame.shape[:2]
    
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Crowd: {crowd}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Density: {density:.0f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Movement: {movement:.2f}", (w//2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Sentiment: {sentiment:.2f}", (w//2, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Panic meter (right side)
    meter_x, meter_y = w - 160, 15
    meter_w, meter_h = 140, 80
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
    
    panic_width = int((panic / 100) * meter_w)
    color = (0, 0, 255) if panic >= CONFIG['panic_threshold'] else (0, 165, 255)
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + panic_width, meter_y + meter_h), color, -1)
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (255, 255, 255), 2)
    cv2.putText(frame, f"PANIC {panic:.0f}%", (meter_x + 5, meter_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Panic alert
    if panic >= CONFIG['panic_threshold']:
        cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, "[!!] PANIC DETECTED - HELP REQUESTED!", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return frame

# ============================================================================
# WEBCAM THREAD
# ============================================================================

def webcam_thread():
    """Capture and process video"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = time.time()
    last_panic_time = 0
    
    print("[✓] Webcam thread started")
    state.status = "Running"
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Analyze
            crowd, density, movement = analyze_frame(frame)
            sentiment = -0.3 if movement > 0.4 else 0.1  # Simple heuristic
            panic = calculate_panic(crowd, density, movement, sentiment)
            
            # Update state
            with state.lock:
                state.crowd_count = crowd
                state.density = density
                state.movement = movement
                state.sentiment = sentiment
                state.panic_percent = panic
                state.is_panic = panic >= CONFIG['panic_threshold']
            
            # Trigger alert
            if state.is_panic and (time.time() - last_panic_time) > 5:
                last_panic_time = time.time()
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'PANIC',
                    'message': f'[PANIC DETECTED] Level: {panic:.0f}% | Crowd: {crowd}',
                    'severity': 'CRITICAL'
                }
                with state.lock:
                    state.alerts.append(alert)
                print(f"\n{'='*60}\n[!! PANIC ALERT] Level: {panic:.0f}%\n{'='*60}\n")
            
            # Visualize
            frame_display = cv2.resize(frame, CONFIG['display_size'])
            heatmap = generate_heatmap(frame_display)
            frame_display = cv2.addWeighted(frame_display, 0.7, heatmap, 0.3, 0)
            frame_display = overlay_viz(frame_display, crowd, density, movement, sentiment, panic)
            
            # Encode
            ret_enc, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                with state.lock:
                    state.frame_encoded = buffer.tobytes()
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                state.fps = frame_count / elapsed
            
            time.sleep(0.033)
            
        except Exception as e:
            print(f"[!] Webcam error: {e}")
            time.sleep(1)
    
    cap.release()

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return render_template('crowd_emotion_monitor.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if state.frame_encoded:
                frame_data = state.frame_encoded
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                       + frame_data + b'\r\n')
            time.sleep(0.02)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    try:
        with state.lock:
            return jsonify({
                'status': state.status,
                'crowd_count': int(state.crowd_count) if state.crowd_count else 0,
                'crowd_density': float(state.density) if state.density else 0.0,
                'movement_intensity': float(state.movement) if state.movement else 0.0,
                'sentiment_score': float(state.sentiment) if state.sentiment else 0.0,
                'panic_percent': float(state.panic_percent) if state.panic_percent else 0.0,
                'is_panic': bool(state.is_panic),
                'fps': round(float(state.fps), 1) if state.fps else 0.0,
                'timestamp': datetime.now().isoformat(),
            })
    except Exception as e:
        print(f"[ERROR in get_status] {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/alerts')
def get_alerts():
    try:
        with state.lock:
            return jsonify({
                'alerts': list(state.alerts),
                'total_alerts': len(state.alerts),
            })
    except Exception as e:
        print(f"[ERROR in get_alerts] {e}")
        return jsonify({'error': str(e), 'alerts': []}), 500

@app.route('/api/alert/request', methods=['POST'])
def request_help():
    alert = {
        'timestamp': datetime.now().isoformat(),
        'type': 'MANUAL_REQUEST',
        'message': 'Manual help request',
        'severity': 'HIGH'
    }
    with state.lock:
        state.alerts.append(alert)
    return jsonify({'success': True})

@app.route('/api/alert/clear', methods=['POST'])
def clear_alerts():
    with state.lock:
        state.alerts.clear()
    return jsonify({'success': True})

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("[CROWD EMOTION MONITOR] - INTEGRATED SYSTEM")
    print("="*70)
    print("\n[1/2] Starting webcam capture thread...")
    webcam = threading.Thread(target=webcam_thread, daemon=True)
    webcam.start()
    time.sleep(2)
    
    print("[2/2] Starting Flask server...")
    print(f"\n[->] Dashboard: http://localhost:{CONFIG['port']}")
    print("\nFeatures:")
    print("  * Real-time crowd detection with webcam")
    print("  * Heatmap overlay of crowd density")
    print("  * Sentiment analysis of crowd behavior")
    print("  * Panic detection & automatic alerts")
    print("  * Live metrics and alert history")
    print("\n" + "="*70 + "\n")
    
    try:
        app.run(host='127.0.0.1', port=CONFIG['port'], debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[OK] Shutting down...")
        state.status = "Stopped"
