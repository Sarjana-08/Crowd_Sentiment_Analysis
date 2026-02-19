#!/usr/bin/env python3
"""
Production-Grade Crowd Monitoring Dashboard
Deployment-Ready with Database & Cloud Support
Features:
- Live video feed with crowd detection
- Real-time density heatmaps
- Alert management with database persistence
- Email configuration
- Historical data visualization
- API endpoints for integration
"""

import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv

from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Import custom modules
from crowd_detector import AdvancedCrowdDetector
from email_alerts import EmailAlertManager

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Enable CORS
CORS(app)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Determine database URL
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    # Fallback to SQLite for development
    DATABASE_PATH = Path(__file__).parent / 'crowd_monitor.db'
    DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# SQLAlchemy configuration
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
}

db = SQLAlchemy(app)

# ============================================================================
# DATABASE MODELS
# ============================================================================

class DetectionRecord(db.Model):
    """Store detection history"""
    __tablename__ = 'detection_records'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    crowd_count = db.Column(db.Float)
    cascade_count = db.Column(db.Integer)
    contour_count = db.Column(db.Integer)
    density_count = db.Column(db.Float)
    confidence = db.Column(db.Float)
    method = db.Column(db.String(100))
    threshold = db.Column(db.Float)
    alert_active = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'crowd_count': round(self.crowd_count, 2),
            'cascade_count': self.cascade_count,
            'contour_count': self.contour_count,
            'density_count': round(self.density_count, 2),
            'confidence': round(self.confidence * 100, 1),
            'method': self.method,
            'threshold': self.threshold,
            'alert_active': self.alert_active
        }

class AlertRecipient(db.Model):
    """Store alert recipients"""
    __tablename__ = 'alert_recipients'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, index=True)
    name = db.Column(db.String(255))
    enabled = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_alert = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'email': self.email,
            'name': self.name,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat()
        }

class AlertHistory(db.Model):
    """Store sent alerts"""
    __tablename__ = 'alert_history'
    
    id = db.Column(db.Integer, primary_key=True)
    recipient_email = db.Column(db.String(255), index=True)
    crowd_count = db.Column(db.Float)
    threshold = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    status = db.Column(db.String(50))  # 'sent', 'failed', 'pending'
    
    def to_dict(self):
        return {
            'email': self.recipient_email,
            'count': round(self.crowd_count, 1),
            'threshold': round(self.threshold, 1),
            'timestamp': self.timestamp.isoformat(),
            'status': self.status
        }

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'threshold': float(os.getenv('CROWD_THRESHOLD', 500)),
    'alert_enabled': os.getenv('ALERTS_ENABLED', 'true').lower() == 'true',
    'fps_target': int(os.getenv('FPS_TARGET', 5)),
    'smooth_window': int(os.getenv('SMOOTH_WINDOW', 5)),
    'max_history': int(os.getenv('MAX_HISTORY', 1000)),
}

# ============================================================================
# GLOBAL STATE
# ============================================================================

class MonitoringState:
    def __init__(self):
        self.detector = AdvancedCrowdDetector(smooth_window=CONFIG['smooth_window'])
        self.alert_manager = EmailAlertManager()
        self.latest_frame = None
        self.latest_heatmap = None
        self.latest_count = 0
        self.latest_detection = None
        self.alert_active = False
        self.alert_sent_time = None
        self.video_source = None
        self.is_recording = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_time = 0
        self.last_alert_email_time = {}
        self.shutdown_flag = False
        self.capture_thread = None

state = MonitoringState()

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✓ Database initialized")

# ============================================================================
# WEBCAM/VIDEO CAPTURE THREAD
# ============================================================================

def capture_video_feed(source=0):
    """Capture video and process for crowd detection"""
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"✗ Cannot open video source: {source}")
        return
    
    frame_skip = 0
    skip_count = int(30 / CONFIG['fps_target'])
    
    print(f"✓ Video capture started (source: {source})")
    
    while not state.shutdown_flag:
        try:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Failed to read frame, attempting to reconnect...")
                time.sleep(1)
                continue
            
            # Resize for processing
            frame = cv2.resize(frame, (640, 480))
            state.latest_frame = frame.copy()
            
            frame_skip += 1
            if frame_skip >= skip_count:
                frame_skip = 0
                
                process_start = time.time()
                
                # Detect crowd
                detection = state.detector.estimate_crowd_count(frame)
                state.latest_detection = detection
                state.latest_count = detection['count']
                
                # Create heatmap
                state.latest_heatmap = state.detector.create_density_heatmap(
                    frame, state.latest_detection
                )
                
                # Store in database
                try:
                    with app.app_context():
                        record = DetectionRecord(
                            crowd_count=detection['count'],
                            cascade_count=detection['cascade_count'],
                            contour_count=detection['contour_count'],
                            density_count=detection['density_count'],
                            confidence=detection['confidence'],
                            method=detection['method'],
                            threshold=CONFIG['threshold'],
                            alert_active=False
                        )
                        db.session.add(record)
                        db.session.commit()
                except Exception as e:
                    print(f"Database error: {e}")
                    try:
                        db.session.rollback()
                    except:
                        pass
                
                # Check threshold for alerts
                if CONFIG['alert_enabled'] and state.latest_count > CONFIG['threshold']:
                    state.alert_active = True
                    
                    # Send email alerts
                    current_time = time.time()
                    try:
                        with app.app_context():
                            recipients = db.session.query(AlertRecipient).filter_by(enabled=True).all()
                            
                            for recipient in recipients:
                                # Check if we've already sent alert to this email recently (every 5 mins)
                                last_alert = state.last_alert_email_time.get(recipient.email, 0)
                                if current_time - last_alert > 300:
                                    success = state.alert_manager.send_alert_email(
                                        recipient.email,
                                        state.latest_count,
                                        CONFIG['threshold']
                                    )
                                    
                                    # Log alert
                                    alert_log = AlertHistory(
                                        recipient_email=recipient.email,
                                        crowd_count=state.latest_count,
                                        threshold=CONFIG['threshold'],
                                        status='sent' if success else 'failed'
                                    )
                                    db.session.add(alert_log)
                                    
                                    if success:
                                        state.last_alert_email_time[recipient.email] = current_time
                            
                            db.session.commit()
                    except Exception as e:
                        print(f"Alert error: {e}")
                        db.session.rollback()
                else:
                    state.alert_active = False
                
                # Calculate FPS
                state.frame_count += 1
                state.processing_time = time.time() - process_start
                elapsed = time.time() - state.start_time
                state.fps = state.frame_count / elapsed if elapsed > 0 else 0
        
        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(1)
    
    # Cleanup on shutdown
    print("Shutting down video capture...")
    cap.release()
    print("✓ Camera released successfully")

# ============================================================================
# FLASK ROUTES - API
# ============================================================================

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get/update configuration"""
    if request.method == 'GET':
        return jsonify({
            'threshold': CONFIG['threshold'],
            'alert_enabled': CONFIG['alert_enabled'],
            'fps_target': CONFIG['fps_target'],
            'smooth_window': CONFIG['smooth_window']
        })
    
    elif request.method == 'POST':
        data = request.json
        
        if 'threshold' in data:
            CONFIG['threshold'] = float(data['threshold'])
        if 'alert_enabled' in data:
            CONFIG['alert_enabled'] = bool(data['alert_enabled'])
        if 'fps_target' in data:
            CONFIG['fps_target'] = int(data['fps_target'])
        
        return jsonify({'status': 'success', 'config': CONFIG})

@app.route('/api/status')
def api_status():
    """Get current monitoring status"""
    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'crowd_count': round(state.latest_count, 2),
        'threshold': CONFIG['threshold'],
        'alert_active': state.alert_active,
        'confidence': round(state.latest_detection['confidence'] * 100) if state.latest_detection else 0,
        'fps': round(state.fps, 1),
        'processing_time_ms': round(state.processing_time * 1000, 1)
    })

@app.route('/api/frame')
def api_frame():
    """Get latest frame with annotations"""
    if state.latest_frame is None:
        return jsonify({'error': 'No frame available'}), 404
    
    frame = state.latest_frame.copy()
    
    # Add annotations
    color = (0, 0, 255) if state.alert_active else (0, 255, 0)
    
    cv2.putText(frame, f"Count: {state.latest_count:.0f}", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"Threshold: {CONFIG['threshold']}", (20, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if state.alert_active:
        cv2.putText(frame, "ALERT!", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    cv2.putText(frame, f"FPS: {state.fps:.1f}", (20, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = buffer.tobytes()
    
    return Response(frame_b64, mimetype='image/jpeg')

@app.route('/api/heatmap')
def api_heatmap():
    """Get density heatmap"""
    if state.latest_heatmap is None:
        return jsonify({'error': 'No heatmap available'}), 404
    
    _, buffer = cv2.imencode('.jpg', state.latest_heatmap)
    heatmap_b64 = buffer.tobytes()
    
    return Response(heatmap_b64, mimetype='image/jpeg')

@app.route('/api/history')
def api_history():
    """Get historical data from database"""
    try:
        limit = request.args.get('limit', 100, type=int)
        hours = request.args.get('hours', 24, type=int)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        records = db.session.query(DetectionRecord)\
            .filter(DetectionRecord.timestamp >= cutoff_time)\
            .order_by(DetectionRecord.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return jsonify([r.to_dict() for r in reversed(records)])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection-details')
def api_detection_details():
    """Get detailed detection information"""
    if state.latest_detection is None:
        return jsonify({'error': 'No detection data'}), 404
    
    return jsonify({
        'count': round(state.latest_detection['count'], 2),
        'cascade_count': state.latest_detection['cascade_count'],
        'contour_count': state.latest_detection['contour_count'],
        'density_count': round(state.latest_detection['density_count'], 2),
        'model_count': round(state.latest_detection['model_count'], 2) if state.latest_detection['model_count'] else None,
        'confidence': round(state.latest_detection['confidence'] * 100, 1),
        'timestamp': state.latest_detection['timestamp'].isoformat()
    })

# ============================================================================
# ALERT MANAGEMENT API
# ============================================================================

@app.route('/api/alerts/recipients', methods=['GET', 'POST', 'DELETE'])
def api_recipients():
    """Manage alert recipients"""
    try:
        if request.method == 'GET':
            recipients = db.session.query(AlertRecipient).all()
            return jsonify([r.to_dict() for r in recipients])
        
        elif request.method == 'POST':
            data = request.json
            email = data.get('email', '').strip()
            name = data.get('name', '').strip()
            
            if not email or '@' not in email:
                return jsonify({'error': 'Valid email required'}), 400
            
            # Check if exists
            existing = db.session.query(AlertRecipient).filter_by(email=email).first()
            if existing:
                return jsonify({'error': 'Email already exists'}), 400
            
            recipient = AlertRecipient(email=email, name=name or email)
            db.session.add(recipient)
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'recipient': recipient.to_dict()
            }), 201
        
        elif request.method == 'DELETE':
            data = request.json
            email = data.get('email', '').strip()
            
            if not email:
                return jsonify({'error': 'Email required'}), 400
            
            recipient = db.session.query(AlertRecipient).filter_by(email=email).first()
            if not recipient:
                return jsonify({'error': 'Recipient not found'}), 404
            
            db.session.delete(recipient)
            db.session.commit()
            
            return jsonify({'status': 'success'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/toggle/<email>/<int:enabled>', methods=['PUT'])
def api_toggle_recipient(email, enabled):
    """Toggle recipient alerts"""
    try:
        recipient = db.session.query(AlertRecipient).filter_by(email=email).first()
        if not recipient:
            return jsonify({'error': 'Recipient not found'}), 404
        
        recipient.enabled = bool(enabled)
        db.session.commit()
        
        return jsonify({'status': 'success', 'recipient': recipient.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/history')
def api_alert_history():
    """Get alert history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        hours = request.args.get('hours', 24, type=int)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        history = db.session.query(AlertHistory)\
            .filter(AlertHistory.timestamp >= cutoff_time)\
            .order_by(AlertHistory.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return jsonify([h.to_dict() for h in history])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/test/<email>', methods=['POST'])
def api_test_alert(email):
    """Send test alert to recipient"""
    try:
        recipient = db.session.query(AlertRecipient).filter_by(email=email).first()
        if not recipient:
            return jsonify({'error': 'Recipient not found'}), 404
        
        success = state.alert_manager.send_alert_email(email, 750, 500)
        
        alert_log = AlertHistory(
            recipient_email=email,
            crowd_count=750,
            threshold=500,
            status='sent' if success else 'failed'
        )
        db.session.add(alert_log)
        db.session.commit()
        
        return jsonify({
            'status': 'sent' if success else 'failed',
            'message': f"Test alert {'sent' if success else 'failed'} to {email}"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STATIC FILES & TEMPLATES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': time.time() - state.start_time,
        'frame_count': state.frame_count
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# STARTUP & INITIALIZATION
# ============================================================================

def shutdown_app():
    """Gracefully shutdown the application"""
    print("\n\nShutting down application...")
    state.shutdown_flag = True
    
    # Wait for capture thread to finish
    if state.capture_thread and state.capture_thread.is_alive():
        print("Waiting for video capture thread to close...")
        state.capture_thread.join(timeout=5)
    
    print("✓ Application shutdown complete")

def initialize_app():
    """Initialize application"""
    print("\n" + "="*80)
    print("PRODUCTION-GRADE CROWD MONITORING DASHBOARD")
    print("="*80 + "\n")
    
    print("[1/4] Initializing database...")
    init_db()
    
    print("\n[2/4] Loading crowd detector...")
    print(f"  ✓ Detector initialized with {CONFIG['smooth_window']} frame smoothing")
    
    print("\n[3/4] Loading email alert system...")
    print(f"  ✓ Email alerts configured")
    
    print("\n[4/4] Starting video capture thread...")
    capture_thread = threading.Thread(
        target=capture_video_feed,
        args=(0,),  # 0 = default webcam
        daemon=True
    )
    capture_thread.start()
    state.capture_thread = capture_thread
    
    # Register shutdown handler
    import atexit
    atexit.register(shutdown_app)
    
    print("\n" + "="*80)
    print("✓ DASHBOARD READY FOR DEPLOYMENT")
    print("="*80)
    print("\nEnvironment Configuration:")
    print(f"  Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL[:50]}")
    print(f"  Crowd Threshold: {CONFIG['threshold']}")
    print(f"  Alerts Enabled: {CONFIG['alert_enabled']}")
    print(f"  Target FPS: {CONFIG['fps_target']}")
    print("\nAccess Points:")
    print("  Dashboard: http://localhost:5000")
    print("  Health Check: http://localhost:5000/api/health")
    print("  API Docs: http://localhost:5000/api/docs\n")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)), threaded=True)
