#!/usr/bin/env python3
"""
Simple Sentiment Dashboard Server - Direct Access
No dependencies on main dashboard_app.py
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os

# Sentiment Analysis
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
    sia = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    sia = None

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Create Flask app
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
CORS(app)

# Config
CONFIG = {
    'sentiment_weight': 0.3,
    'min_sentiment_threshold': -0.5,
}

# ============================================================================
# SENTIMENT FUNCTIONS
# ============================================================================

def analyze_sentiment(text):
    """Analyze sentiment using VADER or TextBlob"""
    if not text:
        return 0.0, 0.5
    
    if VADER_AVAILABLE and sia:
        try:
            scores = sia.polarity_scores(text)
            return scores['compound'], max(scores['pos'], scores['neu'], scores['neg'])
        except:
            pass
    
    if TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            pass
    
    return 0.0, 0.5


def calculate_alert_priority(crowd_count, threshold, context_text=""):
    """Calculate priority combining crowd and sentiment"""
    severity_ratio = min(crowd_count / max(threshold, 1), 2.0)
    severity_score = min(severity_ratio, 1.0)
    
    sentiment_score, _ = analyze_sentiment(context_text)
    sentiment_weight = CONFIG['sentiment_weight']
    sentiment_boost = max(0, sentiment_weight * (1 - sentiment_score) / 2)
    
    priority = severity_score + sentiment_boost
    return min(priority, 1.0), sentiment_score, severity_score


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve sentiment dashboard"""
    return render_template('sentiment_dashboard.html')


@app.route('/api/sentiment/status', methods=['GET'])
def sentiment_status():
    """Check system status"""
    return jsonify({
        'sentiment_analysis_enabled': True,
        'vader_available': VADER_AVAILABLE,
        'textblob_available': TEXTBLOB_AVAILABLE,
        'message': 'Sentiment analysis active'
    })


@app.route('/api/sentiment/analyze', methods=['POST'])
def api_analyze_sentiment():
    """Analyze sentiment"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        score, confidence = analyze_sentiment(text)
        label = 'Negative' if score < -0.3 else ('Positive' if score > 0.3 else 'Neutral')
        
        return jsonify({
            'text': text,
            'sentiment_score': round(score, 3),
            'confidence': round(confidence, 3),
            'sentiment_label': label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alert/priority', methods=['POST'])
def api_calculate_priority():
    """Calculate priority"""
    try:
        data = request.json
        count = data.get('count', 0)
        threshold = data.get('threshold', 500)
        context = data.get('context', '')
        
        priority, sentiment, severity = calculate_alert_priority(count, threshold, context)
        level = 'CRITICAL' if priority > 0.8 else ('HIGH' if priority > 0.6 else 'MODERATE')
        
        return jsonify({
            'crowd_count': count,
            'threshold': threshold,
            'priority': round(priority, 3),
            'sentiment': round(sentiment, 3),
            'severity': round(severity, 3),
            'priority_level': level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alert/message', methods=['POST'])
def api_generate_message():
    """Generate alert message"""
    try:
        data = request.json
        count = data.get('count', 0)
        threshold = data.get('threshold', 500)
        context = data.get('context', '')
        
        priority, sentiment, severity = calculate_alert_priority(count, threshold, context)
        sentiment_label = 'Negative' if sentiment < -0.3 else ('Positive' if sentiment > 0.3 else 'Neutral')
        
        msg = f"ALERT: Crowd {count}\nThreshold: {threshold}\nPriority: {priority:.0%}\nSentiment: {sentiment_label}"
        
        return jsonify({'message': msg})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS DASHBOARD - DIRECT SERVER")
    print("="*70)
    print("\n🌐 Access at: http://localhost:5000")
    print("\n📊 Features:")
    print("   • Sentiment Analyzer")
    print("   • Priority Calculator")
    print("   • Real-Time Feed")
    print("   • API Tester")
    print("\n🔌 API Endpoints:")
    print("   GET  /api/sentiment/status")
    print("   POST /api/sentiment/analyze")
    print("   POST /api/alert/priority")
    print("   POST /api/alert/message")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=False, host='localhost', port=5000, threaded=True)
