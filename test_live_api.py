#!/usr/bin/env python3
"""Test sentiment API endpoints against running dashboard"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

print("="*70)
print("🧠 SENTIMENT ANALYSIS API TESTS - LIVE DASHBOARD")
print("="*70)

# Test 1: Check sentiment status
print("\n[1] Sentiment System Status")
print("-" * 70)
try:
    r = requests.get(f"{BASE_URL}/api/sentiment/status", timeout=5)
    data = r.json()
    print(f"✅ Status: {r.status_code}")
    print(f"Sentiment Analysis Enabled: {data['sentiment_analysis_enabled']}")
    print(f"VADER Available: {data['vader_available']}")
    print(f"TextBlob Available: {data['textblob_available']}")
    print(f"Message: {data['message']}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Analyze negative sentiment
print("\n[2] Analyze Negative Sentiment")
print("-" * 70)
try:
    payload = {"text": "Dangerous crowd with panic and chaos!"}
    r = requests.post(f"{BASE_URL}/api/sentiment/analyze", json=payload, timeout=5)
    data = r.json()
    print(f"✅ Status: {r.status_code}")
    print(f"Text: {data['text']}")
    print(f"Sentiment: {data['sentiment_score']:.3f} ({data['sentiment_label']})")
    print(f"Confidence: {data['confidence']:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Analyze positive sentiment
print("\n[3] Analyze Positive Sentiment")
print("-" * 70)
try:
    payload = {"text": "Happy celebration with excited joyful crowd!"}
    r = requests.post(f"{BASE_URL}/api/sentiment/analyze", json=payload, timeout=5)
    data = r.json()
    print(f"✅ Status: {r.status_code}")
    print(f"Text: {data['text']}")
    print(f"Sentiment: {data['sentiment_score']:.3f} ({data['sentiment_label']})")
    print(f"Confidence: {data['confidence']:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Calculate critical priority
print("\n[4] Calculate Alert Priority (Dangerous)")
print("-" * 70)
try:
    payload = {
        "count": 850,
        "threshold": 500,
        "context": "Reports of panic and crowd pressing barriers"
    }
    r = requests.post(f"{BASE_URL}/api/alert/priority", json=payload, timeout=5)
    data = r.json()
    print(f"✅ Status: {r.status_code}")
    print(f"Count: {data['crowd_count']} / Threshold: {data['threshold']}")
    print(f"Priority: {data['priority']:.1%} → {data['priority_level']}")
    print(f"Should Alert: {data['should_alert']}")
    print(f"Sentiment: {data['sentiment']:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Calculate moderate priority
print("\n[5] Calculate Alert Priority (Happy Event)")
print("-" * 70)
try:
    payload = {
        "count": 2000,
        "threshold": 500,
        "context": "Concert with excited happy crowd enjoying music"
    }
    r = requests.post(f"{BASE_URL}/api/alert/priority", json=payload, timeout=5)
    data = r.json()
    print(f"✅ Status: {r.status_code}")
    print(f"Count: {data['crowd_count']} / Threshold: {data['threshold']}")
    print(f"Priority: {data['priority']:.1%} → {data['priority_level']}")
    print(f"Should Alert: {data['should_alert']}")
    print(f"Sentiment: {data['sentiment']:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 6: Generate alert message
print("\n[6] Generate Alert Message (Critical)")
print("-" * 70)
try:
    payload = {
        "count": 850,
        "threshold": 500,
        "context": "Crowd aggressive, panic spreading"
    }
    r = requests.post(f"{BASE_URL}/api/alert/message", json=payload, timeout=5)
    data = r.json()
    print(f"✅ Status: {r.status_code}")
    print("\nGenerated Alert Message:")
    print(data['message'])
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("✅ ALL SENTIMENT API TESTS COMPLETED!")
print("="*70)
