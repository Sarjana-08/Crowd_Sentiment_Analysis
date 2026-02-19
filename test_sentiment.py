#!/usr/bin/env python3
"""Test sentiment analysis endpoints"""

import requests
import json
import time

# Wait for dashboard to be ready
time.sleep(2)

BASE_URL = "http://localhost:5000"

print("="*70)
print("Testing Sentiment Analysis Integration")
print("="*70)

# Test 1: Check sentiment status
print("\n[1] Checking Sentiment System Status...")
try:
    response = requests.get(f"{BASE_URL}/api/sentiment/status")
    status = response.json()
    print(f"✅ Status: {response.status_code}")
    print(json.dumps(status, indent=2))
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Analyze negative sentiment
print("\n[2] Analyzing Negative Sentiment (Dangerous Situation)...")
try:
    payload = {"text": "Dangerous crowd with panic and chaos reported!"}
    response = requests.post(f"{BASE_URL}/api/sentiment/analyze", json=payload)
    result = response.json()
    print(f"✅ Status: {response.status_code}")
    print(f"Text: {result['text']}")
    print(f"Sentiment Score: {result['sentiment_score']:.3f} (negative)")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Label: {result['sentiment_label']}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Analyze positive sentiment
print("\n[3] Analyzing Positive Sentiment (Happy Celebration)...")
try:
    payload = {"text": "Happy celebration with excited and joyful crowd!"}
    response = requests.post(f"{BASE_URL}/api/sentiment/analyze", json=payload)
    result = response.json()
    print(f"✅ Status: {response.status_code}")
    print(f"Text: {result['text']}")
    print(f"Sentiment Score: {result['sentiment_score']:.3f} (positive)")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Label: {result['sentiment_label']}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Calculate alert priority (high danger)
print("\n[4] Calculating Alert Priority (Dangerous Scenario)...")
try:
    payload = {
        "count": 850,
        "threshold": 500,
        "context": "Reports of panic, crowd pushing barriers"
    }
    response = requests.post(f"{BASE_URL}/api/alert/priority", json=payload)
    result = response.json()
    print(f"✅ Status: {response.status_code}")
    print(f"Crowd Count: {result['crowd_count']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Priority Score: {result['priority']:.1%}")
    print(f"Priority Level: {result['priority_level']}")
    print(f"Should Alert: {result['should_alert']}")
    print(f"Sentiment: {result['sentiment']:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Calculate alert priority (happy event)
print("\n[5] Calculating Alert Priority (Happy Event)...")
try:
    payload = {
        "count": 2000,
        "threshold": 500,
        "context": "Concert with excited happy crowd enjoying music"
    }
    response = requests.post(f"{BASE_URL}/api/alert/priority", json=payload)
    result = response.json()
    print(f"✅ Status: {response.status_code}")
    print(f"Crowd Count: {result['crowd_count']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Priority Score: {result['priority']:.1%}")
    print(f"Priority Level: {result['priority_level']}")
    print(f"Should Alert: {result['should_alert']}")
    print(f"Sentiment: {result['sentiment']:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 6: Generate alert message
print("\n[6] Generating Alert Message...")
try:
    payload = {
        "count": 850,
        "threshold": 500,
        "context": "Crowd aggressive, panic spreading"
    }
    response = requests.post(f"{BASE_URL}/api/alert/message", json=payload)
    result = response.json()
    print(f"✅ Status: {response.status_code}")
    print("\nGenerated Message:")
    print("-" * 70)
    print(result['message'])
    print("-" * 70)
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("✅ All Sentiment Analysis Tests Complete!")
print("="*70)
