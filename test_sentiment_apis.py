#!/usr/bin/env python3
"""Test all Sentiment Analysis API endpoints"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_sentiment_apis():
    """Test all sentiment analysis endpoints"""
    
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS API TEST SUITE")
    print("="*70)
    
    time.sleep(2)  # Give server time to fully start
    
    # Test 1: Status endpoint
    print("\n[TEST 1] System Status Check")
    print("-" * 70)
    try:
        resp = requests.get(f"{BASE_URL}/api/sentiment/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Status endpoint working")
            print(f"   VADER available: {data['vader_available']}")
            print(f"   TextBlob available: {data['textblob_available']}")
        else:
            print(f"❌ Status failed: {resp.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Negative sentiment (dangerous crowd)
    print("\n[TEST 2] Negative Sentiment Analysis")
    print("-" * 70)
    test_cases = [
        ("Dangerous crowd panic chaos stampede rioting", "DANGEROUS SCENARIO"),
        ("Crowd getting violent aggressive hostile", "HOSTILE SCENARIO"),
        ("Emergency evacuation dangerous situation", "EMERGENCY SCENARIO"),
    ]
    
    for text, label in test_cases:
        try:
            resp = requests.post(
                f"{BASE_URL}/api/sentiment/analyze",
                json={"text": text},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                score = data['sentiment_score']
                label_out = data['sentiment_label']
                color = "🔴" if score < -0.5 else "🟠"
                print(f"{color} {label}: Score={score:.3f} ({label_out})")
            else:
                print(f"❌ Error: {resp.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 3: Positive sentiment (happy crowd)
    print("\n[TEST 3] Positive Sentiment Analysis")
    print("-" * 70)
    test_cases = [
        ("Happy celebration excited crowd cheering joy", "HAPPY SCENARIO"),
        ("Festive gathering enjoying celebration party", "FESTIVE SCENARIO"),
        ("Peaceful assembly friendly community gathering", "PEACEFUL SCENARIO"),
    ]
    
    for text, label in test_cases:
        try:
            resp = requests.post(
                f"{BASE_URL}/api/sentiment/analyze",
                json={"text": text},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                score = data['sentiment_score']
                label_out = data['sentiment_label']
                color = "🟢" if score > 0.5 else "🟡"
                print(f"{color} {label}: Score={score:.3f} ({label_out})")
            else:
                print(f"❌ Error: {resp.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 4: Neutral sentiment
    print("\n[TEST 4] Neutral Sentiment Analysis")
    print("-" * 70)
    try:
        resp = requests.post(
            f"{BASE_URL}/api/sentiment/analyze",
            json={"text": "Crowd moving at entrance"},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            score = data['sentiment_score']
            label_out = data['sentiment_label']
            print(f"🟡 NEUTRAL: Score={score:.3f} ({label_out})")
        else:
            print(f"❌ Error: {resp.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Priority calculation
    print("\n[TEST 5] Alert Priority Calculation")
    print("-" * 70)
    
    priority_tests = [
        (850, 500, "Dangerous crowd panic", "HIGH COUNT + NEGATIVE SENTIMENT"),
        (2000, 500, "Happy celebration crowd", "VERY HIGH COUNT + POSITIVE SENTIMENT"),
        (350, 500, "Crowd moving normally", "LOW COUNT + NEUTRAL"),
    ]
    
    for count, threshold, context, description in priority_tests:
        try:
            resp = requests.post(
                f"{BASE_URL}/api/alert/priority",
                json={"count": count, "threshold": threshold, "context": context},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                priority = data['priority']
                level = data['priority_level']
                sentiment = data['sentiment']
                
                priority_badge = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MODERATE': '🟡'
                }.get(level, '⚪')
                
                print(f"{priority_badge} {description}")
                print(f"   Count: {count}, Priority: {priority:.1%}, Level: {level}")
                print(f"   Sentiment: {sentiment:.3f}")
            else:
                print(f"❌ Error: {resp.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 6: Message generation
    print("\n[TEST 6] Alert Message Generation")
    print("-" * 70)
    try:
        resp = requests.post(
            f"{BASE_URL}/api/alert/message",
            json={
                "count": 850,
                "threshold": 500,
                "context": "Dangerous crowd panic"
            },
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            print("Generated Alert Message:")
            print(data['message'])
        else:
            print(f"❌ Error: {resp.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_sentiment_apis()
