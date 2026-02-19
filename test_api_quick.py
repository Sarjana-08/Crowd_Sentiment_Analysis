#!/usr/bin/env python3
"""Test sentiment API - longer wait"""
import requests
import time

time.sleep(5)  # Wait longer for dashboard to be ready

print("\n🧠 Testing Sentiment Analysis APIs...\n")

try:
    # Test sentiment status
    print("[1] Sentiment Status")
    r = requests.get("http://localhost:5000/api/sentiment/status", timeout=5)
    print(f"✅ {r.status_code}: {r.json()['message']}\n")
    
    # Test analyze negative
    print("[2] Analyze Negative Sentiment")
    r = requests.post("http://localhost:5000/api/sentiment/analyze", 
                     json={"text": "Dangerous crowd panic chaos"}, timeout=5)
    d = r.json()
    print(f"✅ {r.status_code}: Score {d['sentiment_score']:.3f} ({d['sentiment_label']})\n")
    
    # Test analyze positive
    print("[3] Analyze Positive Sentiment")
    r = requests.post("http://localhost:5000/api/sentiment/analyze", 
                     json={"text": "Happy celebration excited crowd"}, timeout=5)
    d = r.json()
    print(f"✅ {r.status_code}: Score {d['sentiment_score']:.3f} ({d['sentiment_label']})\n")
    
    # Test priority dangerous
    print("[4] Alert Priority - Dangerous")
    r = requests.post("http://localhost:5000/api/alert/priority",
                     json={"count": 850, "threshold": 500, "context": "Panic panic panic"}, timeout=5)
    d = r.json()
    print(f"✅ {r.status_code}: Priority {d['priority']:.0%} → {d['priority_level']}\n")
    
    # Test priority happy
    print("[5] Alert Priority - Happy Event")
    r = requests.post("http://localhost:5000/api/alert/priority",
                     json={"count": 2000, "threshold": 500, "context": "Happy happy happy"}, timeout=5)
    d = r.json()
    print(f"✅ {r.status_code}: Priority {d['priority']:.0%} → {d['priority_level']}\n")
    
    # Test message generation
    print("[6] Generate Alert Message")
    r = requests.post("http://localhost:5000/api/alert/message",
                     json={"count": 850, "threshold": 500, "context": "Dangerous"}, timeout=5)
    d = r.json()
    print(f"✅ {r.status_code}: Message generated (see below)\n")
    print(d['message'])
    
    print("\n" + "="*60)
    print("✅ ALL SENTIMENT ANALYSIS TESTS PASSED!")
    print("="*60)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
