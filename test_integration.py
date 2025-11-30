#!/usr/bin/env python3
"""
Quick integration test for BLUEmed API server.
Tests that the API server can properly handle requests.
"""

import requests
import json
import sys

API_URL = "http://localhost:8000"
TEST_NOTE = """54-year-old woman with a painful, rapidly growing leg lesion for 1 month.
History includes Crohn's disease, diabetes, hypertension, and previous anterior uveitis.
Examination revealed a 4-cm tender ulcerative lesion with necrotic base and purplish borders,
along with pitting edema and dilated veins. Diagnosed as a venous ulcer."""


def test_health():
    """Test health endpoint."""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"✅ Expert Model: {data['models']['expert']}")
        print(f"✅ Judge Model: {data['models']['judge']}")
        print(f"✅ Retriever: {data['retriever_enabled']}")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Failed: Could not connect to API server")
        print("   Is the server running? Try: python3 api_server.py")
        return False
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return False


def test_analyze():
    """Test analyze endpoint."""
    print("\n" + "="*70)
    print("TEST 2: Analyze Medical Note")
    print("="*70)
    
    try:
        payload = {
            "medical_note": TEST_NOTE,
            "max_rounds": 2
        }
        
        print("📤 Sending request...")
        print(f"   Note length: {len(TEST_NOTE)} characters")
        
        response = requests.post(
            f"{API_URL}/analyze",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120  # Give it 2 minutes
        )
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✅ Request ID: {data['request_id']}")
        print(f"✅ Expert A arguments: {len(data['expertA_arguments'])}")
        print(f"✅ Expert B arguments: {len(data['expertB_arguments'])}")
        print(f"✅ Final Answer: {data['judge_decision']['final_answer']}")
        
        if data['judge_decision'].get('confidence_score'):
            print(f"✅ Confidence: {data['judge_decision']['confidence_score']}/10")
        
        print(f"\n📋 Judge Reasoning (first 200 chars):")
        print(f"   {data['judge_decision']['reasoning'][:200]}...")
        
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Failed: Request timed out (>120s)")
        print("   The analysis is taking too long. Check API server logs.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ Failed: HTTP {e.response.status_code}")
        print(f"   {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("BLUEMED API INTEGRATION TEST")
    print("="*70)
    print(f"\nAPI URL: {API_URL}")
    print(f"Test Note: {len(TEST_NOTE)} characters")
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed. Skipping further tests.")
        sys.exit(1)
    
    # Test 2: Analyze
    if not test_analyze():
        print("\n❌ Analysis test failed.")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe API server is working correctly.")
    print("You can now start the UI with:")
    print("  cd ui/judge_ui")
    print("  streamlit run app.py")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
