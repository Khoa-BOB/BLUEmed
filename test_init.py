#!/usr/bin/env python3
"""
Simple test to verify the API server can start.
This doesn't run a full analysis, just checks initialization.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Checking Imports")
    print("="*70)
    
    try:
        print("Importing config.settings...", end=" ")
        from config.settings import settings
        print("✅")
        
        print("Importing app.graph.graph...", end=" ")
        from app.graph.graph import build_graph
        print("✅")
        
        print("Importing FastAPI...", end=" ")
        from fastapi import FastAPI
        print("✅")
        
        print("Importing langchain_core...", end=" ")
        from langchain_core.messages import HumanMessage
        print("✅")
        
        return True
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


def test_api_models():
    """Test that API models are valid."""
    print("\n" + "="*70)
    print("TEST 2: Validating API Models")
    print("="*70)
    
    try:
        from api_server import DebateRequest, DebateResponse, JudgeDecision
        
        # Test creating a request
        print("Creating test request...", end=" ")
        req = DebateRequest(
            medical_note="Test note",
            max_rounds=2
        )
        print("✅")
        
        # Test creating a judge decision
        print("Creating test judge decision...", end=" ")
        decision = JudgeDecision(
            final_answer="CORRECT",
            confidence_score=8,
            winner="Expert A",
            reasoning="Test reasoning"
        )
        print("✅")
        
        return True
    except Exception as e:
        print(f"\n❌ Model validation error: {e}")
        return False


def test_graph_initialization():
    """Test that the graph can be built."""
    print("\n" + "="*70)
    print("TEST 3: Building Debate Graph")
    print("="*70)
    
    try:
        from config.settings import settings
        from app.graph.graph import build_graph
        
        print("Building graph (this may take a moment)...")
        graph = build_graph(settings)
        print("✅ Graph built successfully")
        
        return True
    except Exception as e:
        print(f"❌ Graph initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all initialization tests."""
    print("\n" + "="*70)
    print("BLUEMED API SERVER - INITIALIZATION TEST")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Cannot proceed.")
        sys.exit(1)
    
    # Test 2: API Models
    if not test_api_models():
        print("\n❌ API model test failed.")
        all_passed = False
    
    # Test 3: Graph initialization
    if not test_graph_initialization():
        print("\n❌ Graph initialization test failed.")
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL INITIALIZATION TESTS PASSED!")
        print("="*70)
        print("\nThe API server is ready to start.")
        print("\nNext steps:")
        print("  1. Start the API server:")
        print("     python3 api_server.py")
        print("\n  2. In another terminal, test the API:")
        print("     python3 test_integration.py")
        print("\n  3. Or use the all-in-one script:")
        print("     ./start.sh")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the errors above before starting the server.")
    
    print("\n" + "="*70 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
