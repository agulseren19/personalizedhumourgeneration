#!/usr/bin/env python3
"""
Check if all personas are visible and accessible
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
import json

def check_personas():
    """Check personas via API"""
    try:
        # Check personas endpoint
        response = requests.get("http://localhost:8000/personas")
        if response.status_code == 200:
            data = response.json()
            
            print("🔍 Checking Personas API...")
            print(f"✅ API Response: {response.status_code}")
            print(f"📊 Total personas found: {len(data.get('personas', []))}")
            print(f"📊 Static personas: {data.get('static_count', 0)}")
            print(f"📊 Dynamic personas: {data.get('dynamic_count', 0)}")
            
            print("\n📋 All Personas:")
            for i, persona in enumerate(data.get('personas', []), 1):
                print(f"{i:2d}. {persona.get('name', 'Unknown')} - {persona.get('description', 'No description')[:50]}...")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking personas: {e}")
        return False

def check_user_preferences():
    """Check if user preferences are working"""
    try:
        # Test user preferences
        test_user = "user_test_123"
        response = requests.get(f"http://localhost:8000/analytics/{test_user}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n👤 User Analytics for {test_user}:")
            print(f"✅ Total interactions: {data.get('total_interactions', 0)}")
            print(f"✅ Favorite persona: {data.get('favorite_persona', 'None')}")
            
            top_personas = data.get('top_personas', [])
            print(f"✅ Top personas ({len(top_personas)}):")
            for persona in top_personas[:5]:
                print(f"   • {persona.get('persona_name', 'Unknown')}: "
                      f"{persona.get('avg_score', 0):.1f}/10 "
                      f"({persona.get('interaction_count', 0)} times)")
            
            return True
        else:
            print(f"❌ Analytics Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking user preferences: {e}")
        return False

if __name__ == "__main__":
    print("🃏 CAH Persona & Preference Checker")
    print("=" * 40)
    
    personas_ok = check_personas()
    prefs_ok = check_user_preferences()
    
    if personas_ok and prefs_ok:
        print("\n✅ All checks passed!")
    else:
        print("\n❌ Some checks failed!")
        sys.exit(1) 