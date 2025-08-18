#!/usr/bin/env python3
"""
Simple debug script to check database and identify host issue
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from models.database import get_db, User
    print("✅ Database models imported successfully")
    
    # Get database session
    db = next(get_db())
    try:
        print("\n📊 All Users in Database:")
        users = db.query(User).all()
        for user in users:
            print(f"  - ID: {user.id}, Email: {user.email}, Username: {user.username}")
        
        # Check for similar usernames
        print("\n🔍 Checking for similar usernames:")
        usernames = [user.username for user in users if user.username]
        emails = [user.email for user in users]
        
        for i, username1 in enumerate(usernames):
            for j, username2 in enumerate(usernames[i+1:], i+1):
                if username1 == username2:
                    print(f"  ⚠️  Duplicate username: {username1}")
        
        for i, email1 in enumerate(emails):
            for j, email2 in enumerate(emails[i+1:], i+1):
                if email1.split('@')[0] == email2.split('@')[0]:
                    print(f"  ⚠️  Similar email prefixes: {email1} and {email2}")
        
        print(f"\n📈 Total users: {len(users)}")
        
    finally:
        db.close()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
