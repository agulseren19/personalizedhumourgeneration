#!/usr/bin/env python3
"""
Debug script to check username confusion between similar email addresses
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    print("🔍 Checking username confusion issue...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test database connection
    from models.database import get_db, User
    
    print("✅ Database models imported successfully")
    
    # Get database session
    db = next(get_db())
    try:
        print("✅ Database connection successful")
        
        # Check specific users with similar email prefixes
        print("\n🔍 Checking users with similar email prefixes...")
        
        # Look for users with 'isilgulseren' prefix
        similar_users = db.query(User).filter(
            User.email.like('isilgulseren%')
        ).all()
        
        print(f"📊 Found {len(similar_users)} users with 'isilgulseren' prefix:")
        for user in similar_users:
            print(f"  - ID: {user.id}")
            print(f"    Email: {user.email}")
            print(f"    Username: {user.username}")
            print(f"    Email prefix: {user.email.split('@')[0]}")
            print()
        
        # Check for users with 'aslihangulseren' prefix
        aslihangulseren_users = db.query(User).filter(
            User.email.like('aslihangulseren%')
        ).all()
        
        print(f"📊 Found {len(aslihangulseren_users)} users with 'aslihangulseren' prefix:")
        for user in aslihangulseren_users:
            print(f"  - ID: {user.id}")
            print(f"    Email: {user.email}")
            print(f"    Username: {user.username}")
            print(f"    Email prefix: {user.email.split('@')[0]}")
            print()
        
        # Check for duplicate usernames
        print("\n🔍 Checking for duplicate usernames:")
        all_users = db.query(User).all()
        usernames = {}
        for user in all_users:
            username = user.username or user.email.split('@')[0]
            if username not in usernames:
                usernames[username] = []
            usernames[username].append(user)
        
        duplicates = {username: users for username, users in usernames.items() if len(users) > 1}
        if duplicates:
            print("⚠️  Found duplicate usernames:")
            for username, users in duplicates.items():
                print(f"  Username '{username}' used by:")
                for user in users:
                    print(f"    - ID: {user.id}, Email: {user.email}")
        else:
            print("✅ No duplicate usernames found")
            
    finally:
        db.close()
        print("✅ Database connection closed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
