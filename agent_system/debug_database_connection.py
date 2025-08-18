#!/usr/bin/env python3
"""
Debug database connection and user data
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, trying to load env vars manually")

def debug_database():
    """Debug database connection and data"""
    
    print("🔍 Debugging Database Connection...")
    
    # Check environment variables
    print("\n1️⃣ Environment Variables:")
    database_url = os.getenv("DATABASE_URL")
    print(f"   DATABASE_URL: {database_url}")
    
    # Try to import and test database
    try:
        print("\n2️⃣ Testing Database Import...")
        from models.database import get_db, User, Game, Persona
        
        print("   ✅ Database models imported successfully")
        
        # Try to get a database session
        print("\n3️⃣ Testing Database Session...")
        try:
            db = next(get_db())
            print("   ✅ Database session created successfully")
            
            # Test user query
            print("\n4️⃣ Testing User Query...")
            try:
                users = db.query(User).all()
                print(f"   ✅ User query successful: {len(users)} users found")
                
                for user in users:
                    print(f"      User ID: {user.id}, Email: {user.email}, Username: {user.username}")
                    
            except Exception as e:
                print(f"   ❌ User query failed: {e}")
            
            # Test game query
            print("\n5️⃣ Testing Game Query...")
            try:
                games = db.query(Game).all()
                print(f"   ✅ Game query successful: {len(games)} games found")
                
                for game in games:
                    print(f"      Game ID: {game.id}, Status: {game.status}")
                    
            except Exception as e:
                print(f"   ❌ Game query failed: {e}")
            
            # Test persona query
            print("\n6️⃣ Testing Persona Query...")
            try:
                personas = db.query(Persona).all()
                print(f"   ✅ Persona query successful: {len(personas)} personas found")
                
            except Exception as e:
                print(f"   ❌ Persona query failed: {e}")
            
            db.close()
            
        except Exception as e:
            print(f"   ❌ Database session failed: {e}")
            
    except ImportError as e:
        print(f"   ❌ Database import failed: {e}")
        
        # Try alternative import paths
        print("\n   🔄 Trying alternative import paths...")
        try:
            sys.path.insert(0, str(current_dir.parent))
            from models.database import get_db, User, Game, Persona
            print("   ✅ Alternative import successful")
        except ImportError as e2:
            print(f"   ❌ Alternative import also failed: {e2}")

if __name__ == "__main__":
    debug_database()
