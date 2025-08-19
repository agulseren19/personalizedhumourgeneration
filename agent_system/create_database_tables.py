#!/usr/bin/env python3
"""
Database Migration Script for CAH System
Creates all necessary tables in PostgreSQL
"""

import os
import sys
from pathlib import Path

# Add parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from models.database import create_database, Base
from config.settings import settings

def main():
    """Create all database tables"""
    try:
        print("🚀 Starting database migration...")
        print(f"📊 Database URL: {settings.database_url}")
        
        # Create all tables
        engine = create_database(settings.database_url)
        
        print("✅ Database migration completed successfully!")
        print("📋 Tables created:")
        
        # List all tables
        inspector = engine.dialect.inspector(engine)
        tables = inspector.get_table_names()
        
        for table in tables:
            print(f"  - {table}")
            
        print(f"\n🎉 Total tables created: {len(tables)}")
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
