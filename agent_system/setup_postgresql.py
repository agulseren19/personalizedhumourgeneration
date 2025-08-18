#!/usr/bin/env python3
"""
PostgreSQL Setup Script for Production CAH System
This script sets up PostgreSQL database and migrates data from SQLite
"""

import os
import sys
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import sqlite3
import json

def create_postgresql_database():
    """Create PostgreSQL database and user"""
    try:
        # On macOS, PostgreSQL uses your system username by default
        import getpass
        system_user = getpass.getuser()
        
        print(f"🔍 Detected system user: {system_user}")
        
        # Try to connect with system username (no password)
        try:
            conn = psycopg2.connect(
                host="localhost",
                port="5432",
                user=system_user,
                database="postgres"
            )
            print(f"✅ Connected as user: {system_user}")
        except:
            # Fallback: try to connect to default database
            try:
                conn = psycopg2.connect(
                    host="localhost",
                    port="5432",
                    user=system_user,
                    database="postgres"
                )
                print(f"✅ Connected to default database as: {system_user}")
            except Exception as e:
                print(f"❌ Could not connect to PostgreSQL: {e}")
                print("💡 Try these steps:")
                print("   1. brew services restart postgresql")
                print("   2. psql postgres")
                print("   3. CREATE USER postgres WITH PASSWORD 'postgres';")
                print("   4. GRANT ALL PRIVILEGES ON DATABASE postgres TO postgres;")
                return False
        
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("SELECT 1 FROM pg_database WHERE datname='cah_db'")
        if not cursor.fetchone():
            cursor.execute("CREATE DATABASE cah_db")
            print("✅ PostgreSQL database 'cah_db' created successfully")
        else:
            print("ℹ️  Database 'cah_db' already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating PostgreSQL database: {e}")
        print("💡 Make sure PostgreSQL is running and accessible")
        return False

def migrate_data_from_sqlite():
    """Migrate data from SQLite to PostgreSQL"""
    try:
        # Connect to SQLite database
        sqlite_conn = sqlite3.connect('agent_humor.db')
        sqlite_cursor = sqlite_conn.cursor()
        
        # Connect to PostgreSQL using system username
        import getpass
        system_user = getpass.getuser()
        pg_engine = create_engine(f"postgresql://{system_user}@localhost:5432/cah_db")
        pg_conn = pg_engine.connect()
        
        # Get list of tables from SQLite
        sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in sqlite_cursor.fetchall()]
        
        print(f"📋 Found tables in SQLite: {tables}")
        
        for table in tables:
            try:
                # Get table schema
                sqlite_cursor.execute(f"PRAGMA table_info({table})")
                columns = sqlite_cursor.fetchall()
                
                # Get data
                sqlite_cursor.execute(f"SELECT * FROM {table}")
                rows = sqlite_cursor.fetchall()
                
                if rows:
                    print(f"🔄 Migrating {len(rows)} rows from table '{table}'...")
                    
                    # Create table in PostgreSQL with better type mapping
                    column_defs = []
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2]
                        col_not_null = "NOT NULL" if col[3] else ""
                        
                        # Map SQLite types to PostgreSQL
                        if col_type == 'INTEGER':
                            pg_type = 'INTEGER'
                        elif col_type == 'TEXT':
                            pg_type = 'TEXT'
                        elif col_type == 'REAL':
                            pg_type = 'DOUBLE PRECISION'
                        elif col_type == 'BLOB':
                            pg_type = 'BYTEA'
                        elif col_type == 'BOOLEAN':
                            pg_type = 'BOOLEAN'
                        else:
                            pg_type = 'TEXT'
                        
                        column_def = f"{col_name} {pg_type} {col_not_null}".strip()
                        column_defs.append(column_def)
                    
                    # Create table
                    create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        {', '.join(column_defs)}
                    )
                    """
                    print(f"🔧 Creating table with SQL: {create_table_sql}")
                    pg_conn.execute(text(create_table_sql))
                    
                    # Insert data - fix the parameter binding issue
                    for row in rows:
                        try:
                            # Convert row to tuple and handle None values
                            row_data = []
                            for value in row:
                                if value is None:
                                    row_data.append(None)
                                elif isinstance(value, str):
                                    row_data.append(value)
                                elif isinstance(value, (int, float)):
                                    row_data.append(value)
                                elif isinstance(value, bool):
                                    row_data.append(value)
                                else:
                                    row_data.append(str(value))
                            
                            # Convert integer booleans to actual booleans for PostgreSQL
                            if table == 'personas' and len(row_data) > 7:
                                # is_active column (index 7) should be boolean
                                if isinstance(row_data[7], int):
                                    row_data[7] = bool(row_data[7])
                            elif table == 'evaluator_personas' and len(row_data) > 6:
                                # is_active column (index 6) should be boolean
                                if isinstance(row_data[6], int):
                                    row_data[6] = bool(row_data[6])
                            
                            # Use proper parameter binding with SQLAlchemy
                            placeholders = ', '.join([f':{i}' for i in range(len(row_data))])
                            insert_sql = f"INSERT INTO {table} VALUES ({placeholders})"
                            
                            # Execute with proper parameter binding
                            params = {str(i): val for i, val in enumerate(row_data)}
                            pg_conn.execute(text(insert_sql), params)
                            print(f"  ✅ Inserted row: {row_data[:3]}...")  # Show first 3 values
                            
                        except Exception as e:
                            print(f"  ❌ Error inserting row: {e}")
                            print(f"  Row data: {row}")
                            # Rollback and continue with next row
                            pg_conn.rollback()
                            continue
                    
                    # Commit the transaction
                    pg_conn.commit()
                    print(f"✅ Table '{table}' migrated successfully")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not migrate table '{table}': {e}")
                continue
        
        sqlite_cursor.close()
        sqlite_conn.close()
        pg_conn.close()
        
        print("✅ Data migration completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during data migration: {e}")
        return False

def setup_production_environment():
    """Set up production environment variables"""
    import getpass
    system_user = getpass.getuser()
    
    env_content = f"""# Production Environment Configuration
# Copy this to .env file (not committed to git)

# Database Configuration
DATABASE_URL=postgresql://{system_user}@localhost:5432/cah_db

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Keys (set these in production)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# AWS Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# Redis Configuration
REDIS_URL=redis://localhost:6379

# CORS Configuration
ALLOWED_ORIGINS=https://cah-frontend.onrender.com,http://localhost:3000

# Security
SECRET_KEY=your-app-secret-key-change-this-in-production
ENVIRONMENT=production

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
"""
    
    try:
        with open('.env.example', 'w') as f:
            f.write(env_content)
        print("✅ Created .env.example file")
        print("💡 Copy this to .env and fill in your actual values")
        return True
    except Exception as e:
        print(f"❌ Error creating .env.example: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up PostgreSQL for Production CAH System...")
    print("=" * 60)
    
    # Step 1: Create PostgreSQL database
    print("\n📊 Step 1: Creating PostgreSQL database...")
    if not create_postgresql_database():
        print("❌ Failed to create database. Exiting.")
        return
    
    # Step 2: Migrate data from SQLite
    print("\n🔄 Step 2: Migrating data from SQLite...")
    if not migrate_data_from_sqlite():
        print("❌ Failed to migrate data. Exiting.")
        return
    
    # Step 3: Set up environment configuration
    print("\n⚙️  Step 3: Setting up environment configuration...")
    if not setup_production_environment():
        print("❌ Failed to set up environment. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("✅ PostgreSQL setup completed successfully!")
    print("\n📋 What was set up:")
    print("   • PostgreSQL database 'cah_db'")
    print("   • Data migrated from SQLite")
    print("   • Production environment template")
    print("\n🔐 Next steps:")
    print("   1. Copy .env.example to .env")
    print("   2. Fill in your actual API keys and secrets")
    print("   3. Run: python migrate_auth_system.py")
    print("   4. Start your system: python start_cah_working.py")
    print("\n🚨 Security Notes:")
    print("   • Change JWT_SECRET_KEY to a strong random string")
    print("   • Set all API keys in production")
    print("   • Never commit .env file to git")

if __name__ == "__main__":
    main()
