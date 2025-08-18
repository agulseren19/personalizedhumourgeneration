#!/usr/bin/env python3
"""
Main entry point for the Agent-Based Humor Generation System
"""

import asyncio
import argparse
import subprocess
import sys
import os
from pathlib import Path

# Add the agent_system directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "agent_system.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])

def start_streamlit_ui():
    """Start the Streamlit UI"""
    print("🎨 Starting Streamlit UI...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "agent_system/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def setup_database():
    """Initialize the database with default data"""
    print("🗄️ Setting up database...")
    
    from models.database import create_database, get_session_local
    from personas.persona_manager import PersonaManager
    
    # Create database
    create_database(settings.database_url)
    
    # Initialize personas
    SessionLocal = get_session_local(settings.database_url)
    db = SessionLocal()
    try:
        persona_manager = PersonaManager(db)
        print("✅ Database setup complete!")
        print(f"📊 Generated {len(persona_manager.get_generation_personas())} generation personas")
        print(f"📊 Generated {len(persona_manager.get_evaluator_personas())} evaluation personas")
    finally:
        db.close()

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    missing_vars = []
    
    # Check API keys
    if not settings.openai_api_key:
        missing_vars.append("OPENAI_API_KEY")
    
    if not settings.anthropic_api_key:
        missing_vars.append("ANTHROPIC_API_KEY")
    
    # At least one API key should be provided
    if not any([settings.openai_api_key, settings.anthropic_api_key, settings.deepseek_api_key]):
        print("❌ Error: At least one LLM API key must be provided")
        print("   Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY")
        return False
    
    if missing_vars:
        print(f"⚠️  Warning: Some API keys are missing: {', '.join(missing_vars)}")
        print("   The system will work with available providers only.")
    
    print("✅ Environment check complete!")
    return True

def run_demo():
    """Run a quick demo of the humor generation system"""
    print("🎭 Running humor generation demo...")
    
    import asyncio
    from models.database import create_database, get_session_local
    from personas.persona_manager import PersonaManager
    from agents.humor_agents import HumorAgentOrchestrator, HumorRequest
    
    async def demo():
        # Setup
        create_database(settings.database_url)
        SessionLocal = get_session_local(settings.database_url)
        db = SessionLocal()
        
        try:
            persona_manager = PersonaManager(db)
            orchestrator = HumorAgentOrchestrator(persona_manager)
            
            # Demo request
            request = HumorRequest(
                context="Office meeting",
                audience="colleagues",
                topic="Monday morning motivation",
                user_id=None,
                humor_type="witty"
            )
            
            print("🤖 Generating humor with multiple agents...")
            result = await orchestrator.generate_and_evaluate_humor(request, num_generators=2, num_evaluators=1)
            
            if result['success']:
                print(f"\n✅ Successfully generated {result['total_generations']} humor options!")
                print(f"🎭 Used personas: {', '.join(result['generation_personas'])}")
                print(f"👩‍⚖️ Evaluated by: {', '.join(result['evaluation_personas'])}")
                
                print("\n🏆 Top Results:")
                for i, ranked_result in enumerate(result['top_results'], 1):
                    generation = ranked_result['generation']
                    avg_scores = ranked_result['average_scores']
                    
                    print(f"\n#{i} - {generation.persona_name}:")
                    print(f"   📝 \"{generation.text}\"")
                    print(f"   🎯 Overall Score: {avg_scores['overall_score']:.1f}/10")
                    print(f"   😂 Humor: {avg_scores['humor_score']:.1f} | "
                          f"🎨 Creativity: {avg_scores['creativity_score']:.1f} | "
                          f"✅ Appropriate: {avg_scores['appropriateness_score']:.1f}")
            else:
                print(f"❌ Demo failed: {result.get('error')}")
        
        finally:
            db.close()
    
    asyncio.run(demo())

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "agent_requirements.txt"])
    print("✅ Dependencies installed!")

def main():
    parser = argparse.ArgumentParser(description="Agent-Based Humor Generation System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Initialize database and check environment')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install dependencies')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start the FastAPI server')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Start the Streamlit UI')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demo')
    
    # Full stack command
    full_parser = subparsers.add_parser('start', help='Start both API and UI (full stack)')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check environment configuration')
    
    args = parser.parse_args()
    
    if args.command == 'install':
        install_dependencies()
    
    elif args.command == 'setup':
        if check_environment():
            setup_database()
    
    elif args.command == 'check':
        check_environment()
    
    elif args.command == 'demo':
        if check_environment():
            setup_database()
            run_demo()
    
    elif args.command == 'api':
        if check_environment():
            setup_database()
            start_api_server()
    
    elif args.command == 'ui':
        start_streamlit_ui()
    
    elif args.command == 'start':
        if check_environment():
            setup_database()
            print("🚀 Starting full stack...")
            print("📡 API will be available at: http://localhost:8000")
            print("🎨 UI will be available at: http://localhost:8501")
            print("📚 API docs will be available at: http://localhost:8000/docs")
            
            # Start API server in background
            import threading
            api_thread = threading.Thread(target=start_api_server)
            api_thread.daemon = True
            api_thread.start()
            
            # Wait a bit for API to start
            import time
            time.sleep(3)
            
            # Start UI (this will block)
            start_streamlit_ui()
    
    else:
        parser.print_help()
        print("\n🎭 Welcome to the Agent-Based Humor Generation System!")
        print("\nQuick start:")
        print("  1. python agent_system/main.py install")
        print("  2. Set your API keys in .env file")
        print("  3. python agent_system/main.py setup")
        print("  4. python agent_system/main.py start")
        print("\nFor a quick demo:")
        print("  python agent_system/main.py demo")

if __name__ == "__main__":
    main() 