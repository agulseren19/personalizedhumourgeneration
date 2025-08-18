#!/usr/bin/env python3
"""
🃏 AI Cards Against Humanity - Complete Startup Script
Runs fixes and automatically starts both backend and frontend
"""

import os
import sys
import subprocess
import signal
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CAHSystemManager:
    """Manages the CAH system startup and shutdown"""
    
    def __init__(self):
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Set up graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum=None, frame=None):
        """Gracefully shutdown all processes"""
        logger.info("🛑 Shutting down CAH system...")
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                logger.info("✅ Frontend stopped")
            except:
                self.frontend_process.kill()
                logger.info("✅ Frontend force-stopped")
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                logger.info("✅ Backend stopped")
            except:
                self.backend_process.kill()
                logger.info("✅ Backend force-stopped")
        
        logger.info("👋 CAH system shutdown complete!")
        sys.exit(0)
    
    def check_prerequisites(self):
        """Check if all required tools are available"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check if in correct directory
        if not (project_root / "agent_system").exists():
            logger.error("❌ agent_system directory not found. Run from project root.")
            return False
        
        if not (project_root / "nextjs-boilerplate-main").exists():
            logger.error("❌ nextjs-boilerplate-main directory not found.")
            return False
        
        # Check OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("⚠️  OPENAI_API_KEY not set. Some features may not work.")
            try:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return False
            except:
                logger.info("Running in non-interactive mode, continuing...")
        else:
            logger.info("✅ OpenAI API key detected")
        
        logger.info("✅ Prerequisites check passed")
        return True
    
    async def run_fixes(self):
        """Run the CAH fixes"""
        logger.info("🔧 Running CAH system fixes...")
        
        try:
            from agent_system.fix_cah_issues import CAHIssueFixer
            
            fixer = CAHIssueFixer()
            await fixer.fix_all_issues()
            
            logger.info("✅ All fixes completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fixes failed: {e}")
            return False
    
    def start_backend(self):
        """Start the backend API server"""
        logger.info("🚀 Starting backend API server...")
        
        try:
            # Kill any existing process on port 8000
            try:
                subprocess.run(["lsof", "-ti:8000"], capture_output=True, check=True)
                subprocess.run(["lsof", "-ti:8000", "|", "xargs", "kill", "-9"], shell=True, capture_output=True)
                time.sleep(2)
            except:
                pass
            
            # Use Python module execution to handle imports properly
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "agent_system.api.main"
            ], cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for backend to be ready
            logger.info("⏳ Waiting for backend to start...")
            for i in range(30):
                if self.backend_process.poll() is not None:
                    # Process died
                    stdout, stderr = self.backend_process.communicate()
                    logger.error(f"❌ Backend process died: {stderr.decode()}")
                    return False
                
                try:
                    import requests
                    response = requests.get("http://localhost:8000", timeout=1)
                    if response.status_code == 200:
                        logger.info("✅ Backend ready at http://localhost:8000")
                        return True
                except:
                    pass
                time.sleep(1)
            
            logger.error("❌ Backend failed to start within 30 seconds")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend development server"""
        logger.info("🚀 Starting frontend development server...")
        
        try:
            # Kill any existing process on port 3000
            try:
                subprocess.run(["lsof", "-ti:3000"], capture_output=True, check=True)
                subprocess.run(["lsof", "-ti:3000", "|", "xargs", "kill", "-9"], shell=True, capture_output=True)
                time.sleep(2)
            except:
                pass
            
            # Check if node_modules exists
            if not (project_root / "nextjs-boilerplate-main" / "node_modules").exists():
                logger.info("📦 Installing frontend dependencies...")
                subprocess.run([
                    "npm", "install"
                ], cwd=project_root / "nextjs-boilerplate-main", check=True)
            
            # Create .env.local if it doesn't exist
            env_file = project_root / "nextjs-boilerplate-main" / ".env.local"
            if not env_file.exists():
                env_file.write_text("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
                logger.info("✅ Created .env.local for frontend")
            
            # Start Next.js dev server
            self.frontend_process = subprocess.Popen([
                "npm", "run", "dev"
            ], cwd=project_root / "nextjs-boilerplate-main", 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for frontend to be ready
            logger.info("⏳ Waiting for frontend to start...")
            for i in range(60):
                if self.frontend_process.poll() is not None:
                    # Process died
                    stdout, stderr = self.frontend_process.communicate()
                    logger.error(f"❌ Frontend process died: {stderr.decode()}")
                    return False
                
                try:
                    import requests
                    response = requests.get("http://localhost:3000", timeout=1)
                    if response.status_code == 200:
                        logger.info("✅ Frontend ready at http://localhost:3000")
                        return True
                except:
                    pass
                time.sleep(2)
            
            logger.error("❌ Frontend failed to start within 2 minutes")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes and show status"""
        logger.info("📊 Monitoring system processes...")
        logger.info("⚠️  Keep this terminal open. Press Ctrl+C to stop the system.")
        
        try:
            while True:
                # Check backend
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("❌ Backend process died unexpectedly")
                    self.shutdown()
                
                # Check frontend  
                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.error("❌ Frontend process died unexpectedly")
                    self.shutdown()
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.shutdown()
    
    async def start_system(self):
        """Start the complete CAH system"""
        logger.info("🃏 AI Cards Against Humanity - Complete System Startup")
        logger.info("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Run fixes
        if not await self.run_fixes():
            return False
        
        # Start backend
        if not self.start_backend():
            return False
        
        # Start frontend
        if not self.start_frontend():
            return False
        
        # Success message
        logger.info("")
        logger.info("🎉 AI Cards Against Humanity is ready!")
        logger.info("=" * 50)
        logger.info("")
        logger.info("🌐 Access your application:")
        logger.info("   • Frontend:  http://localhost:3000")
        logger.info("   • Game:      http://localhost:3000/cah") 
        logger.info("   • API:       http://localhost:8000")
        logger.info("   • API Docs:  http://localhost:8000/docs")
        logger.info("")
        logger.info("📋 What you can do:")
        logger.info("   • Play the CAH game with AI humor generation")
        logger.info("   • Rate responses to train the AI to your preferences")
        logger.info("   • View analytics showing how the AI learns about you")
        logger.info("   • Explore different AI personas and humor styles")
        logger.info("")
        logger.info("🔧 System Status:")
        logger.info("   ✅ Database initialized and fixed")
        logger.info("   ✅ Duplicate personas removed")
        logger.info("   ✅ Interaction counter fixed")
        logger.info("   ✅ Dynamic personas stored")
        logger.info("   ✅ Favorite agent logic working")
        logger.info("")
        
        # Monitor processes
        self.monitor_processes()
        
        return True

async def main():
    """Main entry point"""
    manager = CAHSystemManager()
    success = await manager.start_system()
    
    if not success:
        logger.error("❌ Failed to start CAH system")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        sys.exit(1) 