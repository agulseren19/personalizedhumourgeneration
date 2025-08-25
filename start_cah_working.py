#!/usr/bin/env python3
"""
🃏 CAH Working Startup Script
Replicates the exact functionality of the working start_cah_system.sh
"""

import os
import sys
import subprocess
import signal
import time
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# Add agent_system to Python path
agent_system_path = project_root / "agent_system"
sys.path.insert(0, str(agent_system_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CAHWorkingManager:
    """Exact replica of the working bash script functionality"""
    
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
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
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                logger.info("✅ Backend stopped")
            except:
                self.backend_process.kill()
        
        logger.info("👋 CAH system shutdown complete!")
        sys.exit(0)
    
    def check_prerequisites(self):
        """Check prerequisites exactly like the bash script"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check directories
        if not (project_root / "agent_system").exists():
            logger.error("❌ agent_system directory not found")
            return False
        
        if not (project_root / "nextjs-boilerplate-main").exists():
            logger.error("❌ nextjs-boilerplate-main directory not found")
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
        """Run the CAH fixes first"""
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
    
    def install_dependencies(self):
        """Install dependencies exactly like the bash script"""
        logger.info("📦 Installing dependencies...")
        
        # Install Python dependencies
        try:
            logger.info("Installing Python dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(project_root / "agent_system" / "requirements.txt")
            ], check=True, capture_output=True)
            logger.info("✅ Python dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Python dependencies: {e}")
            return False
        
        # Install Node.js dependencies
        try:
            logger.info("Installing Node.js dependencies...")
            subprocess.run([
                "npm", "install"
            ], cwd=project_root / "nextjs-boilerplate-main", check=True, capture_output=True)
            logger.info("✅ Node.js dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Node.js dependencies: {e}")
            return False
        
        # Create .env.local (exactly like bash script)
        env_file = project_root / "nextjs-boilerplate-main" / ".env.local"
        if not env_file.exists():
            env_file.write_text("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
            logger.info("✅ Created .env.local for frontend")
        
        return True
    
    def start_backend(self):
        """Start backend exactly like the bash script - using cah_crewai_api.py"""
        logger.info("🚀 Starting CrewAI backend API server...")
        
        try:
            # Clean up any existing processes
            subprocess.run(["pkill", "-f", "cah_crewai_api"], capture_output=True)
            subprocess.run(["pkill", "-f", "agent_system.api"], capture_output=True)
            time.sleep(2)
            
            # Start the CrewAI API exactly like the bash script
            # Set up environment to ensure proper Python path
            env = os.environ.copy()
            current_pythonpath = env.get('PYTHONPATH', '')
            additional_paths = [
                str(project_root),
                str(project_root / "agent_system")
            ]
            new_pythonpath = ':'.join(additional_paths + [current_pythonpath] if current_pythonpath else additional_paths)
            env['PYTHONPATH'] = new_pythonpath
            
            self.backend_process = subprocess.Popen([
                sys.executable, str(project_root / "agent_system" / "api" / "cah_crewai_api.py")
            ], cwd=project_root, env=env)
            
            logger.info(f"CrewAI Backend PID: {self.backend_process.pid}")
            
            # Wait for backend to start (exactly like bash script)
            logger.info("⏳ Waiting for backend to start...")
            for i in range(60):  # Increased timeout to 60 seconds
                if self.backend_process.poll() is not None:
                    logger.error("❌ Backend process died")
                    return False
                
                try:
                    import requests
                    response = requests.get("http://localhost:8000", timeout=1)
                    if response.status_code in [200, 404]:  # 404 is OK for root endpoint
                        logger.info("✅ Backend is ready at http://localhost:8000")
                        return True
                except:
                    pass
                
                time.sleep(1)
                
                if i == 59:
                    logger.error("❌ Backend failed to start within 60 seconds")
                    if self.backend_process:
                        self.backend_process.kill()
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start frontend exactly like the bash script"""
        logger.info("🚀 Starting frontend dev server...")
        
        try:
            # Clean up any existing frontend processes
            subprocess.run(["pkill", "-f", "next dev"], capture_output=True)
            time.sleep(2)
            
            # Start Next.js dev server
            self.frontend_process = subprocess.Popen([
                "npm", "run", "dev"
            ], cwd=project_root / "nextjs-boilerplate-main")
            
            logger.info(f"Frontend PID: {self.frontend_process.pid}")
            
            # Wait for frontend to start (like bash script)
            logger.info("⏳ Waiting for frontend to start...")
            for i in range(30):
                if self.frontend_process.poll() is not None:
                    logger.error("❌ Frontend process died")
                    return False
                
                try:
                    import requests
                    response = requests.get("http://localhost:3000", timeout=1)
                    if response.status_code == 200:
                        logger.info("✅ Frontend is ready at http://localhost:3000")
                        return True
                except:
                    pass
                
                time.sleep(2)
                
                if i == 29:
                    logger.error("❌ Frontend failed to start within 60 seconds")
                    if self.frontend_process:
                        self.frontend_process.kill()
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor processes like the bash script"""
        logger.info("📊 System Status:")
        logger.info(f"Backend PID: {self.backend_process.pid}")
        logger.info(f"Frontend PID: {self.frontend_process.pid}")
        logger.info("")
        logger.info("Waiting for shutdown signal...")
        
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("❌ Backend process died unexpectedly")
                    self.shutdown()
                
                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.error("❌ Frontend process died unexpectedly")
                    self.shutdown()
                    
        except KeyboardInterrupt:
            self.shutdown()
    
    async def start_system(self):
        """Start the complete system exactly like the working bash script"""
        logger.info("🃏 Starting AI Cards Against Humanity System...")
        logger.info("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Run fixes first
        if not await self.run_fixes():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Start backend (CrewAI API)
        if not self.start_backend():
            return False
        
        # Start frontend
        if not self.start_frontend():
            return False
        
        # Success message (exactly like bash script)
        logger.info("")
        logger.info("🎉 AI Cards Against Humanity is ready!")
        logger.info("=" * 40)
        logger.info("")
        logger.info("🌐 Access your application:")
        logger.info("   • Frontend: http://localhost:3000")
        logger.info("   • Game:     http://localhost:3000/cah")
        logger.info("   • API:      http://localhost:8000")
        logger.info("")
        logger.info("📋 What you can do:")
        logger.info("   • Play the CAH game with AI humor generation")
        logger.info("   • Rate responses to train the AI to your preferences")
        logger.info("   • View analytics showing how the AI learns about you")
        logger.info("   • Explore different AI personas and humor styles")
        logger.info("")
        logger.info("⚠️  Note: Keep this terminal open. Press Ctrl+C to stop both servers.")
        logger.info("")
        
        # Monitor processes
        self.monitor_processes()
        
        return True

async def main():
    """Main entry point"""
    manager = CAHWorkingManager()
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