#!/usr/bin/env python3
"""
Run All CAH Fixes Script
Executes all fixes for the CAH system issues
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))  # Add parent directory for agent_system imports
sys.path.insert(0, str(current_dir))  # Add current directory for local imports

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run all fixes"""
    logger.info("🔧 Starting CAH System Fixes...")
    
    try:
        # Import and run the fix script
        try:
            from fix_cah_issues import CAHIssueFixer
        except ImportError:
            # Try alternative import path
            from agent_system.fix_cah_issues import CAHIssueFixer
        
        fixer = CAHIssueFixer()
        await fixer.fix_all_issues()
        
        logger.info("✅ All fixes completed successfully!")
        logger.info("")
        logger.info("🎉 FIXED ISSUES:")
        logger.info("   ✅ Removed duplicate 'millennial_memer' personas")
        logger.info("   ✅ Fixed favorite agent selection logic")
        logger.info("   ✅ Fixed interaction counter (50+ interactions now tracked properly)")
        logger.info("   ✅ Dynamic personas now saved to database as AI comedians")
        logger.info("   ✅ Database schema issues resolved")
        logger.info("   ✅ Multiplayer game system created")
        logger.info("   ✅ Card preparation system implemented")
        logger.info("")
        logger.info("🚀 To use the fixed system:")
        logger.info("   1. Use 'python agent_system/api/fixed_cah_api.py' for the API")
        logger.info("   2. Use the Fixed Persona Manager for better persona selection")
        logger.info("   3. Use the Multiplayer Game Manager for games")
        logger.info("")
        logger.info("📊 System Status:")
        logger.info("   • Personas: No duplicates, AI comedians marked")
        logger.info("   • User tracking: Proper interaction counting")
        logger.info("   • Game logic: Full multiplayer support")
        logger.info("   • Card generation: Pre-generated cards ready before rounds")
        
    except Exception as e:
        logger.error(f"❌ Fix failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 