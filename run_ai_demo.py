#!/usr/bin/env python3
"""
🚀 SwaggyStacks AI Trading Demo Launcher 🚀

Quick launcher for the AI trading agent demonstration.
Run this to see your Chinese LLM trading system in action!

Usage:
    python run_ai_demo.py
    
    or
    
    cd backend && python demo_ai_trading_agents.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the AI trading demonstration"""
    
    print("🚀 SwaggyStacks AI Trading Demo Launcher")
    print("=" * 50)
    
    # Find the backend directory
    current_dir = Path(__file__).parent
    backend_dir = current_dir / "backend"
    demo_script = backend_dir / "demo_ai_trading_agents.py"
    
    if not demo_script.exists():
        print(f"❌ Demo script not found at: {demo_script}")
        print("Please run this script from the project root directory.")
        return 1
    
    print(f"📁 Backend directory: {backend_dir}")
    print(f"📄 Demo script: {demo_script}")
    print("\n🎬 Starting AI trading demonstration...")
    print("-" * 50)
    
    try:
        # Change to backend directory and run the demo
        os.chdir(backend_dir)
        
        # Run the demonstration
        result = subprocess.run([
            sys.executable, "demo_ai_trading_agents.py"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Demonstration completed successfully!")
        else:
            print(f"\n⚠️ Demonstration ended with code: {result.returncode}")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Error launching demonstration: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)