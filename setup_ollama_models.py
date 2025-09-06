#!/usr/bin/env python3
"""
Setup script to install Ollama models for M1 MacBook with 8GB RAM
"""

import subprocess
import sys
import time
import asyncio
import httpx

# Models optimized for M1 MacBook 8GB RAM
REQUIRED_MODELS = [
    {
        'name': 'llama3.2:3b',
        'use_case': 'Market Analysis & Trading Chat',
        'memory_mb': 2048,
        'description': 'Lightweight but capable model for analysis'
    },
    {
        'name': 'phi3:mini',
        'use_case': 'Risk Assessment',
        'memory_mb': 1536,
        'description': 'Ultra-efficient model for risk calculations'
    },
    {
        'name': 'qwen2.5-coder:3b',
        'use_case': 'Strategy Generation & Code',
        'memory_mb': 2560,
        'description': 'Coding-optimized model for strategy development'
    }
]

OLLAMA_BASE_URL = "http://localhost:11434"


def print_header():
    print("🤖 Swaggy Stacks AI - Ollama Setup for M1 MacBook")
    print("=" * 55)
    print("🖥️  Optimized for: M1 MacBook with 8GB RAM")
    print("📦 Installing lightweight models for trading AI")
    print()


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is installed:", result.stdout.strip())
            return True
        else:
            print("❌ Ollama command failed")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {str(e)}")
        return False


async def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                print("✅ Ollama service is running")
                return True
            else:
                print(f"❌ Ollama service responded with status {response.status_code}")
                return False
    except httpx.ConnectError:
        print("❌ Cannot connect to Ollama service")
        print("   💡 Try running: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama service: {str(e)}")
        return False


async def get_installed_models():
    """Get list of installed models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                print(f"📦 Currently installed models: {models}")
                return models
            else:
                print("❌ Failed to get model list")
                return []
    except Exception as e:
        print(f"❌ Error getting model list: {str(e)}")
        return []


def install_model(model_name):
    """Install a specific model"""
    print(f"⬇️  Installing {model_name}...")
    print("   This may take several minutes depending on your internet connection...")
    
    try:
        # Use subprocess to run ollama pull
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {model_name}")
            return True
        else:
            print(f"❌ Failed to install {model_name}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Installation of {model_name} timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing {model_name}: {str(e)}")
        return False


def calculate_total_memory():
    """Calculate total memory usage of all models"""
    total_mb = sum(model['memory_mb'] for model in REQUIRED_MODELS)
    total_gb = total_mb / 1024
    return total_mb, total_gb


async def test_model(model_name):
    """Test if a model is working"""
    try:
        # Simple test using the Ollama client
        sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')
        from app.ai.ollama_client import OllamaClient
        
        client = OllamaClient()
        
        # Map model names to our client keys
        model_key_map = {
            'llama3.2:3b': 'analyst',
            'phi3:mini': 'risk',
            'qwen2.5-coder:3b': 'strategist'
        }
        
        model_key = model_key_map.get(model_name, 'chat')
        
        response = await client.generate_response(
            prompt="Hello, please respond briefly to confirm you are working.",
            model_key=model_key,
            max_tokens=50
        )
        
        if response and not response.startswith("Error"):
            print(f"✅ Model {model_name} is working correctly")
            return True
        else:
            print(f"❌ Model {model_name} test failed: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return False


async def main():
    print_header()
    
    # Check prerequisites
    if not check_ollama_installed():
        print("\n📥 To install Ollama:")
        print("   🌐 Visit: https://ollama.ai/download")
        print("   📦 Or use Homebrew: brew install ollama")
        return
    
    if not await check_ollama_service():
        print("\n🚀 To start Ollama service:")
        print("   💻 Run: ollama serve")
        print("   ⏳ Then run this script again")
        return
    
    # Get current models
    installed_models = await get_installed_models()
    
    # Calculate memory requirements
    total_mb, total_gb = calculate_total_memory()
    print(f"\n📊 Memory Requirements:")
    print(f"   Total: {total_mb} MB ({total_gb:.1f} GB)")
    print(f"   Available for system: {8 - total_gb:.1f} GB remaining on 8GB M1")
    
    if total_gb > 6:  # Leave 2GB for system
        print("⚠️  WARNING: Models may use too much RAM for 8GB system")
        print("   Consider using smaller models if you experience issues")
    
    print(f"\n📦 Required Models:")
    for model in REQUIRED_MODELS:
        status = "✅ Installed" if model['name'] in installed_models else "❌ Missing"
        print(f"   {status} {model['name']} - {model['use_case']} ({model['memory_mb']} MB)")
    
    # Install missing models
    missing_models = [model for model in REQUIRED_MODELS 
                     if model['name'] not in installed_models]
    
    if not missing_models:
        print("\n🎉 All required models are already installed!")
    else:
        print(f"\n⬇️  Installing {len(missing_models)} missing models...")
        
        for model in missing_models:
            print(f"\n📦 {model['name']} - {model['description']}")
            success = install_model(model['name'])
            if not success:
                print(f"❌ Failed to install {model['name']}. Continuing with other models...")
    
    # Test all models
    print("\n🧪 Testing all models...")
    
    final_installed = await get_installed_models()
    test_results = {}
    
    for model in REQUIRED_MODELS:
        if model['name'] in final_installed:
            print(f"\n🔍 Testing {model['name']}...")
            success = await test_model(model['name'])
            test_results[model['name']] = success
        else:
            print(f"\n⏭️  Skipping {model['name']} (not installed)")
            test_results[model['name']] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    
    working_models = sum(1 for success in test_results.values() if success)
    total_models = len(REQUIRED_MODELS)
    
    for model in REQUIRED_MODELS:
        name = model['name']
        status = "✅ Working" if test_results.get(name) else "❌ Not Working"
        print(f"   {status} {name}")
    
    print(f"\n🏆 Result: {working_models}/{total_models} models working")
    
    if working_models == total_models:
        print("🎉 Perfect! All AI models are installed and working.")
        print("🚀 You can now start using the Swaggy Stacks AI system.")
    elif working_models > 0:
        print("⚠️  Some models are working. The system will function with reduced capabilities.")
    else:
        print("❌ No models are working. Please check the installation and try again.")
    
    print(f"\n💡 Next Steps:")
    print("   1. Run the test suite: python test_ai_system.py")
    print("   2. Start the backend server: cd backend && python -m uvicorn app.main:app --reload")
    print("   3. Start the frontend: cd frontend && npm run dev")
    print("   4. Access the AI Assistant at: http://localhost:3000/ai")


if __name__ == "__main__":
    asyncio.run(main())