#!/usr/bin/env python3
"""
Test script to verify SEEDS Nexus Academy setup works correctly.
Run this script to test both local environment and package installations.
"""

# Debug: Print immediately to verify script is running
print("🚀 SEEDS Nexus Academy Test Script Starting...")

import sys
import subprocess
import importlib
import os
from pathlib import Path

print("📦 Basic imports successful...")

def test_python_version():
    """Test if Python version is 3.8+"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (meets requirement: 3.8+)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def test_virtual_environment():
    """Test if we're in a virtual environment"""
    print("\n🏠 Testing virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  Not in a virtual environment (recommended but not required)")
        return True

def test_package_installation():
    """Test if required packages can be imported"""
    print("\n📦 Testing package installations...")

    # Essential packages for core functionality
    essential_packages = [
        ('langchain', '0.1.0'),
        ('openai', '1.12.0'),
        ('tiktoken', '0.5.2'),
        ('dotenv', '1.0.0'),
        ('requests', '2.31.0'),
    ]

    # Optional packages for enhanced functionality
    optional_packages = [
        ('pandas', '2.2.0'),
        ('matplotlib', '3.8.2'),
        ('bs4', '4.12.3'),  # beautifulsoup4 imports as bs4
        ('jupyter', '1.0.0'),
    ]

    all_essential_passed = True

    print("Essential packages:")
    for package_name, expected_version in essential_packages:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: Not installed (REQUIRED)")
            all_essential_passed = False

    print("\nOptional packages:")
    for package_name, expected_version in optional_packages:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"⚠️  {package_name}: Not installed (optional)")

    if not all_essential_passed:
        print("\n💡 Install essential packages with: pip install langchain openai tiktoken python-dotenv requests")

    return all_essential_passed

def test_file_structure():
    """Test if all required files and directories exist"""
    print("\n📁 Testing file structure...")

    required_files = [
        'README.md',
        'requirements.txt',
        '.env.template',
        '.gitignore',
        'notebooks/01_tokenization_demo.ipynb',
        'notebooks/02_langchain_concepts.ipynb',
        'notebooks/03_prompt_engineering.ipynb',
        'notebooks/04_simple_agent.ipynb',
        'utils/__init__.py',
        'utils/setup.py',
        'data/sample_text.txt',
        'data/environmental_data.json'
    ]

    all_passed = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: Missing")
            all_passed = False

    return all_passed

def test_env_file():
    """Test environment file setup"""
    print("\n🔐 Testing environment configuration...")

    if Path('.env').exists():
        print("✅ .env file exists")
        # Check if API key is set (without revealing it)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                print("✅ OPENAI_API_KEY is configured")
                return True
            else:
                print("⚠️  OPENAI_API_KEY not configured (add your actual API key)")
                return False
        except ImportError:
            print("⚠️  python-dotenv not installed (run: pip install -r requirements.txt)")
            return False
    else:
        print("⚠️  .env file not found (copy from .env.template and add your API key)")
        return False

def test_langchain_basic():
    """Test basic LangChain functionality"""
    print("\n🦜 Testing LangChain basic functionality...")

    try:
        from langchain.prompts import PromptTemplate
        print("✅ LangChain PromptTemplate imported")

        # Test prompt template creation
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Tell me about {topic} in the context of environmental sustainability."
        )

        formatted_prompt = prompt.format(topic="solar energy")
        print("✅ LangChain PromptTemplate working")

        # Test LLM import (but don't initialize to avoid API issues)
        try:
            from langchain.llms import OpenAI
            print("✅ LangChain OpenAI LLM imported")
        except ImportError as e:
            print(f"⚠️  LangChain OpenAI import failed: {e}")
            return False

        return True
    except ImportError as e:
        print(f"❌ LangChain not installed: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ LangChain test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 SEEDS Nexus Academy - Local Setup Test")
    print("=" * 50)

    tests = [
        test_python_version,
        test_virtual_environment,
        test_file_structure,
        test_package_installation,
        test_env_file,
        test_langchain_basic
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ Your SEEDS Nexus Academy setup is ready!")
        print("\n🚀 Next steps:")
        print("   1. Make sure your .env file has a valid OpenAI API key")
        print("   2. Open any notebook and start learning!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("❌ Please fix the issues above before proceeding")

        if not Path('.env').exists():
            print("\n💡 Quick fix: Copy .env.template to .env and add your OpenAI API key")

if __name__ == "__main__":
    main()
