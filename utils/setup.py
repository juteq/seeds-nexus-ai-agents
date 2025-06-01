"""
SEEDS Nexus Academy - Setup Utilities
Helper functions for environment setup and common operations
"""

import os
import sys
from typing import Optional

def setup_colab_environment():
    """
    Setup function specifically for Google Colab environment
    Installs required packages and handles imports
    """
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        print("🌱 Setting up SEEDS Nexus Academy environment in Google Colab...")

        # Install required packages
        packages = [
            "langchain==0.1.0",
            "langchain-community==0.0.10",
            "langchain-openai==0.0.5",
            "openai==1.12.0",
            "tiktoken==0.5.2",
            "python-dotenv==1.0.0"
        ]

        for package in packages:
            os.system(f"pip install {package}")

        print("✅ Environment setup complete!")
    else:
        print("📝 Local environment detected. Make sure to install requirements.txt")

def get_api_key() -> Optional[str]:
    """
    Get OpenAI API key from environment or user input
    """
    # Try to get from environment first
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            import google.colab
            from google.colab import userdata
            # Try to get from Colab secrets
            try:
                api_key = userdata.get('OPENAI_API_KEY')
            except:
                pass
        except ImportError:
            pass

    if not api_key:
        print("🔑 OpenAI API Key not found in environment.")
        print("Please add your API key to:")
        print("- Google Colab: Go to Secrets (🔒) and add 'OPENAI_API_KEY'")
        print("- Local: Set OPENAI_API_KEY environment variable")
        api_key = input("Or enter your API key here: ").strip()

    return api_key if api_key else None

def verify_setup():
    """
    Verify that all required packages are installed and working
    """
    try:
        import langchain
        import tiktoken
        import openai
        print("✅ All packages successfully imported!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def display_environmental_banner():
    """
    Display the SEEDS Nexus Academy banner
    """
    banner = """
    🌱 SEEDS Nexus AI Agents Academy 🌱
    =====================================
    Learning AI for Environmental Impact

    🌍 Focus: Sustainability & Climate Action
    🤖 Tools: LangChain & OpenAI
    🎯 Goal: Build AI agents for good
    """
    print(banner)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def format_environmental_data(data: dict) -> str:
    """
    Format environmental data for display
    """
    formatted = "🌍 Environmental Data Summary:\n"
    formatted += "=" * 35 + "\n"

    for key, value in data.items():
        if isinstance(value, (int, float)):
            formatted += f"📊 {key}: {value:,.2f}\n"
        else:
            formatted += f"📝 {key}: {value}\n"

    return formatted
