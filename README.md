# SEEDS Nexus AI Agents Academy

Welcome to SEEDS Nexus AI Agents Academy! This project teaches AI agent development using LangChain with a focus on environmental and sustainability themes.

## 🌱 About SEEDS Nexus

SEEDS Nexus is an educational initiative that combines artificial intelligence learning with environmental awareness. Through hands-on exercises and real-world examples, you'll learn to build AI agents while exploring sustainability topics.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Colab account (recommended)
- OpenAI API key

### Setup for Google Colab
1. Open any notebook in Google Colab
2. The notebooks include setup cells that will install all required packages
3. Add your OpenAI API key when prompted

### Setup for Local Development
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.template` to `.env` and add your OpenAI API key

## 📚 Learning Path

Follow these notebooks in order:

1. **01_tokenization_demo.ipynb** - Learn about tokenization and text processing
2. **02_langchain_concepts.ipynb** - Master core LangChain concepts
3. **03_prompt_engineering.ipynb** - Develop effective prompting techniques
4. **04_simple_agent.ipynb** - Build your first AI agent

## 🎯 Learning Objectives

By completing this course, you will:
- Understand how text tokenization works in AI systems
- Master core LangChain components (LLMs, Prompts, Chains, Memory)
- Learn effective prompt engineering techniques
- Build a functional AI agent with environmental knowledge
- Apply AI to sustainability and climate-related problems

## 🌍 Environmental Focus

All examples and exercises use real-world environmental themes:
- Climate change analysis
- Renewable energy planning
- Carbon footprint calculation
- Sustainable lifestyle recommendations
- Green technology assessment

## 📁 Project Structure

```
seeds-nexus-ai-agents/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.template               # Environment variables template
├── notebooks/                  # Jupyter notebooks
│   ├── 01_tokenization_demo.ipynb
│   ├── 02_langchain_concepts.ipynb
│   ├── 03_prompt_engineering.ipynb
│   └── 04_simple_agent.ipynb
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── setup.py
└── data/                       # Sample data files
    ├── sample_text.txt
    └── environmental_data.json
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests to help improve the learning experience.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- Inspired by the urgent need for AI solutions in environmental sustainability

---

**Ready to start learning?** Open the first notebook: `01_tokenization_demo.ipynb`
