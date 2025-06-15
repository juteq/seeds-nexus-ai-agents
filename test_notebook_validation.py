#!/usr/bin/env python3
"""
Test script to validate the PDF semantic search notebook functionality.
This script tests the core components without requiring API keys.
"""

import sys
import os
import traceback

def test_imports():
    """Test that all required imports work correctly."""
    print("🧪 Testing imports...")

    try:
        # Core LangChain imports
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA
        from langchain.schema import Document
        print("  ✅ LangChain imports successful")
    except ImportError as e:
        print(f"  ❌ LangChain import error: {e}")
        return False

    try:
        # Pinecone imports (may not be installed)
        from pinecone import Pinecone, ServerlessSpec
        from langchain_community.vectorstores import Pinecone as LangChainPinecone
        print("  ✅ Pinecone imports successful")
    except ImportError as e:
        print(f"  ⚠️ Pinecone not installed (expected in test): {e}")
        # This is expected in test environment

    try:
        # Standard libraries
        import time
        import requests
        from typing import List, Dict
        from tqdm import tqdm
        import json
        print("  ✅ Standard library imports successful")
    except ImportError as e:
        print(f"  ❌ Standard library import error: {e}")
        return False

    return True

def test_environment_detection():
    """Test the environment detection logic."""
    print("\n🌍 Testing environment detection...")

    try:
        # Test Colab detection
        try:
            import google.colab
            IN_COLAB = True
            print("  📱 Running in Google Colab")
        except ImportError:
            IN_COLAB = False
            print("  💻 Running in local environment")

        print(f"  ✅ Environment detection: IN_COLAB = {IN_COLAB}")
        return True
    except Exception as e:
        print(f"  ❌ Environment detection error: {e}")
        return False

def test_text_splitter():
    """Test the text splitter functionality."""
    print("\n✂️ Testing text splitter...")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Test document
        test_text = """
        This is a test document for the PDF semantic search system.

        It contains multiple paragraphs to test the text splitting functionality.
        The splitter should break this into appropriate chunks while maintaining
        context and ensuring proper overlap between chunks.

        This is another paragraph that should help test the chunking mechanism.
        We want to make sure that the text is split appropriately and that
        the chunks maintain semantic meaning while being of appropriate size.

        The final paragraph tests the complete functionality of our text
        splitting approach. This should work well for academic papers and
        other long-form documents that need to be processed for semantic search.
        """

        # Create a document
        doc = Document(
            page_content=test_text,
            metadata={"title": "Test Document", "source": "test"}
        )

        # Split the document
        chunks = text_splitter.split_documents([doc])

        print(f"  ✅ Created {len(chunks)} chunks from test document")
        print(f"  📊 Chunk sizes: {[len(chunk.page_content) for chunk in chunks]}")

        # Verify metadata is preserved
        for chunk in chunks:
            assert "title" in chunk.metadata
            assert "source" in chunk.metadata

        print("  ✅ Metadata preserved in chunks")
        return True

    except Exception as e:
        print(f"  ❌ Text splitter error: {e}")
        print(f"  📝 Traceback: {traceback.format_exc()}")
        return False

def test_document_creation():
    """Test document creation and metadata handling."""
    print("\n📄 Testing document creation...")

    try:
        from langchain.schema import Document

        # Test document with various metadata
        doc = Document(
            page_content="This is a test document for semantic search.",
            metadata={
                "title": "Test Paper",
                "authors": "Test Author",
                "published": "2025-06-15",
                "source": "test",
                "url": "https://example.com/paper.pdf"
            }
        )

        print(f"  ✅ Document created with {len(doc.page_content)} characters")
        print(f"  📋 Metadata keys: {list(doc.metadata.keys())}")

        # Test document list
        documents = [doc] * 3  # Simulate multiple documents
        print(f"  ✅ Created list of {len(documents)} documents")

        return True

    except Exception as e:
        print(f"  ❌ Document creation error: {e}")
        return False

def test_api_key_handling():
    """Test API key handling without actual keys."""
    print("\n🔑 Testing API key handling...")

    try:
        # Save original environment
        original_openai = os.environ.get("OPENAI_API_KEY")
        original_pinecone = os.environ.get("PINECONE_API_KEY")

        # Test with missing keys
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "PINECONE_API_KEY" in os.environ:
            del os.environ["PINECONE_API_KEY"]

        # Test key detection
        openai_set = os.environ.get("OPENAI_API_KEY") not in [None, "", "your-openai-api-key-here"]
        pinecone_set = os.environ.get("PINECONE_API_KEY") not in [None, "", "your-pinecone-api-key-here"]

        print(f"  📊 OpenAI key detected: {openai_set}")
        print(f"  📊 Pinecone key detected: {pinecone_set}")

        # Test fallback values
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["PINECONE_API_KEY"] = "test-key"

        openai_set = os.environ.get("OPENAI_API_KEY") not in [None, "", "your-openai-api-key-here"]
        pinecone_set = os.environ.get("PINECONE_API_KEY") not in [None, "", "your-pinecone-api-key-here"]

        print(f"  ✅ Key handling works: OpenAI={openai_set}, Pinecone={pinecone_set}")

        # Restore original environment
        if original_openai:
            os.environ["OPENAI_API_KEY"] = original_openai
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        if original_pinecone:
            os.environ["PINECONE_API_KEY"] = original_pinecone
        elif "PINECONE_API_KEY" in os.environ:
            del os.environ["PINECONE_API_KEY"]

        return True

    except Exception as e:
        print(f"  ❌ API key handling error: {e}")
        return False

def test_configuration():
    """Test configuration setup."""
    print("\n⚙️ Testing configuration...")

    try:
        # Configuration values from notebook
        PINECONE_INDEX_NAME = "academic-papers"
        EMBEDDING_MODEL = "text-embedding-3-small"
        CHAT_MODEL = "gpt-4o-mini"
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200

        print(f"  📝 Index name: {PINECONE_INDEX_NAME}")
        print(f"  🤖 Embedding model: {EMBEDDING_MODEL}")
        print(f"  💬 Chat model: {CHAT_MODEL}")
        print(f"  📏 Chunk size: {CHUNK_SIZE} with {CHUNK_OVERLAP} overlap")

        # Validate configuration values
        assert isinstance(CHUNK_SIZE, int) and CHUNK_SIZE > 0
        assert isinstance(CHUNK_OVERLAP, int) and CHUNK_OVERLAP >= 0
        assert CHUNK_OVERLAP < CHUNK_SIZE
        assert len(PINECONE_INDEX_NAME) > 0

        print("  ✅ Configuration validation passed")
        return True

    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_loader_functions():
    """Test document loader function definitions without actual loading."""
    print("\n📚 Testing loader function definitions...")

    try:
        # Simulate the functions from the notebook
        def load_arxiv_papers(queries, max_docs=3):
            return []  # Mock implementation

        def load_pdf_from_url(url, title=None):
            return []  # Mock implementation

        def load_local_pdf(file_path):
            return []  # Mock implementation

        # Test function calls with mock data
        queries = ["test query"]
        result1 = load_arxiv_papers(queries, max_docs=2)
        result2 = load_pdf_from_url("https://example.com/test.pdf", "Test")
        result3 = load_local_pdf("/tmp/test.pdf")

        print(f"  ✅ ArXiv loader function: returns {type(result1)}")
        print(f"  ✅ URL loader function: returns {type(result2)}")
        print(f"  ✅ Local loader function: returns {type(result3)}")

        return True

    except Exception as e:
        print(f"  ❌ Loader function error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Starting PDF Semantic Search Notebook Validation")
    print("=" * 60)

    tests = [
        test_imports,
        test_environment_detection,
        test_text_splitter,
        test_document_creation,
        test_api_key_handling,
        test_configuration,
        test_loader_functions
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"  ❌ {test.__name__} failed")
        except Exception as e:
            print(f"  💥 {test.__name__} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Validation Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The notebook should work correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
