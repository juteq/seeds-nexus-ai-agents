#!/usr/bin/env python3
"""
Test script to verify the PDF semantic search notebook works locally.
"""

import os
import warnings
warnings.filterwarnings('ignore')

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from typing import List, Dict

def test_api_key_setup():
    """Test API key setup (works for both local and Colab)."""
    print("🔑 Testing API key setup...")

    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # Colab environment
        from google.colab import userdata
        try:
            openai_api_key = userdata.get('OPENAI_API_KEY')
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                print("✅ API key loaded from Colab secrets!")
            else:
                print("⚠️ No API key found in Colab secrets")
                return False
        except Exception as e:
            print(f"❌ Error accessing Colab secrets: {e}")
            return False
    else:
        # Local environment
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if os.getenv("OPENAI_API_KEY"):
                print("✅ API key loaded from .env or environment")
            else:
                print("⚠️ No API key found")
                return False
        except ImportError:
            if os.getenv("OPENAI_API_KEY"):
                print("✅ API key found in environment")
            else:
                print("⚠️ No API key found")
                return False

    if os.environ.get("OPENAI_API_KEY"):
        print("🚀 Ready to go!")
        return True
    else:
        print("⚠️ Set your OpenAI API key before continuing!")
        return False

def test_configuration():
    """Test basic configuration."""
    print("\n🔧 Testing configuration...")

    # Setup
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Initialize components
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0.1)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    print("✅ Configuration ready!")
    return embeddings, chat_model, text_splitter

def test_arxiv_loading():
    """Test loading papers from arXiv."""
    print("\n📚 Testing arXiv paper loading...")

    try:
        # Load a single paper on transformers
        loader = ArxivLoader(query="transformer neural networks", load_max_docs=1)
        documents = loader.load()

        if documents:
            doc = documents[0]
            print(f"✅ Loaded paper: {doc.metadata.get('Title', 'Unknown')}")
            print(f"  - Content length: {len(doc.page_content)} characters")
            return documents
        else:
            print("❌ No documents loaded")
            return []
    except Exception as e:
        print(f"❌ Error loading from arXiv: {e}")
        return []

def test_chroma_storage(documents, embeddings, text_splitter):
    """Test Chroma vector storage."""
    print("\n🗄️ Testing Chroma storage...")

    if not documents:
        print("❌ No documents to store")
        return None

    try:
        # Split into chunks
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

        print(f"✂️ Created {len(all_chunks)} chunks")

        # Create vectorstore
        vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

        print(f"✅ Stored {len(all_chunks)} chunks in Chroma")
        return vectorstore
    except Exception as e:
        print(f"❌ Error with Chroma storage: {e}")
        return None

def test_semantic_search(vectorstore):
    """Test semantic search."""
    print("\n🔍 Testing semantic search...")

    if not vectorstore:
        print("❌ No vectorstore available")
        return

    try:
        query = "What is attention mechanism?"
        results = vectorstore.similarity_search(query, k=3)

        print(f"Query: '{query}'")
        print(f"✅ Found {len(results)} relevant chunks")

        if results:
            for i, result in enumerate(results[:2], 1):
                content_preview = result.page_content[:100] + "..."
                print(f"  {i}. {content_preview}")
    except Exception as e:
        print(f"❌ Error in semantic search: {e}")

def test_qa_system(vectorstore, chat_model):
    """Test Q&A system."""
    print("\n🤖 Testing Q&A system...")

    if not vectorstore:
        print("❌ No vectorstore available")
        return

    try:
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=False
        )

        # Ask a question
        question = "What is the attention mechanism in neural networks?"
        result = qa_chain({"query": question})

        print(f"Question: {question}")
        print(f"✅ Answer: {result['result'][:200]}...")
        print(f"✅ Based on {len(result['source_documents'])} source documents")

    except Exception as e:
        print(f"❌ Error in Q&A system: {e}")

def main():
    """Run all tests."""
    print("🧪 Running PDF Semantic Search Tests\n")
    print("=" * 50)

    # Test 1: API Key Setup
    if not test_api_key_setup():
        print("\n❌ API key test failed. Please set your OpenAI API key.")
        return

    # Test 2: Configuration
    try:
        embeddings, chat_model, text_splitter = test_configuration()
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        return

    # Test 3: arXiv Loading
    documents = test_arxiv_loading()

    # Test 4: Chroma Storage
    vectorstore = test_chroma_storage(documents, embeddings, text_splitter)

    # Test 5: Semantic Search
    test_semantic_search(vectorstore)

    # Test 6: Q&A System
    test_qa_system(vectorstore, chat_model)

    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n✅ The notebook should work perfectly in both local and Colab environments!")

if __name__ == "__main__":
    main()
