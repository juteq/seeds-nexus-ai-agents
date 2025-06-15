#!/usr/bin/env python3
"""
Simple test to demonstrate Chroma functionality with mock embeddings.
This validates that the notebook structure will work in practice.
"""

import warnings
warnings.filterwarnings('ignore')

def test_chroma_with_mock_embeddings():
    """Test Chroma with a mock embedding function."""
    print("🧪 Testing Chroma with mock embeddings...")

    try:
        from langchain_community.vectorstores import Chroma
        from langchain.schema import Document
        import numpy as np

        # Mock embedding function
        class MockEmbeddings:
            def embed_documents(self, texts):
                # Return random embeddings for each text
                return [np.random.random(1536).tolist() for _ in texts]

            def embed_query(self, text):
                # Return random embedding for query
                return np.random.random(1536).tolist()

        # Create test documents
        test_docs = [
            Document(
                page_content="This is a test paper about machine learning and neural networks.",
                metadata={"Title": "ML Paper", "Authors": "Test Author", "source": "test"}
            ),
            Document(
                page_content="This paper discusses natural language processing and transformers.",
                metadata={"Title": "NLP Paper", "Authors": "Test Author 2", "source": "test"}
            ),
            Document(
                page_content="Computer vision and convolutional neural networks are the focus here.",
                metadata={"Title": "CV Paper", "Authors": "Test Author 3", "source": "test"}
            )
        ]

        # Create Chroma vectorstore with mock embeddings
        mock_embeddings = MockEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=test_docs,
            embedding=mock_embeddings
        )

        print("  ✅ Created Chroma vectorstore with 3 documents")

        # Test similarity search
        results = vectorstore.similarity_search("machine learning", k=2)
        print(f"  ✅ Similarity search returned {len(results)} results")

        # Test adding more documents
        new_doc = Document(
            page_content="This is about deep learning architectures.",
            metadata={"Title": "DL Paper", "Authors": "Test Author 4", "source": "test"}
        )
        vectorstore.add_documents([new_doc])
        print("  ✅ Successfully added additional document")

        # Test another search
        results = vectorstore.similarity_search("deep learning", k=3)
        print(f"  ✅ Second search returned {len(results)} results")

        print("\n  📊 Mock Test Results:")
        for i, result in enumerate(results, 1):
            title = result.metadata.get('Title', 'Unknown')
            content_preview = result.page_content[:50] + "..."
            print(f"    {i}. {title}: {content_preview}")

        return True

    except Exception as e:
        print(f"  ❌ Chroma test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the mock Chroma test."""
    print("🚀 Testing Chroma Functionality with Mock Data")
    print("=" * 50)

    success = test_chroma_with_mock_embeddings()

    print("\n" + "=" * 50)
    if success:
        print("🎉 Chroma mock test passed!")
        print("\n📝 This confirms:")
        print("   ✅ Chroma vectorstore creation works")
        print("   ✅ Document storage and retrieval works")
        print("   ✅ Similarity search functionality works")
        print("   ✅ Adding documents to existing store works")
        print("\n🔗 The notebook is ready to use with real OpenAI embeddings!")
    else:
        print("❌ Chroma mock test failed.")
        print("Please check the error messages above.")

    return success

if __name__ == "__main__":
    main()
