import unittest
import numpy as np
from src.database import VectorDatabase
from config.settings import CHROMA_COLLECTION_NAME

class TestVectorDatabase(unittest.TestCase):
    def setUp(self):
        self.db = VectorDatabase()
        # Clear any existing data
        self.db.delete_collection()

    def test_metadata(self):
        # Add some test documents
        test_docs = [
            {
                "id": 1,
                "text": "Test document 1",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "source_name": "test1.pdf",
                "title": "Test Document One",
                "chunk_index": 0,
                "total_chunks": 2,
                "section_title": "Introduction",
                "section_type": "heading"
            },
            {
                "id": 2,
                "text": "Test document 2",
                "embedding": np.array([0.4, 0.5, 0.6]),
                "source_name": "test1.pdf",
                "title": "Test Document One",
                "chunk_index": 1,
                "total_chunks": 2,
                "section_title": "Introduction",
                "section_type": "body"
            },
            {
                "id": 3,
                "text": "Another document",
                "embedding": np.array([0.7, 0.8, 0.9]),
                "source_name": "test2.pdf",
                "title": "Test Document Two",
                "chunk_index": 0,
                "total_chunks": 1,
                "section_title": "Summary",
                "section_type": "heading"
            }
        ]
        self.db.add_documents(test_docs)

        # Test get_metadata
        metadata = self.db.get_metadata()
        self.assertEqual(metadata["name"], CHROMA_COLLECTION_NAME)
        self.assertEqual(metadata["count"], 3)

        # Test get_all_documents
        all_docs = self.db.get_all_documents()
        self.assertEqual(len(all_docs["ids"]), 3)
        self.assertEqual(len(all_docs["embeddings"]), 3)
        self.assertEqual(len(all_docs["metadatas"]), 3)
        
        # Verify section metadata is stored correctly
        self.assertEqual(all_docs["metadatas"][0]["section_title"], "Introduction")
        self.assertEqual(all_docs["metadatas"][0]["section_type"], "heading")
        self.assertEqual(all_docs["metadatas"][1]["section_title"], "Introduction")
        self.assertEqual(all_docs["metadatas"][1]["section_type"], "body")

    def test_document_listing(self):
        # Add test documents with source names and chunk information
        test_docs = [
            {
                "id": 1,
                "text": "Chapter 1",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "source_name": "test.pdf",
                "title": "Test Document",
                "chunk_index": 0,
                "total_chunks": 3,
                "section_title": "Chapter One",
                "section_type": "heading"
            },
            {
                "id": 2,
                "text": "Chapter 2",
                "embedding": np.array([0.4, 0.5, 0.6]),
                "source_name": "test.pdf",
                "title": "Test Document",
                "chunk_index": 1,
                "total_chunks": 3,
                "section_title": "Chapter Two",
                "section_type": "heading"
            },
            {
                "id": 3,
                "text": "Chapter 3",
                "embedding": np.array([0.7, 0.8, 0.9]),
                "source_name": "test.pdf",
                "title": "Test Document",
                "chunk_index": 2,
                "total_chunks": 3,
                "section_title": "Chapter Three",
                "section_type": "heading"
            }
        ]
        self.db.add_documents(test_docs)

        # Test list_document_names
        doc_names = self.db.list_document_names()
        self.assertEqual(len(doc_names), 1)  # Should have 1 unique document

        # Verify document information
        doc_info = doc_names[0]
        self.assertEqual(doc_info['source_name'], 'test.pdf')
        self.assertEqual(doc_info['title'], 'Test Document')
        self.assertEqual(doc_info['chunk_count'], 3)
        self.assertEqual(doc_info['total_chunks'], 3)

    def test_get_document_chunks(self):
        # Add test documents with source names and chunk information
        test_docs = [
            {
                "id": 1,
                "text": "First chunk",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "source_name": "test.pdf",
                "title": "Test Document",
                "chunk_index": 0,
                "total_chunks": 2,
                "section_title": "First Section",
                "section_type": "heading"
            },
            {
                "id": 2,
                "text": "Second chunk",
                "embedding": np.array([0.4, 0.5, 0.6]),
                "source_name": "test.pdf",
                "title": "Test Document",
                "chunk_index": 1,
                "total_chunks": 2,
                "section_title": "First Section",
                "section_type": "body"
            }
        ]
        self.db.add_documents(test_docs)

        # Test getting chunks for a specific document
        chunks = self.db.get_document_chunks("test.pdf")
        self.assertEqual(len(chunks), 2)
        
        # Verify chunks are in correct order
        self.assertEqual(chunks[0]['chunk_index'], 0)
        self.assertEqual(chunks[0]['text'], "First chunk")
        self.assertEqual(chunks[0]['title'], "Test Document")
        self.assertEqual(chunks[0]['section_title'], "First Section")
        self.assertEqual(chunks[0]['section_type'], "heading")
        self.assertEqual(chunks[1]['chunk_index'], 1)
        self.assertEqual(chunks[1]['text'], "Second chunk")
        self.assertEqual(chunks[1]['title'], "Test Document")
        self.assertEqual(chunks[1]['section_title'], "First Section")
        self.assertEqual(chunks[1]['section_type'], "body")

        # Test getting chunks for non-existent document
        empty_chunks = self.db.get_document_chunks("nonexistent.pdf")
        self.assertEqual(len(empty_chunks), 0)

    def test_edge_cases(self):
        # Test empty database
        doc_names = self.db.list_document_names()
        self.assertEqual(len(doc_names), 0)

        chunks = self.db.get_document_chunks("any.pdf")
        self.assertEqual(len(chunks), 0)

        # Clear the collection before next test
        self.db.delete_collection()
        
        # Test documents without source names
        test_docs = [
            {
                "id": 1,
                "text": "Test document",
                "embedding": np.array([0.1, 0.2, 0.3])
            }
        ]
        self.db.add_documents(test_docs)

        doc_names = self.db.list_document_names()
        self.assertEqual(len(doc_names), 1)
        self.assertEqual(doc_names[0]['source_name'], "Unknown")
        self.assertEqual(doc_names[0]['title'], "")

        # Clear the collection before next test
        self.db.delete_collection()
        
        # Test documents with missing chunk information
        test_docs = [
            {
                "id": 2,
                "text": "Test document",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "source_name": "test.pdf"
            }
        ]
        self.db.add_documents(test_docs)

        chunks = self.db.get_document_chunks("test.pdf")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['chunk_index'], 0)  # Default value
        self.assertEqual(chunks[0]['total_chunks'], 1)  # Default value
        self.assertEqual(chunks[0]['section_title'], "")  # Default value
        self.assertEqual(chunks[0]['section_type'], "content")  # Default value

        # Clear the collection before next test
        self.db.delete_collection()
        
        # Test documents with missing section metadata
        test_docs = [
            {
                "id": 3,
                "text": "Test document",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "source_name": "test.pdf",
                "title": "Test Document"
            }
        ]
        self.db.add_documents(test_docs)

        chunks = self.db.get_document_chunks("test.pdf")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['section_title'], "")  # Default value
        self.assertEqual(chunks[0]['section_type'], "content")  # Default value

if __name__ == "__main__":
    unittest.main()
