import unittest
import numpy as np
from src.database import VectorDatabase

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
                "embedding": np.array([0.1, 0.2, 0.3])
            },
            {
                "id": 2,
                "text": "Test document 2",
                "embedding": np.array([0.4, 0.5, 0.6])
            }
        ]
        self.db.add_documents(test_docs)

        # Test get_metadata
        metadata = self.db.get_metadata()
        self.assertEqual(metadata["name"], "document_collection")  # Default collection name
        self.assertEqual(metadata["count"], 2)  # Should have 2 documents

        # Test get_all_documents
        all_docs = self.db.get_all_documents()
        self.assertEqual(len(all_docs["ids"]), 2)  # Should have 2 documents
        self.assertEqual(len(all_docs["embeddings"]), 2)  # Should have 2 embeddings
        self.assertEqual(len(all_docs["metadatas"]), 2)  # Should have 2 metadata entries

        # Verify document content
        self.assertIn("1", all_docs["ids"])  # IDs are converted to strings
        self.assertIn("2", all_docs["ids"])
        self.assertEqual(all_docs["metadatas"][0]["text"], "Test document 1")
        self.assertEqual(all_docs["metadatas"][1]["text"], "Test document 2")

if __name__ == "__main__":
    unittest.main()
