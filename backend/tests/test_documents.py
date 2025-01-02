"""Test document processing functionality."""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, PropertyMock
import numpy as np
from rag_backend.documents import DocumentStore, DocumentProcessor, DocumentChunk, ProcessingState

@pytest.fixture
def mock_extract_text():
    with patch('rag_backend.documents.DocumentProcessor._extract_pdf_text') as mock:
        yield mock

class TestProcessingState:
    def test_processing_state_initialization(self):
        """Test ProcessingState initialization and properties."""
        state = ProcessingState(status='processing')
        assert state.status == 'processing'
        assert state.error is None
        assert state.source_name is None
        assert state.chunk_count == 0
        assert state.total_chunks == 0

    def test_processing_state_updates(self):
        """Test ProcessingState updates."""
        state = ProcessingState(status='processing')
        state.status = 'completed'
        state.source_name = 'test.pdf'
        state.chunk_count = 5
        state.total_chunks = 5
        
        assert state.status == 'completed'
        assert state.source_name == 'test.pdf'
        assert state.chunk_count == 5
        assert state.total_chunks == 5

class TestDocumentStore:
    @pytest.mark.usefixtures("mock_extract_text")
    def test_process_and_store_document_with_cleanup(self, mock_extract_text):
        """Test processing and storing a document."""
        # Create mock embeddings
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name
            test_pdf_name = os.path.basename(test_pdf)

        # Configure mock PDF extraction with actual temp filename
        mock_extract_text.return_value = [
            {
                'text': 'Test section 1',
                'metadata': {
                    'source_name': test_pdf_name,
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'section_type': 'content',
                    'chunk_index': 0,
                    'total_chunks': 2
                }
            },
            {
                'text': 'Test section 2',
                'metadata': {
                    'source_name': test_pdf_name,
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'section_type': 'content',
                    'chunk_index': 1,
                    'total_chunks': 2
                }
            }
        ]

        # Create mock EmbeddingGenerator
        mock_embedding_generator = Mock()
        mock_embedding_generator.generate_embeddings.return_value = mock_embeddings

        # Create mock VectorDatabase
        mock_vector_db = Mock()
        mock_vector_db.get_document_chunks.return_value = [
            {
                'id': 1, 
                'text': 'Test section 1',
                'chunk_index': 0,
                'total_chunks': 2
            },
            {
                'id': 2, 
                'text': 'Test section 2',
                'chunk_index': 1,
                'total_chunks': 2
            }
        ]

        try:
            # Initialize store with mocked dependencies
            with patch('rag_backend.documents.EmbeddingGenerator', return_value=mock_embedding_generator):
                with patch('rag_backend.documents.VectorDatabase', return_value=mock_vector_db):
                    store = DocumentStore()
                    state = store.process_and_store_document(test_pdf)

            # Verify state tracking
            assert state.status == 'completed'
            assert state.source_name == test_pdf_name
            assert state.chunk_count == 2
            assert state.total_chunks == 2
            assert state.error is None

            # Verify existing documents were checked and deleted
            mock_vector_db.get_document_chunks.assert_called()
            mock_vector_db.collection.delete.assert_called()

            # Verify chunk consistency was checked
            stored_chunks = mock_vector_db.get_document_chunks.return_value
            assert len(stored_chunks) == 2
            assert all(chunk['chunk_index'] in [0, 1] for chunk in stored_chunks)
            assert all(chunk['total_chunks'] == 2 for chunk in stored_chunks)

            # Verify embeddings were generated once
            mock_embedding_generator.generate_embeddings.assert_called_once()

            # Verify documents were added to database
            mock_vector_db.add_documents.assert_called_once()
            added_docs = mock_vector_db.add_documents.call_args[0][0]
            assert len(added_docs) == 2
            assert added_docs[0]['text'] == 'Test section 1'
            assert added_docs[1]['text'] == 'Test section 2'

        finally:
            # Clean up
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_processing_error_handling(self):
        """Test error handling in document processing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name

        try:
            # Mock an error during processing
            with patch('rag_backend.documents.DocumentProcessor.process_document', side_effect=ValueError("Processing failed")):
                store = DocumentStore()
                with pytest.raises(ValueError):
                    state = store.process_and_store_document(test_pdf)
                
                # Verify error state
                state = store.get_processing_state(os.path.basename(test_pdf))
                assert state is not None
                assert state.status == 'error'
                assert state.error == "Processing failed"

        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_get_documents(self):
        """Test retrieving documents from the store."""
        # Create mock documents
        mock_docs = [
            {
                'source_name': 'test1.pdf',
                'title': 'Test Document 1',
                'chunk_count': 5,
                'total_chunks': 5
            },
            {
                'source_name': 'test2.pdf',
                'title': 'Test Document 2',
                'chunk_count': 3,
                'total_chunks': 3
            }
        ]

        # Create mock VectorDatabase
        mock_vector_db = Mock()
        mock_vector_db.get_all_documents = Mock(return_value=mock_docs)

        # Initialize store with mocked database
        with patch('rag_backend.documents.VectorDatabase', return_value=mock_vector_db):
            store = DocumentStore()
            documents = store.get_documents()

            # Verify documents were retrieved
            assert len(documents) == 2
            assert documents[0]['source_name'] == 'test1.pdf'
            assert documents[1]['source_name'] == 'test2.pdf'

class TestDocumentProcessor:
    def test_process_document(self):
        """Test document processing."""
        processor = DocumentProcessor()

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name

        # Configure mock PDF extraction
        mock_sections = [
            {
                'text': 'Test section 1',
                'metadata': {
                    'source_name': 'test.pdf',
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'section_type': 'content',
                    'chunk_index': 0,
                    'total_chunks': 2
                }
            },
            {
                'text': 'Test section 2',
                'metadata': {
                    'source_name': 'test.pdf',
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'section_type': 'content',
                    'chunk_index': 1,
                    'total_chunks': 2
                }
            }
        ]

        try:
            with patch.object(processor, '_extract_pdf_text', return_value=mock_sections):
                chunks = processor.process_document(test_pdf)

            # Verify chunks were created
            assert len(chunks) == 2
            assert isinstance(chunks[0], DocumentChunk)
            assert isinstance(chunks[1], DocumentChunk)
            
            # Verify chunk content
            assert chunks[0].text == 'Test section 1'
            assert chunks[1].text == 'Test section 2'

            # Verify metadata
            for chunk in chunks:
                assert chunk.metadata['source_name'] == 'test.pdf'
                assert chunk.metadata['title'] == 'Test Document'
                assert chunk.metadata['file_type'] == 'pdf'
                assert chunk.metadata['section_type'] == 'content'

        finally:
            # Clean up
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_chunk_consistency_error(self, mock_extract_text):
        """Test error handling for inconsistent chunks."""
        # Create mock embeddings with wrong total_chunks
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name
            test_pdf_name = os.path.basename(test_pdf)

        # Configure mock PDF extraction with inconsistent total_chunks
        mock_extract_text.return_value = [
            {
                'text': 'Test section 1',
                'metadata': {
                    'source_name': test_pdf_name,
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'section_type': 'content',
                    'chunk_index': 0,
                    'total_chunks': 3  # Wrong total
                }
            },
            {
                'text': 'Test section 2',
                'metadata': {
                    'source_name': test_pdf_name,
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'section_type': 'content',
                    'chunk_index': 1,
                    'total_chunks': 3  # Wrong total
                }
            }
        ]

        # Create mock dependencies
        mock_embedding_generator = Mock()
        mock_embedding_generator.generate_embeddings.return_value = mock_embeddings

        mock_vector_db = Mock()
        mock_vector_db.get_document_chunks.return_value = [
            {'id': 1, 'text': 'Test section 1', 'chunk_index': 0, 'total_chunks': 3},
            {'id': 2, 'text': 'Test section 2', 'chunk_index': 1, 'total_chunks': 3}
        ]

        try:
            # Initialize store with mocked dependencies
            with patch('rag_backend.documents.EmbeddingGenerator', return_value=mock_embedding_generator):
                with patch('rag_backend.documents.VectorDatabase', return_value=mock_vector_db):
                    store = DocumentStore()
                    with pytest.raises(ValueError) as exc_info:
                        state = store.process_and_store_document(test_pdf)
                    assert "total_chunks mismatch" in str(exc_info.value)

        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_error_handling(self):
        """Test error handling in document operations."""
        processor = DocumentProcessor()

        # Test invalid file type
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w+b', delete=False) as f:
            test_file = f.name
            f.write(b'Test content')

        try:
            with pytest.raises(Exception) as exc_info:
                processor.process_document(test_file)
            assert "Unsupported file type" in str(exc_info.value)
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)

        # Test file not found
        nonexistent_file = "nonexistent.pdf"
        with patch('rag_backend.documents.DocumentProcessor._extract_pdf_text', side_effect=FileNotFoundError("No such file or directory: 'nonexistent.pdf'")):
            with pytest.raises(FileNotFoundError) as exc_info:
                processor.process_document(nonexistent_file)
            assert "File not found" in str(exc_info.value)
