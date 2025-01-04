"""Test document processing functionality."""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, PropertyMock, call
import numpy as np
from src.documents import DocumentStore, DocumentProcessor, DocumentChunk, ProcessingState
from src.document_analyzer import Heading, DocumentSection

@pytest.fixture
def mock_document_analyzer():
    mock = Mock()
    # Configure analyze_document to return analysis results
    mock.analyze_document.return_value = {
        'classification': 'policies_procedures',
        'toc': [{'text': 'Test Document', 'level': 1, 'children': []}],
        'headings': [
            Heading(
                text='Test Document',
                level=1,
                start_pos=0,
                end_pos=12
            ),
            Heading(
                text='Section 1',
                level=2,
                start_pos=13,
                end_pos=22
            )
        ],
        'sections': [
            DocumentSection(
                text='Test Document\nThis is the introduction.',
                start_pos=0,
                end_pos=40,
                heading=Heading(
                    text='Test Document',
                    level=1,
                    start_pos=0,
                    end_pos=12
                )
            ),
            DocumentSection(
                text='Section 1\nThis is section 1 content.',
                start_pos=41,
                end_pos=80,
                heading=Heading(
                    text='Section 1',
                    level=2,
                    start_pos=41,
                    end_pos=50
                )
            )
        ]
    }
    return mock

class TestDocumentProcessor:
    def test_section_chunking(self, mock_document_analyzer):
        """Test chunking of sections with size control."""
        processor = DocumentProcessor(analyzer=mock_document_analyzer)
        
        # Create a section with text longer than chunk size (500)
        long_text = "This is a very long section that needs to be split into multiple chunks. " * 20  # Make it longer
        section = DocumentSection(
            text=long_text,
            start_pos=0,
            end_pos=len(long_text),
            heading=Heading(
                text="Long Section",
                level=1,
                start_pos=0,
                end_pos=11
            )
        )
        
        # Base metadata for testing
        metadata = {
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'file_type': 'pdf',
            'classification': 'policies_procedures',
            'toc': json.dumps([{'text': 'Test Document', 'level': 1, 'children': []}])
        }
        
        # Split section into chunks
        chunks = processor._split_section(section, metadata)
        
        # Verify chunks
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(len(chunk.text) <= 500 for chunk in chunks)  # Default chunk size
        
        # Verify metadata preservation
        for chunk in chunks:
            assert chunk.metadata['source_name'] == 'test.pdf'
            assert chunk.metadata['title'] == 'Test Document'
            assert chunk.metadata['section_title'] == 'Long Section'
            assert chunk.metadata['section_level'] == 1
            assert isinstance(chunk.metadata['chunk_index'], int)
            assert chunk.metadata['total_chunks'] == len(chunks)

    def test_chunk_size_settings(self, mock_document_analyzer):
        """Test chunk size configuration and updates."""
        processor = DocumentProcessor(analyzer=mock_document_analyzer)
        
        # Create test section with very long text
        section = DocumentSection(
            text="Test content " * 200,  # Make it much longer
            start_pos=0,
            end_pos=2400,
            heading=Heading(
                text="Test Section",
                level=1,
                start_pos=0,
                end_pos=11
            )
        )
        
        metadata = {
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'file_type': 'pdf'
        }
        
        # Test with default settings (500)
        chunks1 = processor._split_section(section, metadata)
        chunk_count1 = len(chunks1)
        assert all(len(chunk.text) <= 500 for chunk in chunks1)  # Verify default size limit
        
        # Update settings to smaller chunk size (100)
        processor.settings['document_processing']['chunk_size'] = 100
        processor.chunk_size = 100  # Update processor's chunk size
        processor._init_text_splitter()  # Reinitialize splitter with new size
        
        # Test with updated settings
        chunks2 = processor._split_section(section, metadata)
        chunk_count2 = len(chunks2)
        
        # Smaller chunk size should result in more chunks
        assert chunk_count2 > chunk_count1
        assert all(len(chunk.text) <= 100 for chunk in chunks2)  # Verify new size limit

    def test_pdf_processing(self, mock_document_analyzer):
        """Test processing of PDF documents."""
        processor = DocumentProcessor(analyzer=mock_document_analyzer)
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name
            
        try:
            # Mock PDF text extraction
            with patch('src.documents.DocumentProcessor._extract_pdf_text') as mock_extract:
                mock_extract.return_value = ('Test Document', 'Test content\nMore content')
                
                # Process document
                chunks = processor.process_document(test_pdf)
                
                # Verify chunks
                assert len(chunks) > 0
                assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
                assert all('source_name' in chunk.metadata for chunk in chunks)
                assert all('title' in chunk.metadata for chunk in chunks)
                assert all('classification' in chunk.metadata for chunk in chunks)
                assert all('toc' in chunk.metadata for chunk in chunks)
                
                # Verify document analysis was used
                mock_document_analyzer.analyze_document.assert_called_once()
                
        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_docx_processing(self, mock_document_analyzer):
        """Test processing of DOCX documents."""
        processor = DocumentProcessor(analyzer=mock_document_analyzer)
        
        # Create a temporary DOCX file
        with tempfile.NamedTemporaryFile(suffix='.docx', mode='w+b', delete=False) as f:
            test_docx = f.name
            
        try:
            # Mock DOCX text extraction
            with patch('src.documents.DocumentProcessor._extract_docx_text') as mock_extract:
                mock_extract.return_value = ('Test Document', 'Test content\nMore content')
                
                # Process document
                chunks = processor.process_document(test_docx)
                
                # Verify chunks
                assert len(chunks) > 0
                assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
                assert all('source_name' in chunk.metadata for chunk in chunks)
                assert all('title' in chunk.metadata for chunk in chunks)
                assert all('classification' in chunk.metadata for chunk in chunks)
                assert all('toc' in chunk.metadata for chunk in chunks)
                
                # Verify document analysis was used
                mock_document_analyzer.analyze_document.assert_called_once()
                
        finally:
            if os.path.exists(test_docx):
                os.remove(test_docx)

    def test_metadata_preservation(self, mock_document_analyzer):
        """Test preservation of metadata through processing pipeline."""
        processor = DocumentProcessor(analyzer=mock_document_analyzer)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_file = f.name
            
        try:
            # Mock text extraction
            with patch('src.documents.DocumentProcessor._extract_pdf_text') as mock_extract:
                mock_extract.return_value = ('Test Document', 'Test content\nMore content')
                
                # Process document
                chunks = processor.process_document(test_file)
                
                # Verify metadata in all chunks
                for chunk in chunks:
                    # Basic metadata
                    assert chunk.metadata['source_name'] == os.path.basename(test_file)
                    assert chunk.metadata['title'] == 'Test Document'
                    assert chunk.metadata['file_type'] == 'pdf'
                    
                    # Analysis metadata
                    assert chunk.metadata['classification'] == 'policies_procedures'
                    assert isinstance(chunk.metadata['toc'], str)  # JSON string
                    toc = json.loads(chunk.metadata['toc'])
                    assert isinstance(toc, list)
                    
                    # Section metadata
                    if 'section_title' in chunk.metadata:
                        assert isinstance(chunk.metadata['section_title'], str)
                        assert isinstance(chunk.metadata['section_level'], int)
                    
                    # Chunk metadata
                    assert isinstance(chunk.metadata['chunk_index'], int)
                    assert isinstance(chunk.metadata['total_chunks'], int)
                    assert 0 <= chunk.metadata['chunk_index'] < chunk.metadata['total_chunks']
                
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

class TestDocumentStore:
    def test_process_and_store_document(self, mock_document_analyzer):
        """Test complete document processing and storage pipeline."""
        # Create mock embeddings
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Create mock EmbeddingGenerator
        mock_embedding_generator = Mock()
        mock_embedding_generator.generate_embeddings.return_value = mock_embeddings
        
        # Create mock VectorDatabase
        mock_vector_db = Mock()
        mock_vector_db.get_document_chunks.side_effect = [
            [],  # First call for existing chunks
            [  # Second call for verification
                {
                    'id': 'chunk1',
                    'text': 'Test Document\nThis is the introduction.',
                    'chunk_index': 0,
                    'total_chunks': 2,
                    'classification': 'policies_procedures',
                    'toc': json.dumps([{'text': 'Test Document', 'level': 1, 'children': []}])
                },
                {
                    'id': 'chunk2',
                    'text': 'Section 1\nThis is section 1 content.',
                    'chunk_index': 1,
                    'total_chunks': 2,
                    'classification': 'policies_procedures',
                    'toc': json.dumps([{'text': 'Test Document', 'level': 1, 'children': []}])
                }
            ]
        ]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_file = f.name
            
        try:
            # Initialize store with mocked dependencies
            with patch('src.documents.EmbeddingGenerator', return_value=mock_embedding_generator):
                with patch('src.documents.VectorDatabase', return_value=mock_vector_db):
                    processor = DocumentProcessor(analyzer=mock_document_analyzer)
                    store = DocumentStore(processor=processor)
                    
                    # Mock text extraction
                    with patch('src.documents.DocumentProcessor._extract_pdf_text') as mock_extract:
                        mock_extract.return_value = ('Test Document', 'Test content\nMore content')
                        
                        # Process and store document
                        state = store.process_and_store_document(test_file)
                        
                        # Verify state
                        assert state.status == 'completed'
                        assert state.source_name == os.path.basename(test_file)
                        assert state.error is None
                        assert state.classification == 'policies_procedures'
                        assert isinstance(state.toc, list)
                        
                        # Verify embeddings were generated
                        mock_embedding_generator.generate_embeddings.assert_called_once()
                        
                        # Verify documents were stored
                        mock_vector_db.add_documents.assert_called_once()
                        
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_error_handling(self):
        """Test error handling in document processing."""
        store = DocumentStore()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            store.process_and_store_document("nonexistent.pdf")
        
        # Test with unsupported file type
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w+b', delete=False) as f:
            test_file = f.name
            
        try:
            with pytest.raises(ValueError) as exc_info:
                store.process_and_store_document(test_file)
            assert "Unsupported file type" in str(exc_info.value)
            
            # Verify error state
            state = store.get_processing_state(os.path.basename(test_file))
            assert state is not None
            assert state.status == 'error'
            assert state.error is not None
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_document_info_retrieval(self, mock_document_analyzer):
        """Test retrieval of document information."""
        # Create mock chunks
        mock_chunks = [{
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'chunk_index': 0,
            'total_chunks': 2,
            'classification': 'policies_procedures',
            'toc': json.dumps([{'text': 'Test Document', 'level': 1, 'children': []}])
        }]
        
        # Create mock VectorDatabase
        mock_vector_db = Mock()
        mock_vector_db.get_document_chunks.return_value = mock_chunks
        
        # Initialize store with mocked database
        with patch('src.documents.VectorDatabase', return_value=mock_vector_db):
            store = DocumentStore()
            
            # Get document info
            info = store.get_document_info('test.pdf')
            
            # Verify info
            assert info is not None
            assert info['source_name'] == 'test.pdf'
            assert info['title'] == 'Test Document'
            assert info['classification'] == 'policies_procedures'
            assert isinstance(info['toc'], list)  # Verify TOC is parsed from JSON
            assert info['chunk_count'] == 1
            assert info['total_chunks'] == 2
