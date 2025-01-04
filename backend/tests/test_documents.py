"""Test document processing functionality."""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, PropertyMock
import numpy as np
from src.documents import DocumentStore, DocumentProcessor, DocumentChunk, ProcessingState

@pytest.fixture
def mock_extract_text():
    with patch('src.documents.DocumentProcessor._extract_pdf_text') as mock:
        mock.return_value = (
            'Test Document',  # title
            'Test section 1\nTest section 2',  # full_text
            [  # sections
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
        )
        yield mock

@pytest.fixture
def mock_document_analyzer():
    mock = Mock()
    mock.analyze_document.return_value = {
        'classification': 'policies_procedures',
        'toc': [{'text': 'Test Document', 'level': 1, 'children': []}],
        'headings': [{'text': 'Test Document', 'level': 1, 'start_pos': 0, 'end_pos': 12}]
    }
    return mock

class TestProcessingState:
    def test_processing_state_initialization(self):
        """Test ProcessingState initialization and properties."""
        state = ProcessingState(status='processing')
        assert state.status == 'processing'
        assert state.error is None
        assert state.source_name is None
        assert state.chunk_count == 0
        assert state.total_chunks == 0
        assert state.classification is None
        assert state.toc is None

    def test_processing_state_updates(self):
        """Test ProcessingState updates."""
        state = ProcessingState(status='processing')
        state.status = 'completed'
        state.source_name = 'test.pdf'
        state.chunk_count = 5
        state.total_chunks = 5
        state.classification = 'policies_procedures'
        state.toc = [{'text': 'Test', 'level': 1, 'children': []}]
        
        assert state.status == 'completed'
        assert state.source_name == 'test.pdf'
        assert state.chunk_count == 5
        assert state.total_chunks == 5
        assert state.classification == 'policies_procedures'
        assert len(state.toc) == 1

class TestDocumentStore:
    def test_process_and_store_document_with_cleanup(self, mock_extract_text, mock_document_analyzer):
        """Test processing and storing a document."""
        # Create mock embeddings
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name
            test_pdf_name = os.path.basename(test_pdf)

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
                'total_chunks': 2,
                'classification': 'policies_procedures',
                'toc': json.dumps([{'text': 'Test Document', 'level': 1, 'children': []}])  # TOC as JSON string
            },
            {
                'id': 2,
                'text': 'Test section 2',
                'chunk_index': 1,
                'total_chunks': 2,
                'classification': 'policies_procedures',
                'toc': json.dumps([{'text': 'Test Document', 'level': 1, 'children': []}])  # TOC as JSON string
            }
        ]

        try:
            # Initialize store with mocked dependencies
            with patch('src.documents.EmbeddingGenerator', return_value=mock_embedding_generator):
                with patch('src.documents.VectorDatabase', return_value=mock_vector_db):
                    # Create processor with mock analyzer
                    processor = DocumentProcessor(analyzer=mock_document_analyzer)
                    store = DocumentStore(processor=processor)
                    state = store.process_and_store_document(test_pdf)

            # Verify state tracking
            assert state.status == 'completed'
            assert state.source_name == test_pdf_name
            assert state.chunk_count == 2
            assert state.total_chunks == 2
            assert state.error is None
            assert state.classification == 'policies_procedures'
            assert len(state.toc) == 1
            assert state.toc[0]['text'] == 'Test Document'

            # Verify document analysis was called
            mock_document_analyzer.analyze_document.assert_called_once()

            # Verify existing documents were checked and deleted
            mock_vector_db.get_document_chunks.assert_called()
            mock_vector_db.collection.delete.assert_called()

            # Verify chunk consistency was checked
            stored_chunks = mock_vector_db.get_document_chunks.return_value
            assert len(stored_chunks) == 2
            assert all(chunk['chunk_index'] in [0, 1] for chunk in stored_chunks)
            assert all(chunk['total_chunks'] == 2 for chunk in stored_chunks)
            assert all(chunk['classification'] == 'policies_procedures' for chunk in stored_chunks)
            assert all('toc' in chunk for chunk in stored_chunks)
            # Verify TOC is stored as JSON string
            assert all(isinstance(chunk['toc'], str) for chunk in stored_chunks)
            # Verify TOC can be parsed back to list
            assert all(isinstance(json.loads(chunk['toc']), list) for chunk in stored_chunks)

            # Verify embeddings were generated once
            mock_embedding_generator.generate_embeddings.assert_called_once()

            # Verify documents were added to database
            mock_vector_db.add_documents.assert_called_once()
            added_docs = mock_vector_db.add_documents.call_args[0][0]
            assert len(added_docs) == 2
            assert added_docs[0]['text'] == 'Test section 1'
            assert added_docs[1]['text'] == 'Test section 2'
            assert all('classification' in doc for doc in added_docs)
            assert all('toc' in doc for doc in added_docs)
            # Verify TOC is stored as JSON string
            assert all(isinstance(doc['toc'], str) for doc in added_docs)
            # Verify TOC can be parsed back to list
            assert all(isinstance(json.loads(doc['toc']), list) for doc in added_docs)

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
            with patch('src.documents.DocumentProcessor.process_document', side_effect=ValueError("Processing failed")):
                store = DocumentStore()
                with pytest.raises(ValueError):
                    state = store.process_and_store_document(test_pdf)
                
                # Verify error state
                state = store.get_processing_state(os.path.basename(test_pdf))
                assert state is not None
                assert state.status == 'error'
                assert state.error == "Processing failed"
                assert state.classification is None
                assert state.toc is None

        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_get_documents(self):
        """Test retrieving documents from the store."""
        # Create mock documents with TOC as JSON string
        mock_docs = [
            {
                'source_name': 'test1.pdf',
                'title': 'Test Document 1',
                'chunk_count': 5,
                'total_chunks': 5,
                'classification': 'policies_procedures',
                'toc': json.dumps([{'text': 'Test Document 1', 'level': 1, 'children': []}])
            },
            {
                'source_name': 'test2.pdf',
                'title': 'Test Document 2',
                'chunk_count': 3,
                'total_chunks': 3,
                'classification': 'human_resources',
                'toc': json.dumps([{'text': 'Test Document 2', 'level': 1, 'children': []}])
            }
        ]

        # Create mock VectorDatabase
        mock_vector_db = Mock()
        mock_vector_db.get_all_documents = Mock(return_value=mock_docs)

        # Initialize store with mocked database
        with patch('src.documents.VectorDatabase', return_value=mock_vector_db):
            store = DocumentStore()
            documents = store.get_documents()

            # Verify documents were retrieved
            assert len(documents) == 2
            assert documents[0]['source_name'] == 'test1.pdf'
            assert documents[1]['source_name'] == 'test2.pdf'
            assert all('classification' in doc for doc in documents)
            assert all('toc' in doc for doc in documents)
            # Verify TOC is stored as JSON string
            assert all(isinstance(doc['toc'], str) for doc in documents)
            # Verify TOC can be parsed back to list
            assert all(isinstance(json.loads(doc['toc']), list) for doc in documents)

    def test_get_document_info_toc_handling(self):
        """Test TOC handling in get_document_info."""
        # Create mock chunks with TOC as JSON string
        mock_chunks = [{
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'chunk_count': 1,
            'total_chunks': 1,
            'classification': 'policies_procedures',
            'toc': json.dumps([{'text': 'Test Document', 'level': 1, 'children': []}])
        }]

        # Create mock VectorDatabase
        mock_vector_db = Mock()
        mock_vector_db.get_document_chunks = Mock(return_value=mock_chunks)

        # Initialize store with mocked database
        with patch('src.documents.VectorDatabase', return_value=mock_vector_db):
            store = DocumentStore()
            doc_info = store.get_document_info('test.pdf')

            # Verify document info
            assert doc_info is not None
            assert doc_info['source_name'] == 'test.pdf'
            assert doc_info['title'] == 'Test Document'
            assert doc_info['classification'] == 'policies_procedures'
            # Verify TOC is parsed back to list
            assert isinstance(doc_info['toc'], list)
            assert len(doc_info['toc']) == 1
            assert doc_info['toc'][0]['text'] == 'Test Document'

class TestDocumentProcessor:
    def test_create_sections_from_analysis(self):
        """Test section creation with and without headings."""
        processor = DocumentProcessor()
        
        # Test data
        full_text = "Title\n1. First Section\nContent\n1.1. Subsection\nMore content"
        file_path = "test.pdf"
        title = "Test Document"
        
        # Test with headings
        analysis = {
            'headings': [
                {'text': 'Title', 'level': 1, 'start_pos': 0, 'end_pos': 5},
                {'text': '1. First Section', 'level': 1, 'start_pos': 6, 'end_pos': 20},
                {'text': '1.1. Subsection', 'level': 2, 'start_pos': 28, 'end_pos': 41}
            ]
        }
        
        # Test with PDF fallback
        fallback_sections = ["Page 1 content", "Page 2 content"]
        sections = processor._create_sections_from_analysis(
            full_text=full_text,
            analysis=analysis,
            file_path=file_path,
            title=title,
            file_type='pdf',
            fallback_sections=fallback_sections
        )
        
        # Verify heading-based sections
        assert len(sections) == 3
        assert sections[0]['metadata']['section_title'] == 'Title'
        assert sections[1]['metadata']['section_title'] == '1. First Section'
        assert sections[2]['metadata']['section_title'] == '1.1. Subsection'
        
        # Test without headings (fallback)
        analysis_no_headings = {'headings': []}
        fallback_sections = ["Page 1", "Page 2"]
        sections = processor._create_sections_from_analysis(
            full_text=full_text,
            analysis=analysis_no_headings,
            file_path=file_path,
            title=title,
            file_type='pdf',
            fallback_sections=fallback_sections
        )
        
        # Verify fallback sections
        assert len(sections) == 2
        assert sections[0]['metadata']['chunk_index'] == 0
        assert sections[1]['metadata']['chunk_index'] == 1
        assert sections[0]['metadata']['total_chunks'] == 2
        assert 'section_title' not in sections[0]['metadata']

    def test_consistent_section_handling(self, mock_document_analyzer):
        """Test consistent section handling across document types."""
        processor = DocumentProcessor()
        
        # Configure mock analyzer with same headings for both formats
        mock_document_analyzer.analyze_document.return_value = {
            'classification': 'policies_procedures',
            'toc': [
                {
                    'text': 'Test Document',
                    'level': 1,
                    'children': [
                        {
                            'text': '1. First Section',
                            'level': 1,
                            'children': [
                                {
                                    'text': '1.1. Subsection',
                                    'level': 2,
                                    'children': []
                                }
                            ]
                        }
                    ]
                }
            ],
            'headings': [
                {
                    'text': 'Test Document',
                    'level': 1,
                    'start_pos': 0,
                    'end_pos': 12
                },
                {
                    'text': '1. First Section',
                    'level': 1,
                    'start_pos': 13,
                    'end_pos': 30
                },
                {
                    'text': '1.1. Subsection',
                    'level': 2,
                    'start_pos': 31,
                    'end_pos': 50
                }
            ]
        }
        
        # Test PDF processing
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            pdf_path = f.name
            
        # Test DOCX processing
        with tempfile.NamedTemporaryFile(suffix='.docx', mode='w+b', delete=False) as f:
            docx_path = f.name
            
        try:
            # Configure mock extractors with section titles
            pdf_extract = (
                'Test Document',
                'Test Document\n1. First Section\n1.1. Subsection',
                [
                    {
                        'text': 'Test Document',
                        'metadata': {
                            'source_name': 'test.pdf',
                            'title': 'Test Document',
                            'file_type': 'pdf',
                            'section_type': 'content',
                            'section_title': 'Test Document',
                            'chunk_index': 0,
                            'total_chunks': 3
                        }
                    },
                    {
                        'text': '1. First Section',
                        'metadata': {
                            'source_name': 'test.pdf',
                            'title': 'Test Document',
                            'file_type': 'pdf',
                            'section_type': 'content',
                            'section_title': '1. First Section',
                            'chunk_index': 1,
                            'total_chunks': 3
                        }
                    },
                    {
                        'text': '1.1. Subsection',
                        'metadata': {
                            'source_name': 'test.pdf',
                            'title': 'Test Document',
                            'file_type': 'pdf',
                            'section_type': 'content',
                            'section_title': '1.1. Subsection',
                            'chunk_index': 2,
                            'total_chunks': 3
                        }
                    }
                ]
            )
            
            docx_extract = (
                'Test Document',
                'Test Document\n1. First Section\n1.1. Subsection',
                [
                    {
                        'text': 'Test Document',
                        'metadata': {
                            'source_name': 'test.docx',
                            'title': 'Test Document',
                            'file_type': 'docx',
                            'section_type': 'content',
                            'section_title': 'Test Document',
                            'chunk_index': 0,
                            'total_chunks': 3
                        }
                    },
                    {
                        'text': '1. First Section',
                        'metadata': {
                            'source_name': 'test.docx',
                            'title': 'Test Document',
                            'file_type': 'docx',
                            'section_type': 'content',
                            'section_title': '1. First Section',
                            'chunk_index': 1,
                            'total_chunks': 3
                        }
                    },
                    {
                        'text': '1.1. Subsection',
                        'metadata': {
                            'source_name': 'test.docx',
                            'title': 'Test Document',
                            'file_type': 'docx',
                            'section_type': 'content',
                            'section_title': '1.1. Subsection',
                            'chunk_index': 2,
                            'total_chunks': 3
                        }
                    }
                ]
            )
            
            with patch.object(processor, '_extract_pdf_text', return_value=pdf_extract):
                with patch.object(processor, '_extract_docx_text', return_value=docx_extract):
                    # Process both formats
                    pdf_chunks = processor.process_document(pdf_path)
                    docx_chunks = processor.process_document(docx_path)
                    
                    # Verify consistent section handling
                    assert len(pdf_chunks) == len(docx_chunks)
                    
                    for pdf_chunk, docx_chunk in zip(pdf_chunks, docx_chunks):
                        # Verify section titles are preserved
                        assert pdf_chunk.metadata['section_title'] == docx_chunk.metadata['section_title']
                        # Verify section numbers are preserved
                        if '1.' in pdf_chunk.metadata['section_title']:
                            assert '1.' in docx_chunk.metadata['section_title']
                        # Verify metadata structure
                        assert set(pdf_chunk.metadata.keys()) == set(docx_chunk.metadata.keys())
                        assert pdf_chunk.metadata['total_chunks'] == docx_chunk.metadata['total_chunks']
                        
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists(docx_path):
                os.remove(docx_path)

    def test_process_document(self, mock_document_analyzer):
        """Test document processing."""
        # Create processor with mock analyzer
        processor = DocumentProcessor(analyzer=mock_document_analyzer)

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name

        # Configure mock document analysis with sections
        mock_document_analyzer.analyze_document.return_value = {
            'classification': 'policies_procedures',
            'toc': [
                {
                    'text': 'Test Document',
                    'level': 1,
                    'children': [
                        {
                            'text': '1. First Section',
                            'level': 1,
                            'children': [
                                {
                                    'text': '1.1. Subsection One',
                                    'level': 2,
                                    'children': []
                                }
                            ]
                        }
                    ]
                }
            ],
            'headings': [
                {
                    'text': 'Test Document',
                    'level': 1,
                    'start_pos': 0,
                    'end_pos': 12
                },
                {
                    'text': '1. First Section',
                    'level': 1,
                    'start_pos': 13,
                    'end_pos': 30
                },
                {
                    'text': '1.1. Subsection One',
                    'level': 2,
                    'start_pos': 31,
                    'end_pos': 50
                }
            ]
        }

        # Configure mock PDF extraction with sections
        mock_extract = (
            'Test Document',  # title
            'Test Document\n1. First Section\n1.1. Subsection One',  # full_text
            [  # sections
                {
                    'text': 'Test Document\nThis is the introduction.',
                    'metadata': {
                        'source_name': 'test.pdf',
                        'title': 'Test Document',
                        'file_type': 'pdf',
                        'section_type': 'content',
                        'section_title': 'Test Document',
                        'chunk_index': 0,
                        'total_chunks': 3
                    }
                },
                {
                    'text': '1. First Section\nThis is the first main section.',
                    'metadata': {
                        'source_name': 'test.pdf',
                        'title': 'Test Document',
                        'file_type': 'pdf',
                        'section_type': 'content',
                        'section_title': '1. First Section',
                        'chunk_index': 1,
                        'total_chunks': 3
                    }
                },
                {
                    'text': '1.1. Subsection One\nThis is a subsection.',
                    'metadata': {
                        'source_name': 'test.pdf',
                        'title': 'Test Document',
                        'file_type': 'pdf',
                        'section_type': 'content',
                        'section_title': '1.1. Subsection One',
                        'chunk_index': 2,
                        'total_chunks': 3
                    }
                }
            ]
        )

        try:
            with patch.object(processor, '_extract_pdf_text', return_value=mock_extract):
                chunks = processor.process_document(test_pdf)

            # Verify document analysis was called
            mock_document_analyzer.analyze_document.assert_called_once()

            # Verify chunks were created with proper sections
            assert len(chunks) == 3
            assert isinstance(chunks[0], DocumentChunk)
            assert isinstance(chunks[1], DocumentChunk)
            assert isinstance(chunks[2], DocumentChunk)
            
            # Verify chunk content and section titles
            assert chunks[0].metadata['section_title'] == 'Test Document'
            assert chunks[1].metadata['section_title'] == '1. First Section'
            assert chunks[2].metadata['section_title'] == '1.1. Subsection One'
            
            # Verify section numbers are preserved
            assert '1.' in chunks[1].metadata['section_title']
            assert '1.1.' in chunks[2].metadata['section_title']

            # Verify common metadata
            for i, chunk in enumerate(chunks):
                assert chunk.metadata['source_name'] == 'test.pdf'
                assert chunk.metadata['title'] == 'Test Document'
                assert chunk.metadata['file_type'] == 'pdf'
                assert chunk.metadata['section_type'] == 'content'
                assert chunk.metadata['classification'] == 'policies_procedures'
                assert 'toc' in chunk.metadata
                # Verify TOC is stored as JSON string
                assert isinstance(chunk.metadata['toc'], str)
                # Verify TOC can be parsed back to list
                toc = json.loads(chunk.metadata['toc'])
                assert isinstance(toc, list)
                assert len(toc) == 1
                assert toc[0]['text'] == 'Test Document'

        finally:
            # Clean up
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_chunk_consistency_error(self, mock_extract_text, mock_document_analyzer):
        """Test error handling for inconsistent chunks."""
        # Create mock embeddings with wrong total_chunks
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name
            test_pdf_name = os.path.basename(test_pdf)

        # Configure mock PDF extraction with inconsistent total_chunks
        mock_extract_text.return_value = (
            'Test Document',  # title
            'Test section 1\nTest section 2',  # full_text
            [  # sections
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
        )

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
            with patch('src.documents.EmbeddingGenerator', return_value=mock_embedding_generator):
                with patch('src.documents.VectorDatabase', return_value=mock_vector_db):
                    # Create processor with mock analyzer
                    processor = DocumentProcessor(analyzer=mock_document_analyzer)
                    store = DocumentStore(processor=processor)
                    with pytest.raises(ValueError) as exc_info:
                        state = store.process_and_store_document(test_pdf)
                    assert "total_chunks mismatch" in str(exc_info.value)

        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_get_title_from_content(self):
        """Test title extraction from content with various edge cases."""
        processor = DocumentProcessor()
        
        # Test empty content
        assert processor._get_title_from_content("") is None
        assert processor._get_title_from_content(None) is None
        assert processor._get_title_from_content("   ") is None
        
        # Test content with only whitespace and newlines
        assert processor._get_title_from_content("\n\n\t  \n") is None
        
        # Test valid title
        assert processor._get_title_from_content("Valid Title\nContent below") == "Valid Title"
        
        # Test title with punctuation (should not be considered a title)
        assert processor._get_title_from_content("Not a title.\nContent") is None
        
        # Test lowercase start (should not be considered a title)
        assert processor._get_title_from_content("lowercase start\nContent") is None
        
        # Test excluded starts
        assert processor._get_title_from_content("The document title\nContent") is None
        assert processor._get_title_from_content("This is a test\nContent") is None

    def test_docx_processing_errors(self):
        """Test DOCX processing error handling."""
        processor = DocumentProcessor()
        
        # Test invalid DOCX file
        with tempfile.NamedTemporaryFile(suffix='.docx', mode='w+b', delete=False) as f:
            f.write(b'Not a valid DOCX file')
            test_file = f.name
            
        try:
            with pytest.raises(ValueError) as exc_info:
                processor._extract_docx_text(test_file)
            assert "Failed to open DOCX file" in str(exc_info.value)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
                
        # Test DOCX with missing paragraphs attribute
        mock_doc = Mock(spec=['core_properties'])  # No paragraphs attribute
        with patch('src.documents.Document', return_value=mock_doc):
            with pytest.raises(ValueError) as exc_info:
                processor._extract_docx_text("test.docx")
            assert "Invalid DOCX file: document has no paragraphs" in str(exc_info.value)
            
        # Test DOCX with property access error
        mock_doc = Mock(spec=['paragraphs', 'core_properties'])
        mock_doc.paragraphs = []  # Empty but iterable
        type(mock_doc).core_properties = PropertyMock(side_effect=Exception("Property access error"))
        
        with patch('src.documents.Document', return_value=mock_doc):
            # Should not raise exception, should handle property error gracefully
            title, full_text, sections = processor._extract_docx_text("test.docx")
