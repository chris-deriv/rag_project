"""Test document processing functionality."""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, PropertyMock, call
import numpy as np
from src.documents import DocumentStore, DocumentProcessor, DocumentChunk, ProcessingState
from src.document_analyzer import Heading, DocumentSection

[Previous content up to test_section_chunking remains unchanged...]

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

[Rest of the file remains unchanged...]
