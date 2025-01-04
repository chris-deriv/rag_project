"""Test document analyzer functionality."""
import pytest
from src.document_analyzer import DocumentAnalyzer, Heading
from src.config.constants import DOCUMENT_CLASSIFICATIONS

@pytest.fixture
def analyzer():
    return DocumentAnalyzer()

class TestHeadingExtraction:
    def test_markdown_headings(self, analyzer):
        """Test extraction of markdown-style headings."""
        text = """# Main Title
Some content
## Section 1
Content 1
### Subsection 1.1
Content 1.1
## Section 2
Content 2"""
        
        headings = analyzer.extract_headings(text)
        assert len(headings) == 4
        assert [h.text for h in headings] == [
            'Main Title',
            'Section 1',
            'Subsection 1.1',
            'Section 2'
        ]
        assert [h.level for h in headings] == [1, 2, 3, 2]

    def test_numbered_headings(self, analyzer):
        """Test extraction of numbered headings."""
        text = """1. First Section
Content
1.1. Subsection
More content
2. Second Section
Content"""
        
        headings = analyzer.extract_headings(text)
        assert len(headings) == 3
        assert [h.text for h in headings] == [
            'First Section',
            'Subsection',
            'Second Section'
        ]
        assert [h.level for h in headings] == [1, 2, 1]

    def test_mixed_heading_styles(self, analyzer):
        """Test extraction of mixed heading styles."""
        text = """# Document Title
1. First Section
## 1.1 Subsection
IMPORTANT NOTICE:
Section Header:
2. Second Section"""
        
        headings = analyzer.extract_headings(text)
        assert len(headings) == 6
        assert all(isinstance(h, Heading) for h in headings)

    def test_empty_document(self, analyzer):
        """Test handling of empty documents."""
        assert analyzer.extract_headings("") == []
        assert analyzer.extract_headings("\n\n") == []

class TestTableOfContents:
    def test_toc_generation(self, analyzer):
        """Test generation of table of contents."""
        headings = [
            Heading("Main Title", 1, 0, 10),
            Heading("Section 1", 2, 20, 30),
            Heading("Subsection 1.1", 3, 40, 50),
            Heading("Section 2", 2, 60, 70)
        ]
        
        toc = analyzer.build_toc(headings)
        assert len(toc) == 1  # One top-level entry
        assert toc[0]['text'] == "Main Title"
        assert len(toc[0]['children']) == 2  # Two sections
        assert toc[0]['children'][0]['text'] == "Section 1"
        assert len(toc[0]['children'][0]['children']) == 1  # One subsection
        assert toc[0]['children'][0]['children'][0]['text'] == "Subsection 1.1"

    def test_toc_with_missing_levels(self, analyzer):
        """Test TOC generation with missing heading levels."""
        headings = [
            Heading("Title", 1, 0, 10),
            Heading("Subsection", 3, 20, 30)  # Missing level 2
        ]
        
        toc = analyzer.build_toc(headings)
        assert len(toc) == 1
        assert toc[0]['text'] == "Title"
        # Subsection should still be included
        assert len(toc[0]['children']) == 1
        assert toc[0]['children'][0]['text'] == "Subsection"

    def test_empty_toc(self, analyzer):
        """Test TOC generation with no headings."""
        assert analyzer.build_toc([]) == []

class TestDocumentClassification:
    @pytest.mark.parametrize("content,title,expected_class", [
        ("This is a policy document outlining procedures", "Company Policy", "policies_procedures"),
        ("Financial report for Q1 2024 revenue and expenses", "Q1 Report", "finance_accounting"),
        ("Employee handbook for HR policies", "HR Manual", "human_resources"),
        ("IT system architecture and network setup", "System Design", "it_technology"),
        ("Customer service guidelines and support procedures", "Service Manual", "customer_service_ops"),
        ("Risk assessment and mitigation strategies", "Risk Report", "risk_management"),
        ("Random unclassifiable content", "Misc Document", "miscellaneous")
    ])
    def test_document_classification(self, analyzer, content, title, expected_class):
        """Test classification of documents based on content and title."""
        classification = analyzer.classify_document(content, title)
        assert classification == expected_class
        assert classification in DOCUMENT_CLASSIFICATIONS

    def test_empty_document_classification(self, analyzer):
        """Test classification of empty documents."""
        classification = analyzer.classify_document("", "")
        assert classification == "miscellaneous"

class TestDocumentAnalysis:
    def test_complete_document_analysis(self, analyzer):
        """Test complete document analysis including TOC and classification."""
        document = """# HR Policy Manual
## Employee Guidelines
### Code of Conduct
All employees must follow these guidelines...
### Leave Policy
Annual leave and sick leave policies...
## Performance Reviews
### Review Process
The annual review process includes..."""

        result = analyzer.analyze_document(document, "HR Policy Manual")
        
        assert 'classification' in result
        assert result['classification'] == 'human_resources'
        
        assert 'toc' in result
        assert len(result['toc']) == 1  # One top-level entry
        assert result['toc'][0]['text'] == "HR Policy Manual"
        assert len(result['toc'][0]['children']) == 2  # Two main sections
        
        assert 'headings' in result
        assert len(result['headings']) == 7  # Total number of headings
        assert all(isinstance(h['text'], str) for h in result['headings'])
        assert all(isinstance(h['level'], int) for h in result['headings'])

    def test_error_handling(self, analyzer):
        """Test error handling in document analysis."""
        # Test with None input
        result = analyzer.analyze_document(None, None)
        assert result['classification'] == 'miscellaneous'
        assert result['toc'] == []
        assert result['headings'] == []

        # Test with invalid input type
        result = analyzer.analyze_document(123, "Test")  # Non-string input
        assert result['classification'] == 'miscellaneous'
        assert result['toc'] == []
        assert result['headings'] == []
