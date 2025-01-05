"""Test document analyzer functionality."""
import pytest
from src.document_analyzer import DocumentAnalyzer, Heading, DocumentSection
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
        """Test extraction of numbered headings with preserved numbers."""
        text = """1. First Section
Content
1.1. Subsection One
More content
1.2. Subsection Two
Even more content
2. Second Section
Content
2.1.a Additional Details
Final content"""
        
        headings = analyzer.extract_headings(text)
        assert len(headings) == 5
        assert [h.text for h in headings] == [
            '1 First Section',
            '1.1 Subsection One',
            '1.2 Subsection Two',
            '2 Second Section',
            '2.1.a Additional Details'
        ]
        assert [h.level for h in headings] == [1, 2, 2, 1, 3]

    def test_section_number_formats(self, analyzer):
        """Test various section number formats."""
        text = """A. Overview
A.1. Background
A.1.1. Historical Context
I. Major Section
I.1. Subsection
1.a Additional Point
Section 2.1: Methodology
Chapter 3: Results
Appendix B.1: References"""
        
        headings = analyzer.extract_headings(text)
        
        # Verify section numbers are preserved
        texts = [h.text for h in headings]
        assert 'A Overview' in texts
        assert 'A.1 Background' in texts
        assert 'A.1.1 Historical Context' in texts
        assert 'I Major Section' in texts
        assert 'I.1 Subsection' in texts
        assert '1.a Additional Point' in texts
        assert 'Section 2.1 Methodology' in texts
        assert 'Chapter 3 Results' in texts
        assert 'Appendix B.1 References' in texts
        
        # Verify correct level assignment
        levels = [h.level for h in headings]
        assert levels[0] == 1  # A
        assert levels[1] == 2  # A.1
        assert levels[2] == 3  # A.1.1
        assert levels[3] == 2  # I (Roman numerals are major sections)
        assert levels[4] == 3  # I.1
        assert levels[5] == 2  # 1.a
        assert levels[6] == 2  # Section 2.1
        assert levels[7] == 1  # Chapter 3
        assert levels[8] == 2  # Appendix B.1

    def test_mixed_document_format(self, analyzer):
        """Test consistent heading extraction across document formats."""
        text = """# Main Title

1. Introduction
This is an introduction.

1.1. Background
Some background information.

Section 2: Methodology
The methodology section.

2.1. Process Steps
Step by step guide.

RESULTS AND DISCUSSION
Important findings.

Appendix A.1: Data Tables
Reference data."""
        
        headings = analyzer.extract_headings(text)
        
        # Verify all heading styles are captured
        texts = [h.text for h in headings]
        assert 'Main Title' in texts  # Markdown
        assert '1 Introduction' in texts  # Numbered
        assert '1.1 Background' in texts  # Sub-numbered
        assert 'Section 2 Methodology' in texts  # Section format
        assert '2.1 Process Steps' in texts  # Sub-numbered
        assert 'RESULTS AND DISCUSSION' in texts  # All caps
        assert 'Appendix A.1 Data Tables' in texts  # Appendix format
        
        # Verify proper hierarchy
        levels = [h.level for h in headings]
        assert levels[0] == 1  # Main Title
        assert levels[1] == 1  # 1. Introduction
        assert levels[2] == 2  # 1.1. Background
        assert levels[3] == 1  # Section 2
        assert levels[4] == 2  # 2.1. Process
        assert levels[5] == 1  # RESULTS
        assert levels[6] == 2  # Appendix A.1

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

class TestSectionExtraction:
    def test_section_extraction_with_headings(self, analyzer):
        """Test extraction of sections from document with headings."""
        text = """# Main Title
This is the introduction.

## Section 1
This is section 1 content.

## Section 2
This is section 2 content."""

        headings = analyzer.extract_headings(text)
        sections = analyzer.extract_sections(text, headings)
        
        assert len(sections) == 3  # Main + 2 sections
        assert sections[0].text.startswith("# Main Title")
        assert sections[1].text.startswith("## Section 1")
        assert sections[2].text.startswith("## Section 2")
        
        # Verify section positions
        assert sections[0].start_pos == 0
        assert sections[0].end_pos < sections[1].start_pos
        assert sections[1].end_pos < sections[2].start_pos
        
        # Verify heading references
        assert sections[0].heading.text == "Main Title"
        assert sections[0].heading.level == 1
        assert sections[1].heading.text == "Section 1"
        assert sections[1].heading.level == 2
        assert sections[2].heading.text == "Section 2"
        assert sections[2].heading.level == 2

    def test_section_extraction_without_headings(self, analyzer):
        """Test extraction of sections from document without headings."""
        text = "This is a document without any headings.\nIt should be one section."
        
        sections = analyzer.extract_sections(text, [])
        
        assert len(sections) == 1
        assert sections[0].text == text
        assert sections[0].start_pos == 0
        assert sections[0].end_pos == len(text)
        assert sections[0].heading is None

    def test_section_extraction_with_empty_sections(self, analyzer):
        """Test handling of empty sections between headings."""
        text = """# Title 1

# Title 2

# Title 3
Content 3"""

        headings = analyzer.extract_headings(text)
        sections = analyzer.extract_sections(text, headings)
        
        # Should only include non-empty sections
        assert len(sections) == 2  # Title 1 and Title 3 (with content)
        assert sections[0].heading.text == "Title 1"
        assert sections[1].heading.text == "Title 3"
        assert "Content 3" in sections[1].text

class TestTableOfContents:
    def test_complex_toc_generation(self, analyzer):
        """Test generation of complex hierarchical table of contents."""
        headings = [
            Heading("Test Document Title", 1, 0, 10),
            Heading("1. Introduction", 2, 20, 30),
            Heading("1.1. Background", 3, 40, 50),
            Heading("1.2. Purpose", 3, 60, 70),
            Heading("2. Methodology", 2, 80, 90),
            Heading("2.1. Process Steps", 3, 100, 110),
            Heading("2.1.1. Planning", 4, 120, 130),
            Heading("2.1.2. Implementation", 4, 140, 150),
            Heading("2.2. Data Collection", 3, 160, 170),
            Heading("2.2.1. Primary Sources", 4, 180, 190),
            Heading("2.2.1.1. Interviews", 5, 200, 210),
            Heading("2.2.1.2. Surveys", 5, 220, 230),
            Heading("2.2.2. Secondary Sources", 4, 240, 250),
            Heading("3. Results", 2, 260, 270),
            Heading("3.1. Key Findings", 3, 280, 290),
            Heading("3.2. Analysis", 3, 300, 310),
            Heading("Appendix A: Reference Data", 2, 320, 330),
            Heading("A.1. Data Tables", 3, 340, 350),
            Heading("A.2. Methodology Details", 3, 360, 370),
            Heading("GLOSSARY", 2, 380, 390)
        ]
        
        toc = analyzer.build_toc(headings)
        
        # Verify root structure
        assert len(toc) == 1  # Only the title at root level
        assert toc[0]['text'] == "Test Document Title"
        root = toc[0]
        
        # Verify main sections
        assert len(root['children']) == 5  # Introduction, Methodology, Results, Appendix, Glossary
        
        # Verify Introduction section
        intro = root['children'][0]
        assert intro['text'] == "1. Introduction"
        assert len(intro['children']) == 2  # Background and Purpose
        assert intro['children'][0]['text'] == "1.1. Background"
        assert intro['children'][1]['text'] == "1.2. Purpose"
        
        # Verify Methodology section
        method = root['children'][1]
        assert method['text'] == "2. Methodology"
        assert len(method['children']) == 2  # Process Steps and Data Collection
        
        # Verify Process Steps subsection
        process = method['children'][0]
        assert process['text'] == "2.1. Process Steps"
        assert len(process['children']) == 2  # Planning and Implementation
        assert process['children'][0]['text'] == "2.1.1. Planning"
        assert process['children'][1]['text'] == "2.1.2. Implementation"
        
        # Verify Data Collection subsection
        data = method['children'][1]
        assert data['text'] == "2.2. Data Collection"
        assert len(data['children']) == 2  # Primary and Secondary Sources
        
        # Verify Primary Sources sub-subsection
        primary = data['children'][0]
        assert primary['text'] == "2.2.1. Primary Sources"
        assert len(primary['children']) == 2  # Interviews and Surveys
        assert primary['children'][0]['text'] == "2.2.1.1. Interviews"
        assert primary['children'][1]['text'] == "2.2.1.2. Surveys"
        
        # Verify Results section
        results = root['children'][2]
        assert results['text'] == "3. Results"
        assert len(results['children']) == 2  # Key Findings and Analysis
        assert results['children'][0]['text'] == "3.1. Key Findings"
        assert results['children'][1]['text'] == "3.2. Analysis"
        
        # Verify Appendix section
        appendix = root['children'][3]
        assert appendix['text'] == "Appendix A: Reference Data"
        assert len(appendix['children']) == 2  # Data Tables and Methodology Details
        assert appendix['children'][0]['text'] == "A.1. Data Tables"
        assert appendix['children'][1]['text'] == "A.2. Methodology Details"
        
        # Verify Glossary section
        assert root['children'][4]['text'] == "GLOSSARY"

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
        with open('tests/test_data/test.md', 'r') as f:
            document = f.read()

        result = analyzer.analyze_document(document, "Test Document Title")
        
        # Verify TOC structure
        assert len(result['toc']) == 1  # Only title at root
        root = result['toc'][0]
        assert root['text'] == "Test Document Title"
        
        # Verify main sections
        assert len(root['children']) == 5  # Introduction, Methodology, Results, Appendix, Glossary
        
        # Verify Introduction section
        intro = root['children'][0]
        assert intro['text'] == "1. Introduction"
        assert len(intro['children']) == 2  # Background and Purpose
        
        # Verify Methodology section with deep nesting
        method = root['children'][1]
        assert method['text'] == "2. Methodology"
        assert len(method['children']) == 2  # Process Steps and Data Collection
        
        # Verify Process Steps subsection
        process = method['children'][0]
        assert process['text'] == "2.1. Process Steps"
        assert len(process['children']) == 2  # Planning and Implementation
        
        # Verify Data Collection subsection with deep nesting
        data = method['children'][1]
        assert data['text'] == "2.2. Data Collection"
        assert len(data['children']) == 2  # Primary and Secondary Sources
        
        # Verify Primary Sources with deepest nesting
        primary = data['children'][0]
        assert primary['text'] == "2.2.1. Primary Sources"
        assert len(primary['children']) == 2  # Interviews and Surveys
        
        # Verify headings
        assert len(result['headings']) == 20  # Total number of headings
        assert all(isinstance(h, Heading) for h in result['headings'])
        assert result['headings'][0].text == "Test Document Title"
        assert result['headings'][0].level == 1
        
        # Verify sections
        assert len(result['sections']) == 20  # One section per heading
        assert all(isinstance(s, DocumentSection) for s in result['sections'])
        assert all(s.heading is not None for s in result['sections'])
        assert result['sections'][0].heading.text == "Test Document Title"
        assert "methodology" in result['sections'][4].text.lower()

    def test_error_handling(self, analyzer):
        """Test error handling in document analysis."""
        # Test with None input
        result = analyzer.analyze_document(None, None)
        assert result['classification'] == 'miscellaneous'
        assert result['toc'] == []
        assert result['headings'] == []
        assert result['sections'] == []

        # Test with invalid input type
        result = analyzer.analyze_document(123, "Test")  # Non-string input
        assert result['classification'] == 'miscellaneous'
        assert result['toc'] == []
        assert result['headings'] == []
        assert result['sections'] == []
