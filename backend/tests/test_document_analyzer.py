"""Test document analyzer functionality."""
import pytest
import re
from src.document_analyzer import DocumentAnalyzer, Heading, DocumentSection
import os

@pytest.fixture
def document_analyzer():
    """Create a document analyzer instance."""
    return DocumentAnalyzer()

def test_heading_extraction_from_real_pdf():
    """Test heading extraction from a real PDF document with clear structure."""
    analyzer = DocumentAnalyzer()
    
    # Read the real PDF content
    pdf_path = os.path.join("tests", "test_data", "sDR_IT_disaster_recovery_plan_template.pdf")
    with open(pdf_path, 'rb') as f:
        from pypdf import PdfReader
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            print(f"\nPage content (length: {len(page_text)}):")
            print("-" * 80)
            print(page_text[:500] + "..." if len(page_text) > 500 else page_text)
            print("-" * 80)
            text += page_text + "\n"
        
        print(f"\nTotal document length: {len(text)}")
        
        # Print lines that look like headings
        print("\nPotential heading lines:")
        print("-" * 80)
        for line in text.split('\n'):
            line = line.strip()
            # Remove "All Rights Reserved" prefix if present
            if line.startswith("All Rights Reserved"):
                line = line[line.find("TechTarget") + len("TechTarget"):].strip()
            if not line:
                continue
                
            if re.match(r'^\s*\d+(?:\.\d+)*\s+\w', line):
                print(f"Numbered line: {line}")
            elif re.match(r'^(?:Information Technology Statement of Intent|Policy Statement|Objectives|Key Personnel Contact Info|External Contacts|Notification Calling Tree)', line):
                print(f"Special section: {line}")
        print("-" * 80)
    
    # Extract headings using unstructured
    headings = analyzer.extract_headings(text, file_path=pdf_path)
    
    print("\nExtracted headings:")
    print("-" * 80)
    for h in headings:
        print(f"Level {h.level}: {h.text}")
    print("-" * 80)
    
    # Verify main section headings are found
    expected_main_sections = [
        "1 Plan Overview",
        "2 Emergency Response",
        "3 Media",
        "4 Insurance",
        "5 Financial and Legal Issues",
        "6 DRP Exercising"
    ]
    
    found_main_sections = [h.text for h in headings if h.level == 1 and h.text.startswith(("1 ", "2 ", "3 ", "4 ", "5 ", "6 "))]
    
    # Verify all main sections were found
    for section in expected_main_sections:
        assert section in found_main_sections, f"Main section '{section}' not found in extracted headings"
    
    # Verify subsections for Section 1
    expected_subsections_1 = [
        "1.1 Plan Updating",
        "1.2 Plan Documentation Storage",
        "1.3 Backup Strategy",
        "1.4 Risk Management"
    ]
    
    found_subsections_1 = [h.text for h in headings if h.level == 2 and h.text.startswith("1.")]
    
    # Verify all subsections of section 1 were found
    for subsection in expected_subsections_1:
        assert subsection in found_subsections_1, f"Subsection '{subsection}' not found in extracted headings"
    
    # Verify subsections for Section 2
    expected_subsections_2 = [
        "2.1 Alert, escalation and plan invocation",
        "2.2 Disaster Recovery Team",
        "2.3 Emergency Alert, Escalation and DRP Activation"
    ]
    
    found_subsections_2 = [h.text for h in headings if h.level == 2 and h.text.startswith("2.")]
    
    # Verify all subsections of section 2 were found
    for subsection in expected_subsections_2:
        assert subsection in found_subsections_2, f"Subsection '{subsection}' not found in extracted headings"
    
    # Verify some third-level headings
    expected_level3_sections = [
        "2.1.1 Plan Triggering Events",
        "2.1.2 Assembly Points",
        "2.1.3 Activation of Emergency Response Team"
    ]
    
    found_level3_sections = [h.text for h in headings if h.level == 3]
    
    # Verify third-level sections were found
    for section in expected_level3_sections:
        assert section in found_level3_sections, f"Level 3 section '{section}' not found in extracted headings"

def test_heading_extraction_from_bom_pdf():
    """Test heading extraction from the BOM business model PDF."""
    analyzer = DocumentAnalyzer()
    
    # Read the PDF content
    pdf_path = os.path.join("tests", "test_data", "BOM_business_model.pdf")
    with open(pdf_path, 'rb') as f:
        from pypdf import PdfReader
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Extract headings using unstructured
    headings = analyzer.extract_headings(text, file_path=pdf_path)
    
    # Verify main section headings are found
    expected_main_sections = [
        "1 Business synopsis",
        "2 2013 Goals"
    ]
    
    found_main_sections = [h.text for h in headings if h.level == 1]
    
    # Verify all main sections were found
    for section in expected_main_sections:
        assert section in found_main_sections, f"Main section '{section}' not found in extracted headings"
    
    # Verify subsections for Section 1
    expected_subsections_1 = [
        "1.1 Legal structure",
        "1.2 How we make money",
        "1.3 Broker codes",
        "1.4 Client accounts and payments",
        "1.5 Regulations",
        "1.6 Bets",
        "1.7 Underlyings and markets",
        "1.8 Trading restrictions",
        "1.9 IT",
        "1.10 Marketing"
    ]
    
    found_subsections_1 = [h.text for h in headings if h.level == 2 and h.text.startswith("1.")]
    
    # Verify all subsections of section 1 were found
    for subsection in expected_subsections_1:
        assert subsection in found_subsections_1, f"Subsection '{subsection}' not found in extracted headings"
    
    # Verify subsections for Section 2
    expected_subsections_2 = [
        "2.1 Development of the Web API",
        "2.2 Mojolicious iteration 2",
        "2.3 New Charting system",
        "2.4 Integrate Tick trades into main betting interface"
    ]
    
    found_subsections_2 = [h.text for h in headings if h.level == 2 and h.text.startswith("2.")]
    
    # Verify all subsections of section 2 were found
    for subsection in expected_subsections_2:
        assert subsection in found_subsections_2, f"Subsection '{subsection}' not found in extracted headings"
    
    # Verify some third-level headings under Marketing
    expected_marketing_subsections = [
        "1.10.1 Data & Analytics",
        "1.10.2 Affiliate Program",
        "1.10.3 Free gift/bonus codes",
        "1.10.4 White labels",
        "1.10.5 Other sites"
    ]
    
    found_marketing_subsections = [h.text for h in headings if h.level == 3 and h.text.startswith("1.10.")]
    
    # Verify marketing subsections were found
    for section in expected_marketing_subsections:
        assert section in found_marketing_subsections, f"Marketing subsection '{section}' not found in extracted headings"

def test_toc_building_from_real_pdf():
    """Test TOC building from a real PDF document with clear structure."""
    analyzer = DocumentAnalyzer()
    
    # Read the real PDF content
    pdf_path = os.path.join("tests", "test_data", "sDR_IT_disaster_recovery_plan_template.pdf")
    with open(pdf_path, 'rb') as f:
        from pypdf import PdfReader
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Extract headings and build TOC
    headings = analyzer.extract_headings(text)
    toc = analyzer.build_toc(headings)
    
    # Verify TOC structure
    assert len(toc) > 0, "TOC should not be empty"
    
    # Find main sections in TOC
    main_sections = {}
    for entry in toc:
        if entry['text'].startswith(("1 ", "2 ", "3 ", "4 ", "5 ", "6 ")):
            main_sections[entry['text']] = entry
    
    # Verify section 1 and its subsections
    section1 = main_sections.get("1 Plan Overview")
    assert section1 is not None, "Section 1 not found in TOC"
    assert len(section1['children']) == 4, "Section 1 should have 4 subsections"
    
    # Verify section 2 and its subsections
    section2 = main_sections.get("2 Emergency Response")
    assert section2 is not None, "Section 2 not found in TOC"
    assert len(section2['children']) >= 3, "Section 2 should have at least 3 subsections"
    
    # Verify section 2.1 and its subsections
    section2_1 = None
    for child in section2['children']:
        if child['text'] == "2.1 Alert, escalation and plan invocation":
            section2_1 = child
            break
    
    assert section2_1 is not None, "Section 2.1 not found in TOC"
    assert len(section2_1['children']) >= 3, "Section 2.1 should have at least 3 subsections"
    
    # Verify correct level assignment
    def verify_levels(entries, expected_level):
        for entry in entries:
            assert entry['level'] == expected_level, f"Entry {entry['text']} has incorrect level"
            if entry['children']:
                verify_levels(entry['children'], expected_level + 1)
    
    verify_levels(toc, 1)

def test_section_extraction_from_real_pdf():
    """Test section extraction from a real PDF document."""
    analyzer = DocumentAnalyzer()
    
    # Read the real PDF content
    pdf_path = os.path.join("tests", "test_data", "sDR_IT_disaster_recovery_plan_template.pdf")
    with open(pdf_path, 'rb') as f:
        from pypdf import PdfReader
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Extract headings and sections
    headings = analyzer.extract_headings(text)
    sections = analyzer.extract_sections(text, headings)
    
    # Verify sections were extracted
    assert len(sections) > 0, "Sections should not be empty"
    
    # Verify section 1 content
    section1 = None
    for section in sections:
        if section.heading and section.heading.text == "1 Plan Overview":
            section1 = section
            break
    
    assert section1 is not None, "Section 1 not found in extracted sections"
    assert "Plan Overview" in section1.text, "Section 1 text should contain 'Plan Overview'"
    assert section1.heading.level == 1, "Section 1 should be level 1"
    
    # Verify section 2.1 content
    section2_1 = None
    for section in sections:
        if section.heading and section.heading.text == "2.1 Alert, escalation and plan invocation":
            section2_1 = section
            break
    
    assert section2_1 is not None, "Section 2.1 not found in extracted sections"
    assert "Alert" in section2_1.text, "Section 2.1 text should contain 'Alert'"
    assert section2_1.heading.level == 2, "Section 2.1 should be level 2"
    
    # Verify sections maintain order
    section_texts = [s.heading.text if s.heading else "" for s in sections]
    
    # Verify some key sections appear in correct order
    def index_of(text):
        return next((i for i, t in enumerate(section_texts) if text in t), -1)
    
    idx_1 = index_of("1 Plan Overview")
    idx_2 = index_of("2 Emergency Response")
    idx_3 = index_of("3 Media")
    
    assert idx_1 < idx_2 < idx_3, "Sections should maintain document order"
