"""Create test files for document processing tests."""
import os
from fpdf import FPDF
from docx import Document

def create_test_pdf():
    """Create a test PDF with a hierarchical structure."""
    pdf = FPDF()
    
    # Set document info
    pdf.set_title("Test Document Title")
    
    # Add a page
    pdf.add_page()
    
    # Add main title
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "Test Document Title", ln=True, align="C")
    
    # Add Section 1
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "1. Introduction", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This is the introduction section of our test document. It provides an overview of the content.")
    
    # Add Subsection 1.1
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 15, "1.1. Background", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This subsection provides background information about the test document.")
    
    # Add Section 2
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "2. Methodology", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This section describes the methodology used in our test document.")
    
    # Add Subsection 2.1
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 15, "2.1. Process Steps", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This subsection outlines the steps involved in our process.")
    
    # Add Subsection 2.2
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 15, "2.2. Data Collection", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This subsection describes how data was collected.")
    
    # Add Section 3
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "3. Results", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This section presents the results of our analysis.")
    
    # Add Appendix
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "Appendix A: Reference Data", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This appendix contains reference data and additional information.")
    
    # Save the file
    output_path = os.path.join("tests", "test_data", "test.pdf")
    pdf.output(output_path)
    print(f"Created test PDF at: {output_path}")

def create_test_docx():
    """Create a test DOCX with a hierarchical structure."""
    doc = Document()
    
    # Set core properties
    doc.core_properties.title = "Test DOCX Title"
    
    # Add main title
    doc.add_heading("Test DOCX Title", 0)
    
    # Add Section 1
    doc.add_heading("1. Introduction", 1)
    doc.add_paragraph("This is the introduction section of our test document. It provides an overview of the content.")
    
    # Add Subsection 1.1
    doc.add_heading("1.1. Background", 2)
    doc.add_paragraph("This subsection provides background information about the test document.")
    
    # Add Section 2
    doc.add_heading("2. Methodology", 1)
    doc.add_paragraph("This section describes the methodology used in our test document.")
    
    # Add Subsection 2.1
    doc.add_heading("2.1. Process Steps", 2)
    doc.add_paragraph("This subsection outlines the steps involved in our process.")
    
    # Add Subsection 2.2
    doc.add_heading("2.2. Data Collection", 2)
    doc.add_paragraph("This subsection describes how data was collected.")
    
    # Add Section 3
    doc.add_heading("3. Results", 1)
    doc.add_paragraph("This section presents the results of our analysis.")
    
    # Add Appendix
    doc.add_heading("Appendix A: Reference Data", 1)
    doc.add_paragraph("This appendix contains reference data and additional information.")
    
    # Save the file
    output_path = os.path.join("tests", "test_data", "test.docx")
    doc.save(output_path)
    print(f"Created test DOCX at: {output_path}")

def create_test_markdown():
    """Create a test Markdown file with a hierarchical structure."""
    content = """# Test Markdown Document

## 1. Introduction
This is the introduction section of our test document. It provides an overview of the content.

### 1.1. Background
This subsection provides background information about the test document.

## 2. Methodology
This section describes the methodology used in our test document.

### 2.1. Process Steps
This subsection outlines the steps involved in our process.

### 2.2. Data Collection
This subsection describes how data was collected.

## 3. Results
This section presents the results of our analysis.

## Appendix A: Reference Data
This appendix contains reference data and additional information.
"""
    
    output_path = os.path.join("tests", "test_data", "test.md")
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Created test Markdown at: {output_path}")

if __name__ == "__main__":
    # Create test directory if it doesn't exist
    os.makedirs(os.path.join("tests", "test_data"), exist_ok=True)
    
    # Create test files
    create_test_pdf()
    create_test_docx()
    create_test_markdown()
    
    # Create test files with different names for fallback tests
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Test content")
    pdf.output(os.path.join("tests", "test_data", "test_document.pdf"))
    
    doc = Document()
    doc.add_paragraph("Test content")
    doc.save(os.path.join("tests", "test_data", "test_document.docx"))
    
    print("All test files created successfully!")
