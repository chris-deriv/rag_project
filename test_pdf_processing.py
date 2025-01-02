import os
from fpdf import FPDF

def create_test_pdf():
    """Create a test PDF with a known title and content."""
    pdf = FPDF()
    
    # Set document info
    pdf.set_title("Test Document Title")
    
    # Add a page
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Test Document Title", ln=True, align="C")
    
    # Add content
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This is a test document.\n\nIt contains multiple paragraphs to test the chunking functionality.\n\nEach paragraph should be processed correctly.")
    
    # Save the file
    output_path = "uploads/test.pdf"
    os.makedirs("uploads", exist_ok=True)
    pdf.output(output_path)
    print(f"Created test PDF at: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_pdf()
