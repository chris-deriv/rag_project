from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Set font and size
    c.setFont("Helvetica", 12)
    
    # Write content
    y = 750  # Start from top of page
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Test PDF Document")
    y -= 30
    
    # Section 1
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Section 1: Introduction")
    y -= 20
    
    c.setFont("Helvetica", 12)
    text = "This is the introduction section of our test document. It contains multiple sentences to test the chunking functionality."
    for line in text.split('\n'):
        c.drawString(50, y, line)
        y -= 15
    y -= 20
    
    # Section 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Section 2: Main Content")
    y -= 20
    
    c.setFont("Helvetica", 12)
    text = """Here is the main content of our document. It includes several paragraphs to ensure proper text extraction and processing.

This paragraph tests how well the system handles multiple paragraphs within a section. It should maintain the structure while splitting into appropriate chunks."""
    for line in text.split('\n'):
        c.drawString(50, y, line)
        y -= 15
    y -= 20
    
    # Section 3
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Section 3: Conclusion")
    y -= 20
    
    c.setFont("Helvetica", 12)
    text = "Finally, we conclude our test document. This section helps verify that the document processing works end-to-end."
    for line in text.split('\n'):
        c.drawString(50, y, line)
        y -= 15
    
    # Save the PDF
    c.save()

if __name__ == '__main__':
    create_test_pdf('uploads/test.pdf')
