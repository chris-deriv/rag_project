"""Constants for the RAG application."""

# Document Classifications
DOCUMENT_CLASSIFICATIONS = {
    'policies_procedures': 'Policies and Procedures',
    'legal_compliance': 'Legal and Compliance',
    'finance_accounting': 'Finance and Accounting',
    'human_resources': 'Human Resources',
    'it_technology': 'IT and Technology',
    'product_growth': 'Product and Growth',
    'customer_service_ops': 'Customer Service and Operations',
    'training_knowledge': 'Training and Knowledge',
    'risk_management': 'Risk Management',
    'customer_client': 'Customer and Client Documents',
    'strategic_planning': 'Strategic Planning',
    'internal_communication': 'Internal Communication',
    'project_management': 'Project Management',
    'cost_procurement': 'Cost Control and Procurement',
    'data_analytics': 'Data, Analytics and Reports',
    'security': 'Security',
    'governance': 'Governance',
    'miscellaneous': 'Miscellaneous'
}

# Document Structure
HEADING_PATTERNS = [
    # Numbered section patterns (capture both number and text)
    r'^(\d+\.(?:\d+)*)\s+(.+)$',  # Basic numbered (1.1, 1.2.3, etc.)
    r'^(\d+\.(?:\d+)*[A-Za-z]?)\s+(.+)$',  # With optional letter (1.1a, 2.3b)
    r'^([A-Z]\.(?:\d+)*)\s+(.+)$',  # Letter-based (A.1, B.2.1)
    
    # Standard heading markers
    r'^#{1,6}\s+(.+)$',  # Markdown headings
    
    # Common document sections (capture number and text)
    r'^(?:Section|Chapter|Part)\s+(\d+(?:\.\d+)*):?\s*(.+)$',  # Section 1.1: Title
    r'^(?:Appendix)\s+([A-Z](?:\.\d+)*):?\s*(.+)$',  # Appendix A.1: Title
    
    # Special formats
    r'^([IVX]+\.(?:\d+)*)\s+(.+)$',  # Roman numerals (I.1, IV.2)
    r'^([A-Z][A-Za-z\s]+):\s*(.+)$',  # Title case with content: Text
    r'^([A-Z][A-Z\s]+(?:\s|$))(.+)?$'  # All caps with optional content
]

# Section number format mapping
SECTION_NUMBER_FORMATS = {
    'numeric': r'^\d+\.(?:\d+)*$',  # 1.1, 1.2.3
    'alpha': r'^[A-Z]\.(?:\d+)*$',  # A.1, B.2.1
    'roman': r'^[IVX]+\.(?:\d+)*$',  # I.1, IV.2.1
    'mixed': r'^\d+\.(?:\d+)*[A-Za-z]?$'  # 1.1a, 2.3b
}

# API Response Keys
RESPONSE_KEYS = {
    'TOC_KEY': 'table_of_contents',
    'CONTENT_KEY': 'content',
    'METADATA_KEY': 'metadata'
}
