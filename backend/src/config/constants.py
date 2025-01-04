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
    # Markdown headings (must come first)
    r'^(#{1,6})\s+(.+)$',  # Captures level and text
    
    # Common document sections (must come before numbered sections)
    r'^((?:Section|Chapter|Part)\s+\d+(?:\.\d+)*):?\s*(.+)$',  # Section 1.1: Title
    r'^(Appendix\s+[A-Z](?:\.\d+)*):?\s*(.+)$',  # Appendix A.1: Title
    
    # Letter-based sections (must come before numeric)
    r'^([A-Z]\.?)\s+(.+)$',  # A., B., etc.
    r'^([A-Z](?:\.\d+)+\.?)\s+(.+)$',  # A.1., A.1.1., etc.
    r'^([IVX]+\.?)\s+(.+)$',  # I., II., etc.
    r'^([IVX]+(?:\.\d+)+\.?)\s+(.+)$',  # I.1., IV.2., etc.
    
    # Mixed alphanumeric sections (must come before pure numeric)
    r'^(\d+(?:\.\d+)*\.[a-z]\.?)\s+(.+)$',  # 2.1.a., etc.
    r'^(\d+\.[a-z]\.?)\s+(.+)$',  # 1.a., etc.
    
    # Numbered sections with subsections
    r'^(\d+(?:\.\d+)*\.?)\s+(.+)$',  # 1., 1.1., 1.1.1., etc.
    
    # Special formats (must come last)
    r'^([A-Z][A-Z\s]+[A-Z]):?\s*(.+)?$',  # ALL CAPS: Text
    r'^([A-Z][A-Z\s]+(?:\s+[A-Z])+)(?:\s+|$)(.*)$',  # ALL CAPS MULTIPLE WORDS
    r'^([A-Z][a-z]+\s+[A-Z][a-z]+):(.*)$',  # Title Case Header: Text
]

# Section number format mapping
SECTION_NUMBER_FORMATS = {
    'numeric': r'^\d+(?:\.\d+)*\.?$',  # 1, 1.1, 1.1.1
    'alpha': r'^[A-Z](?:\.\d+)*\.?$',  # A, A.1, A.1.1
    'roman': r'^[IVX]+(?:\.\d+)*\.?$',  # I, I.1, IV.2
    'mixed': r'^\d+(?:\.\d+)*[a-z]\.?$',  # 1a, 1.1a, 2.1.a
    'section': r'^(?:Section|Chapter|Part|Appendix)\s+(?:\d+|\w+)(?:\.\d+)*\.?$'  # Section 1.1, Appendix A.1
}

# API Response Keys
RESPONSE_KEYS = {
    'TOC_KEY': 'table_of_contents',
    'CONTENT_KEY': 'content',
    'METADATA_KEY': 'metadata'
}
