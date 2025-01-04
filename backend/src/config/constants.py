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
    r'^#{1,6}\s+(.+)$',  # Markdown headings
    r'^(\d+\.(?:\d+)*)\s+(.+)$',  # Numbered headings (1.1, 1.2, etc.)
    r'^[A-Z][A-Za-z\s]+:$',  # Title case followed by colon
    r'^[A-Z][A-Z\s]+(?:\s|$)',  # All caps text
    r'^(?:Section|Chapter|Part)\s+\d+:?\s*(.+)$',  # Section/Chapter headings
]

# API Response Keys
RESPONSE_KEYS = {
    'TOC_KEY': 'table_of_contents',
    'CONTENT_KEY': 'content',
    'METADATA_KEY': 'metadata'
}
