"""Document analysis and classification for the RAG application."""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from .config.constants import (
    DOCUMENT_CLASSIFICATIONS,
    HEADING_PATTERNS,
    SECTION_NUMBER_FORMATS
)

logger = logging.getLogger(__name__)

@dataclass
class Heading:
    """Represents a document heading with its level and content."""
    text: str
    level: int
    start_pos: int
    end_pos: int

@dataclass
class DocumentSection:
    """Represents a section of the document with its heading and content."""
    heading: Heading
    content: str
    subsections: List['DocumentSection']

class DocumentAnalyzer:
    """Analyzes documents for classification and structure."""

    def __init__(self):
        """Initialize the document analyzer."""
        self.heading_patterns = [re.compile(pattern) for pattern in HEADING_PATTERNS]
        
    def _get_section_level(self, section_number: str) -> int:
        """Determine heading level from section number format."""
        if not section_number:
            return 1
            
        # Check each section number format
        for format_name, pattern in SECTION_NUMBER_FORMATS.items():
            if re.match(pattern, section_number):
                if format_name in ['numeric', 'alpha', 'mixed']:
                    # Count dots for hierarchical depth
                    return section_number.count('.') + 1
                elif format_name == 'roman':
                    # Roman numerals usually indicate major sections
                    return section_number.count('.') + 2
        return 1

    def extract_headings(self, text: str) -> List[Heading]:
        """Extract headings from document text with preserved section numbers."""
        headings = []
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            original_line = line
            line = line.strip()
            if not line:
                current_pos += len(original_line) + 1
                continue
            
            heading_found = False
            # Try each heading pattern
            for pattern in self.heading_patterns:
                match = re.match(pattern, line)
                if match:
                    if line.startswith('#'):
                        # Markdown heading
                        level = len(line) - len(line.lstrip('#'))
                        heading_text = line.lstrip('#').strip()
                        section_number = ''
                    elif len(match.groups()) == 2:
                        # Pattern with section number and text
                        section_number = match.group(1).strip()
                        heading_text = match.group(2).strip()
                        level = self._get_section_level(section_number)
                    else:
                        # Pattern with just text (e.g., ALL CAPS)
                        heading_text = match.group(1).strip()
                        section_number = ''
                        level = 1
                    
                    # Combine section number and text if both exist
                    full_text = f"{section_number} {heading_text}" if section_number else heading_text
                    
                    heading = Heading(
                        text=full_text.strip(),
                        level=level,
                        start_pos=current_pos,
                        end_pos=current_pos + len(original_line)
                    )
                    headings.append(heading)
                    heading_found = True
                    break
            
            current_pos += len(original_line) + 1
        
        return headings

    def build_toc(self, headings: List[Heading]) -> List[Dict]:
        """Build table of contents from headings."""
        if not headings:
            return []

        def add_to_toc(heading: Heading, current_level: List[Dict], level_map: Dict[int, Dict]) -> None:
            entry = {
                'text': heading.text,
                'level': heading.level,
                'children': []
            }
            
            if heading.level == 1:
                current_level.append(entry)
                level_map[1] = entry
            else:
                # Find the closest parent level
                parent_level = heading.level - 1
                while parent_level > 0:
                    if parent_level in level_map:
                        level_map[parent_level]['children'].append(entry)
                        level_map[heading.level] = entry
                        break
                    parent_level -= 1
                if parent_level == 0:
                    # No parent found, add at root level
                    current_level.append(entry)
                    level_map[heading.level] = entry

        toc = []
        level_map = {}  # Track the last entry at each level
        
        for heading in headings:
            add_to_toc(heading, toc, level_map)
            
        return toc

    def classify_document(self, text: str, title: str) -> str:
        """
        Classify document based on content and title.
        Returns the classification key from DOCUMENT_CLASSIFICATIONS.
        """
        # Prepare text for classification
        combined_text = f"{title}\n{text}".lower()
        
        # Define classification keywords with weights
        classification_keywords = {
            'human_resources': {
                'high': ['hr', 'human resource', 'employee', 'personnel', 'staff', 'recruitment', 'policy manual'],
                'medium': ['benefits', 'payroll', 'training', 'workplace', 'leave'],
                'low': ['policy', 'management', 'guidelines']
            },
            'customer_service_ops': {
                'high': ['customer service', 'support', 'service desk', 'helpdesk', 'operations'],
                'medium': ['customer', 'client', 'ticket', 'inquiry'],
                'low': ['help', 'assistance', 'response']
            },
            'finance_accounting': {
                'high': ['finance', 'accounting', 'budget', 'financial', 'revenue', 'expense'],
                'medium': ['cost', 'profit', 'loss', 'balance'],
                'low': ['money', 'payment', 'price']
            },
            'it_technology': {
                'high': ['it', 'technology', 'software', 'hardware', 'system', 'network', 'cyber'],
                'medium': ['computer', 'data', 'security', 'infrastructure'],
                'low': ['digital', 'online', 'electronic']
            },
            'risk_management': {
                'high': ['risk', 'mitigation', 'assessment', 'control', 'audit'],
                'medium': ['compliance', 'security', 'safety'],
                'low': ['review', 'evaluation', 'monitoring']
            },
            'training_knowledge': {
                'high': ['training', 'learning', 'education', 'knowledge', 'course'],
                'medium': ['workshop', 'seminar', 'instruction', 'guide'],
                'low': ['manual', 'documentation', 'reference']
            },
            'product_growth': {
                'high': ['product', 'growth', 'development', 'market', 'feature', 'roadmap'],
                'medium': ['strategy', 'innovation', 'launch', 'release'],
                'low': ['plan', 'improvement', 'update']
            },
            'legal_compliance': {
                'high': ['legal', 'compliance', 'regulation', 'law', 'statute', 'regulatory'],
                'medium': ['requirements', 'obligations', 'guidelines'],
                'low': ['policy', 'rules']
            },
            'policies_procedures': {
                'high': ['policy', 'procedure', 'guideline', 'protocol', 'standard operating'],
                'medium': ['compliance', 'requirements', 'rules', 'regulations'],
                'low': ['process', 'steps', 'instructions']
            }
        }
        
        # Score each classification
        scores = {category: 0 for category in DOCUMENT_CLASSIFICATIONS.keys()}
        
        for category, keywords in classification_keywords.items():
            # High priority keywords (weight: 5)
            for keyword in keywords.get('high', []):
                count = combined_text.count(keyword)
                scores[category] += count * 5
            
            # Medium priority keywords (weight: 3)
            for keyword in keywords.get('medium', []):
                count = combined_text.count(keyword)
                scores[category] += count * 3
            
            # Low priority keywords (weight: 1)
            for keyword in keywords.get('low', []):
                count = combined_text.count(keyword)
                scores[category] += count
                
        # Get category with highest score
        max_score = max(scores.values())
        if max_score > 0:
            # If there's a tie, prefer more specific categories
            max_categories = [cat for cat, score in scores.items() if score == max_score]
            if len(max_categories) > 1:
                priority_order = [
                    'human_resources',
                    'customer_service_ops',
                    'finance_accounting',
                    'it_technology',
                    'risk_management',
                    'training_knowledge',
                    'product_growth',
                    'legal_compliance',
                    'policies_procedures',
                    'miscellaneous'
                ]
                for category in priority_order:
                    if category in max_categories:
                        return category
            return max_categories[0]
        else:
            return 'miscellaneous'

    def analyze_document(self, text: str, title: str) -> Dict:
        """
        Analyze document to extract structure and classify content.
        
        Returns:
            Dict containing:
            - classification: document category
            - toc: table of contents
            - headings: list of extracted headings
        """
        try:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            if not isinstance(title, str):
                title = str(title) if title is not None else ""
                
            # Extract headings
            headings = self.extract_headings(text)
            
            # Build table of contents
            toc = self.build_toc(headings)
            
            # Classify document
            classification = self.classify_document(text, title)
            
            return {
                'classification': classification,
                'toc': toc,
                'headings': [
                    {
                        'text': h.text,
                        'level': h.level,
                        'start_pos': h.start_pos,
                        'end_pos': h.end_pos
                    }
                    for h in headings
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {
                'classification': 'miscellaneous',
                'toc': [],
                'headings': []
            }

# Global analyzer instance
document_analyzer = DocumentAnalyzer()
