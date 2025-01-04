"""Document analysis and classification for the RAG application."""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from .config.constants import DOCUMENT_CLASSIFICATIONS, HEADING_PATTERNS

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
        
    def extract_headings(self, text: str) -> List[Heading]:
        """Extract headings from document text."""
        headings = []
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                current_pos += len(line) + 1
                continue
                
            # Check each heading pattern
            for pattern in self.heading_patterns:
                match = pattern.match(line)
                if match:
                    # Determine heading level
                    if line.startswith('#'):
                        level = len(line) - len(line.lstrip('#'))
                    elif re.match(r'^\d+\.', line):
                        # Count the dots to determine level for numbered headings
                        level = line.count('.') + 1
                    else:
                        level = 1
                        
                    # Extract heading text
                    if match.groups():
                        heading_text = match.group(1)
                    else:
                        heading_text = line.rstrip(':')
                    
                    heading = Heading(
                        text=heading_text.strip(),
                        level=level,
                        start_pos=current_pos,
                        end_pos=current_pos + len(line)
                    )
                    headings.append(heading)
                    break
                    
            current_pos += len(line) + 1
            
        return headings

    def build_toc(self, headings: List[Heading]) -> List[Dict]:
        """Build table of contents from headings."""
        def add_to_toc(heading: Heading, current_level: List[Dict], current_depth: int) -> None:
            while current_depth >= len(current_level):
                current_level.append([])
            
            entry = {
                'text': heading.text,
                'level': heading.level,
                'children': []
            }
            
            if heading.level == 1:
                current_level[0].append(entry)
            else:
                # Find appropriate parent
                parent_level = current_level[heading.level - 2]
                if parent_level:
                    parent_level[-1]['children'].append(entry)
                else:
                    # No parent found, add at current level
                    current_level[heading.level - 1].append(entry)

        toc_levels: List[List[Dict]] = []
        for heading in headings:
            add_to_toc(heading, toc_levels, heading.level)
            
        return toc_levels[0] if toc_levels else []

    def classify_document(self, text: str, title: str) -> str:
        """
        Classify document based on content and title.
        Returns the classification key from DOCUMENT_CLASSIFICATIONS.
        """
        # Prepare text for classification
        combined_text = f"{title}\n{text}".lower()
        
        # Define classification keywords
        classification_keywords = {
            'policies_procedures': ['policy', 'procedure', 'guideline', 'protocol', 'standard operating'],
            'legal_compliance': ['legal', 'compliance', 'regulation', 'law', 'statute', 'regulatory'],
            'finance_accounting': ['finance', 'accounting', 'budget', 'financial', 'revenue', 'expense'],
            'human_resources': ['hr', 'human resources', 'employee', 'personnel', 'staff', 'recruitment'],
            'it_technology': ['it', 'technology', 'software', 'hardware', 'system', 'network', 'cyber'],
            'product_growth': ['product', 'growth', 'development', 'market', 'feature', 'roadmap'],
            'customer_service_ops': ['customer service', 'operation', 'support', 'service desk'],
            'training_knowledge': ['training', 'learning', 'education', 'knowledge', 'course'],
            'risk_management': ['risk', 'mitigation', 'assessment', 'control', 'audit'],
            'customer_client': ['client', 'customer', 'account', 'engagement'],
            'strategic_planning': ['strategy', 'planning', 'objective', 'goal', 'initiative'],
            'internal_communication': ['memo', 'announcement', 'internal', 'communication'],
            'project_management': ['project', 'milestone', 'deliverable', 'timeline'],
            'cost_procurement': ['cost', 'procurement', 'purchase', 'vendor', 'supplier'],
            'data_analytics': ['data', 'analytics', 'report', 'metric', 'dashboard'],
            'security': ['security', 'protection', 'safeguard', 'access control'],
            'governance': ['governance', 'board', 'committee', 'oversight']
        }
        
        # Score each classification
        scores = {category: 0 for category in DOCUMENT_CLASSIFICATIONS.keys()}
        
        for category, keywords in classification_keywords.items():
            for keyword in keywords:
                count = combined_text.count(keyword)
                scores[category] += count
                
        # Get category with highest score
        max_score = max(scores.values())
        if max_score > 0:
            category = max(scores.items(), key=lambda x: x[1])[0]
        else:
            category = 'miscellaneous'
            
        return category

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
