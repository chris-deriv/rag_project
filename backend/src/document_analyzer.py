"""Document analysis and classification for the RAG application."""
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from langchain.docstore.document import Document
from .config.constants import (
    DOCUMENT_CLASSIFICATIONS,
    HEADING_PATTERNS
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
    text: str
    start_pos: int
    end_pos: int
    heading: Optional[Heading] = None

class DocumentAnalyzer:
    """Analyzes documents for classification and structure."""

    def __init__(self):
        """Initialize the document analyzer."""
        logger.info("Initialized DocumentAnalyzer")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
            
        # Replace form feeds and other special characters
        text = text.replace('\f', '\n')
        text = text.replace('\r', '\n')
        
        # Split into lines, clean each line while preserving markdown
        lines = []
        for line in text.split('\n'):
            # Preserve markdown heading markers
            if line.strip().startswith('#'):
                # Only normalize spaces after the # markers
                hash_count = len(re.match(r'^#+', line.strip()).group())
                line = '#' * hash_count + ' ' + re.sub(r'\s+', ' ', line.strip()[hash_count:]).strip()
            else:
                # Preserve dots in section numbers
                if re.match(r'^\d+\.(?:\d+)?\.?[a-z]?\.?\s', line.strip()):
                    # Keep original spacing for numbered sections with letters
                    line = line.strip()
                else:
                    line = re.sub(r'\s+', ' ', line).strip()
            if line:
                lines.append(line)
                
        return '\n'.join(lines)

    def _extract_section_info(self, text: str, start_pos: int) -> Tuple[str, int, int]:
        """Extract section level and clean heading text."""
        # Markdown headings (must come first)
        markdown_match = re.match(r'^(#{1,6})\s+(.+)$', text)
        if markdown_match:
            level = len(markdown_match.group(1))
            return markdown_match.group(2).strip(), level, len(text)

        # Common section patterns
        patterns = [
            # Mixed formats (must come first)
            (r'^(\d+)\.(\d+)\.([a-z])\.\s*(.+)$', 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # 2.1.a
            (r'^(\d+)\.([a-z])\.\s*(.+)$', 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # 1.a
            
            # Letter sections with subsections
            (r'^([A-Z])\.(\d+)\.(\d+)\.\s*(.+)$', 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # A.1.1
            (r'^([A-Z])\.(\d+)\.\s*(.+)$', 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # A.1
            (r'^([A-Z])\.\s*(.+)$', 1, lambda m: f"{m.group(1)} {m.group(2)}"),  # A
            
            # Roman numerals with subsections
            (r'^([IVX]+)\.(\d+)\.\s*(.+)$', 3, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # IV.1
            (r'^([IVX]+)\.\s*(.+)$', 2, lambda m: f"{m.group(1)} {m.group(2)}"),  # IV
            
            # Numeric sections with subsections
            (r'^(\d+)\.(\d+)\.([a-z])\.\s*(.+)$', 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # 2.1.a
            (r'^(\d+)\.(\d+)\.\s*(.+)$', 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # 2.1
            (r'^(\d+)\.\s*(.+)$', 1, lambda m: f"{m.group(1)} {m.group(2)}"),  # 1
            
            # Special sections
            (r'^(?:Section|Chapter)\s+(\d+(?:\.\d+)*):?\s*(.+)$', 1, lambda m: f"{m.group(0).split(':')[0]} {m.group(2)}"),  # Section 1.1
            (r'^Appendix\s+([A-Z](?:\.\d+)*):?\s*(.+)$', 2, lambda m: f"{m.group(0).split(':')[0]} {m.group(2)}"),  # Appendix A.1
            
            # ALL CAPS sections (treat as major sections)
            (r'^([A-Z][A-Z\s]+(?::[A-Z\s]*)?)\s*(.*)$', 1, lambda m: m.group(1).strip() + (' ' + m.group(2) if m.group(2) else '')),
        ]
        
        # Try each pattern
        for pattern, level, formatter in patterns:
            match = re.match(pattern, text)
            if match:
                # Special handling for mixed formats to preserve dots
                if '.a.' in text or '.a ' in text:
                    # Extract the original format up to the content
                    prefix = text[:text.index(' ')].strip()
                    content = text[text.index(' '):].strip()
                    # Increase level for letter suffixes
                    if re.search(r'\.[a-z]\.?$', prefix):
                        level += 1
                    return f"{prefix} {content}", level, len(text)
                    
                # Special handling for Roman numerals
                if re.match(r'^[IVX]+\.', text):
                    if '.' in text[:-1]:  # Has subsection (e.g., I.1)
                        level = 3  # Roman numeral subsections are level 3
                    else:
                        level = 2  # Plain Roman numerals are level 2
                    
                return formatter(match), level, len(text)
                
        return text.strip(), 1, len(text)

    def extract_headings(self, text: str) -> List[Heading]:
        """Extract headings from document text."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        headings = []
        current_pos = 0
        
        # Split text into lines and clean up
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # Skip empty lines and likely non-heading content
            if not line or len(line) > 200:  # Skip very long lines
                current_pos += len(line) + 1
                continue
                
            lines.append((line, current_pos))
            current_pos += len(line) + 1
        
        # Process each line
        for line, pos in lines:
            # Try to extract section info
            clean_text, level, length = self._extract_section_info(line, pos)
            
            # Only add if it looks like a heading
            if clean_text and (
                line.startswith('#') or  # Markdown
                re.match(r'^[A-Z]\.', line) or  # Letter sections
                re.match(r'^\d+\.', line) or  # Numbered sections
                re.match(r'^[IVX]+\.', line) or  # Roman numerals
                re.match(r'^(?:Section|Chapter|Part|Appendix)', line) or  # Special sections
                re.match(r'^[A-Z][A-Z\s]+(?::|$)', line)  # ALL CAPS
            ):
                heading = Heading(
                    text=clean_text,
                    level=level,
                    start_pos=pos,
                    end_pos=pos + length
                )
                headings.append(heading)
        
        logger.info(f"Extracted {len(headings)} headings from document")
        return headings

    def extract_sections(self, text: str, headings: List[Heading]) -> List[DocumentSection]:
        """Extract document sections based on headings."""
        if not headings:
            # If no headings found, treat entire text as one section if not empty
            text = text.strip() if isinstance(text, str) else str(text).strip()
            return [DocumentSection(
                text=text,
                start_pos=0,
                end_pos=len(text),
                heading=None
            )] if text and not text.isspace() else []
            
        sections = []
        for i in range(len(headings)):
            # Calculate section boundaries
            start_pos = headings[i].start_pos
            end_pos = headings[i+1].start_pos if i < len(headings)-1 else len(text)
            
            # Extract section text (including heading)
            section_text = text[start_pos:end_pos].strip()
            
            # Only include sections with content beyond the heading
            heading_text = text[start_pos:headings[i].end_pos].strip()
            content_text = text[headings[i].end_pos:end_pos].strip()
            
            if content_text:  # Only include if there's content beyond the heading
                section = DocumentSection(
                    text=section_text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(section_text),  # Fix position calculation
                    heading=headings[i]
                )
                sections.append(section)
                
        logger.info(f"Extracted {len(sections)} sections from document")
        return sections

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
        # Handle non-string inputs
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not isinstance(title, str):
            title = str(title) if title is not None else ""
            
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
                'high': ['it', 'technology', 'software', 'hardware', 'system', 'network', 'cyber', 'disaster recovery'],
                'medium': ['computer', 'data', 'security', 'infrastructure', 'backup', 'recovery'],
                'low': ['digital', 'online', 'electronic']
            },
            'risk_management': {
                'high': ['risk', 'mitigation', 'assessment', 'control', 'audit', 'disaster'],
                'medium': ['compliance', 'security', 'safety', 'emergency'],
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
            - sections: list of document sections
        """
        try:
            # Handle non-string inputs
            if not isinstance(text, str) or text is None:
                return {
                    'classification': 'miscellaneous',
                    'toc': [],
                    'headings': [],
                    'sections': []
                }
            if not isinstance(title, str):
                title = str(title) if title is not None else ""
                
            # Clean and normalize text
            text = self.clean_text(text)
            
            # Extract headings
            headings = self.extract_headings(text)
            logger.info(f"Extracted {len(headings)} headings from document")
            
            # Extract sections
            sections = self.extract_sections(text, headings)
            
            # Build table of contents
            toc = self.build_toc(headings)
            
            # Classify document
            classification = self.classify_document(text, title)
            logger.info(f"Classified document as: {classification}")
            
            return {
                'classification': classification,
                'toc': toc,
                'headings': headings,
                'sections': sections
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {
                'classification': 'miscellaneous',
                'toc': [],
                'headings': [],
                'sections': []
            }

# Global analyzer instance
document_analyzer = DocumentAnalyzer()
