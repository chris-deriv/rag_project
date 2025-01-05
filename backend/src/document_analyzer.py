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
        patterns = []
        
        # Mixed formats (must come first)
        patterns.extend([
            # With trailing dot
            (r'^(\d+)\.(\d+)\.([a-z])\.\s*(.+)$', lambda m: 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # 2.1.a.
            (r'^(\d+)\.([a-z])\.\s*(.+)$', lambda m: 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # 1.a.
            
            # Without trailing dot but with dot before letter
            (r'^(\d+)\.(\d+)\.([a-z])\s+(.+)$', lambda m: 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # 2.1.a
            (r'^(\d+)\.([a-z])\s+(.+)$', lambda m: 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # 1.a
            
            # Without any dots after numbers
            (r'^(\d+)\.(\d+)([a-z])\s+(.+)$', lambda m: 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # 2.1a
            (r'^(\d+)([a-z])\s+(.+)$', lambda m: 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # 1a
        ])
        
        # Letter sections with subsections
        patterns.extend([
            (r'^([A-Z])\.(\d+)\.(\d+)\.\s*(.+)$', lambda m: 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # A.1.1
            (r'^([A-Z])\.(\d+)\.\s*(.+)$', lambda m: 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # A.1
            (r'^([A-Z])\.\s*(.+)$', lambda m: 1, lambda m: f"{m.group(1)} {m.group(2)}"),  # A
        ])
        
        # Roman numerals with subsections
        patterns.extend([
            (r'^([IVX]+)\.(\d+)\.\s*(.+)$', lambda m: 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # IV.1
            (r'^([IVX]+)\.\s*(.+)$', lambda m: 1, lambda m: f"{m.group(1)} {m.group(2)}"),  # IV
        ])
        
        # Numeric sections with subsections
        patterns.extend([
            (r'^(\d+)\.(\d+)\.(\d+)\.(\d+)\.\s*(.+)$', lambda m: 4, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)}.{m.group(4)} {m.group(5)}"),  # 1.1.1.1
            (r'^(\d+)\.(\d+)\.(\d+)\.\s*(.+)$', lambda m: 3, lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)} {m.group(4)}"),  # 1.1.1
            (r'^(\d+)\.(\d+)\.\s*(.+)$', lambda m: 2, lambda m: f"{m.group(1)}.{m.group(2)} {m.group(3)}"),  # 1.1
            (r'^(\d+)\.\s*(.+)$', lambda m: 1, lambda m: f"{m.group(1)} {m.group(2)}"),  # 1
        ])
        
        # Special sections
        patterns.extend([
            (r'^(?:Section|Chapter)\s+(\d+(?:\.\d+)*):?\s*(.+)$', lambda m: len(m.group(1).split('.')), lambda m: f"{m.group(0).split(':')[0]} {m.group(2)}"),  # Section 1.1
            (r'^Appendix\s+([A-Z](?:\.\d+)*):?\s*(.+)$', lambda m: 2, lambda m: f"{m.group(0).split(':')[0]} {m.group(2)}"),  # Appendix A.1 (always level 2)
        ])
        
        # ALL CAPS sections (treat as major sections)
        patterns.append(
            (r'^([A-Z][A-Z\s]+(?::[A-Z\s]*)?)\s*(.*)$', lambda m: 1, lambda m: m.group(1).strip() + (' ' + m.group(2) if m.group(2) else ''))
        )
        
        # Try each pattern
        for pattern, get_level, formatter in patterns:
            match = re.match(pattern, text)
            if match:
                # Get the level from the pattern-specific function
                level = get_level(match)
                logger.info(f"Matched pattern: {pattern} -> level {level}")
                
                # Format the text
                clean_text = formatter(match)
                logger.info(f"Formatted text: {clean_text}")
                
                return clean_text, level, len(text)
                
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
            if clean_text:
                is_heading = (
                    line.startswith('#') or  # Markdown
                    re.match(r'^[A-Z]\.', line) or  # Letter sections
                    re.match(r'^\d+\.', line) or  # Numbered sections
                    re.match(r'^[IVX]+\.', line) or  # Roman numerals
                    re.match(r'^(?:Section|Chapter|Part|Appendix)', line) or  # Special sections
                    re.match(r'^[A-Z][A-Z\s]+(?::|$)', line)  # ALL CAPS
                )
                
                if is_heading:
                    logger.info(f"Found heading: '{clean_text}' (level {level})")
                    heading = Heading(
                        text=clean_text,
                        level=level,
                        start_pos=pos,
                        end_pos=pos + length
                    )
                    headings.append(heading)
                else:
                    logger.debug(f"Skipping non-heading line: '{line}'")
        
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
            
            # Extract section text and check content
            heading_text = text[start_pos:headings[i].end_pos].strip()
            content_text = text[headings[i].end_pos:end_pos].strip()
            section_text = heading_text + '\n' + content_text if content_text else heading_text
            
            # Log section details for debugging
            logger.info(f"\nProcessing section {i+1}/{len(headings)}:")
            logger.info(f"Heading: {heading_text}")
            logger.info(f"Content length: {len(content_text)}")
            logger.info(f"Start pos: {start_pos}, End pos: {end_pos}")
            
            # Include section if:
            # 1. It has direct content beyond the heading, or
            # 2. It's the first section (document title), or
            # 3. It's a parent section (has subsections), or
            # 4. It's the last section with content
            is_first_section = i == 0
            is_last_with_content = i == len(headings)-1 and content_text
            has_subsections = (i < len(headings)-1 and headings[i+1].level > headings[i].level)
            
            if content_text or is_first_section or has_subsections or is_last_with_content:
                # Calculate actual end position based on content
                actual_end_pos = start_pos + len(heading_text) + (len(content_text) + 1 if content_text else 0)
                
                section = DocumentSection(
                    text=section_text,
                    start_pos=start_pos,
                    end_pos=actual_end_pos,
                    heading=headings[i]
                )
                sections.append(section)
                logger.info(f"Added section:")
                logger.info(f"  Heading: {heading_text}")
                logger.info(f"  Content length: {len(content_text)}")
                logger.info(f"  Start: {start_pos}, End: {actual_end_pos}")
                
        logger.info(f"Extracted {len(sections)} sections from document")
        return sections

    def build_toc(self, headings: List[Heading]) -> List[Dict]:
        """Build table of contents from headings."""
        if not headings:
            return []

        def add_to_toc(heading: Heading, current_level: List[Dict], level_map: Dict[int, Dict], is_first_heading: bool) -> None:
            entry = {
                'text': heading.text,
                'level': heading.level,
                'children': []
            }
            logger.info(f"Adding TOC entry: {heading.text} (level {heading.level})")
            
            if heading.level == 1 and is_first_heading:
                # Only the first level 1 heading becomes the root
                logger.info("Adding as root entry")
                current_level.append(entry)
                level_map[1] = entry
            else:
                # Find the closest parent level
                parent_level = heading.level - 1
                parent_found = False
                
                while parent_level > 0:
                    if parent_level in level_map:
                        logger.info(f"Found parent at level {parent_level}")
                        level_map[parent_level]['children'].append(entry)
                        level_map[heading.level] = entry
                        parent_found = True
                        break
                    parent_level -= 1
                
                if not parent_found:
                    # If no parent found and not the first heading,
                    # add as child of root if it exists
                    if current_level and current_level[0]['level'] == 1:
                        logger.info("Adding as child of root")
                        current_level[0]['children'].append(entry)
                        level_map[heading.level] = entry
                    else:
                        # No root exists, add at root level
                        logger.info("No root found, adding at root level")
                        current_level.append(entry)
                        level_map[heading.level] = entry

        toc = []
        level_map = {}  # Track the last entry at each level
        
        # Process headings
        for i, heading in enumerate(headings):
            add_to_toc(heading, toc, level_map, i == 0)
            
        # Log final TOC structure
        logger.info(f"Built TOC with {len(toc)} root entries")
        if toc:
            logger.info(f"Root entry: {toc[0]['text']}")
            logger.info(f"Number of root children: {len(toc[0]['children'])}")
            
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
            
            result = {
                'classification': classification,
                'toc': toc,
                'headings': headings,
                'sections': sections
            }
            logger.info(f"Document analysis complete - TOC has {len(toc)} top-level entries")
            if toc:
                logger.info(f"First TOC entry: {toc[0]['text']}")
            return result
            
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
