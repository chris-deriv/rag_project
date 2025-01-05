"""Document analysis and classification for the RAG application."""
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from langchain.docstore.document import Document
from unstructured.partition.pdf import partition_pdf
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
        
        # Split into lines and process
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            
            # Remove "All Rights Reserved" prefix if present
            if line.startswith("All Rights Reserved"):
                prefix_end = line.find("TechTarget")
                if prefix_end != -1:
                    line = line[prefix_end + len("TechTarget"):].strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Clean the line while preserving section numbers
            if re.match(r'^\s*\d+(?:\.\d+)*\s+\w', line):
                # Remove trailing dots and page numbers for numbered sections
                line = re.sub(r'\s*\.+\s*\d*\s*$', '', line)
            else:
                line = re.sub(r'\s+', ' ', line)
            
            lines.append(line)
                
        return '\n'.join(lines)

    def _clean_heading_text(self, text: str) -> str:
        """Clean heading text by removing trailing dots and page numbers."""
        # Remove trailing dots and page numbers
        text = re.sub(r'\.{3,}\s*\d*\s*$', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_headings(self, text: str, file_path: Optional[str] = None) -> List[Heading]:
        """Extract headings from document text."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        try:
            # Use unstructured.io if file path is provided
            if file_path:
                logger.info("Using unstructured.io for heading extraction")
                elements = partition_pdf(file_path)
                headings = []
                current_pos = 0
                
                # First pass: Extract TOC entries to understand the document structure
                toc_entries = {}
                in_toc = False
                
                for element in elements:
                    element_text = str(element)
                    if hasattr(element, 'category') and element.category == 'Title':
                        # Check for TOC start
                        if "Table of Contents" in element_text:
                            in_toc = True
                            continue
                        
                        # Process TOC entries
                        if in_toc:
                            # Check for end of TOC
                            if "Information Technology Statement" in element_text:
                                in_toc = False
                                continue
                            
                            # Extract section numbers and titles from TOC
                            toc_match = re.match(r'^\s*(\d+(?:\.\d+)*)\s+([^\.]+?)(?:\.{2,}|\s{3,})\s*\d*\s*$', element_text)
                            if toc_match:
                                section_num = toc_match.group(1)
                                section_title = toc_match.group(2).strip()
                                # Clean up title
                                section_title = re.sub(r'\s*\.+\s*\d*\s*$', '', section_title)
                                section_title = re.sub(r'\s+', ' ', section_title).strip()
                                section_title = re.sub(r'[.,:;]+$', '', section_title).strip()
                                toc_entries[section_num] = section_title
                
                # Second pass: Extract actual headings
                for element in elements:
                    element_text = str(element)
                    if hasattr(element, 'category') and element.category == 'Title':
                        # Try to extract section number and level
                        match = re.match(r'^\s*(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.+\s*\d*\s*)?$', element_text)
                        if match:
                            section_num = match.group(1)
                            section_text = match.group(2).strip()
                            level = len(section_num.split('.'))
                            
                            # Skip if this appears to be a TOC entry
                            if re.search(r'\.{3,}\s*\d+\s*$', element_text):
                                continue
                            
                            # Skip if this is just a page number
                            if section_text.isdigit():
                                continue
                            
                            # Skip if this is part of a form or table
                            if any(form_word in section_text.lower() for form_word in ["form", "table", "figure"]):
                                continue
                            
                            # Clean up section text
                            section_text = re.sub(r'\s*\.+\s*\d*\s*$', '', section_text)
                            section_text = re.sub(r'\s+', ' ', section_text).strip()
                            section_text = re.sub(r'[.,:;]+$', '', section_text).strip()
                            
                            # Use TOC title if available
                            if section_num in toc_entries:
                                section_text = toc_entries[section_num]
                            
                            # Remove any page numbers from the end
                            section_text = re.sub(r'\s+\d+\s*$', '', section_text)
                            
                            # Remove any trailing dots
                            section_text = re.sub(r'\s*\.+\s*$', '', section_text)
                            
                            # Construct the final heading text
                            heading_text = f"{section_num} {section_text}"
                            
                            # Skip if this is a duplicate heading
                            if any(h.text == heading_text for h in headings):
                                continue
                                
                            heading = Heading(
                                text=heading_text,
                                level=level,
                                start_pos=current_pos,
                                end_pos=current_pos + len(element_text)
                            )
                            headings.append(heading)
                            logger.info(f"Found heading: {heading.text} (level {level})")
                    
                    current_pos += len(element_text) + 1
                
                # If unstructured.io didn't find any headings, fall back to text-based extraction
                if not headings:
                    logger.info("No headings found with unstructured.io, falling back to text-based extraction")
                    return self._extract_headings_from_text(text)
                    
                return headings
            
            # Fallback to text-based extraction
            logger.info("Using text-based heading extraction")
            return self._extract_headings_from_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting headings: {str(e)}")
            return []

    def _extract_headings_from_text(self, text: str) -> List[Heading]:
        """Extract headings from text using pattern matching."""
        headings = []
        current_pos = 0
        
        # First pass: Extract TOC entries to understand the document structure
        toc_entries = {}
        in_toc = False
        
        # Split text into lines for processing
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for TOC start
            if "Table of Contents" in line:
                in_toc = True
                i += 1
                continue
            
            # Process TOC entries
            if in_toc:
                # Remove "All Rights Reserved" prefix if present
                if line.startswith("All Rights Reserved"):
                    prefix_end = line.find("TechTarget")
                    if prefix_end != -1:
                        line = line[prefix_end + len("TechTarget"):].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Check for end of TOC (usually starts with "Information Technology Statement")
                if line.startswith("Information Technology Statement"):
                    in_toc = False
                    i += 1
                    continue
                
                # Extract section numbers and titles from TOC
                toc_match = re.match(r'^\s*(\d+(?:\.\d+)*)\s+([^\.]+?)(?:\.{2,}|\s{3,})\s*\d*\s*$', line)
                if toc_match:
                    section_num = toc_match.group(1)
                    section_title = toc_match.group(2).strip()
                    # Clean up title
                    section_title = re.sub(r'\s*\.+\s*\d*\s*$', '', section_title)
                    section_title = re.sub(r'\s+', ' ', section_title).strip()
                    section_title = re.sub(r'[.,:;]+$', '', section_title).strip()
                    toc_entries[section_num] = section_title
            
            i += 1
        
        # Second pass: Extract actual headings using TOC info
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                current_pos += 1
                i += 1
                continue
            
            # Remove "All Rights Reserved" prefix if present
            if line.startswith("All Rights Reserved"):
                prefix_end = line.find("TechTarget")
                if prefix_end != -1:
                    line = line[prefix_end + len("TechTarget"):].strip()
            
            # Skip if empty after cleaning
            if not line:
                current_pos += 1
                i += 1
                continue
                
            # Skip if this is just a page number
            if line.isdigit():
                current_pos += len(line) + 1
                i += 1
                continue
                
            # Skip if this is part of a form or table
            if any(form_word in line.lower() for form_word in ["form", "table", "figure"]):
                current_pos += len(line) + 1
                i += 1
                continue
                
            # Try to match numbered sections
            numbered_match = re.match(r'^\s*(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.+\s*\d*\s*)?$', line)
            if numbered_match:
                section_num = numbered_match.group(1)
                section_text = numbered_match.group(2).strip()
                
                # Skip if this appears to be a TOC entry
                if re.search(r'\.{3,}\s*\d+\s*$', line):
                    current_pos += len(line) + 1
                    i += 1
                    continue
                
                # Skip if this is just a page number
                if section_text.isdigit():
                    current_pos += len(line) + 1
                    i += 1
                    continue
                
                # Skip if this is part of a form or table
                if any(form_word in section_text.lower() for form_word in ["form", "table", "figure"]):
                    current_pos += len(line) + 1
                    i += 1
                    continue
                
                # Determine level based on section number
                section_parts = section_num.split('.')
                level = len(section_parts)
                
                # Clean up section text
                section_text = re.sub(r'\s*\.+\s*\d*\s*$', '', section_text)
                
                # For main sections (level 1), ensure we have the full heading
                if level == 1:
                    # Look ahead to find any continuation of the heading
                    next_line_index = i + 1
                    while next_line_index < len(lines):
                        next_line = lines[next_line_index].strip()
                        
                        # Skip empty lines and page headers
                        if not next_line or next_line.startswith("All Rights Reserved"):
                            next_line_index += 1
                            continue
                            
                        # Stop if we hit another numbered section
                        if re.match(r'^\s*\d+(?:\.\d+)*\s+', next_line):
                            break
                            
                        # Stop if we hit a subsection marker
                        if re.match(r'^\s*\d+\.\d+', next_line):
                            break
                            
                        # Add this line to the heading text if it's not too long
                        # (to avoid capturing paragraph content)
                        if len(next_line) < 100:  # Reasonable length for a heading
                            section_text = f"{section_text} {next_line}"
                            i = next_line_index  # Skip the lines we've consumed
                            
                        next_line_index += 1
                        
                        # Stop after looking ahead a reasonable number of lines
                        if next_line_index > i + 3:
                            break
                
                # Clean up any remaining dots and page numbers
                section_text = re.sub(r'\s*\.+\s*\d*\s*$', '', section_text)
                
                # Remove any trailing punctuation
                section_text = re.sub(r'[.,:;]+$', '', section_text).strip()
                
                # Use TOC title if available
                if section_num in toc_entries:
                    section_text = toc_entries[section_num]
                
                # Clean up any remaining whitespace
                section_text = re.sub(r'\s+', ' ', section_text).strip()
                
                # Remove any page numbers from the end
                section_text = re.sub(r'\s+\d+\s*$', '', section_text)
                
                # Remove any trailing dots
                section_text = re.sub(r'\s*\.+\s*$', '', section_text)
                
                # Construct the final heading text
                heading_text = f"{section_num} {section_text}"
                
                # Skip if this is a duplicate heading
                if any(h.text == heading_text for h in headings):
                    current_pos += len(line) + 1
                    i += 1
                    continue
                
                heading = Heading(
                    text=heading_text,
                    level=level,
                    start_pos=current_pos,
                    end_pos=current_pos + len(line)
                )
                headings.append(heading)
                logger.info(f"Found numbered section: {heading.text} (level {level})")
            
            current_pos += len(line) + 1
            i += 1
        
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
                logger.info(f"Root entry: {toc[0]['text']}")
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
