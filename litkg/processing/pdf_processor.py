"""PDF processing with LLM Sherpa and advanced chunking strategies."""

import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Advanced PDF processor with LLM Sherpa integration.

    Features:
    - Document structure preservation
    - Semantic chunking strategies
    - Multi-format support
    - Hierarchical content extraction
    """

    def __init__(self, config):
        """Initialize PDF processor."""
        self.config = config
        self.sherpa_processor = None
        self.fallback_processors = []
        self._initialize_processors()

    def _initialize_processors(self):
        """Initialize PDF processing engines."""
        # Try LLM Sherpa first (best structure preservation)
        try:
            from llmsherpa.readers import LayoutPDFReader
            sherpa_api_url = self.config.get("sherpa_api_url", "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all")
            self.sherpa_processor = LayoutPDFReader(sherpa_api_url)
            logger.info("LLM Sherpa processor initialized")
        except ImportError:
            logger.warning("LLM Sherpa not available. Install with: pip install llmsherpa")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Sherpa: {e}")

        # Initialize fallback processors
        self._initialize_fallback_processors()

    def _initialize_fallback_processors(self):
        """Initialize fallback PDF processors."""
        # PyMuPDF (fast and accurate)
        try:
            import fitz  # PyMuPDF
            self.fallback_processors.append(('pymupdf', self._extract_with_pymupdf))
            logger.info("PyMuPDF processor available")
        except ImportError:
            pass

        # pdfplumber (good for tables)
        try:
            import pdfplumber
            self.fallback_processors.append(('pdfplumber', self._extract_with_pdfplumber))
            logger.info("pdfplumber processor available")
        except ImportError:
            pass

        # PyPDF2 (basic fallback)
        try:
            import PyPDF2
            self.fallback_processors.append(('pypdf2', self._extract_with_pypdf2))
            logger.info("PyPDF2 processor available")
        except ImportError:
            pass

    def extract_text(self,
                    filepath: Union[str, Path],
                    preserve_structure: bool = True,
                    chunking_strategy: str = "semantic") -> List[str]:
        """
        Extract text from PDF with optional structure preservation.

        Args:
            filepath: Path to PDF file
            preserve_structure: Whether to preserve document structure
            chunking_strategy: "semantic", "fixed", "hierarchical"

        Returns:
            List of text chunks
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")

        logger.info(f"Processing PDF: {filepath}")

        # Try LLM Sherpa first if structure preservation is needed
        if preserve_structure and self.sherpa_processor:
            try:
                return self._extract_with_sherpa(filepath, chunking_strategy)
            except Exception as e:
                logger.warning(f"LLM Sherpa failed, falling back: {e}")

        # Try fallback processors
        for processor_name, processor_func in self.fallback_processors:
            try:
                logger.info(f"Trying {processor_name} processor")
                text_chunks = processor_func(filepath, chunking_strategy)
                if text_chunks:
                    logger.info(f"Successfully processed with {processor_name}")
                    return text_chunks
            except Exception as e:
                logger.warning(f"{processor_name} failed: {e}")

        # Final fallback - simple text extraction
        logger.warning("All processors failed, using basic text extraction")
        return self._basic_text_extraction(filepath)

    def _extract_with_sherpa(self, filepath: Path, chunking_strategy: str) -> List[str]:
        """Extract text using LLM Sherpa with structure preservation."""
        if not self.sherpa_processor:
            raise RuntimeError("LLM Sherpa not available")

        # Parse document
        doc = self.sherpa_processor.read_pdf(str(filepath))

        if chunking_strategy == "hierarchical":
            return self._hierarchical_chunking_sherpa(doc)
        elif chunking_strategy == "semantic":
            return self._semantic_chunking_sherpa(doc)
        else:
            return self._fixed_chunking_sherpa(doc)

    def _hierarchical_chunking_sherpa(self, doc) -> List[str]:
        """Hierarchical chunking based on document structure."""
        chunks = []

        # Extract sections hierarchically
        for section in doc.sections():
            section_text = f"# {section.title}\n\n"
            section_text += section.to_text()

            # Split large sections
            if len(section_text) > 3000:
                sub_chunks = self._split_text_semantic(section_text, max_length=2000)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section_text)

        # Add tables separately
        for table in doc.tables():
            table_text = f"TABLE:\n{table.to_text()}"
            chunks.append(table_text)

        return chunks

    def _semantic_chunking_sherpa(self, doc) -> List[str]:
        """Semantic chunking preserving meaning boundaries."""
        chunks = []

        # Process each chunk from Sherpa
        for chunk in doc.chunks():
            chunk_text = chunk.to_text()

            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                continue

            # Merge short adjacent chunks
            if chunks and len(chunks[-1]) < 1000 and len(chunk_text) < 1000:
                chunks[-1] += "\n\n" + chunk_text
            else:
                chunks.append(chunk_text)

        # Post-process: split overly long chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 3000:
                sub_chunks = self._split_text_semantic(chunk, max_length=2000)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _fixed_chunking_sherpa(self, doc) -> List[str]:
        """Fixed-size chunking from Sherpa document."""
        full_text = doc.to_text()
        return self._split_text_fixed(full_text, chunk_size=2000, overlap=200)

    def _extract_with_pymupdf(self, filepath: Path, chunking_strategy: str) -> List[str]:
        """Extract text using PyMuPDF."""
        import fitz

        doc = fitz.open(str(filepath))
        pages_text = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()

            # Clean up text
            text = self._clean_text(text)
            if text.strip():
                pages_text.append(text)

        doc.close()

        # Combine pages and chunk
        full_text = "\n\n".join(pages_text)

        if chunking_strategy == "semantic":
            return self._split_text_semantic(full_text)
        else:
            return self._split_text_fixed(full_text)

    def _extract_with_pdfplumber(self, filepath: Path, chunking_strategy: str) -> List[str]:
        """Extract text using pdfplumber (good for tables)."""
        import pdfplumber

        chunks = []

        with pdfplumber.open(str(filepath)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    text = self._clean_text(text)
                    chunks.append(f"Page {page_num + 1}:\n{text}")

                # Extract tables separately
                tables = page.extract_tables()
                for i, table in enumerate(tables):
                    if table:
                        table_text = self._format_table(table)
                        chunks.append(f"Table {i+1} from Page {page_num + 1}:\n{table_text}")

        # Post-process chunks
        if chunking_strategy == "semantic":
            return self._merge_and_split_semantic(chunks)
        else:
            return self._merge_and_split_fixed(chunks)

    def _extract_with_pypdf2(self, filepath: Path, chunking_strategy: str) -> List[str]:
        """Extract text using PyPDF2 (basic fallback)."""
        import PyPDF2

        text_parts = []

        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    text = self._clean_text(text)
                    text_parts.append(f"Page {page_num + 1}:\n{text}")

        full_text = "\n\n".join(text_parts)

        if chunking_strategy == "semantic":
            return self._split_text_semantic(full_text)
        else:
            return self._split_text_fixed(full_text)

    def _split_text_semantic(self, text: str, max_length: int = 2000) -> List[str]:
        """Split text based on semantic boundaries."""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed max length
            if len(current_chunk) + len(paragraph) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Paragraph itself is too long, split by sentences
                    sentences = self._split_into_sentences(paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > max_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                # Sentence is extremely long, split by words
                                chunks.append(sentence[:max_length])
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def _split_text_fixed(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to end at a word boundary
            if end < len(text):
                # Look for the last space within overlap distance
                for i in range(end, max(end - overlap, start), -1):
                    if text[i].isspace():
                        end = i
                        break

            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append(chunk)

            start = end - overlap if end < len(text) else end

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Fix hyphenated words split across lines
        text = re.sub(r'-\s*\n\s*', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def _format_table(self, table: List[List[str]]) -> str:
        """Format table data as text."""
        if not table:
            return ""

        formatted_rows = []
        for row in table:
            # Clean and join cells
            clean_row = [str(cell).strip() if cell else "" for cell in row]
            formatted_rows.append(" | ".join(clean_row))

        return "\n".join(formatted_rows)

    def _merge_and_split_semantic(self, chunks: List[str]) -> List[str]:
        """Merge small chunks and split large ones semantically."""
        merged_chunks = []

        for chunk in chunks:
            # If chunk is small and we have existing chunks, try to merge
            if len(chunk) < 500 and merged_chunks and len(merged_chunks[-1]) < 1500:
                merged_chunks[-1] += "\n\n" + chunk
            elif len(chunk) > 3000:
                # Split large chunks
                sub_chunks = self._split_text_semantic(chunk, max_length=2000)
                merged_chunks.extend(sub_chunks)
            else:
                merged_chunks.append(chunk)

        return merged_chunks

    def _merge_and_split_fixed(self, chunks: List[str]) -> List[str]:
        """Merge and split chunks using fixed-size strategy."""
        full_text = "\n\n".join(chunks)
        return self._split_text_fixed(full_text)

    def _basic_text_extraction(self, filepath: Path) -> List[str]:
        """Basic text extraction as final fallback."""
        try:
            # Try reading as text file (for converted PDFs)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                return self._split_text_fixed(text)
        except Exception:
            # Return error placeholder
            return [f"Could not extract text from {filepath}. Please ensure the file is a valid PDF."]

    def get_document_metadata(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Extract document metadata."""
        filepath = Path(filepath)
        metadata = {
            "filename": filepath.name,
            "file_size": filepath.stat().st_size,
            "processing_method": "unknown",
            "page_count": 0,
            "has_tables": False,
            "has_images": False
        }

        # Try to get metadata from PyMuPDF
        try:
            import fitz
            doc = fitz.open(str(filepath))
            metadata.update({
                "processing_method": "pymupdf",
                "page_count": doc.page_count,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            })

            # Check for tables and images
            for page in doc:
                if page.get_images():
                    metadata["has_images"] = True
                # Basic table detection (look for grid-like patterns)
                text = page.get_text()
                if "|" in text or "\t" in text:
                    metadata["has_tables"] = True

            doc.close()

        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")

        return metadata

    def is_available(self) -> bool:
        """Check if PDF processing is available."""
        return len(self.fallback_processors) > 0 or self.sherpa_processor is not None

    def get_available_processors(self) -> List[str]:
        """Get list of available processors."""
        processors = []
        if self.sherpa_processor:
            processors.append("llm_sherpa")
        processors.extend([name for name, _ in self.fallback_processors])
        return processors