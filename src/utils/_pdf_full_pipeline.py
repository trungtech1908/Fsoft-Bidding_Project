"""
Complete PDF to Qdrant RAG Pipeline
Handles native and scanned PDFs with tables, images, charts, formulas
Uses open-source tools: pdfplumber, Marker, Tesseract, BLIP-2, sentence-transformers
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image

import math
import re
from qdrant_client.http import models as rest

# Core PDF processing
import pdfplumber
import fitz  # PyMuPDF for image extraction

# Markdown parsing
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

# Embeddings and vector DB
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


@dataclass
class PDFChunk:
    """Unified chunk structure for Qdrant"""
    # Core content
    text: str  # Markdown text for embedding
    chunk_id: str
    doc_id: str
    chunk_index: int

    # Structure metadata
    section_hierarchy: List[str]
    header_level: int

    # Position
    page_num: int
    bbox: Dict[str, float]

    # Content classification
    chunk_type: str  # 'text', 'table', 'image', 'mixed', 'formula'
    contains_table: bool = False
    contains_image: bool = False
    contains_formula: bool = False

    # Table data
    table_data: Optional[List[Dict]] = None
    table_schema: Optional[List[str]] = None
    table_markdown: Optional[str] = None

    # Image data
    image_path: Optional[str] = None
    image_caption: Optional[str] = None

    # Formula data
    latex_formula: Optional[str] = None

    # Context windows
    prev_chunk_text: Optional[str] = None
    next_chunk_text: Optional[str] = None

    # Metrics
    token_count: int = 0

    # Custom metadata
    metadata: Optional[Dict] = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dict, removing None values"""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class PDFTypeDetector:
    """Detect if PDF is native, scanned, or mixed"""

    @staticmethod
    def detect(pdf_path: str) -> str:
        """Returns: 'native', 'scanned', or 'mixed'"""
        with pdfplumber.open(pdf_path) as pdf:
            native_pages = 0
            scanned_pages = 0

            for page in pdf.pages:
                text = page.extract_text()
                images = page.images

                # Heuristic: if page has text and few images, it's native
                if text and len(text.strip()) > 50:
                    native_pages += 1
                # If page has images but little/no text, it's scanned
                elif len(images) > 0 and (not text or len(text.strip()) < 50):
                    scanned_pages += 1

            total = len(pdf.pages)
            if scanned_pages > total * 0.7:
                return "scanned"
            elif native_pages > total * 0.7:
                return "native"
            else:
                return "mixed"


class NativePDFProcessor:
    """Process native/digital PDFs with pdfplumber"""

    def __init__(self, pdf_path: str, output_dir: str = "extracted_content"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

    def extract_to_markdown(self) -> str:
        """Extract PDF content as Markdown"""
        markdown_parts = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                md = self._process_page(page, page_num)
                if md:
                    markdown_parts.append(md)

        return "\n\n---\n\n".join(markdown_parts)

    def _process_page(self, page, page_num: int) -> str:
        """Process a single page"""
        markdown_lines = []

        # Extract tables first to know their positions
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables]

        # Extract text outside tables
        text = page.extract_text() or ""

        # Extract images
        images = self._extract_images(page, page_num)

        # Simple text processing (you can enhance with better layout analysis)
        markdown_lines.append(f"## Page {page_num}\n")
        markdown_lines.append(text)

        # Add tables
        for i, table in enumerate(tables):
            try:
                md_table = self._table_to_markdown(table)
                markdown_lines.append(f"\n{md_table}\n")
            except Exception as e:
                print(f"Warning: Failed to extract table {i} on page {page_num}: {e}")

        # Add images with captions
        for img_info in images:
            markdown_lines.append(f"\n![{img_info['caption']}]({img_info['path']})\n")

        return "\n".join(markdown_lines)

    def _table_to_markdown(self, table) -> str:
        """Convert pdfplumber table to Markdown"""
        data = table.extract()
        if not data or len(data) < 2:
            return ""

        headers = data[0]
        rows = data[1:]

        # Create markdown table
        md_lines = []
        header_line = "| " + " | ".join(str(h or "") for h in headers) + " |"
        md_lines.append(header_line)

        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        md_lines.append(separator)

        for row in rows:
            row_line = "| " + " | ".join(str(cell or "") for cell in row) + " |"
            md_lines.append(row_line)

        return "\n".join(md_lines)

    def _extract_images(self, page, page_num: int) -> List[Dict]:
        """Extract images from page"""
        images_info = []

        # Use PyMuPDF for better image extraction
        doc = fitz.open(self.pdf_path)
        fitz_page = doc[page_num - 1]

        for img_index, img in enumerate(fitz_page.get_images()):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Save image
                img_filename = f"page{page_num}_img{img_index}.png"
                img_path = self.images_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # Generate caption
                caption = self._generate_image_caption(img_path)

                images_info.append({
                    "path": f"images/{img_filename}",
                    "caption": caption
                })
            except Exception as e:
                print(f"Warning: Failed to extract image {img_index} on page {page_num}: {e}")

        doc.close()
        return images_info

    def _generate_image_caption(self, image_path: Path) -> str:
        """Generate caption for image using BLIP-2"""
        if not BLIP_AVAILABLE:
            return "Image"

        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

            image = Image.open(image_path).convert('RGB')
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            return caption
        except Exception as e:
            print(f"Warning: Failed to generate caption: {e}")
            return "Image"


class ScannedPDFProcessor:
    """Process scanned PDFs using OCR (Marker, MinerU, or Tesseract)"""

    def __init__(self, pdf_path: str, ocr_engine: str = "marker"):
        self.pdf_path = pdf_path
        self.ocr_engine = ocr_engine  # 'marker', 'mineru', 'tesseract'

    def extract_to_markdown(self) -> str:
        """Extract scanned PDF using OCR"""
        if self.ocr_engine == "marker":
            return self._use_marker()
        elif self.ocr_engine == "mineru":
            return self._use_mineru()
        elif self.ocr_engine == "tesseract":
            return self._use_tesseract()
        else:
            raise ValueError(f"Unknown OCR engine: {self.ocr_engine}")

    def _use_marker(self) -> str:
        """Use Marker for OCR (best quality)"""
        try:
            # Marker command-line usage
            output_dir = Path("marker_output")
            output_dir.mkdir(exist_ok=True)

            cmd = [
                "marker_single",
                self.pdf_path,
                str(output_dir),
                "--batch_multiplier", "2"
            ]

            subprocess.run(cmd, check=True)

            # Read generated markdown
            md_file = output_dir / f"{Path(self.pdf_path).stem}.md"
            if md_file.exists():
                return md_file.read_text(encoding='utf-8')
            else:
                raise FileNotFoundError("Marker output not found")

        except Exception as e:
            print(f"Error using Marker: {e}")
            print("Falling back to Tesseract...")
            return self._use_tesseract()

    def _use_mineru(self) -> str:
        """Use MinerU for fast OCR"""
        try:
            # MinerU command-line usage
            output_dir = Path("mineru_output")
            output_dir.mkdir(exist_ok=True)

            cmd = [
                "magic-pdf",
                "-p", self.pdf_path,
                "-o", str(output_dir),
                "-m", "auto"
            ]

            subprocess.run(cmd, check=True)

            # Read generated markdown
            md_file = output_dir / "auto" / f"{Path(self.pdf_path).stem}.md"
            if md_file.exists():
                return md_file.read_text(encoding='utf-8')
            else:
                raise FileNotFoundError("MinerU output not found")

        except Exception as e:
            print(f"Error using MinerU: {e}")
            print("Falling back to Tesseract...")
            return self._use_tesseract()

    def _use_tesseract(self) -> str:
        """Use Tesseract OCR (most compatible)"""
        try:
            # Use ocrmypdf to add OCR layer
            output_pdf = Path("ocr_output.pdf")

            cmd = [
                "ocrmypdf",
                "--skip-text",
                "--deskew",
                self.pdf_path,
                str(output_pdf)
            ]

            subprocess.run(cmd, check=True)

            # Now extract text from OCR'd PDF
            processor = NativePDFProcessor(str(output_pdf))
            markdown = processor.extract_to_markdown()

            # Cleanup
            output_pdf.unlink()

            return markdown

        except Exception as e:
            raise RuntimeError(f"All OCR methods failed: {e}")


class MarkdownParser:
    """Parse Markdown structure using markdown-it-py"""

    def __init__(self, markdown_text: str):
        self.markdown_text = markdown_text
        # Enable tables and other components
        self.md_parser = MarkdownIt("commonmark").enable('table')

    def parse_structure(self) -> List[Dict]:
        """Parse Markdown into hierarchical sections"""
        # DEBUG: Check if text actually exists
        if not self.markdown_text or not self.markdown_text.strip():
            print("Warning: Markdown text is empty.")
            return []

        tokens = self.md_parser.parse(self.markdown_text)
        tree = SyntaxTreeNode(tokens)

        sections = []
        current_hierarchy = []
        # Initialize with a default section to catch content before the first header
        current_section = {"content": [], "header": "Introduction", "level": 0}

        for node in tree.walk():
            if node.type == "heading":
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append({
                        "hierarchy": current_hierarchy.copy(),
                        "header": current_section["header"],
                        "level": current_section["level"],
                        "content": "\n".join(current_section["content"]),
                        "contains_table": any("|" in c for c in current_section["content"]),
                        "contains_image": any("![" in c for c in current_section["content"])
                    })

                # Start new section
                try:
                    level = int(node.tag.replace('h', '')) # Get h1, h2, h3...
                except:
                    level = 1

                header_text = self._extract_text_from_node(node)

                # Update hierarchy
                if level <= len(current_hierarchy):
                    current_hierarchy = current_hierarchy[:level-1] + [header_text]
                else:
                    current_hierarchy.append(header_text)

                current_section = {
                    "content": [],
                    "header": header_text,
                    "level": level
                }

            # Catch paragraphs, tables, fences (code), and handle list items via recursion
            # We don't explicitly check for "list_item" because the walker will visit
            # the "paragraph" inside the list item anyway.
            elif node.type in ["paragraph", "table", "fence", "code_block", "blockquote", "html_block"]:
                content = self._render_node(node)
                if content and content.strip():
                    current_section["content"].append(content)

        # Save last section
        if current_section["content"]:
            sections.append({
                "hierarchy": current_hierarchy,
                "header": current_section["header"],
                "level": current_section["level"],
                "content": "\n".join(current_section["content"]),
                "contains_table": any("|" in c and "---" in c for c in current_section["content"]),
                "contains_image": any("![" in c for c in current_section["content"])
            })

        return sections

    def _extract_text_from_node(self, node) -> str:
        """Extract text from a node recursively"""
        if hasattr(node, 'content') and node.content:
            return node.content

        text_parts = []
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                text_parts.append(self._extract_text_from_node(child))

        return " ".join(t for t in text_parts if t)

    def _render_node(self, node) -> str:
        """Render a node back to Markdown"""
        if node.type == "paragraph":
            return self._extract_text_from_node(node)
        elif node.type == "table":
            return node.markup if hasattr(node, 'markup') else self._extract_text_from_node(node)
        elif node.type in ["fence", "code_block"]:
            # Handle code blocks
            lang = node.info if hasattr(node, 'info') else ""
            content = self._extract_text_from_node(node)
            return f"```{lang}\n{content}\n```"
        elif node.type == "blockquote":
            return f"> {self._extract_text_from_node(node)}"
        elif node.type == "html_block":
            return node.content

        return ""
class SmartChunker:
    """Create optimized chunks from parsed sections"""

    def __init__(self, doc_id: str, metadata: Dict = None):
        self.doc_id = doc_id
        self.metadata = metadata or {}

    def create_chunks(self, sections: List[Dict],
                     chunk_size: int = 512,
                     overlap: int = 50) -> List[PDFChunk]:
        """Create chunks with overlap and context"""
        chunks = []
        chunk_idx = 0

        for section in sections:
            section_chunks = self._chunk_section(
                section, chunk_size, overlap, chunk_idx
            )
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)

        # Add context windows
        self._add_context_windows(chunks)

        return chunks

    def _chunk_section(self, section: Dict, chunk_size: int,
                       overlap: int, start_idx: int) -> List[PDFChunk]:
        """Chunk a single section"""
        content = section["content"]
        hierarchy = section["hierarchy"]
        level = section["level"]

        # Estimate tokens (4 chars ≈ 1 token)
        char_size = chunk_size * 4

        if len(content) <= char_size:
            # Small section - single chunk
            return [PDFChunk(
                text=content,
                chunk_id=f"{self.doc_id}_chunk_{start_idx}",
                doc_id=self.doc_id,
                chunk_index=start_idx,
                section_hierarchy=hierarchy,
                header_level=level,
                page_num=1,  # You can track actual page numbers
                bbox={"x0": 0, "y0": 0, "x1": 0, "y1": 0},
                chunk_type=self._classify_chunk_type(section),
                contains_table=section.get("contains_table", False),
                contains_image=section.get("contains_image", False),
                token_count=len(content) // 4,
                metadata=self.metadata
            )]

        # Large section - split with overlap
        sub_chunks = self._split_with_overlap(content, char_size, overlap * 4)

        result = []
        for i, sub_content in enumerate(sub_chunks):
            chunk = PDFChunk(
                text=sub_content,
                chunk_id=f"{self.doc_id}_chunk_{start_idx + i}",
                doc_id=self.doc_id,
                chunk_index=start_idx + i,
                section_hierarchy=hierarchy + [f"Part {i+1}"],
                header_level=level + 1,
                page_num=1,
                bbox={"x0": 0, "y0": 0, "x1": 0, "y1": 0},
                chunk_type=self._classify_chunk_type(section),
                contains_table=section.get("contains_table", False) and "|" in sub_content,
                contains_image=section.get("contains_image", False) and "![" in sub_content,
                token_count=len(sub_content) // 4,
                metadata=self.metadata
            )
            result.append(chunk)

        return result

    def _classify_chunk_type(self, section: Dict) -> str:
        """Classify chunk type based on content"""
        has_table = section.get("contains_table", False)
        has_image = section.get("contains_image", False)
        has_formula = "$$" in section.get("content", "")

        if has_table and has_image:
            return "mixed"
        elif has_table:
            return "table"
        elif has_image:
            return "image"
        elif has_formula:
            return "formula"
        else:
            return "text"

    def _split_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text with overlap at sentence boundaries"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > chunk_size and current_chunk:
                # Save chunk
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else ''
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _add_context_windows(self, chunks: List[PDFChunk], window_size: int = 200):
        """Add prev/next context to chunks"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev = chunks[i-1].text[:window_size]
                chunk.prev_chunk_text = prev + "..." if len(chunks[i-1].text) > window_size else prev

            if i < len(chunks) - 1:
                next_text = chunks[i+1].text[:window_size]
                chunk.next_chunk_text = next_text + "..." if len(chunks[i+1].text) > window_size else next_text

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

class PDFToQdrantPipeline:
    """Complete pipeline: PDF → Markdown → Chunks → Qdrant"""

    def __init__(self,
                 embedding_model: str = "Qwen/Qwen3-0.6B",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
    
    def process_pdf(self, 
                    pdf_path: str,
                    doc_id: Optional[str] = None,
                    metadata: Optional[Dict] = None,
                    chunk_size: int = 512,
                    overlap: int = 50,
                    ocr_engine: str = "marker") -> List[PDFChunk]:
        """Process PDF end-to-end"""
        
        # Generate doc_id
        if doc_id is None:
            doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:16]
        
        print(f"Processing PDF: {pdf_path}")
        
        # Step 1: Detect PDF type
        pdf_type = PDFTypeDetector.detect(pdf_path)
        print(f"PDF Type: {pdf_type}")
        
        # Step 2: Extract to Markdown
        if pdf_type == "native":
            processor = NativePDFProcessor(pdf_path)
            markdown = processor.extract_to_markdown()
        elif pdf_type == "scanned":
            processor = ScannedPDFProcessor(pdf_path, ocr_engine=ocr_engine)
            markdown = processor.extract_to_markdown()
        else:  # mixed
            # Process native first, then OCR failed pages
            processor = NativePDFProcessor(pdf_path)
            markdown = processor.extract_to_markdown()
        
        print(f"✓ Extracted {len(markdown)} characters")
        
        # Step 3: Parse Markdown structure
        parser = MarkdownParser(markdown)
        sections = parser.parse_structure()
        print(f"✓ Parsed {len(sections)} sections")
        
        # Step 4: Create chunks
        chunker = SmartChunker(doc_id, metadata)
        chunks = chunker.create_chunks(sections, chunk_size, overlap)
        print(f"✓ Created {len(chunks)} chunks")
        
        return chunks
    
    def upload_to_qdrant(self, 
                        chunks: List[PDFChunk],
                        collection_name: str = "documents"):
        """Upload chunks to Qdrant"""
        
        # Create collection if not exists
        try:
            self.qdrant_client.get_collection(collection_name)
        except:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created collection: {collection_name}")
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        # Create points
        points = [
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=chunk.to_dict()
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Upload
        print("Number of chunks:", len(chunks))
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"✓ Uploaded {len(points)} points to Qdrant")
    
    def search(self, 
              query: str,
              collection_name: str = "documents", # <--- VERIFY THIS MATCHES UPLOAD
              filters: Optional[Dict] = None,
              limit: int = 5) -> List[Dict]:
        
        # 1. Inspect Collection
        try:
            col_info = self.qdrant_client.get_collection(collection_name)
            if col_info.points_count == 0:
                print(f"WARNING: Collection '{collection_name}' is empty.")
                return []
        except Exception:
            print(f"ERROR: Collection '{collection_name}' does not exist.")
            return []

        # 2. Encode
        query_vector = self.embedding_model.encode([query])[0]
        
        # 3. Search
        results = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            query_filter=filters,
            limit=limit,
            with_payload=True
        ).points
        
        # 4. Return formatted
        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text", hit.payload.get("page_content", "")),
                "metadata": hit.payload
            }
            for hit in results
        ]
    

    def search_bm25(self, 
                    query: str,
                    collection_name: str = "documents",
                    limit: int = 5) -> List[Dict]:
        """
        Primitive BM25 search using Qdrant filters for retrieval 
        and local Python logic for ranking.
        """
        # 1. Clean and tokenize query into unique keywords
        keywords = list(set(re.findall(r'\w+', query.lower())))
        if not keywords:
            return []

        # 2. Retrieve candidate pool from Qdrant
        # We fetch a larger pool (e.g., 100) to ensure we have enough to rank
        try:
            # We try to find documents containing ANY of the keywords
            candidates, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=rest.Filter(
                    should=[
                        rest.FieldCondition(key="text", match=rest.MatchText(text=word))
                        for word in keywords
                    ]
                ),
                limit=100,
                with_payload=True
            )
        except Exception:
            # Fallback if 'MatchText' is not supported by your field configuration
            candidates, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True
            )

        if not candidates:
            return []

        # 3. BM25 Scoring Logic
        k1 = 1.5  # Term frequency saturation
        b = 0.75  # Length normalization
        
        # Pre-process candidate pool stats
        docs = []
        for cand in candidates:
            text = cand.payload.get("text", cand.payload.get("page_content", "")).lower()
            tokens = re.findall(r'\w+', text)
            docs.append({
                "tokens": tokens,
                "len": len(tokens),
                "payload": cand.payload
            })

        N = len(docs)
        avg_dl = sum(d["len"] for d in docs) / N if N > 0 else 1
        
        scored_results = []
        for doc in docs:
            score = 0.0
            for word in keywords:
                # Term Frequency (TF): occurrences in this doc
                tf = doc["tokens"].count(word)
                if tf == 0: continue
                
                # Document Frequency (DF): occurrences in the candidate pool
                df = sum(1 for d in docs if word in d["tokens"])
                
                # Inverse Document Frequency (IDF) - Smoothed
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                
                # Full BM25 Formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc["len"] / avg_dl))
                score += idf * (numerator / denominator)
            
            if score > 0:
                scored_results.append({
                    "score": round(score, 4),
                    "text": doc["payload"].get("text", doc["payload"].get("page_content", "")),
                    "metadata": doc["payload"]
                })

        # 4. Sort by score and return top results
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]