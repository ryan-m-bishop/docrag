from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import hashlib
from datetime import datetime


class DocumentIndexer:
    def __init__(self, embeddings, vectordb, chunk_size=512, chunk_overlap=50):
        self.embeddings = embeddings
        self.vectordb = vectordb
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Splitter for markdown with headers
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        # Fallback splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def index_directory(
        self,
        collection_name: str,
        source_dir: Path,
        file_pattern: str = "**/*.md"
    ) -> int:
        """Index all files matching pattern in directory"""
        files = list(source_dir.glob(file_pattern))
        total_chunks = 0

        for file_path in files:
            chunks = self.process_file(file_path, collection_name)
            total_chunks += len(chunks)

        return total_chunks

    def process_file(self, file_path: Path, collection_name: str) -> List[Dict[str, Any]]:
        """Process a single file into chunks and index"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Detect file type and chunk accordingly
        if file_path.suffix == '.md':
            chunks = self._chunk_markdown(content)
        else:
            chunks = self.text_splitter.split_text(content)

        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Skip empty chunks
                continue

            doc_id = self._generate_id(file_path, i)
            documents.append({
                'id': doc_id,
                'text': chunk,
                'metadata': {
                    'source': str(file_path),
                    'chunk_index': i,
                    'collection': collection_name,
                    'indexed_at': datetime.now().isoformat()
                }
            })

        if not documents:
            return []

        # Generate embeddings
        texts = [doc['text'] for doc in documents]
        embeddings = self.embeddings.embed(texts)

        # Add to vector DB
        self.vectordb.add_documents(collection_name, documents, embeddings.tolist())

        return documents

    def _chunk_markdown(self, content: str) -> List[str]:
        """Chunk markdown while preserving structure"""
        try:
            md_chunks = self.md_splitter.split_text(content)
            # Further split if chunks are too large
            final_chunks = []
            for chunk in md_chunks:
                chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                if len(chunk_text) > self.chunk_size * 2:
                    final_chunks.extend(self.text_splitter.split_text(chunk_text))
                else:
                    final_chunks.append(chunk_text)
            return final_chunks
        except Exception:
            # Fallback to regular splitting
            return self.text_splitter.split_text(content)

    def _generate_id(self, file_path: Path, chunk_index: int) -> str:
        """Generate unique ID for a chunk"""
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
