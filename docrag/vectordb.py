import lancedb
from pathlib import Path
from typing import List, Dict, Any, Optional


class VectorDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))

    def create_collection(self, collection_name: str):
        """Create a new collection (table) if it doesn't exist"""
        # LanceDB will create table on first insert
        pass

    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ):
        """
        Add documents with embeddings to a collection

        documents format: [
            {
                'id': 'doc_1',
                'text': 'content',
                'metadata': {'source': 'file.md', 'section': 'API'},
            }
        ]
        embeddings: List of embedding vectors
        """
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['vector'] = embedding

        table_name = f"collection_{collection_name}"

        # Create or append to table
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(documents)
        else:
            self.db.create_table(table_name, documents)

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in a collection"""
        table_name = f"collection_{collection_name}"

        if table_name not in self.db.table_names():
            return []

        table = self.db.open_table(table_name)
        results = table.search(query_embedding).limit(limit).to_list()

        return results

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        table_name = f"collection_{collection_name}"
        if table_name in self.db.table_names():
            self.db.drop_table(table_name)

    def list_collections(self) -> List[str]:
        """List all collections"""
        return [
            name.replace("collection_", "")
            for name in self.db.table_names()
            if name.startswith("collection_")
        ]

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        table_name = f"collection_{collection_name}"
        return table_name in self.db.table_names()
