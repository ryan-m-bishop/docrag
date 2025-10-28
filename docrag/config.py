from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
import json
from datetime import datetime


class CollectionMetadata(BaseModel):
    name: str
    source_type: str  # 'local', 'url', 'git'
    source_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    doc_count: int = 0
    description: Optional[str] = None


class GlobalConfig(BaseModel):
    active_collections: List[str] = []
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50


class ConfigManager:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.home() / ".docrag"
        self.config_file = self.base_path / "config.json"
        self.collections_dir = self.base_path / "collections"
        self.vectordb_dir = self.base_path / "vectordb"

    def init(self):
        """Initialize the docrag directory structure"""
        self.base_path.mkdir(exist_ok=True)
        self.collections_dir.mkdir(exist_ok=True)
        self.vectordb_dir.mkdir(exist_ok=True)

        if not self.config_file.exists():
            config = GlobalConfig()
            self.save_config(config)

    def load_config(self) -> GlobalConfig:
        if not self.config_file.exists():
            return GlobalConfig()
        with open(self.config_file) as f:
            return GlobalConfig(**json.load(f))

    def save_config(self, config: GlobalConfig):
        with open(self.config_file, 'w') as f:
            json.dump(config.model_dump(), f, indent=2, default=str)

    def get_collection_dir(self, name: str) -> Path:
        return self.collections_dir / name

    def get_collection_metadata(self, name: str) -> Optional[CollectionMetadata]:
        metadata_file = self.get_collection_dir(name) / "metadata.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file) as f:
            return CollectionMetadata(**json.load(f))

    def save_collection_metadata(self, metadata: CollectionMetadata):
        collection_dir = self.get_collection_dir(metadata.name)
        collection_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = collection_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)
