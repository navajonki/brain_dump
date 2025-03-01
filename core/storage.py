from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase
from config import config
from utils import get_logger
import os

logger = get_logger(__name__)

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config.paths.output_dir
        ))
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=config.openai.embedding_model
        )
    
    def ingest(self, chunks: List[Dict[str, Any]]):
        collection = self.client.get_or_create_collection(
            name=config.vector_db_settings["collection_name"],
            embedding_function=self.openai_ef
        )
        
        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks]
        )

class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
    
    def ingest(self, chunks: List[Dict[str, Any]]):
        with self.driver.session() as session:
            for chunk in chunks:
                session.execute_write(self._create_chunk_node, chunk)
                for topic in chunk["metadata"]["topics"]:
                    session.execute_write(self._create_topic_relationship, 
                                       chunk["id"], topic)
    
    def _create_chunk_node(self, tx, chunk: Dict[str, Any]):
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.summary = $summary,
            c.source_file = $source_file,
            c.timestamp_range = $timestamp_range
        """
        tx.run(query, 
               id=chunk["id"],
               text=chunk["text"],
               summary=chunk["metadata"]["summary"],
               source_file=chunk["metadata"]["source_file"],
               timestamp_range=chunk["metadata"]["timestamp_range"])
    
    def _create_topic_relationship(self, tx, chunk_id: str, topic: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (t:Topic {name: $topic})
        MERGE (c)-[:HAS_TOPIC]->(t)
        """
        tx.run(query, chunk_id=chunk_id, topic=topic) 