# src/rag_engine/vector_store.py
import chromadb
from chromadb.config import Settings
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# ── Embedding Model ────────────────────────────────────────────────
class EmbeddingModel:
    """
    Loads sentence transformer model for converting text to vectors.
    all-MiniLM-L6-v2 is lightweight (80MB) and runs well on your GPU.
    It converts any text into a 384-dimensional vector.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded")

    def embed(self, texts: list) -> list:
        """Convert list of texts to list of vectors"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,        # Process 32 chunks at a time (safe for 8GB RAM)
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for better similarity search
        )
        return embeddings.tolist()

    def embed_single(self, text: str) -> list:
        """Convert single text to vector (used for query embedding)"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding.tolist()


# ── Vector Store ───────────────────────────────────────────────────
class MSMEVectorStore:
    """
    ChromaDB vector store with 3 separate collections:
    1. regulatory_docs  — RBI/SEBI/Government regulations
    2. tax_docs         — GST/Income Tax documents
    3. financial_data   — Business-specific financial data
    
    Separating collections means queries go to the RIGHT place.
    A tax question won't search through loan regulations.
    """

    def __init__(self, persist_dir: str = "./data/chromadb"):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir
        )

        self.embedding_model = EmbeddingModel()

        # Create or load collections
        self.collections = {
            "regulatory": self.client.get_or_create_collection(
                name="regulatory_docs",
                metadata={"description": "RBI, SEBI, Government regulations for MSMEs"}
            ),
            "tax": self.client.get_or_create_collection(
                name="tax_docs",
                metadata={"description": "GST and Income Tax documents"}
            ),
            "scheme": self.client.get_or_create_collection(
                name="scheme_docs",
                metadata={"description": "Government schemes like MUDRA, Udyam"}
            ),
            "research": self.client.get_or_create_collection(
                name="research_docs",
                metadata={"description": "SIDBI reports and industry benchmarks"}
            ),
            "financial": self.client.get_or_create_collection(
                name="financial_data",
                metadata={"description": "Business-specific GST and bank data"}
            )
        }

        print("✅ Vector store initialized with 5 collections")

    def _get_collection(self, category: str):
        """Route to correct collection based on document category"""
        routing = {
            "regulatory": "regulatory",
            "tax": "tax",
            "scheme": "scheme",
            "research": "research",
            "compliance": "scheme",
            "financial": "financial",
            "general": "regulatory"
        }
        return self.collections.get(routing.get(category, "regulatory"))

    def ingest_knowledge_base(self, chunks_file: str = "data/processed/knowledge_base_chunks.json"):
        """
        Ingests all 441 document chunks into ChromaDB.
        Embeds each chunk and stores with metadata.
        """
        print("\n" + "="*50)
        print("Ingesting Knowledge Base into Vector Store")
        print("="*50)

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Total chunks to ingest: {len(chunks)}")

        # Group chunks by category for efficient batch processing
        category_groups = {}
        for chunk in chunks:
            category = chunk["metadata"]["category"]
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(chunk)

        total_ingested = 0

        for category, category_chunks in category_groups.items():
            print(f"\n📥 Ingesting {len(category_chunks)} chunks — category: {category}")

            collection = self._get_collection(category)

            # Process in batches of 50 (safe for 8GB RAM)
            batch_size = 50
            for i in range(0, len(category_chunks), batch_size):
                batch = category_chunks[i:i + batch_size]

                texts = [c["text"] for c in batch]
                ids = [c["chunk_id"] for c in batch]

                # Build metadata for each chunk
                metadatas = []
                for c in batch:
                    metadatas.append({
                        "source_file": c["metadata"]["source_file"],
                        "category": c["metadata"]["category"],
                        "authority": c["metadata"]["authority"],
                        "chunk_index": str(c["chunk_index"]),
                        "topics": ", ".join(c["metadata"]["topics"][:3]),
                        "relevance_keywords": ", ".join(c["metadata"]["relevance_keywords"])
                    })

                # Generate embeddings
                embeddings = self.embedding_model.embed(texts)

                # Check for duplicate IDs and skip them
                existing = collection.get(ids=ids)
                existing_ids = set(existing["ids"])
                
                new_indices = [j for j, id in enumerate(ids) if id not in existing_ids]
                
                if not new_indices:
                    print(f"   ⏭️  Batch {i//batch_size + 1}: already ingested, skipping")
                    continue

                # Filter to only new items
                new_texts = [texts[j] for j in new_indices]
                new_ids = [ids[j] for j in new_indices]
                new_metadatas = [metadatas[j] for j in new_indices]
                new_embeddings = [embeddings[j] for j in new_indices]

                collection.add(
                    documents=new_texts,
                    embeddings=new_embeddings,
                    metadatas=new_metadatas,
                    ids=new_ids
                )

                total_ingested += len(new_indices)
                print(f"   ✅ Batch {i//batch_size + 1}: {len(new_indices)} chunks ingested")

        print(f"\n✅ Total ingested: {total_ingested} chunks")
        return total_ingested

    def ingest_financial_data(self, business_id: str, gst_data: list, bank_data: list):
        """
        Ingests business-specific financial data as searchable text.
        Each business gets its own searchable financial summary.
        """
        collection = self.collections["financial"]
        documents = []
        ids = []
        metadatas = []

        # Convert GST records to searchable text summaries
        for record in gst_data:
            text = (
                f"GST Return for {record.get('business_name', '')} "
                f"Period: {record.get('return_period', '')}. "
                f"Taxable value: Rs {record.get('taxable_value', 0):,.0f}. "
                f"Total tax collected: Rs {record.get('total_tax_collected', 0):,.0f}. "
                f"ITC utilized: Rs {record.get('itc_utilized', 0):,.0f}. "
                f"Net GST payable: Rs {record.get('net_gst_payable', 0):,.0f}. "
                f"Filed on time: {record.get('filed_on_time', True)}. "
                f"Late by {record.get('late_filing_days', 0)} days."
            )
            doc_id = f"{business_id}_gst_{record.get('return_period', '').replace('-', '_')}"
            documents.append(text)
            ids.append(doc_id)
            metadatas.append({
                "business_id": business_id,
                "data_type": "gst_return",
                "period": record.get("return_period", ""),
                "category": "financial",
                "authority": "business_data"
            })

        if documents:
            embeddings = self.embedding_model.embed(documents)
            
            # Check existing before adding
            existing = collection.get(ids=ids)
            existing_ids = set(existing["ids"])
            new_indices = [i for i, id in enumerate(ids) if id not in existing_ids]
            
            if new_indices:
                collection.add(
                    documents=[documents[i] for i in new_indices],
                    embeddings=[embeddings[i] for i in new_indices],
                    metadatas=[metadatas[i] for i in new_indices],
                    ids=[ids[i] for i in new_indices]
                )
                print(f"✅ Ingested {len(new_indices)} GST records for {business_id}")

    def semantic_search(self, query: str, category: str = None,
                        n_results: int = 5) -> list:
        """
        Searches for relevant chunks using semantic similarity.
        If category specified, searches only that collection.
        Otherwise searches all collections.
        """
        query_embedding = self.embedding_model.embed_single(query)
        results = []

        collections_to_search = (
            [self._get_collection(category)] if category
            else list(self.collections.values())
        )

        for collection in collections_to_search:
            if collection.count() == 0:
                continue

            try:
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, collection.count()),
                    include=["documents", "metadatas", "distances"]
                )

                for i, doc in enumerate(result["documents"][0]):
                    results.append({
                        "text": doc,
                        "metadata": result["metadatas"][0][i],
                        "score": 1 - result["distances"][0][i],  # Convert distance to similarity
                        "collection": collection.name
                    })
            except Exception as e:
                continue

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n_results]

    def get_stats(self):
        """Shows how many chunks are in each collection"""
        print("\n📊 Vector Store Statistics:")
        print("-"*35)
        total = 0
        for name, collection in self.collections.items():
            count = collection.count()
            total += count
            print(f"   {name:15} : {count:4} chunks")
        print(f"   {'TOTAL':15} : {total:4} chunks")
        print("-"*35)


# ── Main Runner ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Initialize vector store
    store = MSMEVectorStore()

    # Ingest knowledge base documents
    store.ingest_knowledge_base()

    # Ingest financial data for each business
    print("\n📥 Ingesting business financial data...")
    gst_df = pd.read_csv("data/processed/gst_returns.csv")
    
    for business_id in gst_df["business_id"].unique():
        biz_gst = gst_df[gst_df["business_id"] == business_id].to_dict("records")
        store.ingest_financial_data(business_id, biz_gst, [])
        print(f"   ✅ {business_id} financial data ingested")

    # Show stats
    store.get_stats()

    # Test semantic search
    print("\n🔍 Testing Semantic Search...")
    print("="*50)

    test_queries = [
        ("What is the GST rate and how do I file returns?", "tax"),
        ("How can I get a MUDRA loan for my business?", "scheme"),
        ("What relief options do I have if I cannot repay my loan?", "regulatory"),
    ]

    for query, category in test_queries:
        print(f"\nQuery: '{query}'")
        results = store.semantic_search(query, category=category, n_results=2)
        for i, r in enumerate(results):
            print(f"  Result {i+1} (score: {r['score']:.3f}):")
            print(f"  Source: {r['metadata'].get('source_file', 'N/A')}")
            print(f"  Text: {r['text'][:150]}...")
        print()