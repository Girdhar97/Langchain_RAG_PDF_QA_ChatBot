# src/backend/db/faiss_index.py
"""
FAISS vector index management for similarity search in RAG pipelines.
"""

from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# from backend.utils.logging_config import setup_logger
from utils.logging_config import setup_logger
logger = setup_logger()

def build_faiss_index(embeddings):
    logger.info(f"Building FAISS index for embeddings of shape {embeddings.shape}")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info("FAISS index built and embeddings added")

    return index

def search_similar_chunks(index, embedding_model, query, all_chunks_with_meta, k=3):
    """
    Given a user query, embed and search top-k similar chunks (with metadata).
    Args:
        index: FAISS index object
        embedding_model: sentence transformer model
        query: str, user question
        all_chunks_with_meta: list of dict, each dict has 'chunk', 'source_pdf', etc.
        k: int, number of results
    Returns:
        matches: list of dict, each dict merges all_chunks_with_meta[idx] + {'distance': ...}
    """
    query_emb = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, k)
    matched_indices = indices[0]
    matched_distances = distances[0]
    matches = []
    for idx, dist in zip(matched_indices, matched_distances):
        chunk_data = dict(all_chunks_with_meta[idx])  # copy the dict
        chunk_data['distance'] = float(dist)
        matches.append(chunk_data)

    return matches 


# # Test Block:
# if __name__ == "__main__":
#     import sys
#     from backend.utils.pdf_utils import extract_pdfs_with_metadata
#     from backend.utils.preprocessing import preprocess_all_documents
#     from backend.utils.chunking import chunk_documents
#     from backend.api.llm.embedding import get_embedding_model, generate_embeddings

#     # Usage: python -m backend.db.faiss_index <pdf1> [<pdf2> ...] [chunk_size]
#     args = sys.argv[1:]
#     if not args:
#         print("Usage: python -m backend.db.faiss_index <pdf1> [<pdf2> ...] [chunk_size]")
#         sys.exit(1)
#     if args[-1].isdigit():
#         chunk_size = int(args[-1])
#         pdf_input = args[:-1]
#     else:
#         chunk_size = 150
#         pdf_input = args

#     print(f"Extracting and preparing {len(pdf_input)} PDF(s)...")
#     pdf_results = extract_pdfs_with_metadata(pdf_input)
#     texts = {fn: d["text"] for fn, d in pdf_results.items()}
#     cleaned = preprocess_all_documents(texts)
#     all_chunks_with_meta = chunk_documents(cleaned, chunk_size=chunk_size)

#     print(f"\nTotal chunks: {len(all_chunks_with_meta)} (chunk_size={chunk_size})")
#     chunk_texts = [c["chunk"] for c in all_chunks_with_meta]

#     # Embedding
#     print("\nLoading embedding model and building index...")
#     embedding_model = get_embedding_model()
#     embeddings = generate_embeddings(embedding_model, chunk_texts)
#     index = build_faiss_index(embeddings)

#     import os
#     import json

#     # save index and metadata for later use
#     os.makedirs("/mnt/s/Projects/Langchain_RAG_PDF_QA_ChatBot/outputs/", exist_ok=True)

#     import datetime
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     faiss_index_path = f"/mnt/s/Projects/Langchain_RAG_PDF_QA_ChatBot/outputs/faiss_indices_{timestamp}.idx"
#     chunks_metadata_path = f"/mnt/s/Projects/Langchain_RAG_PDF_QA_ChatBot/outputs/chunks_metadata_{timestamp}.json"

#     faiss.write_index(index, faiss_index_path)
#     with open(chunks_metadata_path, "w", encoding="utf-8") as f:
#         json.dump(all_chunks_with_meta, f, indent=2, ensure_ascii=False)

#     logger.info("FAISS index built and chunkmetadata saved")

#     # Run real queries
#     test_queries = [
#         "What is the main topic?",
#         "Tell me about generative AI.",
#     ]
#     for query in test_queries:
#         print(f"\nQuery: '{query}'\n{'-'*60}")
#         matches = search_similar_chunks(
#             index, embedding_model, query, all_chunks_with_meta, k=3
#         )
#         print("Top results (with metadata):")
#         for i, res in enumerate(matches, 1):
#             print(f"  {i}. [{res['source_pdf']}, chunk {res['chunk_index']}, dist={res['distance']:.4f}]")
#             print(f"     {res['chunk'][:120]}...")
#     print("\nTest complete! Check 'logs/rag_backend.log' for full debug info.")

