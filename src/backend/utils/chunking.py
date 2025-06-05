# src/backend/utils/chunking.py

# - fixed/window chunking enhanced to - Semantic chunking: NLTK/sentence-based chunking
"""
Text chunking utility functions for RAG pipeline.

This module provides functions to split large text documents into smaller,
manageable chunks for embedding and retrieval operations.
"""
from pathlib import Path
# from backend.utils.logging_config import setup_logger
from utils.logging_config import setup_logger
import nltk
nltk.download('punkt_tab')
import re

logger = setup_logger()

# # Fixed Window Chunking:
# def split_text(text, chunk_size=150):
#     """
#     Split text into chunks of approximately equal word count.
    
#     Divides input text into smaller chunks by splitting on whitespace
#     and grouping words into chunks of the specified size. This is useful
#     for processing large documents in RAG (Retrieval-Augmented Generation)
#     pipelines where embeddings have token limits.
    
#     Parameters
#     ----------
#     text : str
#         The input text to be split into chunks.
#     chunk_size : int, optional
#         The target number of words per chunk (default: 400).
#         Actual chunk sizes may vary slightly based on word boundaries.
        
#     Returns
#     -------
#     list of str
#         A list of text chunks, where each chunk contains approximately
#         chunk_size words. The last chunk may contain fewer words.
#         Returns an empty list if input text is empty.
        
#     Raises
#     ------
#     Exception
#         If an error occurs during the chunking process.
#         The original exception is logged and re-raised.
    
#     Notes
#     -----
#     - Chunking is performed at word boundaries, preserving complete words
#     - Whitespace is normalized (multiple spaces become single spaces)
#     - Original paragraph structure is not preserved
#     - For semantic chunking or overlap between chunks, consider using
#       more advanced chunking strategies
#     - Chunk size of 400 words ≈ 500-600 tokens for most LLMs
#     """

#     logger.info(f"Starting text chunking, input length: {len(text)} chars, chunk_size: {chunk_size}")
#     words = text.split()
#     chunks = []
#     try:
#         for i in range(0, len(words), chunk_size):
#             chunk = " ".join(words[i:i+chunk_size])
#             chunks.append(chunk)
#             logger.debug(f"Created chunk {len(chunks)} with {len(chunk.split())} words")
#         logger.info(f"Chunking complete: {len(chunks)} chunks generated")
#         return chunks
#     except Exception as e:
#         logger.error(f"Error during chunking: {str(e)}")
#         raise

# Heading-Based Chunking (Structural Chunking for Parent-Child Links):
def extract_headings(text):
    # Find heading lines (simple: all-caps, starts with number, etc.)
    lines = text.split('\n')
    headings = []
    for line in lines:
        if re.match(r'^(\d+\.\s+|[A-Z][A-Z\s]{5,})', line.strip()):
            headings.append(line.strip())
    return headings


# NLTK Sentence-Based Chunking (Semantic Chunking):
def semantic_sentence_chunk_with_parent(text, sentences_per_chunk=5):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    parents = []

    # Use headings in text, assign parent for each chunk
    headings = extract_headings(text)
    current_parent = None
    sentence_idx = 0
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i+sentences_per_chunk])
        # Simple parent assignment: previous heading or None
        # (Advanced: Track sentence indices near headings)
        for h in headings:
            if h in chunk:
                current_parent = h
        parents.append(current_parent)
        chunks.append(chunk)

    return chunks, parents


## Combine chunking with metadata attachment:
# def chunk_documents(texts: dict, chunk_size=150):
#     """
#     Chunk multiple documents (dict {filename: text}) and attach metadata.
#     Returns: List[dict] — each dict: {'chunk': str, 'source_pdf': filename, 'chunk_index': int}
#     """
#     chunks_with_meta = []
#     for filename, text in texts.items():
#         # print(f"{filename}: length={len(text)}")    # debugging
#         if not text.strip():
#             logger.warning(f"Skipping {filename}: text empty after preprocessing")
#             continue
#         chunks = split_text(text, chunk_size)
#         for idx, chunk in enumerate(chunks):
#             chunks_with_meta.append({
#                 "chunk": chunk,
#                 "source_pdf": filename,
#                 "chunk_index": idx
#                 # Optionally: add more metadata like page number if trackable
#             })
#     return chunks_with_meta

# Combine chunking with metadata attachment and Parent-Child Links:
def chunk_documents(texts: dict, chunk_size=150):
    chunks_with_meta = []
    for filename, text in texts.items():
        if not text.strip():
            logger.warning(f"Skipping {filename}: text empty after preprocessing")
            continue
        chunks, parents = semantic_sentence_chunk_with_parent(text, sentences_per_chunk=5)
        for idx, (chunk, parent_heading) in enumerate(zip(chunks, parents)):
            chunks_with_meta.append({
                "chunk": chunk,
                "parent": parent_heading,
                "source_pdf": filename,
                "chunk_index": idx
            })
    return chunks_with_meta


# # Test the chunking function when run as a script

# if __name__ == "__main__":
#     import sys
#     from backend.utils.pdf_utils import extract_pdfs_with_metadata
#     from backend.utils.preprocessing import preprocess_all_documents

#     if len(sys.argv) < 2:
#         print("Usage: python -m backend.utils.chunking <pdf1> [<pdf2> ...] [chunk_size]")
#         sys.exit(1)
#     pdf_input = sys.argv[1:]
#     chunk_size = int(sys.argv[-1]) if len(sys.argv) > 2 and sys.argv[-1].isdigit() else 150
#     if chunk_size != 150:
#         pdf_input = pdf_input[:-1]

#     # Extract
#     pdf_results = extract_pdfs_with_metadata(pdf_input)
#     texts = {fn: data["text"] for fn, data in pdf_results.items()}
#     cleaned = preprocess_all_documents(texts)

#     # Chunk
#     all_chunks = chunk_documents(cleaned, chunk_size=chunk_size)
#     print(f"\nTotal chunks: {len(all_chunks)} (chunk_size={chunk_size})")
#     chunk_sample = all_chunks[0]
#     print(f"\nSample chunk: {chunk_sample['source_pdf']} [idx={chunk_sample['chunk_index']}]\n----\n{chunk_sample['chunk'][:200]}...\n----")
#     print("\nSummary per PDF (chunk counts):")
#     from collections import Counter
#     counts = Counter([c['source_pdf'] for c in all_chunks])
#     for fn, count in counts.items():
#         print(f"  {fn}: {count} chunks")

