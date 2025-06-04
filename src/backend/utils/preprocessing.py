# src/backend/utils/preprocessing.py
"""
Document preprocessing utilities for cleaning and normalizing text.

WHY THIS FILE:
- Remove noise from extracted text
- Normalize whitespace, encoding
- Remove headers/footers patterns
- Improve downstream quality (chunks, embeddings)

When using with extract_pdfs_with_metadata(), 
pass only the .text for each file!
Example:
  texts = {fn: d['text'] for fn, d in pdf_results.items()}
  cleaned = preprocess_all_documents(texts)
"""

import re
from typing import List
# from backend.utils.logging_config import setup_logger
from utils.logging_config import setup_logger

logger = setup_logger()


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Operations:
    1. Remove excessive whitespace
    2. Normalize line breaks
    3. Remove special characters (optional)
    4. Fix encoding issues
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    logger.debug(f"Cleaning text (length: {len(text)} chars)")
    
    # Remove null bytes (can cause encoding issues)
    text = text.replace('\x00', '')
    
    # Normalize line breaks (Windows/Unix/Mac)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    logger.debug(f"Cleaned text (length: {len(text)} chars)")
    return text


def remove_headers_footers(text: str, patterns: List[str] = None) -> str:
    """
    Remove common header/footer patterns.
    
    WHY THIS FUNCTION:
    - PDFs often have page numbers, headers, footers
    - These repeat on every page → noise in embeddings
    - Removal improves retrieval quality
    
    Args:
        text: Input text
        patterns: Custom regex patterns to remove
        
    Returns:
        Text with patterns removed
        
    Example patterns:
        - "Page \d+ of \d+"
        - "Confidential - Do Not Share"
        - Email footers, signatures
    """
    if patterns is None:
        # Default patterns (customize based on your PDFs)
        patterns = [
            r'Page \d+ of \d+',  # Page numbers
            r'© \d{4}.*',  # Copyright notices
            r'\d+\s*/\s*\d+',  # Page fractions (1/10, 2/10)
        ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace for consistent processing.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Multiple spaces → single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove spaces at line starts/ends
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    return text


def preprocess_document(text: str, 
                       remove_headers: bool = True,
                       custom_patterns: List[str] = None) -> str:
    """
    Complete preprocessing pipeline for documents.
    
    WHY THIS FUNCTION:
    - Single entry point for all preprocessing
    - Consistent processing across all documents
    - Production-ready with logging
    
    Args:
        text: Raw extracted text
        remove_headers: Whether to remove headers/footers
        custom_patterns: Additional patterns to remove
        
    Returns:
        Preprocessed text ready for chunking
    """
    logger.info("Starting document preprocessing")
    original_length = len(text)
    
    # Step 1: Basic cleaning
    text = clean_text(text)
    
    # Step 2: Remove headers/footers
    if remove_headers:
        text = remove_headers_footers(text, custom_patterns)
    
    # Step 3: Normalize whitespace
    text = normalize_whitespace(text)
    
    final_length = len(text)
    removed_chars = original_length - final_length
    
    if original_length > 0:
        percent = f"({removed_chars / original_length * 100:.2f}%)"
    else:
        percent = "(n/a%)"

    logger.info(f"Preprocessing complete: {removed_chars} chars removed {percent}")
    
    return text


# Preprocess batch/multiple pdfs:
def preprocess_multiple_documents(texts: dict[str, str], remove_headers: bool = True, custom_patterns: List[str] = None) -> dict[str, str]:
    """
    Preprocess multiple documents given as a dict: {filename: text}.
    Returns a dict {filename: cleaned_text}.
    """
    cleaned_texts = {}
    for filename, text in texts.items():
        cleaned_texts[filename] = preprocess_document(text, remove_headers, custom_patterns)
    return cleaned_texts

def preprocess_all_documents(texts, remove_headers=True, custom_patterns=None):
    """
    Smart preprocessing entrypoint.
    Accepts: string or Dict[str, str]
    Returns: preprocessed string or Dict[str, str] (preserves type)
    """
    if isinstance(texts, dict):
        # Multiple docs case
        return {fn: preprocess_document(t, remove_headers, custom_patterns) for fn, t in texts.items()}
    elif isinstance(texts, str):
        # Single doc case
        return preprocess_document(texts, remove_headers, custom_patterns)
    else:
        raise TypeError("Input must be str or Dict[str, str]")


# # # Test Block:
# if __name__ == "__main__":
#     import sys
#     from backend.utils.pdf_utils import extract_pdfs_with_metadata

#     if len(sys.argv) < 2:
#         print("Usage: python -m backend.utils.preprocessing <pdf1> [<pdf2> ...]")
#         sys.exit(1)

#     pdf_input = sys.argv[1:]
#     if len(pdf_input) == 1:
#         pdf_input = pdf_input[0]  # handle single-string for single PDF

#     pdf_results = extract_pdfs_with_metadata(pdf_input)
#     texts = {fn: d['text'] for fn, d in pdf_results.items()}
#     cleaned = preprocess_all_documents(texts)
#     print("\n===== Preprocessing Summary =====")
#     for fn, text in cleaned.items():
#         orig_len = len(pdf_results[fn]["text"])
#         print(f"\n{fn}: {orig_len} → {len(text)} chars cleaned\nPreview:\n{'-'*40}\n{text[:300]}\n{'-'*40}")
#     print("\n=========== All Done ============")



