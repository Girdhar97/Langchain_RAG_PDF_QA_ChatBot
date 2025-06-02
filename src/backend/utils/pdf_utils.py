# src/backend/utils/pdf_utils.py
"""
PDF utility functions for text extraction with multiple file support.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import pypdf
# from backend.utils.logging_config import setup_logger
from utils.logging_config import setup_logger

logger = setup_logger()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as string
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        pypdf.errors.PdfReadError: If PDF is corrupted
    """
    logger.info(f"Extracting text from: {pdf_path}")
    
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += page_text + "\n"
            logger.debug(f"Page {page_num} extracted: {len(page_text)} chars")
        
        logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
    
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except pypdf.errors.PdfReadError as e:
        logger.error(f"Corrupted PDF or read error: {pdf_path}, {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error extracting PDF {pdf_path}: {str(e)}")
        raise


def extract_text_from_multiple_pdfs(pdf_paths: List[str]) -> Dict[str, str]:
    """
    Extract text from multiple PDF files.
    
    WHY THIS FUNCTION:
    - Batch processing of PDFs
    - Returns organized structure {filename: text}
    - Track which text came from which PDF
    - Error handling per file (one failure doesn't break all)
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        Dictionary mapping filename to extracted text
        
    Example:
        >>> paths = ["doc1.pdf", "doc2.pdf"]
        >>> result = extract_text_from_multiple_pdfs(paths)
        >>> result
        {'doc1.pdf': 'text content...', 'doc2.pdf': 'text content...'}
    """
    logger.info(f"Starting batch extraction for {len(pdf_paths)} PDFs")
    
    results = {}
    failed_files = []
    
    for pdf_path in pdf_paths:
        filename = Path(pdf_path).name
        
        try:
            text = extract_text_from_pdf(pdf_path)
            results[filename] = text
            logger.info(f"✓ Successfully processed: {filename}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {filename}: {str(e)}")
            failed_files.append((filename, str(e)))
            # Continue processing other files
            continue
    
    # Log summary
    logger.info(f"Batch extraction complete: {len(results)}/{len(pdf_paths)} successful")
    if failed_files:
        logger.warning(f"Failed files: {failed_files}")
    
    return results


def get_pdf_metadata(pdf_path: str) -> Dict:
    """
    Get metadata about a PDF file.
    
    WHY THIS FUNCTION:
    - Useful for validation (file size limits)
    - Show user info before processing
    - Debugging & logging
    - Production monitoring
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with metadata:
        - pages: Number of pages
        - file_size_mb: File size in MB
        - title: Document title (if available)
        - author: Author (if available)
    """
    logger.debug(f"Getting metadata for: {pdf_path}")
    
    try:
        # File system metadata
        file_path = Path(pdf_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # PDF metadata
        reader = pypdf.PdfReader(pdf_path)
        
        metadata = {
            "filename": file_path.name,
            "pages": len(reader.pages),
            "file_size_mb": round(file_size_mb, 2),
            "title": reader.metadata.title if reader.metadata else "Unknown",
            "author": reader.metadata.author if reader.metadata else "Unknown",
        }
        
        logger.debug(f"Metadata: {metadata}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata for {pdf_path}: {str(e)}")
        return {
            "filename": Path(pdf_path).name,
            "error": str(e)
        }

def extract_pdfs_with_metadata(pdf_input):
    """
    Universal extractor for one or multiple PDFs.
    Accepts: single file path (str) OR list of paths.
    Returns:
      - If input is a single path: {filename: {'text': ..., 'meta': ...}}
      - If input is a list: {filename: {'text': ..., 'meta': ...}, ...}
    """
    results = {}
    if isinstance(pdf_input, str):
        # Single file
        filename = Path(pdf_input).name
        text = extract_text_from_pdf(pdf_input)
        meta = get_pdf_metadata(pdf_input)
        results[filename] = {'text': text, 'meta': meta}
    elif isinstance(pdf_input, list):
        batch = extract_text_from_multiple_pdfs(pdf_input)
        for pdf_path in pdf_input:
            filename = Path(pdf_path).name
            text = batch.get(filename, "")
            meta = get_pdf_metadata(pdf_path)
            results[filename] = {'text': text, 'meta': meta}
    else:
        raise TypeError("Input must be a str PDF path or list of paths")
    return results


# # # Test block
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python -m backend.utils.pdf_utils <pdf1> [<pdf2> ...]")
#         sys.exit(1)
#     pdf_input = sys.argv[1:]
#     if len(pdf_input) == 1:
#         pdf_input = pdf_input[0]   # Pass just the string for single file
#         print("Testing single PDF extraction with metadata...\n")
#     else:
#         print(f"Testing batch extraction for {len(pdf_input)} PDFs with metadata...\n")
#     results = extract_pdfs_with_metadata(pdf_input)
#     print("\n===== Extraction Summary =====")
#     for filename, data in results.items():
#         meta = data['meta']
#         text = data['text']
#         print(f"\n{filename}:")
#         print(f"  Pages: {meta.get('pages', 'N/A')}")
#         print(f"  Size: {meta.get('file_size_mb', 'N/A')} MB")
#         print(f"  Title: {meta.get('title', 'Unknown')}")
#         print(f"  Author: {meta.get('author', 'Unknown')}")
#         print(f"  Extracted: {len(text)} characters")
#         print(f"  Preview:\n{'-'*40}\n{text[:200]}...\n{'-'*40}")
#     print("\n========== All Done ==========")

