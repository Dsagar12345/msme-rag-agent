# src/data_pipeline/document_processor.py
import pdfplumber
import os
import json
from datetime import datetime

# ── PDF Reader ─────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"   ❌ Error reading {pdf_path}: {e}")
        return ""
    return text.strip()

# ── Financial Smart Chunker ────────────────────────────────────────
def chunk_financial_document(text: str, doc_name: str,
                              chunk_size: int = 500,
                              overlap: int = 100) -> list:
    """
    Fixed chunker - splits by character count directly
    """
    chunks = []
    chunk_index = 0
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Don't cut in the middle of a word
        if end < text_length:
            while end > start and text[end] not in [' ', '\n', '.']:
                end -= 1

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_id": f"{doc_name}_chunk_{chunk_index}",
                "source": doc_name,
                "text": chunk_text,
                "chunk_index": chunk_index,
                "char_count": len(chunk_text)
            })
            chunk_index += 1

        # Move forward with overlap
        start = end - overlap

    return chunks

# ── Document Tagger ────────────────────────────────────────────────
def tag_document(doc_name: str) -> dict:
    """
    Tags each document with metadata so RAG knows
    WHAT TYPE of question to route to it
    """
    tags_map = {
        "rbi_msme_resolution": {
            "category": "regulatory",
            "topics": ["loan restructuring", "MSME relief", "covid relief", 
                      "NPA", "stressed assets", "resolution framework"],
            "authority": "RBI",
            "relevance": ["loan", "repayment", "restructure", "stress", "relief"]
        },
        "gst_concept_guide": {
            "category": "tax",
            "topics": ["GST", "goods and services tax", "CGST", "SGST", 
                      "IGST", "input tax credit", "ITC", "returns", "filing"],
            "authority": "CBIC",
            "relevance": ["gst", "tax", "filing", "return", "itc", "invoice"]
        },
        "rbi_priority_lending": {
            "category": "regulatory",
            "topics": ["priority sector", "lending", "MSME loans", 
                      "bank loans", "credit", "working capital"],
            "authority": "RBI",
            "relevance": ["loan", "credit", "bank", "priority", "lending"]
        },
        "mudra_scheme_guide": {
            "category": "scheme",
            "topics": ["MUDRA loan", "Shishu", "Kishore", "Tarun", 
                      "microfinance", "startup loan", "small loan"],
            "authority": "MUDRA",
            "relevance": ["mudra", "loan", "microfinance", "startup", "shishu"]
        },
        "income_tax_small_business": {
            "category": "tax",
            "topics": ["income tax", "ITR", "advance tax", "TDS", 
                      "Section 44AD", "presumptive taxation", "deductions"],
            "authority": "Income Tax Department",
            "relevance": ["income tax", "itr", "advance tax", "tds", "deduction"]
        },
        "sidbi_msme_pulse": {
            "category": "research",
            "topics": ["MSME trends", "credit growth", "industry benchmark", 
                      "sector performance", "NPA trends"],
            "authority": "SIDBI",
            "relevance": ["benchmark", "industry", "trend", "sector", "growth"]
        },
        "udyam_registration_guide": {
            "category": "compliance",
            "topics": ["Udyam registration", "MSME registration", 
                      "Udyog Aadhaar", "registration process", "certificate"],
            "authority": "Ministry of MSME",
            "relevance": ["udyam", "registration", "certificate", "msme registration"]
        }
    }
    
    # Match doc name to tags
    for key, tags in tags_map.items():
        if key in doc_name.lower():
            return tags
    
    # Default tags if no match
    return {
        "category": "general",
        "topics": ["MSME", "finance", "business"],
        "authority": "Unknown",
        "relevance": ["msme", "finance"]
    }

# ── Main Processor ─────────────────────────────────────────────────
def process_all_documents(knowledge_base_path: str = "data/knowledge_base",
                           output_path: str = "data/processed") -> list:
    """
    Processes all PDFs in knowledge_base folder:
    1. Extracts text
    2. Chunks it smartly
    3. Tags with metadata
    4. Saves as JSON for RAG ingestion
    """
    print("="*50)
    print("Processing Knowledge Base Documents")
    print("="*50)
    
    all_chunks = []
    pdf_files = [f for f in os.listdir(knowledge_base_path) 
                 if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in knowledge_base folder!")
        return []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(knowledge_base_path, pdf_file)
        doc_name = pdf_file.replace('.pdf', '')
        
        print(f"\n📄 Processing: {pdf_file}")
        
        # Step 1: Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"   ⚠️  No text extracted — skipping")
            continue
        print(f"   ✅ Extracted {len(text):,} characters")
        
        # Step 2: Chunk the text
        chunks = chunk_financial_document(text, doc_name)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Step 3: Tag with metadata
        tags = tag_document(doc_name)
        
        # Step 4: Add metadata to each chunk
        for chunk in chunks:
            chunk.update({
                "metadata": {
                    "source_file": pdf_file,
                    "doc_name": doc_name,
                    "category": tags["category"],
                    "topics": tags["topics"],
                    "authority": tags["authority"],
                    "relevance_keywords": tags["relevance"],
                    "processed_at": datetime.now().isoformat()
                }
            })
        
        all_chunks.extend(chunks)
        print(f"   ✅ Tagged as: {tags['category']} | {tags['authority']}")
    
    # Save all chunks to JSON
    output_file = os.path.join(output_path, "knowledge_base_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print(f"✅ Processing Complete!")
    print(f"   Documents processed: {len(pdf_files)}")
    print(f"   Total chunks created: {len(all_chunks)}")
    print(f"   Saved to: {output_file}")
    print("="*50)
    
    return all_chunks

if __name__ == "__main__":
    chunks = process_all_documents()
    
    # Show a sample chunk so you can see what the RAG will work with
    if chunks:
        print("\n📋 Sample Chunk (what RAG will retrieve):")
        print("-"*40)
        sample = chunks[10] if len(chunks) > 10 else chunks[0]
        print(f"ID: {sample['chunk_id']}")
        print(f"Source: {sample['metadata']['source_file']}")
        print(f"Category: {sample['metadata']['category']}")
        print(f"Authority: {sample['metadata']['authority']}")
        print(f"Text preview: {sample['text'][:200]}...")
        print("-"*40)