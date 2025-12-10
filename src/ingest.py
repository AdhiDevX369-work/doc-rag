from pathlib import Path
from pypdf import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
import shutil
import re
import hashlib
import warnings
import json

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from config import (
    DATA_DIR, DB_DIR, ALLOWED_EXTENSIONS, EMBEDDING_MODEL, 
    CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, BOOK_METADATA, DEDUP_THRESHOLD
)


def get_text_hash(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r'\x00', '', text)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


def is_toc_page(text: str) -> bool:
    text_start = text[:500].lower()
    
    if 'table of contents' in text_start or 'contents' in text_start[:100]:
        dot_pattern = r'\.\s*\.+\s*\d+'
        if len(re.findall(dot_pattern, text)) > 2:
            return True
        
        page_ref_pattern = r'\.\s+\d+\s*$'
        if len(re.findall(page_ref_pattern, text, re.MULTILINE)) > 3:
            return True
    
    chapter_lines = re.findall(r'^\d+\.\s+[A-Z].*?\d+\s*$', text, re.MULTILINE)
    if len(chapter_lines) > 5:
        return True
    
    numbered_items = re.findall(r'(?:Chapter|Part|Section)\s+\d+', text, re.IGNORECASE)
    if len(numbered_items) > 3:
        lines_with_pages = re.findall(r'.*\d+\s*$', text, re.MULTILINE)
        if len(lines_with_pages) > 5:
            return True
    
    return False


def extract_toc_content(pages: list[dict], book_title: str) -> str:
    toc_pages = []
    
    for page in pages:
        if is_toc_page(page["text"]):
            toc_pages.append((page.get("page", 0), page["text"]))
    
    if not toc_pages:
        for page in pages[:20]:
            text = page["text"]
            if re.search(r'^\d+\.\s+[A-Z].*?\.\s*\.+\s*\d+', text, re.MULTILINE):
                toc_pages.append((page.get("page", 0), text))
            elif len(re.findall(r'Chapter\s+\d+', text, re.IGNORECASE)) > 2:
                toc_pages.append((page.get("page", 0), text))
    
    if toc_pages:
        toc_pages.sort(key=lambda x: x[0])
        return "\n\n".join([p[1] for p in toc_pages])
    return ""


def detect_section_info(text: str, page_num: int) -> dict:
    patterns = [
        (r'(?:^|\n)\s*CHAPTER\s+(\d+)[:\.\s]*([^\n]*)', 'chapter'),
        (r'(?:^|\n)\s*Chapter\s+(\d+)[:\.\s]*([^\n]*)', 'chapter'),
        (r'(?:^|\n)\s*(\d+)\.\s+([A-Z][^\n]{5,50})\s*\n', 'chapter'),
        (r'(?:^|\n)\s*PART\s+([IVXLCDM]+|\d+)[:\.\s]*([^\n]*)', 'part'),
        (r'(?:^|\n)\s*Part\s+([IVXLCDM]+|\d+)[:\.\s]*([^\n]*)', 'part'),
        (r'(?:^|\n)\s*Appendix\s+([A-Z]|\d+)[:\.\s]*([^\n]*)', 'appendix'),
    ]
    
    for pattern, section_type in patterns:
        match = re.search(pattern, text[:500])
        if match:
            num = match.group(1)
            title = match.group(2).strip() if match.group(2) else ""
            try:
                num = int(num)
            except ValueError:
                pass
            return {
                "section_type": section_type,
                "section_num": num,
                "section_title": title[:100] if title else f"{section_type.title()} {num}",
            }
    return {}


def get_book_metadata(filename: str) -> dict:
    if filename in BOOK_METADATA:
        return BOOK_METADATA[filename].copy()
    
    name = Path(filename).stem
    name = re.sub(r'[_-]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return {
        "title": name,
        "author": "Unknown",
        "publisher": "Unknown",
    }


def extract_pdf_text(file_path: Path) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []
    total_pages = len(reader.pages)
    current_section = {}
    book_meta = get_book_metadata(file_path.name)
    
    print(f"  📖 {book_meta['title']} by {book_meta['author']}")
    print(f"  Processing {total_pages} pages...")
    
    raw_pages = []
    for i, page in enumerate(tqdm(reader.pages, desc=f"  Extracting", leave=False), 1):
        try:
            page_text = page.extract_text()
        except Exception as e:
            continue
            
        if not page_text or len(page_text.strip()) < 50:
            continue
        
        page_text = clean_text(page_text)
        if not page_text:
            continue
        
        raw_pages.append({"text": page_text, "page": i, "total_pages": total_pages})
    
    toc_content = extract_toc_content(raw_pages, book_meta["title"])
    if toc_content:
        print(f"  📑 Found Table of Contents")
        pages.append({
            "text": f"TABLE OF CONTENTS for {book_meta['title']} by {book_meta['author']}:\n\n{toc_content}",
            "page": 0,
            "total_pages": total_pages,
            "book_title": book_meta["title"],
            "author": book_meta["author"],
            "publisher": book_meta["publisher"],
            "source_file": file_path.name,
            "content_type": "table_of_contents",
        })
    else:
        print(f"  ⚠️  No Table of Contents found")
    
    for page_data in raw_pages:
        page_text = page_data["text"]
        i = page_data["page"]
        
        if is_toc_page(page_text):
            continue
        
        section_info = detect_section_info(page_text, i)
        if section_info:
            current_section = section_info
        
        page_entry = {
            "text": page_text,
            "page": i,
            "total_pages": total_pages,
            "book_title": book_meta["title"],
            "author": book_meta["author"],
            "publisher": book_meta["publisher"],
            "source_file": file_path.name,
            "content_type": "content",
        }
        
        if current_section:
            page_entry.update(current_section)
            
        pages.append(page_entry)
    
    print(f"  ✓ Extracted {len(pages)} pages (including TOC)")
    return pages


def extract_epub_text(file_path: Path) -> list[dict]:
    book = epub.read_epub(str(file_path), options={"ignore_ncx": True})
    chapters = []
    chapter_num = 0
    book_meta = get_book_metadata(file_path.name)
    
    toc_items = []
    try:
        toc = book.toc
        if toc:
            for item in toc:
                if hasattr(item, 'title'):
                    toc_items.append(item.title)
    except:
        pass
    
    if toc_items:
        toc_text = f"TABLE OF CONTENTS for {book_meta['title']}:\n\n"
        for i, title in enumerate(toc_items, 1):
            toc_text += f"{i}. {title}\n"
        
        chapters.append({
            "text": toc_text,
            "chapter": 0,
            "chapter_title": "Table of Contents",
            "book_title": book_meta["title"],
            "author": book_meta["author"],
            "publisher": book_meta["publisher"],
            "source_file": file_path.name,
            "content_type": "table_of_contents",
        })
    
    items = [item for item in book.get_items() if item.get_type() == 9]
    print(f"  📖 {book_meta['title']} by {book_meta['author']}")
    print(f"  Processing {len(items)} sections...")
    
    for item in tqdm(items, desc=f"  Sections", leave=False):
        try:
            soup = BeautifulSoup(item.get_content(), "lxml")
            
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator="\n")
            text = clean_text(text)
            
            if not text or len(text) < 100:
                continue
            
            chapter_num += 1
            
            title_tag = soup.find(["h1", "h2", "h3"])
            chapter_title = ""
            if title_tag:
                chapter_title = clean_text(title_tag.get_text())[:100]
            
            if not chapter_title:
                section_info = detect_section_info(text, chapter_num)
                if section_info:
                    chapter_title = section_info.get("section_title", "")
            
            if not chapter_title:
                chapter_title = f"Section {chapter_num}"
            
            chapters.append({
                "text": text,
                "chapter": chapter_num,
                "chapter_title": chapter_title,
                "book_title": book_meta["title"],
                "author": book_meta["author"],
                "publisher": book_meta["publisher"],
                "source_file": file_path.name,
                "content_type": "content",
            })
        except Exception as e:
            continue
    
    print(f"  ✓ Extracted {len(chapters)} chapters")
    return chapters


def extract_txt_text(file_path: Path) -> list[dict]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    book_meta = get_book_metadata(file_path.name)
    return [{
        "text": text,
        "page": 1,
        "book_title": book_meta["title"],
        "author": book_meta["author"],
        "source_file": file_path.name,
        "content_type": "content",
    }]


def extract_text_with_metadata(file_path: Path) -> list[dict]:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext == ".epub":
        return extract_epub_text(file_path)
    elif ext == ".txt":
        return extract_txt_text(file_path)
    return []


def get_documents():
    documents = []
    files = sorted([f for f in DATA_DIR.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS])
    
    print(f"\n{'='*60}")
    print(f"Found {len(files)} files to process")
    print(f"{'='*60}\n")
    
    for file_path in files:
        print(f"\n📂 {file_path.name}")
        sections = extract_text_with_metadata(file_path)
        total_chars = sum(len(s["text"]) for s in sections)
        print(f"  Total: {total_chars:,} characters")
        documents.extend(sections)
    
    return documents


def deduplicate_chunks(chunks: list[str], metadatas: list[dict]) -> tuple[list[str], list[dict]]:
    seen_hashes = {}
    unique_chunks = []
    unique_meta = []
    duplicates = 0
    
    for chunk, meta in zip(chunks, metadatas):
        chunk_hash = get_text_hash(chunk)
        
        if chunk_hash in seen_hashes:
            duplicates += 1
            continue
        
        seen_hashes[chunk_hash] = True
        unique_chunks.append(chunk)
        unique_meta.append(meta)
    
    if duplicates > 0:
        print(f"  Removed {duplicates} duplicate chunks")
    
    return unique_chunks, unique_meta


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    
    toc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 2,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n"],
    )
    
    chunks = []
    metadatas = []
    
    books = {}
    for doc in documents:
        book = doc.get("book_title", "Unknown")
        if book not in books:
            books[book] = []
        books[book].append(doc)
    
    for book_title, book_docs in books.items():
        print(f"\n  Chunking: {book_title}")
        book_chunks = 0
        
        for doc in tqdm(book_docs, desc=f"    Processing", leave=False):
            content_type = doc.get("content_type", "content")
            
            if content_type == "table_of_contents":
                doc_chunks = toc_splitter.split_text(doc["text"])
            else:
                doc_chunks = splitter.split_text(doc["text"])
            
            for i, chunk in enumerate(doc_chunks):
                chunk = chunk.strip()
                if len(chunk) < 50:
                    continue
                
                metadata = {
                    "book_title": doc.get("book_title", "Unknown"),
                    "author": doc.get("author", "Unknown"),
                    "source_file": doc.get("source_file", ""),
                    "chunk_index": i,
                    "total_chunks_in_section": len(doc_chunks),
                    "content_type": content_type,
                }
                
                if "page" in doc:
                    metadata["page"] = doc["page"]
                if "total_pages" in doc:
                    metadata["total_pages"] = doc["total_pages"]
                if "chapter" in doc:
                    metadata["chapter"] = doc["chapter"]
                if "chapter_title" in doc:
                    metadata["chapter_title"] = doc["chapter_title"]
                if "section_type" in doc:
                    metadata["section_type"] = doc["section_type"]
                if "section_num" in doc:
                    metadata["section_num"] = doc["section_num"]
                if "section_title" in doc:
                    metadata["section_title"] = doc["section_title"]
                
                chunks.append(chunk)
                metadatas.append(metadata)
                book_chunks += 1
        
        print(f"    ✓ {book_chunks} chunks created")
    
    chunks, metadatas = deduplicate_chunks(chunks, metadatas)
    
    return chunks, metadatas


def create_vectorstore(chunks, metadatas):
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating embeddings for {len(chunks)} chunks...")
    print(f"{'='*60}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )
    
    batch_size = 500
    vectorstore = None
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch_chunks = chunks[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        
        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=batch_chunks,
                embedding=embeddings,
                metadatas=batch_meta,
                collection_name=CHROMA_COLLECTION,
                persist_directory=str(DB_DIR),
            )
        else:
            vectorstore.add_texts(texts=batch_chunks, metadatas=batch_meta)
    
    return vectorstore


def main():
    print("\n" + "="*60)
    print("   DOCUMENT INGESTION SYSTEM")
    print("="*60)
    
    print(f"\nScanning: {DATA_DIR}")
    documents = get_documents()
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total sections: {len(documents)}")
    total_chars = sum(len(d["text"]) for d in documents)
    print(f"Total characters: {total_chars:,}")
    
    toc_count = sum(1 for d in documents if d.get("content_type") == "table_of_contents")
    print(f"Table of Contents sections: {toc_count}")
    
    if not documents:
        print("\n❌ No documents found. Add PDF, EPUB, or TXT files to the data/ folder.")
        return
    
    print(f"\n{'='*60}")
    print("CHUNKING")
    print(f"{'='*60}")
    chunks, metadatas = chunk_documents(documents)
    print(f"\nTotal unique chunks: {len(chunks)}")
    
    create_vectorstore(chunks, metadatas)
    
    print(f"\n{'='*60}")
    print("✅ INGESTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Vector store saved to: {DB_DIR}")
    print(f"Total chunks indexed: {len(chunks)}")
    
    books = set(m.get("book_title", "Unknown") for m in metadatas)
    print(f"\nBooks indexed:")
    for book in sorted(books):
        count = sum(1 for m in metadatas if m.get("book_title") == book)
        toc = sum(1 for m in metadatas if m.get("book_title") == book and m.get("content_type") == "table_of_contents")
        print(f"  📖 {book}: {count} chunks ({toc} TOC)")


if __name__ == "__main__":
    main()
