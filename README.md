## 🧠 Graph RAG System - Advanced Document Intelligence Platform

A comprehensive Retrieval-Augmented Generation (RAG) system that transforms Excel inventory data and PDF documents into an intelligent, searchable knowledge graph. The system combines semantic vector search, lexical BM25 retrieval, AI-powered document understanding, and relationship mapping to provide accurate, contextual answers with full traceability.

### ✨ Key Features

- **🤖 AI-Powered Document Understanding**: Automatic document type detection and structured field extraction using Claude AI
- **🔍 Hybrid Search**: Combines semantic vector search (FAISS) with lexical BM25 for optimal retrieval
- **🕸️ Knowledge Graph**: Builds explicit relationships between Excel data and PDF documents using NetworkX
- **🌐 Neo4j Integration**: Export and visualize relationships in Neo4j graph database
- **💻 Modern Web Interface**: Beautiful, responsive Flask web application with real-time statistics
- **🔄 Dynamic Updates**: Hot-reload system for adding new documents without restart
- **📊 Advanced Analytics**: Comprehensive system statistics and relationship analysis

### 💻 Quick Start

#### Prerequisites
- Python 3.8+
- Anthropic API key (for AI features)

#### Installation

1. **Clone and setup environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure API key**:
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
# Or create a .env file with: ANTHROPIC_API_KEY=your_key_here
```

3. **Start the web interface**:
```bash
python start_web_interface.py
# Or directly: python app.py
```

4. **Access the application**:
   - Open your browser to: http://localhost:5001
   - The system will automatically build indices from your data

### 📁 Data Management

#### Adding Documents
- **Excel files**: Place in `data/excel/` directory
- **PDF documents**: Place anywhere under `data/pdfs/` (supports subdirectories)
- Supported PDF types: GRNs, Invoices, Purchase Orders, Proforma Invoices, Quotations, etc.

#### Updating the System
After adding new documents, trigger a system rebuild:




## ⁉️ How it works (precise, step-by-step)

### 1) Data ingestion
- Excel: `ABC_Book_Stores_Inventory_Register.xlsx` is read and key text fields are normalized (Unicode normalization, whitespace cleanup). Each row becomes an “Excel chunk” with a compact header (supplier, customer, book, totals, PO, GRN) plus the raw row dict.
- PDFs: Every `.pdf` under `data/pdfs/` is opened with PyMuPDF (text extraction page by page). The full text is then used for document typing and field extraction (below).

Where to look in code:
- `GraphRAGSystem.load_excel_data(...)`
- `GraphRAGSystem.load_pdf_documents(...)`

### 2) PDF document understanding
For each PDF:
- Document type detection: If the Anthropic client is available, the model classifies a document as one of: `grn`, `invoice`, `purchase_order`, `proforma_invoice`, `quotation`, `delivery_note`, `receipt`, `contract`, `statement`, or `other`. If not, a robust filename/content heuristic is used.
- Field extraction: If the Anthropic client is available, the model extracts a structured JSON with fields like `document_number`, `date`, `supplier_name`, `customer_name`, `items`, `total_amount`, `tax_amount`, etc. If the API is unavailable or times out, a regex fallback extracts common fields.

Why this matters: questions like “What is the GST amount for PO-CP-202509-011?” can be answered because the system either (a) extracts `tax_amount` as a field, or (b) retrieves the PDF content around totals/taxes for the LLM to compute from context.

Where to look in code:
- `PDFProcessor.detect_document_type(...)`
- `PDFProcessor.extract_structured_data(...)` and `_extract_basic_fields(...)`

### 3) Advanced Chunking & Hybrid Indexing
- **PDF Processing**: Text split into overlapping windows (1200 chars, 250-char overlap) with metadata including document type and AI-extracted structured fields
- **Excel Processing**: Each row becomes a single chunk with normalized header information and raw data
- **Vector Index**: All chunks embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) and indexed in FAISS with cosine similarity
- **BM25 Index**: Parallel lexical search index for exact token matching and keyword retrieval
- **Hybrid Fusion**: Combines vector similarity + BM25 scores + token boosts for optimal retrieval

**Why Hybrid?** 
- Vector search excels at semantic understanding (e.g., "GST amount for purchase orders")
- BM25 handles exact matches and specific identifiers (e.g., `PO-CP-202509-011`)
- Fusion algorithm: `final_score = vector_similarity + token_boost + 0.4 × BM25_score`

Where to look in code:
- Vector index build: `GraphRAGSystem.build_system(...)`
- BM25: `_build_bm25(...)`

### 4) Intelligent Retrieval Pipeline
The system processes queries through a sophisticated multi-stage pipeline:

1. **Query Analysis**: Detects query intent (aggregation, address lookup, specific document search)
2. **Candidate Retrieval**: 
   - Embeds query using same transformer model
   - Searches FAISS vector index for semantic matches
   - Retrieves wide candidate set (3x final k)
3. **BM25 Scoring**: Computes lexical relevance scores for all candidates
4. **Score Fusion**: `final_score = vector_similarity + token_boost + 0.4 × BM25_score`
5. **Intent-Specific Processing**:
   - **Address queries**: Uses supplier→address cache for direct lookup
   - **Aggregation queries**: Expands search to include all relevant data
   - **Document-specific**: Enhances token matching for exact references
6. **Result Ranking**: Returns top-k chunks with similarity scores

**Advanced Features**:
- Unicode normalization for consistent matching
- Supplier address caching for fast lookups
- Dynamic k adjustment based on query complexity
- Fallback mechanisms for API timeouts

Where to look in code:
- `GraphRAGSystem.search(...)` (fusion logic and ranking)
- `GraphRAGSystem.search_and_answer(...)` (query orchestration and special intents)

### 5) Answer generation (what you finally read)
- The top-k chunks (from Excel and PDFs) are formatted as “context” and sent to the LLM (Anthropic) with instructions to answer directly, cite data points, and explain relationships if relevant.
- If the Anthropic API isn’t configured, the app will show search results and relationships without an LLM-composed paragraph.

How GST answers are produced:
- Retrieval brings in the relevant PO/GRN PDF chunk(s) where totals/taxes are shown.
- If `tax_amount` was extracted, that’s strongly represented in the context.
- The LLM reads the figures in context to produce “GST amount = X”, often also naming the source document (filename).

Where to look in code:
- `GraphRAGSystem.query_with_claude(...)`

### 6) Relationship graph (explainability layer)
- Entities are extracted from Excel rows (e.g., `po_number`, `grn_code`, `supplier_name`, `customer_name`, `book_title`, etc.).
- Entities are also extracted from each PDF’s structured fields (e.g., `document_number`, `supplier_name`, `tax_amount`, etc.).
- Relationships are created between semantically matching types (e.g., Excel `po_number` ↔ PDF `po_number`/`document_number`; `supplier_name` ↔ `vendor_name`), with a basic similarity-based strength.
- The result is a NetworkX graph kept in-memory for the app’s “related documents” view.

This graph is used to:
- Enrich answers with “related documents” derived from connected entities.
- Provide an explainable path from a question to linked artifacts across sources.

Where to look in code:
- Entity extraction: `GraphRelationshipBuilder.extract_entities_from_excel(...)` and `extract_entities_from_pdf(...)`
- Relationship build: `GraphRelationshipBuilder.build_relationships(...)`
- Graph creation: `create_network_graph(...)`

### 7) Neo4j (optional)
You can export the in-memory graph to Neo4j to visualize and query via Cypher.

Steps:
1) Start Neo4j locally (e.g., `bolt://localhost:7687`).
2) Call `GraphRAGSystem.export_to_neo4j(...)` with URI, user, password.
3) Open Neo4j Browser, explore nodes/edges, run the prebuilt queries from `generate_neo4j_cypher_queries()`.

What gets exported:
- Nodes labeled by source (Excel/Pdf) and type (e.g., `Supplier_name`, `Document_number`, etc.).
- Edges with relationship type (e.g., `SAME_DOCUMENT_REFERENCE`) and a `strength` score.

## Screenshots 


1) Home screen and a sample question
   - <img width="1470" height="834" alt="Screenshot 2025-10-05 at 3 58 28 PM" src="https://github.com/user-attachments/assets/c9bb72e6-7845-401c-81bc-5bf88db5eade" />


2) Correct answer for a PO GST question 
   - <img width="1470" height="833" alt="Screenshot 2025-10-05 at 3 59 01 PM" src="https://github.com/user-attachments/assets/b530be83-a056-45c7-aae8-6f3602f3cc6c" />


3) Search results and “Related documents” panel
   - <img width="1470" height="832" alt="Screenshot 2025-10-05 at 3 59 12 PM" src="https://github.com/user-attachments/assets/89f6cacd-7d6e-4302-aa7c-415b7af2a8a1" />


4) Neo4j graph view with nodes/edges
   - <img width="1470" height="688" alt="Screenshot 2025-10-05 at 3 59 50 PM" src="https://github.com/user-attachments/assets/32999bbb-dd39-48b4-97c8-213e47dfd098" />




