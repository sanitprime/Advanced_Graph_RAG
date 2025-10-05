## 🧠 Graph RAG System - Excel + PDF Intelligence with Neo4j

This project turns your Excel inventory and PDF documents (POs, GRNs, invoices, etc.) into a searchable knowledge system. It builds a vector index for semantic retrieval and a relationship graph for explainability, then uses an LLM to answer natural-language questions with citations.

Use this README as an implementation guide. It explains exactly how data is ingested, how retrieval works, how relationships are formed, and how answers are produced end-to-end.

### 💻 Quick start

```bash
1) Create environment and install deps (if not already)
cd /Users/khushiagrawal/Desktop/graph_rag
source venv/bin/activate
pip install -r requirements.txt

2) Ensure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY="<your_key>"

3) Start the web interface
python start_web_interface.py

4) Open the app
#    http://localhost:5001
```

### ✨ Adding or updating data
Place new PDFs anywhere under `data/pdfs/` (subfolders are fine). The system scans recursively.

After adding PDFs, call the reload endpoint to rebuild indices without restarting:
```bash
curl -X POST http://localhost:5001/api/reload
```
This triggers a fresh end-to-end build: Excel load → PDF processing → graph build → embeddings + indices.

Tip: You can also just restart the app, but `/api/reload` is faster for iterative uploads.

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

### 3) Chunking and embeddings (vector index) + BM25
- PDF text is split into overlapping character windows (e.g., 1200 chars with 250-char overlap). Each window becomes a “PDF chunk” with metadata: source filename, document type, and the structured field JSON (if available).
- Excel rows are represented as single chunks (one per row) with a concise header + row data.
- All chunks (Excel + PDF) are embedded using `sentence-transformers` (`all-MiniLM-L6-v2`). The embeddings are indexed in FAISS (`IndexFlatL2`).
- In parallel, a simple BM25 structure is built over the raw chunk texts for lexical matching.

Why two indices? Vector search is great for semantics (e.g., “GST for this PO”), while BM25 helps with exact tokens (e.g., `PO-CP-202509-011`). We fuse them to balance recall and precision.

Where to look in code:
- Vector index build: `GraphRAGSystem.build_system(...)`
- BM25: `_build_bm25(...)`

### 4) Hybrid retrieval pipeline
Given your question, the system:
1. Embeds the query and searches FAISS for a wide candidate set.
2. Computes BM25 scores over the same candidates.
3. Combines scores: base vector similarity + small token boost + 0.4 × BM25 score.
4. Applies light intent handling (e.g., address queries may widen `k` and use a supplier→address cache built from extracted fields).
5. Returns the top-k ranked chunks (Excel rows and PDF text windows).

This is why PO-specific questions work: the PO id token improves BM25; semantic phrasing is handled by embeddings; together they surface the correct PDF window(s).

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


## File map
- `graph_rag_system.py`: Core pipeline (ingestion, PDF understanding, chunking, embeddings, BM25, hybrid retrieval, LLM answering, graph build, Neo4j export)
- `app.py`: Flask API (`/api/query`, `/api/stats`, `/api/health`, and `/api/reload` to rebuild indices)
- `templates/index.html`: Web UI (query input, live stats, results, relationships)
- `start_web_interface.py`: Helper launcher for the web app
- `demo_script.py`: Programmatic demo of building and querying
- `requirements.txt`: Dependencies
- `data/excel/`: Excel inventory source(s)
- `data/pdfs/`: Your PDFs (POs, GRNs, invoices, etc.)

## Troubleshooting
- “New PDFs aren’t reflected in answers”: Call `POST /api/reload` to rebuild indices after adding files.
- “No Anthropic key”: Set `ANTHROPIC_API_KEY`. Without it, you’ll still get search/relationships, but not full LLM-written answers.
- “Vector/BM25 mismatch”: Ensure filenames contain helpful tokens (e.g., PO IDs) and PDF text is extractable (selectable text, not just images). If PDFs are scans, add OCR before indexing.

That’s it. With this pipeline, even PO-specific questions like “What is the GST amount for PO-XXXX?” work reliably because the system combines token-sensitive BM25, semantic embeddings, structured field extraction, and a relationship graph for context and explainability.

