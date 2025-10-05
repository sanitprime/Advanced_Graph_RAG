#!/usr/bin/env python3
"""
Graph RAG System - Excel + PDF Integration with Relationships
A comprehensive RAG system that creates explicit relationships between Excel inventory data and PDF documents.
"""

import os
import logging
import pandas as pd
import numpy as np
import json
import time
import fitz  # PyMuPDF
import uuid
import re
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic
import networkx as nx
from collections import defaultdict
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
from unicodedata import normalize as ucnorm

logger = logging.getLogger(__name__)

class Config:
    CHUNK_SIZE = 1200
    OVERLAP = 250
    BM25_K1 = 1.5
    BM25_B = 0.75

class PDFProcessor:
    """Dynamic PDF processing with AI-powered document type detection and field extraction"""
    
    def __init__(self, claude_api_key=None):
        self.claude_api_key = claude_api_key
        if claude_api_key:
            self.claude_client = Anthropic(api_key=claude_api_key)
        else:
            self.claude_client = None
        self.document_types = {}  # Will be populated dynamically
        self.field_patterns = {}  # Will be learned dynamically
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF with error handling"""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"Page {page_num + 1}:\n{page_text}")
            
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def detect_document_type(self, text: str, filename: str) -> str:
        """AI-powered document type detection"""
        if not self.claude_client:
            # Fallback to simple pattern matching
            text_lower = text.lower()
            filename_lower = filename.lower()
            
            if 'grn' in filename_lower or 'goods receipt' in text_lower:
                return 'grn'
            elif 'invoice' in filename_lower or 'inv-' in filename_lower or 'bill' in text_lower:
                return 'invoice'
            elif 'po-' in filename_lower or 'purchase order' in text_lower:
                return 'purchase_order'
            elif 'proforma' in text_lower or 'pi-' in filename_lower:
                return 'proforma_invoice'
            else:
                return 'unknown'
        
        # Use Claude for intelligent document type detection
        prompt = f"""Analyze this document and determine its type. The filename is: {filename}

Document content (first 2000 characters):
{text[:2000]}

Based on the content and filename, classify this document into one of these categories:
- grn (Goods Receipt Note)
- invoice (Invoice/Bill)
- purchase_order (Purchase Order)
- proforma_invoice (Proforma Invoice)
- quotation (Quotation/Quote)
- delivery_note (Delivery Note)
- receipt (Receipt)
- contract (Contract/Agreement)
- statement (Account Statement)
- other (Other document type)

Respond with ONLY the category name, nothing else."""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            doc_type = response.content[0].text.strip().lower()
            return doc_type if doc_type in ['grn', 'invoice', 'purchase_order', 'proforma_invoice', 'quotation', 'delivery_note', 'receipt', 'contract', 'statement', 'other'] else 'unknown'
        except Exception as e:
            logger.error(f"Error in Claude document type detection: {e}")
            return 'unknown'
    
    def extract_structured_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """AI-powered dynamic field extraction from PDF text"""
        if not self.claude_client:
            # Fallback to basic regex patterns
            return self._extract_basic_fields(text, doc_type)
        
        # Use Claude for intelligent field extraction
        prompt = f"""Extract all relevant structured data from this {doc_type} document.

Document content:
{text[:3000]}

Please extract the following information in JSON format. If a field is not found, use null:
- document_number (any reference number like GRN, Invoice No, PO No, etc.)
- date (any date mentioned)
- amount (any monetary value)
- supplier_name (vendor/supplier name)
- customer_name (buyer/customer name)
- company_name (company issuing the document)
- address (any address mentioned)
- phone (phone numbers)
- email (email addresses)
- items (list of items/products mentioned)
- quantity (quantities mentioned)
- unit_price (unit prices)
- total_amount (total monetary value)
- tax_amount (tax values)
- due_date (payment due date)
- payment_terms (payment terms)
- shipping_address (delivery address)
- billing_address (billing address)
- notes (any additional notes or comments)

Return ONLY a valid JSON object with the extracted fields. If a field is not found, use null."""

        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"  Attempting Claude API call (attempt {attempt + 1}/{max_retries})...")
                
                response = self._claude_api_call_with_timeout(prompt)
                
                # Parse JSON response
                import json
                structured_data = json.loads(response.content[0].text.strip())
                logger.info(f"  ✅ Claude extraction successful")
                return structured_data
                
            except TimeoutError:
                logger.warning(f"  ⏰ Claude API call timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"  ⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"  ❌ All Claude API attempts failed due to timeout")
                    
            except Exception as e:
                logger.error(f"  ❌ Error in Claude field extraction (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"  ⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"  ❌ All Claude API attempts failed")
        
        # If all attempts fail, use fallback
        logger.info(f"  🔄 Falling back to basic field extraction...")
        return self._extract_basic_fields(text, doc_type)
    
    def _claude_api_call_with_timeout(self, prompt):
        """Make Claude API call with timeout using ThreadPoolExecutor (cross-platform)."""
        if not self.claude_client:
            raise RuntimeError("Claude client not initialized")
        with ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(
                self.claude_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return fut.result(timeout=30)
    
    def _extract_basic_fields(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Fallback basic field extraction using regex patterns"""
        structured_data = {}
        
        # Common patterns for various document types
        patterns = {
            'document_number': [
                r'(?:GRN|Invoice|PO|PI|Quote|Delivery)\s*(?:No|Number|Code)[:\s]*([A-Z0-9-]+)',
                r'Reference[:\s]*([A-Z0-9-]+)',
                r'Doc\s*No[:\s]*([A-Z0-9-]+)'
            ],
            'date': [
                r'(?:Date|Dated)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{1,2}\s+\w+\s+\d{4})'
            ],
            'amount': [
                r'(?:Total|Amount|Sum)[:\s]*(\d+\.?\d*)',
                r'(\$\d+\.?\d*)',
                r'(?:Rs\.?|INR)[:\s]*(\d+\.?\d*)'
            ],
            'supplier_name': [
                r'(?:Supplier|Vendor|From)[:\s]*([A-Za-z\s&.,]+)',
                r'Company[:\s]*([A-Za-z\s&.,]+)'
            ],
            'customer_name': [
                r'(?:Customer|Buyer|To)[:\s]*([A-Za-z\s&.,]+)',
                r'Bill\s+To[:\s]*([A-Za-z\s&.,]+)'
            ]
        }
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    structured_data[field] = match.group(1).strip()
                    break
        
        return structured_data

class GraphRelationshipBuilder:
    """Builds explicit relationships between documents and data"""
    
    def __init__(self):
        self.relationships = []
        self.entity_map = {}
        self.document_connections = {}
        
    def _normalize_name(self, value: Any) -> str:
        """Normalize entity names for efficient equality matching."""
        if value is None:
            return ""
        s = str(value)
        s = ucnorm('NFKC', s.replace('\xa0', ' '))
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s
        
    def extract_entities_from_excel(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract entities from Excel data"""
        entities = []
        
        for idx, row in df.iterrows():
            row_entities = {
                'grn_code': row.get('GRN Code'),
                'po_number': row.get('Purchase Order No.'),
                'invoice_number': row.get('Sales Inv No.'),
                'supplier_name': row.get('Supplier Name'),
                'customer_name': row.get('Customer Name'),
                'book_title': row.get('Book Title'),
                'author': row.get('Author'),
                'isbn': row.get('ISBN'),
                'store_location': row.get('Store Location'),
                'store_code': row.get('Store Code')
            }
            
            for key, value in row_entities.items():
                if pd.notna(value) and str(value).strip():
                    entity_id = f"excel_{key}_{idx}_{str(value).replace(' ', '_')}"
                    entities.append({
                        'id': entity_id,
                        'name': str(value),
                        'type': key,
                        'source': 'excel',
                        'row_index': idx,
                        'attributes': {
                            'source_file': 'ABC_Book_Stores_Inventory_Register.xlsx',
                            'row_index': idx,
                            'column': key
                        }
                    })
                    
                    self.entity_map[entity_id] = {
                        'name': str(value),
                        'type': key,
                        'source': 'excel',
                        'row_index': idx
                    }
        
        return entities
    
    def extract_entities_from_pdf(self, pdf_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dynamic entity extraction from PDF documents"""
        entities = []
        
        for chunk in pdf_chunks:
            if chunk.get('structured_data'):
                structured = chunk['structured_data']
                doc_type = chunk.get('document_type', 'unknown')
                source = chunk['source']
                
                # Dynamically extract all non-null fields as entities
                for field_name, field_value in structured.items():
                    if field_value is not None and str(field_value).strip():
                        # Create entity ID based on field type and value
                        entity_id = f"pdf_{field_name}_{str(field_value).replace(' ', '_').replace('-', '_')}"
                        
                        # Determine entity type based on field name
                        entity_type = self._map_field_to_entity_type(field_name, doc_type)
                        
                        entities.append({
                            'id': entity_id,
                            'name': str(field_value),
                            'type': entity_type,
                            'source': 'pdf',
                            'document_type': doc_type,
                            'source_file': source,
                            'field_name': field_name,
                            'attributes': structured
                        })
                        
                        self.entity_map[entity_id] = {
                            'name': str(field_value),
                            'type': entity_type,
                            'source': 'pdf',
                            'document_type': doc_type,
                            'field_name': field_name
                        }
        
        return entities
    
    def _map_field_to_entity_type(self, field_name: str, doc_type: str) -> str:
        """Map field names to standardized entity types"""
        field_mapping = {
            'document_number': 'document_number',
            'grn_code': 'document_number',
            'invoice_number': 'document_number',
            'po_number': 'document_number',
            'quote_number': 'document_number',
            'date': 'date',
            'grn_date': 'date',
            'invoice_date': 'date',
            'po_date': 'date',
            'amount': 'amount',
            'total_amount': 'amount',
            'supplier_name': 'supplier_name',
            'vendor_name': 'supplier_name',
            'customer_name': 'customer_name',
            'buyer_name': 'customer_name',
            'company_name': 'company_name',
            'address': 'address',
            'phone': 'contact_info',
            'email': 'contact_info',
            'items': 'items',
            'quantity': 'quantity',
            'unit_price': 'price',
            'tax_amount': 'tax',
            'due_date': 'date',
            'payment_terms': 'terms',
            'shipping_address': 'address',
            'billing_address': 'address',
            'notes': 'notes'
        }
        
        return field_mapping.get(field_name, f"{doc_type}_{field_name}")
    
    def build_relationships(self, excel_entities: List[Dict], pdf_entities: List[Dict]) -> List[Dict[str, Any]]:
        """Efficient relationship building using indexed joins to avoid cross-product scans."""
        relationships = []
        
        # Build indexes: (type, normalized_name) -> [entities]
        pdf_index: Dict[tuple, List[Dict]] = defaultdict(list)
        for e in pdf_entities:
            key = (e['type'], self._normalize_name(e['name']))
            pdf_index[key].append(e)
        
        # Similar/compatible types mapping
        similar_types = {
            'document_number': {'grn_code', 'invoice_number', 'po_number', 'quote_number', 'document_number'},
            'supplier_name': {'vendor_name', 'company_name', 'supplier_name'},
            'customer_name': {'buyer_name', 'customer_name'},
            'amount': {'total_amount', 'amount'},
            'date': {'grn_date', 'invoice_date', 'po_date', 'due_date', 'date'}
        }
        
        def candidate_pdf_types(excel_type: str) -> List[str]:
            if excel_type in similar_types:
                return list(similar_types[excel_type])
            # Also allow exact type matches by default
            return [excel_type]
        
        # Join by normalized name across compatible types
        for ex in excel_entities:
            ex_name_norm = self._normalize_name(ex['name'])
            if not ex_name_norm:
                continue
            for pdf_t in candidate_pdf_types(ex['type']):
                for pdf_entity in pdf_index.get((pdf_t, ex_name_norm), []):
                    relationship_type = self._get_relationship_type(ex['type'], pdf_entity['type'])
                    relationships.append({
                        'from': ex['id'],
                        'to': pdf_entity['id'],
                        'type': relationship_type,
                        'strength': 1.0,
                        'description': f"Excel {ex['type']} '{ex['name']}' relates to PDF {pdf_entity['type']} '{pdf_entity['name']}'"
                    })
        
        # Create relationships within Excel/PDF data using star topology
        relationships.extend(self._build_within_source_relationships(excel_entities, 'excel', 'SAME_ROW'))
        relationships.extend(self._build_within_source_relationships(pdf_entities, 'pdf', 'SAME_DOCUMENT'))
        
        return relationships
    
    def _should_create_relationship(self, excel_type: str, pdf_type: str, excel_name: str, pdf_name: str) -> bool:
        """Determine if two entities should have a relationship"""
        # Exact type matches
        if excel_type == pdf_type:
            return True
        
        # Similar type matches
        similar_types = {
            'document_number': ['grn_code', 'invoice_number', 'po_number', 'quote_number'],
            'supplier_name': ['vendor_name', 'company_name'],
            'customer_name': ['buyer_name'],
            'amount': ['total_amount', 'amount'],
            'date': ['grn_date', 'invoice_date', 'po_date', 'due_date']
        }
        
        for base_type, variants in similar_types.items():
            if (excel_type in variants and pdf_type in variants) or \
               (excel_type == base_type and pdf_type in variants) or \
               (pdf_type == base_type and excel_type in variants):
                return True
        
        return False
    
    def _get_relationship_type(self, excel_type: str, pdf_type: str) -> str:
        """Get the relationship type between two entity types"""
        if excel_type == pdf_type:
            return f"SAME_{excel_type.upper()}"
        
        # Map to common relationship types
        type_mapping = {
            ('document_number', 'grn_code'): 'SAME_DOCUMENT_REFERENCE',
            ('document_number', 'invoice_number'): 'SAME_DOCUMENT_REFERENCE',
            ('document_number', 'po_number'): 'SAME_DOCUMENT_REFERENCE',
            ('supplier_name', 'vendor_name'): 'SAME_SUPPLIER',
            ('customer_name', 'buyer_name'): 'SAME_CUSTOMER',
            ('amount', 'total_amount'): 'SAME_AMOUNT',
            ('date', 'grn_date'): 'SAME_DATE',
            ('date', 'invoice_date'): 'SAME_DATE',
            ('date', 'po_date'): 'SAME_DATE'
        }
        
        return type_mapping.get((excel_type, pdf_type), f"RELATED_{excel_type.upper()}_{pdf_type.upper()}")
    
    def _calculate_relationship_strength(self, excel_name: str, pdf_name: str, excel_type: str, pdf_type: str) -> float:
        """Calculate relationship strength between two entities"""
        # Exact match
        if excel_name == pdf_name:
            return 1.0
        
        # Normalize names for comparison
        excel_norm = str(excel_name).strip().lower()
        pdf_norm = str(pdf_name).strip().lower()
        
        if excel_norm == pdf_norm:
            return 1.0
        
        # Partial match for names
        if excel_type in ['supplier_name', 'customer_name', 'company_name'] and pdf_type in ['supplier_name', 'customer_name', 'company_name']:
            # Check for partial name matches
            excel_words = set(excel_norm.split())
            pdf_words = set(pdf_norm.split())
            
            if excel_words and pdf_words:
                overlap = len(excel_words.intersection(pdf_words))
                total = len(excel_words.union(pdf_words))
                return overlap / total if total > 0 else 0.0
        
        # Date similarity
        if excel_type in ['date', 'grn_date', 'invoice_date', 'po_date'] and pdf_type in ['date', 'grn_date', 'invoice_date', 'po_date']:
            # Simple date comparison (could be enhanced with proper date parsing)
            if excel_norm == pdf_norm:
                return 1.0
        
        return 0.0
    
    def _build_within_source_relationships(self, entities: List[Dict], source: str, relationship_type: str) -> List[Dict]:
        """Build relationships within the same data source using a star topology to avoid O(m^2)."""
        relationships = []
        
        if source == 'excel':
            # Group by row index
            row_groups = {}
            for entity in entities:
                row_idx = entity.get('row_index', 0)
                if row_idx not in row_groups:
                    row_groups[row_idx] = []
                row_groups[row_idx].append(entity)
            
            for row_idx, row_entities in row_groups.items():
                if not row_entities:
                    continue
                # Prefer a hub with a document_number type if present
                hub = next((e for e in row_entities if e.get('type') == 'document_number'), row_entities[0])
                for e in row_entities:
                    if e['id'] == hub['id']:
                        continue
                    relationships.append({
                        'from': hub['id'],
                        'to': e['id'],
                        'type': relationship_type,
                        'strength': 1.0,
                        'description': f"Both entities from Excel row {row_idx}"
                    })
        
        elif source == 'pdf':
            # Group by source file
            doc_groups = {}
            for entity in entities:
                source_file = entity.get('source_file', 'unknown')
                if source_file not in doc_groups:
                    doc_groups[source_file] = []
                doc_groups[source_file].append(entity)
            
            for source_file, doc_entities in doc_groups.items():
                if not doc_entities:
                    continue
                # Prefer a hub with document_number if present
                hub = next((e for e in doc_entities if e.get('type') == 'document_number'), doc_entities[0])
                for e in doc_entities:
                    if e['id'] == hub['id']:
                        continue
                    relationships.append({
                        'from': hub['id'],
                        'to': e['id'],
                        'type': relationship_type,
                        'strength': 1.0,
                        'description': f"Both entities from PDF document {source_file}"
                    })
        
        return relationships
    
    def create_network_graph(self, entities: List[Dict], relationships: List[Dict]) -> nx.Graph:
        """Create a NetworkX graph for visualization"""
        G = nx.Graph()
        
        for entity in entities:
            G.add_node(
                entity['id'],
                name=entity['name'],
                type=entity['type'],
                source=entity['source'],
                **entity.get('attributes', {})
            )
        
        for rel in relationships:
            G.add_edge(
                rel['from'],
                rel['to'],
                type=rel['type'],
                strength=rel['strength'],
                description=rel['description']
            )
        
        return G

class GraphRAGSystem:
    """Main Graph RAG System with Excel + PDF Integration"""
    
    def __init__(self, claude_api_key: str = None):
        self.claude_api_key = claude_api_key
        self.pdf_processor = PDFProcessor(claude_api_key)
        self.graph_builder = GraphRelationshipBuilder()
        self.faiss_index = None
        self.metadata_list = None
        self.claude_client = Anthropic(api_key=claude_api_key) if claude_api_key else None
        self.df = None
        self.graph = None
        self.entities = []
        self.relationships = []
        # Field index and BM25 data structures
        self.field_index = {
            'supplier_to_addresses': defaultdict(set)
        }
        self._bm25_docs = []           # list[list[str]] tokenized docs
        self._bm25_df = {}             # term -> doc frequency
        self._bm25_idf = {}            # term -> idf
        self._bm25_doc_len = []        # list[int]
        self._bm25_avgdl = 0.0
        
    def load_excel_data(self, excel_path: str):
        """Load Excel data"""
        logger.info("📊 Loading Excel data...")
        self.df = pd.read_excel(excel_path)
        # Normalize key text fields to avoid unicode/whitespace mismatches
        def _norm_text(val: Any) -> Any:
            if pd.isna(val):
                return val
            s = str(val)
            s = ucnorm('NFKC', s.replace('\xa0', ' '))  # replace NBSP, normalize unicode
            s = re.sub(r"\s+", " ", s).strip()       # collapse whitespace
            return s

        for col in [
            'Supplier Name', 'Customer Name', 'Book Title', 'Author', 'Publisher',
            'GRN Code', 'Purchase Order No.', 'Sales Inv No.'
        ]:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(_norm_text)
        logger.info(f"✅ Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        return self.df
    
    def load_pdf_documents(self, pdf_folder: str) -> List[Dict[str, Any]]:
        """Load and process all PDF documents"""
        logger.info("📄 Loading PDF documents...")
        
        all_chunks = []
        pdf_files = []
        
        for root, dirs, files in os.walk(pdf_folder):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(pdf_files)} PDF files")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_path)
            logger.info(f"  📄 Processing {filename} ({i}/{len(pdf_files)})...")
            
            try:
                text = self.pdf_processor.extract_pdf_text(pdf_path)
                if text:
                    doc_type = self.pdf_processor.detect_document_type(text, filename)
                    logger.info(f"    📋 Document type: {doc_type}")
                    logger.info(f"    🔍 Extracting structured data...")
                    structured_data = self.pdf_processor.extract_structured_data(text, doc_type)
                    logger.info(f"    ✅ Processing completed")
                    # Populate field index for supplier -> address
                    try:
                        supplier_name = structured_data.get('supplier_name') if structured_data else None
                        address_val = structured_data.get('address') if structured_data else None
                        if supplier_name and address_val:
                            sup_norm = re.sub(r"\s+", " ", ucnorm('NFKC', str(supplier_name).replace('\xa0',' '))).strip().lower()
                            addr_norm = re.sub(r"\s+", " ", ucnorm('NFKC', str(address_val).replace('\xa0',' '))).strip()
                            if sup_norm and addr_norm:
                                self.field_index['supplier_to_addresses'][sup_norm].add(addr_norm)
                    except Exception:
                        pass
                    
                    # Create overlapping sliding-window chunks over full text
                    full_text = text
                    chunk_size = Config.CHUNK_SIZE  # characters
                    overlap = Config.OVERLAP      # characters
                    total_len = len(full_text)
                    j = 0
                    start = 0
                    while start < total_len:
                        end = min(start + chunk_size, total_len)
                        chunk_text = full_text[start:end].strip()
                        if chunk_text:
                            chunk = {
                                'id': f"{filename}_{j}",
                                'source': filename,
                                'type': 'pdf',
                                'document_type': doc_type,
                                'text': chunk_text,
                                'structured_data': structured_data,
                                'chunk_index': j,
                                'total_chunks': None
                            }
                            all_chunks.append(chunk)
                            j += 1
                        if end == total_len:
                            break
                        start = end - overlap
                else:
                    logger.warning(f"    ⚠️  No text extracted from {filename}")
                    continue
                        
            except Exception as e:
                logger.warning(f"⚠️ Error processing {pdf_path}: {e}")
        
        logger.info(f"✅ Created {len(all_chunks)} PDF chunks from {len(pdf_files)} documents")
        return all_chunks
    
    def create_excel_chunks(self) -> List[Dict]:
        """Create enhanced Excel chunks"""
        if self.df is None:
            raise ValueError("Excel data not loaded")
        
        chunks = []
        for idx, row in self.df.iterrows():
            supplier = row.get('Supplier Name', '')
            customer = row.get('Customer Name', '')
            title = row.get('Book Title', '')
            total_purchase = row.get('Total Purchase amount', '')
            po = row.get('Purchase Order No.', '')
            grn = row.get('GRN Code', '')
            header = (
                f"Supplier: {supplier} | Customer: {customer} | Book: {title} | "
                f"Total Purchase amount: {total_purchase} | PO: {po} | GRN: {grn}"
            )
            chunk = {
                'id': f"excel_row_{idx}",
                'source': 'ABC_Book_Stores_Inventory_Register.xlsx',
                'type': 'excel',
                'document_type': 'inventory_register',
                'row_index': idx,
                'text': header + "\n" + str(row.to_dict()),
                'raw_data': row.to_dict()
            }
            chunks.append(chunk)
        
        return chunks
    
    def build_system(self, excel_path: str, pdf_folder: str):
        """Build the complete system with relationships"""
        logger.info("🔧 Building Graph RAG System...")
        
        # Load data
        self.load_excel_data(excel_path)
        pdf_chunks = self.load_pdf_documents(pdf_folder)
        
        # Extract entities
        logger.info("🔍 Extracting entities...")
        excel_entities = self.graph_builder.extract_entities_from_excel(self.df)
        pdf_entities = self.graph_builder.extract_entities_from_pdf(pdf_chunks)
        
        logger.info(f"✅ Extracted {len(excel_entities)} Excel entities")
        logger.info(f"✅ Extracted {len(pdf_entities)} PDF entities")
        
        # Build relationships
        logger.info("🔗 Building relationships...")
        relationships = self.graph_builder.build_relationships(excel_entities, pdf_entities)
        
        logger.info(f"✅ Created {len(relationships)} relationships")
        
        # Create network graph
        logger.info("🕸️ Creating network graph...")
        all_entities = excel_entities + pdf_entities
        self.graph = self.graph_builder.create_network_graph(all_entities, relationships)
        
        self.entities = all_entities
        self.relationships = relationships
        
        logger.info(f"✅ Network graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Build vector index for semantic search
        logger.info("🔢 Building vector index...")
        excel_chunks = self.create_excel_chunks()
        all_chunks = excel_chunks + pdf_chunks
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [chunk['text'] for chunk in all_chunks]
        # Batch encode and L2-normalize for cosine/IP search
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        # Normalize to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        
        dim = embeddings.shape[1]
        # Use HNSW for large datasets, otherwise flat IP (cosine)
        if len(embeddings) > 10000:
            self.faiss_index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        
        self.metadata_list = all_chunks
        
        logger.info(f"✅ Vector index built with {len(all_chunks)} chunks")
        # Build BM25 structures over chunk texts
        self._build_bm25([c['text'] for c in all_chunks])
        
        return all_entities, relationships
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid retrieval: vector search fused with BM25 over chunk text.
        Optionally restrict candidates based on supplier names in query."""
        if self.faiss_index is None:
            raise ValueError("System not built. Call build_system() first.")
        
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        # Normalize query to unit length for cosine/IP
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
        q_emb = q_emb / q_norm
        search_k = max(k * 3, 30)
        D, I = self.faiss_index.search(q_emb, search_k)
        
        results = []
        q_tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) > 2]
        # Supplier filter if query mentions a supplier entity
        supplier_match = None
        m = re.search(r"address\s+of\s+(.+)$", query, re.IGNORECASE)
        if m:
            supplier_match = m.group(1).strip().lower()
        # BM25 scores for same candidate pool
        bm25_scores = self._bm25_search(query, candidate_indices=I[0].tolist())
        for i, idx in enumerate(I[0]):
            chunk = self.metadata_list[idx]
            # With IP/cosine, D holds similarity directly
            base_sim = float(D[0][i])
            text_lower = chunk['text'].lower()
            boost = 0.0
            if any(tok in text_lower for tok in q_tokens):
                boost += 0.1
            # Fuse with BM25 score (scaled)
            bm25 = bm25_scores.get(int(idx), 0.0)
            adjusted = base_sim + boost + 0.4 * bm25
            # Optional supplier filter: keep chunks containing the supplier string
            if supplier_match and supplier_match not in text_lower:
                continue
            results.append({
                'text': chunk['text'],
                'source': chunk['source'],
                'type': chunk['type'],
                'document_type': chunk.get('document_type', 'unknown'),
                'row_index': chunk.get('row_index'),
                'similarity': adjusted,
                'distance': 1.0 - base_sim
            })
        results.sort(key=lambda r: r['similarity'], reverse=True)
        return results[:k]

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 2]

    def _build_bm25(self, docs: List[str], k1: float = Config.BM25_K1, b: float = Config.BM25_B):
        # Tokenize documents
        tokenized = [self._tokenize(t) for t in docs]
        self._bm25_docs = tokenized
        N = len(tokenized)
        df = {}
        doc_len = []
        for tokens in tokenized:
            doc_len.append(len(tokens))
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._bm25_df = df
        self._bm25_avgdl = float(sum(doc_len)) / N if N else 0.0
        # idf (BM25 variant)
        idf = {}
        for term, dfi in df.items():
            idf[term] = max(0.0, np.log((N - dfi + 0.5) / (dfi + 0.5) + 1))
        self._bm25_idf = idf
        self._bm25_doc_len = doc_len

    def _bm25_search(self, query: str, top_n: int = 100, candidate_indices: Optional[List[int]] = None,
                      k1: float = Config.BM25_K1, b: float = Config.BM25_B) -> Dict[int, float]:
        if not self._bm25_docs:
            return {}
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return {}
        # term frequencies per doc for candidate set only
        candidates = candidate_indices if candidate_indices is not None else list(range(len(self._bm25_docs)))
        scores = {}
        for di in candidates:
            tokens = self._bm25_docs[di]
            if not tokens:
                continue
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            dl = self._bm25_doc_len[di] or 1
            for qt in q_tokens:
                if qt not in tf:
                    continue
                idf = self._bm25_idf.get(qt, 0.0)
                freq = tf[qt]
                denom = freq + k1 * (1 - b + b * (dl / (self._bm25_avgdl or 1)))
                score += idf * ((freq * (k1 + 1)) / denom)
            # scale score down to be compatible with cosine similarities
            scores[di] = score / 10.0
        return scores
    
    def find_related_documents(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find documents related through graph relationships"""
        if not self.graph:
            return []
        
        search_results = self.search(query, k=10)
        related_docs = []
        processed_nodes = set()
        
        for result in search_results:
            if result['type'] == 'excel':
                row_entities = [e for e in self.entities if e.get('row_index') == result.get('row_index')]
            else:
                source = result['source']
                row_entities = [e for e in self.entities if e.get('source_file') == source]
            
            for entity in row_entities:
                entity_id = entity['id']
                if entity_id in self.graph.nodes() and entity_id not in processed_nodes:
                    processed_nodes.add(entity_id)
                    
                    neighbors = list(self.graph.neighbors(entity_id))
                    
                    for neighbor_id in neighbors:
                        if neighbor_id not in processed_nodes:
                            neighbor_data = self.graph.nodes[neighbor_id]
                            edge_data = self.graph.edges[entity_id, neighbor_id]
                            
                            related_docs.append({
                                'entity_name': entity['name'],
                                'related_entity': neighbor_data.get('name', 'Unknown'),
                                'relationship_type': edge_data.get('type', 'unknown'),
                                'relationship_strength': edge_data.get('strength', 0),
                                'related_source': neighbor_data.get('source', 'unknown'),
                                'related_type': neighbor_data.get('type', 'unknown'),
                                'description': edge_data.get('description', '')
                            })
                            
                            processed_nodes.add(neighbor_id)
        
        related_docs.sort(key=lambda x: x['relationship_strength'], reverse=True)
        return related_docs[:max_results]
    
    def query_with_claude(self, query: str, context_chunks: List[Dict]) -> str:
        """Query Claude API with context"""
        if not self.claude_api_key:
            return "Claude API not available - showing search results only"
        
        context_parts = []
        for chunk in context_chunks:
            if chunk['type'] == 'excel':
                context_parts.append(f"Excel Row: {chunk['text']}")
            else:
                context_parts.append(f"PDF Document ({chunk['document_type']}): {chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert data analyst for ABC Book Stores inventory and document management system.

Use the following data to answer the question comprehensively. Pay attention to:
1. Excel data relationships and business context
2. PDF document types and their specific information
3. Connections between Excel records and PDF documents
4. Business process workflows

Context Data:
{context}

Question: {query}

Please provide a detailed answer that:
1. Directly addresses the question
2. Cites specific data points and sources
3. Explains relationships between different data sources
4. Provides business insights where applicable

Answer:"""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error querying Claude: {str(e)}"
    
    def search_and_answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Complete search and answer pipeline"""
        print(f"🔍 Searching for: '{query}'")

        # Increase k substantially for aggregation-style queries (e.g., totals across suppliers/customers)
        agg_patterns = [
            r"\btotal\b",
            r"\bsum\b",
            r"\baggregate\b",
            r"\bacross\s+all\b"
        ]
        is_agg = any(re.search(p, query, re.IGNORECASE) for p in agg_patterns) and \
                 any(re.search(p, query, re.IGNORECASE) for p in [r"supplier", r"customer", r"suppliers", r"customers"]) 

        # If aggregation intent detected, raise k to all Excel rows (plus buffer)
        k_effective = max(k, 10)
        if is_agg and isinstance(self.df, pd.DataFrame):
            excel_rows = len(self.df)
            # Ensure we retrieve at least all excel chunks; add small buffer for PDFs
            k_effective = max(k_effective, excel_rows + 20)

        # Address-style queries: try to extract address from retrieved chunks (no special boosts)
        addr_intent = re.search(r"\baddress\b|\blocation\b", query, re.IGNORECASE)
        supplier_target = None
        m = re.search(r"address\s+of\s+(.+)$", query, re.IGNORECASE)
        if m:
            supplier_target = m.group(1).strip()

        # Semantic search
        # If address intent, retrieve all chunks to maximize recall
        if addr_intent:
            k_effective = len(self.metadata_list) if isinstance(self.metadata_list, list) else max(k_effective, 100)
        search_results = self.search(query, k_effective)

        # If address intent, first try exact match from field index
        if addr_intent and supplier_target:
            sup_norm = re.sub(r"\s+", " ", ucnorm('NFKC', supplier_target.replace('\xa0',' '))).strip().lower()
            addr_set = self.field_index['supplier_to_addresses'].get(sup_norm)
            if addr_set:
                candidate = sorted(addr_set, key=len, reverse=True)[0]
                answer = f"Address: {candidate}"
                related_docs = self.find_related_documents(query, max_results=10)
                return {
                    'query': query,
                    'answer': answer,
                    'search_results': self.search(query, k_effective),
                    'related_documents': related_docs,
                    'num_results': len(self.metadata_list) if isinstance(self.metadata_list, list) else 0,
                    'num_relationships': len(related_docs)
                }

        # If address intent, scan retrieved chunks for address-like text
        if addr_intent:
            addr_regexes = [
                r"\b\d{1,4}[^\n,]{0,40},[^\n]{0,80}\b\d{6}\b",  # typical Indian address with 6-digit PIN
                r"\b[A-Z0-9\-]{1,10}[^\n,]{0,40},[^\n]{0,120}\b"    # fallback general pattern
            ]
            candidate = None
            best_len = 0
            q_sup = (supplier_target or '').lower()
            for res in search_results:
                if res.get('type') != 'pdf':
                    continue
                text = res.get('text', '')
                low = text.lower()
                if q_sup and q_sup not in low:
                    continue
                for rgx in addr_regexes:
                    for m2 in re.finditer(rgx, text, flags=re.IGNORECASE):
                        addr = m2.group(0).strip()
                        # prefer longer plausible addresses
                        if len(addr) > best_len:
                            best_len = len(addr)
                            candidate = addr
            if candidate:
                answer = f"Address: {candidate}"
                related_docs = self.find_related_documents(query, max_results=10)
                return {
                    'query': query,
                    'answer': answer,
                    'search_results': search_results,
                    'related_documents': related_docs,
                    'num_results': len(search_results),
                    'num_relationships': len(related_docs)
                }
        
        # Find related documents
        related_docs = self.find_related_documents(query, max_results=10)
        
        # Get answer from Claude
        answer = self.query_with_claude(query, search_results)
        
        return {
            'query': query,
            'answer': answer,
            'search_results': search_results,
            'related_documents': related_docs,
            'num_results': len(search_results),
            'num_relationships': len(related_docs)
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        if not self.graph:
            return {"error": "System not built"}
        
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }
        
        # Node type breakdown
        node_types = {}
        source_types = {}
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node_type = node_data.get('type', 'unknown')
            source = node_data.get('source', 'unknown')
            
            node_types[node_type] = node_types.get(node_type, 0) + 1
            source_types[source] = source_types.get(source, 0) + 1
        
        stats['node_types'] = node_types
        stats['source_types'] = source_types
        
        # Edge type breakdown
        edge_types = {}
        for edge in self.graph.edges(data=True):
            edge_type = edge[2].get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        stats['edge_types'] = edge_types
        
        return stats
    
    def export_graph_data(self, output_file: str = "graph_data.json"):
        """Export graph data for visualization"""
        if not self.graph:
            return {"error": "Graph not built"}
        
        nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'label': node_data.get('name', node_id),
                'type': node_data.get('type', 'unknown'),
                'source': node_data.get('source', 'unknown'),
                'group': node_data.get('source', 'unknown')
            })
        
        edges = []
        for edge in self.graph.edges(data=True):
            edges.append({
                'from': edge[0],
                'to': edge[1],
                'type': edge[2].get('type', 'unknown'),
                'strength': edge[2].get('strength', 0),
                'description': edge[2].get('description', '')
            })
        
        graph_data = {
            'nodes': nodes,
            'edges': edges,
            'statistics': self.get_system_statistics()
        }
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"✅ Graph data exported to {output_file}")
        return graph_data
    
    def export_to_neo4j(self, neo4j_uri: str = "bolt://localhost:7687", 
                       neo4j_user: str = "neo4j", neo4j_password: str = "password",
                       clear_database: bool = True) -> Dict[str, Any]:
        """Export graph data to Neo4j database for visualization"""
        if not self.graph:
            return {"error": "Graph not built"}
        
        try:
            # Connect to Neo4j
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            
            with driver.session() as session:
                if clear_database:
                    # Clear existing data
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("🗑️ Cleared existing Neo4j data")
                
                # Create nodes
                node_count = 0
                for node_id, node_data in self.graph.nodes(data=True):
                    # Determine node labels
                    labels = [node_data.get('source', 'Unknown').title()]
                    if node_data.get('type'):
                        labels.append(node_data.get('type').title())
                    if node_data.get('document_type'):
                        labels.append(node_data.get('document_type').title())
                    
                    # Create node with properties
                    properties = {
                        'id': node_id,
                        'name': node_data.get('name', 'Unknown'),
                        'type': node_data.get('type', 'unknown'),
                        'source': node_data.get('source', 'unknown'),
                        **node_data.get('attributes', {})
                    }
                    
                    # Remove None values
                    properties = {k: v for k, v in properties.items() if v is not None}
                    
                    labels_str = ':'.join(labels)
                    cypher = f"CREATE (n:{labels_str}) SET n = $properties"
                    session.run(cypher, properties=properties)
                    node_count += 1
                
                # Create relationships
                rel_count = 0
                for edge in self.graph.edges(data=True):
                    from_node = edge[0]
                    to_node = edge[1]
                    edge_data = edge[2]
                    
                    relationship_type = edge_data.get('type', 'RELATED')
                    properties = {
                        'strength': edge_data.get('strength', 0),
                        'description': edge_data.get('description', '')
                    }
                    
                    cypher = f"""
                    MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                    CREATE (a)-[r:{relationship_type}]->(b)
                    SET r = $properties
                    """
                    session.run(cypher, from_id=from_node, to_id=to_node, properties=properties)
                    rel_count += 1
                
                driver.close()
                
                result = {
                    'success': True,
                    'nodes_created': node_count,
                    'relationships_created': rel_count,
                    'neo4j_uri': neo4j_uri
                }
                
                logger.info(f"✅ Exported {node_count} nodes and {rel_count} relationships to Neo4j")
                logger.info(f"🌐 Neo4j Browser: http://localhost:7474")
                logger.info(f"🔗 Connection: {neo4j_uri}")
                
                return result
                
        except Exception as e:
            error_msg = f"Error exporting to Neo4j: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def generate_neo4j_cypher_queries(self) -> List[str]:
        """Generate useful Cypher queries for exploring the graph"""
        queries = [
            # Basic exploration queries
            "MATCH (n) RETURN n LIMIT 25",
            "MATCH (n) RETURN labels(n) as NodeTypes, count(n) as Count ORDER BY Count DESC",
            "MATCH ()-[r]->() RETURN type(r) as RelationshipTypes, count(r) as Count ORDER BY Count DESC",
            
            # Document relationships
            "MATCH (n:Excel)-[r]->(m:Pdf) RETURN n.name, type(r), m.name LIMIT 10",
            "MATCH (n:Pdf)-[r]->(m:Excel) RETURN n.name, type(r), m.name LIMIT 10",
            
            # Supplier analysis
            "MATCH (n:Supplier_name) RETURN n.name, n.source, count(n) as Frequency ORDER BY Frequency DESC",
            "MATCH (n:Supplier_name)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 10",
            
            # Document type analysis
            "MATCH (n) WHERE n.document_type IS NOT NULL RETURN n.document_type, count(n) as Count ORDER BY Count DESC",
            "MATCH (n:Grn)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 10",
            "MATCH (n:Invoice)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 10",
            
            # Relationship strength analysis
            "MATCH ()-[r]->() WHERE r.strength > 0.8 RETURN type(r), count(r) as StrongRelationships ORDER BY StrongRelationships DESC",
            "MATCH (n)-[r]->(m) WHERE r.strength = 1.0 RETURN n.name, type(r), m.name LIMIT 10",
            
            # Path finding
            "MATCH path = (n:Excel)-[*1..3]-(m:Pdf) RETURN path LIMIT 5",
            "MATCH (n:Supplier_name)-[*1..2]-(m) RETURN n.name, m.name, m.type LIMIT 10",
            
            # Statistics
            "MATCH (n) RETURN count(n) as TotalNodes",
            "MATCH ()-[r]->() RETURN count(r) as TotalRelationships",
            "MATCH (n) RETURN labels(n) as Labels, count(n) as Count ORDER BY Count DESC"
        ]
        
        return queries

def main():
    """Demo the Graph RAG System"""
    logger.info("🚀 Graph RAG System Demo")
    logger.info("📊 Excel + PDF Integration with Relationships")
    logger.info("=" * 60)
    
    # Initialize system
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    rag = GraphRAGSystem(claude_api_key)
    
    # Build system
    excel_path = "data/excel/ABC_Book_Stores_Inventory_Register.xlsx"
    pdf_folder = "data/pdfs"
    
    if not os.path.exists(excel_path) or not os.path.exists(pdf_folder):
        logger.error("❌ Required data files not found")
        return
    
    entities, relationships = rag.build_system(excel_path, pdf_folder)
    
    # Show statistics
    stats = rag.get_system_statistics()
    logger.info(f"\n📊 System Statistics:")
    logger.info(f"  • Total nodes: {stats['total_nodes']}")
    logger.info(f"  • Total edges: {stats['total_edges']}")
    logger.info(f"  • Graph density: {stats['density']:.3f}")
    logger.info(f"  • Connected components: {stats['connected_components']}")
    
    logger.info(f"\n📋 Node Types:")
    for node_type, count in stats['node_types'].items():
        logger.info(f"  • {node_type}: {count}")
    
    logger.info(f"\n🔗 Relationship Types:")
    for edge_type, count in stats['edge_types'].items():
        logger.info(f"  • {edge_type}: {count}")
    
    # Test queries
    test_queries = [
        "What books are available in the inventory?",
        "Show me GRN documents from 2024",
        "Find purchase orders and their related inventory",
        "What invoices are available?"
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"❓ Query: {query}")
        logger.info('='*60)
        
        result = rag.search_and_answer(query)
        logger.info(f"📊 Found {result['num_results']} search results")
        logger.info(f"🔗 Found {result['num_relationships']} related documents")
        
        if result['answer']:
            logger.info(f"\n🤖 Answer: {result['answer']}")
        else:
            logger.info(f"\n📋 Search Results:")
            for i, search_result in enumerate(result['search_results'][:3]):
                logger.info(f"  {i+1}. {search_result['type']} - {search_result['source']} (similarity: {search_result['similarity']:.3f})")
                logger.info(f"     {search_result['text'][:100]}...")
    
    # Export graph data
    logger.info(f"\n📁 Exporting graph data...")
    rag.export_graph_data()
    
    logger.info(f"\n🎉 Graph RAG System Demo completed!")
    logger.info(f"📊 System has {stats['total_nodes']} nodes and {stats['total_edges']} relationships")

if __name__ == "__main__":
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("⚠️  No Anthropic API key found!")
        logger.info("🔑 To enable AI-powered features:")
        logger.info("   1. Get your API key from: https://console.anthropic.com/")
        logger.info("   2. Set it as environment variable: export ANTHROPIC_API_KEY='your_key'")
        logger.info("\n🤖 Running with basic functionality (no AI features)...")
    
    rag = GraphRAGSystem(api_key)
    rag.build_system("data/excel/ABC_Book_Stores_Inventory_Register.xlsx", "data/pdfs")

    stats = rag.get_system_statistics()
    logger.info("\n📈 SYSTEM STATISTICS 📈")
    logger.info(json.dumps(stats, indent=2))

    query = "Find invoices linked with supplier XYZ Books"
    result = rag.search_and_answer(query)

    logger.info("\n🧠 AI ANSWER 🧠")
    logger.info(result["answer"])

    logger.info("\n🔗 RELATED DOCUMENTS 🔗")
    for rel in result["related_documents"]:
        logger.info(f"- {rel['entity_name']} ↔ {rel['related_entity']} ({rel['relationship_type']}) [{rel['relationship_strength']}]")

    