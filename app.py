#!/usr/bin/env python3
"""
Graph RAG System Web Interface
Simple Flask frontend for easy querying
"""

from flask import Flask, render_template, request, jsonify
import os
import json
from dotenv import load_dotenv
from graph_rag_system import GraphRAGSystem

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Global variable to store the RAG system
rag_system = None

def initialize_system():
    """Initialize the Graph RAG system"""
    global rag_system
    if rag_system is None:
        print("🔧 Initializing Graph RAG System...")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        rag_system = GraphRAGSystem(api_key)
        rag_system.build_system("data/excel/ABC_Book_Stores_Inventory_Register.xlsx", "data/pdfs")
        print("✅ System initialized successfully!")

@app.route('/api/reload', methods=['POST'])
def reload_system():
    """Rebuild the Graph RAG system (use when new PDFs are added)"""
    try:
        global rag_system
        print("🔄 Reloading Graph RAG System...")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        # Recreate the system to ensure a clean rebuild
        rag_system = GraphRAGSystem(api_key)
        rag_system.build_system("data/excel/ABC_Book_Stores_Inventory_Register.xlsx", "data/pdfs")
        print("✅ Reload completed!")
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize system on startup
print("🚀 Starting Graph RAG Web Interface...")
initialize_system()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests"""
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        # Initialize system if not already done
        initialize_system()
        
        # Process query
        result = rag_system.search_and_answer(query_text)
        
        return jsonify({
            'success': True,
            'query': query_text,
            'answer': result['answer'],
            'num_results': result['num_results'],
            'num_relationships': result['num_relationships'],
            'related_documents': result['related_documents'][:10]  # Limit to 10
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    """Get system statistics"""
    try:
        initialize_system()
        stats = rag_system.get_system_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Graph RAG System is running'})

if __name__ == '__main__':
    print("🚀 Starting Graph RAG Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
