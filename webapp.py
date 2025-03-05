"""
A simple webapp for comparing outputs from different LLM pipeline runs.
"""

import os
import json
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from core.prompts import template_registry
from utils.file_ops import load_json_file

app = Flask(__name__)

# Base directory for output files
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

@app.route('/')
def home():
    """Home page with list of available model runs"""
    # Get all model directories
    models = _get_models()
    return render_template('index.html', models=models)

@app.route('/model/<model_name>')
def model_sessions(model_name):
    """List all sessions for a specific model"""
    sessions = _get_sessions(model_name)
    return render_template('sessions.html', model=model_name, sessions=sessions)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Compare view for selected sessions"""
    if request.method == 'POST':
        # Get selected sessions for comparison
        selected = request.form.getlist('sessions')
        if len(selected) < 2:
            return render_template('error.html', 
                                  message="Please select at least 2 sessions to compare")
        
        # Parse selection format: "model_name/session_id"
        selections = [s.split('/') for s in selected]
        sessions_data = {}
        
        for model, session in selections:
            sessions_data[f"{model}/{session}"] = _load_session_data(model, session)
            
        return render_template('compare.html', sessions=sessions_data)
    
    # GET request - show selection form
    models = _get_models()
    all_sessions = {}
    
    for model in models:
        all_sessions[model] = _get_sessions(model)
    
    return render_template('select.html', models=models, sessions=all_sessions)

@app.route('/api/session/<model>/<session>')
def api_session(model, session):
    """API endpoint to get session data"""
    data = _load_session_data(model, session)
    return jsonify(data)

def _get_models():
    """Get list of all model directories in output"""
    return [d for d in os.listdir(OUTPUT_DIR) 
            if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and not d.startswith('.')]

def _get_sessions(model_name):
    """Get all sessions for a given model"""
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(model_dir):
        return []
    
    return [d for d in os.listdir(model_dir) 
            if os.path.isdir(os.path.join(model_dir, d)) and not d.startswith('.')]

def _load_session_data(model, session):
    """Load all data for a specific session"""
    session_dir = os.path.join(OUTPUT_DIR, model, session)
    
    data = {
        "model": model,
        "session_id": session,
        "facts": [],
        "relationships": None,
        "chunks": []
    }
    
    # Load facts
    facts_dir = os.path.join(session_dir, "facts", "facts")
    if os.path.exists(facts_dir):
        facts = []
        for file in os.listdir(facts_dir):
            if file.endswith('.json'):
                fact_path = os.path.join(facts_dir, file)
                try:
                    fact_data = load_json_file(fact_path)
                    facts.append(fact_data)
                except Exception as e:
                    print(f"Error loading {fact_path}: {e}")
        data["facts"] = facts
    
    # Load relationships
    rel_path = os.path.join(session_dir, "relationships.json")
    if os.path.exists(rel_path):
        try:
            data["relationships"] = load_json_file(rel_path)
        except Exception as e:
            print(f"Error loading relationships: {e}")
    
    # Load chunks
    chunks_dir = os.path.join(session_dir, "chunks")
    if os.path.exists(chunks_dir):
        chunks = []
        for file in os.listdir(chunks_dir):
            if file.endswith('.json'):
                chunk_path = os.path.join(chunks_dir, file)
                try:
                    chunk_data = load_json_file(chunk_path)
                    chunks.append(chunk_data)
                except Exception as e:
                    print(f"Error loading {chunk_path}: {e}")
        data["chunks"] = chunks
    
    return data

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)