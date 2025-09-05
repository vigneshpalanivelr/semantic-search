#!/usr/bin/env python3
"""
Simple Web API for DevOps Build Failure Analyzer
=================================================

A lightweight Flask API that exposes the semantic search functionality
for integration with CI/CD systems, dashboards, or other tools.

Usage:
    python3 api.py

Endpoints:
    GET  /health              - Health check
    POST /analyze             - Analyze build failure
    GET  /categories          - Get available error categories
    GET  /stats               - Get system statistics
"""

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
    from flask import Flask, request, jsonify
    from flask_cors import CORS

import json
import os
from datetime import datetime
from semantic_search import QueryEngine, OutputFormatter

app = Flask(__name__)
CORS(app)  # Enable CORS for web integration

# Global analyzer instance
analyzer = None

def initialize_analyzer():
    """Initialize the query engine with knowledge base"""
    global analyzer
    if analyzer is None:
        print("Initializing Build Failure Analyzer...")
        analyzer = QueryEngine()

        # Load default data file
        data_file = "pipeline-failure-log.json"
        if os.path.exists(data_file):
            analyzer.load_knowledge_base(data_file)
            print(f"Loaded knowledge base from {data_file}")
        else:
            print(f"Warning: Data file {data_file} not found")

    return analyzer

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        analyzer = initialize_analyzer()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system': 'DevOps Build Failure Analyzer',
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_failure():
    """Analyze a build failure"""
    try:
        # Get request data
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing required field: query',
                'example': {'query': 'docker build failed out of memory'}
            }), 400

        query = data['query']
        top_k = data.get('top_k', 3)

        # Validate input
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400

        if top_k < 1 or top_k > 10:
            return jsonify({'error': 'top_k must be between 1 and 10'}), 400

        # Analyze failure
        analyzer = initialize_analyzer()
        result = analyzer.analyze_failure(query, top_k)

        # Add API metadata
        result['api_info'] = {
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': 0,  # Could add timing here
            'version': '1.0.0'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available error categories and their automated actions"""
    try:
        analyzer = initialize_analyzer()

        categories = {
            'resource': {
                'description': 'Memory, CPU, or disk resource issues',
                'keywords': ['memory', 'oom', 'cpu', 'disk', 'killed'],
                'action': analyzer.automated_actions.get('resource', '')
            },
            'build': {
                'description': 'Compilation and build-time errors',
                'keywords': ['compile', 'syntax', 'dependency', 'module'],
                'action': analyzer.automated_actions.get('build', '')
            },
            'test': {
                'description': 'Test execution failures',
                'keywords': ['test failed', 'assertion', 'timeout'],
                'action': analyzer.automated_actions.get('test', '')
            },
            'deployment': {
                'description': 'Deployment and orchestration issues',
                'keywords': ['deploy', 'kubernetes', 'helm', 'imagepull'],
                'action': analyzer.automated_actions.get('deployment', '')
            },
            'network': {
                'description': 'Network connectivity problems',
                'keywords': ['network', 'connection', 'timeout', 'dns'],
                'action': analyzer.automated_actions.get('network', '')
            },
            'security': {
                'description': 'Security scan failures and vulnerabilities',
                'keywords': ['security', 'vulnerability', 'cve'],
                'action': analyzer.automated_actions.get('security', '')
            },
            'storage': {
                'description': 'Storage and artifact repository issues',
                'keywords': ['artifactory', 'storage', 'disk space'],
                'action': analyzer.automated_actions.get('storage', '')
            }
        }

        return jsonify({
            'categories': categories,
            'total_categories': len(categories),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get categories',
            'details': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        # This is a simplified stats endpoint
        # In a real system, you'd track actual usage metrics

        stats = {
            'system_info': {
                'status': 'operational',
                'embedding_model': 'all-MiniLM-L6-v2',
                'database': 'ChromaDB',
                'version': '1.0.0'
            },
            'knowledge_base': {
                'total_failures': 'Unknown',  # Would query actual DB
                'categories_covered': 7,
                'last_updated': 'Unknown'
            },
            'api_usage': {
                'uptime': 'Unknown',
                'total_queries': 'Unknown',
                'avg_response_time': 'Unknown'
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({
            'error': 'Failed to get stats',
            'details': str(e)
        }), 500

@app.route('/', methods=['GET'])
def api_info():
    """API documentation endpoint"""
    return jsonify({
        'name': 'DevOps Build Failure Analyzer API',
        'version': '1.0.0',
        'description': 'Semantic search API for CI/CD pipeline failure analysis',
        'endpoints': {
            'GET /health': 'Health check',
            'POST /analyze': 'Analyze single build failure',
            'GET /categories': 'Get error categories and actions',
            'GET /stats': 'Get system statistics',
            'GET /': 'API documentation'
        },
        'example_usage': {
            'analyze': {
                'method': 'POST',
                'url': '/analyze',
                'body': {
                    'query': 'docker build failed out of memory',
                    'top_k': 3
                }
            }
        }
    })

if __name__ == '__main__':
    print("Starting DevOps Build Failure Analyzer API...")
    print("📖 API Documentation: http://localhost:5000/")
    print("Try: curl -X POST http://localhost:5000/analyze -H 'Content-Type: application/json' -d '{\"query\":\"docker build failed\"}'")

    app.run(host='0.0.0.0', port=5000, debug=True)