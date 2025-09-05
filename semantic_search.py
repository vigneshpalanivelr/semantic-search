#!/usr/bin/env python3
"""
DevOps CI Build Failure Analyzer - Semantic Search & RAG System
================================================================

A simplified system that transforms CI/CD build failures into actionable insights
using semantic search to find similar past failures and their solutions.

Core Components:
1. Log Collector (simulated with JSON files)
2. Error Processor (parsing and extraction)
3. Vector Database (ChromaDB for semantic search)
4. Query Engine (semantic matching + automated suggestions)
"""

import json
import os
import subprocess
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import dependencies (install via setup.sh)
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print("Required dependencies not found!")
    print("Please run the setup script first:")
    print("   chmod +x setup.sh")
    print("   ./setup.sh")
    print("   source activate_env.sh")
    sys.exit(1)

# === Configuration ===
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and good quality
DATA_FILE = "pipeline-failure-log.json"

# === Setup Vector Database ===
def setup_database():
    """Initialize ChromaDB with persistent storage"""
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))

    # Reset collection for fresh start
    if "build_failures" in [c.name for c in client.list_collections()]:
        client.delete_collection("build_failures")

    collection = client.create_collection("build_failures")
    return client, collection

# === Error Processing (Component 2 from our design) ===
class ErrorProcessor:
    """Processes and cleans error messages from CI/CD logs"""

    @staticmethod
    def extract_key_info(error_record: Dict) -> str:
        """Extract the most relevant information for semantic search"""
        key_parts = [
            error_record.get('error_msg', ''),
            error_record.get('details', ''),
            error_record.get('error_type', ''),
            error_record.get('stage', ''),
        ]
        return ' '.join(filter(None, key_parts))

    @staticmethod
    def categorize_error(error_text: str) -> str:
        """Simple error categorization based on keywords"""
        text = error_text.lower()

        if any(keyword in text for keyword in ['memory', 'oom', '137', 'killed']):
            return 'resource'
        elif any(keyword in text for keyword in ['compile', 'syntax', 'module not found']):
            return 'build'
        elif any(keyword in text for keyword in ['test failed', 'assertion', 'timeout']):
            return 'test'
        elif any(keyword in text for keyword in ['deploy', 'kubernetes', 'imagepull']):
            return 'deployment'
        elif any(keyword in text for keyword in ['network', 'connection', 'timeout']):
            return 'network'
        elif any(keyword in text for keyword in ['security', 'vulnerability', 'cve']):
            return 'security'
        elif any(keyword in text for keyword in ['artifactory', 'storage', 'disk']):
            return 'storage'
        else:
            return 'general'

# === Vector Database (Component 3 from our design) ===
class VectorDatabase:
    """Handles embedding generation and semantic search"""

    def __init__(self):
        self.client, self.collection = setup_database()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.processor = ErrorProcessor()

    def ingest_errors(self, json_file_path: str):
        """Load and process error logs into vector database"""
        print(f"Loading errors from {json_file_path}...")

        with open(json_file_path, 'r') as f:
            data = json.load(f)

        doc_id = 0
        total_errors = 0

        for error_type, error_list in data.items():
            for error_record in error_list:
                # Extract text for embedding
                search_text = self.processor.extract_key_info(error_record)

                # Generate embedding
                embedding = self.model.encode([search_text])[0].tolist()

                # Store in vector database
                self.collection.add(
                    ids=[f"error_{doc_id}"],
                    embeddings=[embedding],
                    documents=[search_text],
                    metadatas=[{
                        "original_record": json.dumps(error_record),
                        "error_type": error_type,
                        "category": self.processor.categorize_error(search_text),
                        "product": error_record.get('product', 'unknown'),
                        "severity": error_record.get('severity', 'medium'),
                        "timestamp": error_record.get('timestamp', '')
                    }]
                )

                doc_id += 1
                total_errors += 1

        print(f"Indexed {total_errors} build failures")

    def search_similar_failures(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar past failures using semantic search"""
        print(f"Searching for: '{query}'")

        # Generate query embedding
        query_embedding = self.model.encode([query])[0].tolist()

        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Process and format results
        matches = []
        for i, metadata in enumerate(results['metadatas'][0]):
            original_record = json.loads(metadata['original_record'])

            match = {
                'similarity_score': round(1 - results['distances'][0][i], 4),
                'confidence': self._calculate_confidence(1 - results['distances'][0][i]),
                'category': metadata['category'],
                'error_info': {
                    'type': metadata['error_type'],
                    'product': original_record.get('product', 'N/A'),
                    'stage': original_record.get('stage', 'N/A'),
                    'message': original_record.get('error_msg', 'N/A'),
                    'details': original_record.get('details', 'N/A'),
                    'severity': original_record.get('severity', 'medium')
                },
                'solution': {
                    'suggestion': original_record.get('suggestion', 'No specific suggestion available'),
                    'contact_teams': original_record.get('contact_team', []),
                    'estimated_resolution_time': original_record.get('resolution_time', 'Unknown'),
                },
                'references': {
                    'pipeline_url': original_record.get('pipeline_url', ''),
                    'timestamp': original_record.get('timestamp', ''),
                    'host': original_record.get('host', 'Unknown')
                }
            }

            matches.append(match)

        return matches

    def _calculate_confidence(self, similarity: float) -> str:
        """Convert similarity score to confidence level"""
        if similarity >= 0.8:
            return "high"
        elif similarity >= 0.6:
            return "medium"
        else:
            return "low"

# === Query Engine (Component 4 from our design) ===
class QueryEngine:
    """Main interface for querying build failures"""

    def __init__(self):
        self.vector_db = VectorDatabase()
        self.automated_actions = {
            'resource': "Resource Issue: Check memory/CPU allocation and contact DevOps team",
            'build': "Build Issue: Review code changes and dependency configurations",
            'test': "Test Issue: Check test environment and data setup",
            'deployment': "Deployment Issue: Verify deployment configuration and resources",
            'network': "Network Issue: Check connectivity and firewall rules",
            'security': "Security Issue: Address vulnerabilities before proceeding",
            'storage': "Storage Issue: Check disk space and clean up if needed"
        }

    def load_knowledge_base(self, json_file: str):
        """Load historical failure data"""
        self.vector_db.ingest_errors(json_file)

    def analyze_failure(self, query: str, top_results: int = 3) -> Dict:
        """Analyze a build failure and provide recommendations"""

        # Find similar past failures
        similar_failures = self.vector_db.search_similar_failures(query, top_results)

        if not similar_failures:
            return {
                'query': query,
                'status': 'no_matches',
                'message': 'No similar failures found in knowledge base',
                'suggestion': 'Consider adding this failure to the knowledge base for future reference'
            }

        # Get automated action based on most similar failure
        primary_match = similar_failures[0]
        category = primary_match['category']
        automated_action = self.automated_actions.get(category, "Manual investigation required")

        # Prepare comprehensive response
        analysis_result = {
            'query': query,
            'status': 'matches_found',
            'summary': {
                'most_likely_cause': primary_match['error_info']['type'].replace('_', ' ').title(),
                'confidence': primary_match['confidence'],
                'automated_action': automated_action,
                'estimated_resolution': primary_match['solution']['estimated_resolution_time']
            },
            'similar_failures': similar_failures,
            'recommendations': self._generate_recommendations(similar_failures)
        }

        return analysis_result

    def _generate_recommendations(self, failures: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on similar failures"""
        recommendations = []

        # Collect unique suggestions
        suggestions = set()
        teams = set()

        for failure in failures:
            if failure['solution']['suggestion']:
                suggestions.add(failure['solution']['suggestion'])
            teams.update(failure['solution']['contact_teams'])

        # Primary recommendations
        for suggestion in list(suggestions)[:3]:  # Top 3 unique suggestions
            recommendations.append(f"{suggestion}")

        # Contact information
        if teams:
            team_list = ', '.join(sorted(teams))
            recommendations.append(f"Contact teams: {team_list}")

        return recommendations

# === User Interface (Simplified from our design) ===
class OutputFormatter:
    """Handles different output formats for the analysis results"""

    @staticmethod
    def format_console_output(result: Dict) -> str:
        """Format results for console display"""
        if result['status'] == 'no_matches':
            return f"""
Query: "{result['query']}"
{result['message']}
{result['suggestion']}
"""

        output = f"""
Build Failure Analysis
{'=' * 50}
Query: "{result['query']}"

SUMMARY
{'-' * 20}
Most Likely Cause: {result['summary']['most_likely_cause']}
Confidence: {result['summary']['confidence'].upper()}
Automated Action: {result['summary']['automated_action']}
Est. Resolution Time: {result['summary']['estimated_resolution']}

RECOMMENDATIONS
{'-' * 20}"""

        for rec in result['recommendations']:
            output += f"\n{rec}"

        output += f"""

SIMILAR PAST FAILURES
{'-' * 30}"""

        for i, failure in enumerate(result['similar_failures'], 1):
            output += f"""

#{i} - Similarity: {failure['similarity_score']} ({failure['confidence']} confidence)
├─ Product: {failure['error_info']['product']}
├─ Stage: {failure['error_info']['stage']}
├─ Error: {failure['error_info']['message']}
├─ Solution: {failure['solution']['suggestion']}
└─ Teams: {', '.join(failure['solution']['contact_teams'])}"""

            if failure['references']['pipeline_url']:
                output += f"\n   Pipeline: {failure['references']['pipeline_url']}"

        return output

    @staticmethod
    def format_json_output(result: Dict) -> str:
        """Format results as JSON for API integration"""
        return json.dumps(result, indent=2)

# === CLI Interface ===
def main():
    parser = argparse.ArgumentParser(
        description="DevOps CI Build Failure Analyzer - Semantic Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 semantic_search.py                                    # Demo mode
  python3 semantic_search.py -q "docker out of memory"         # Analyze specific failure
  python3 semantic_search.py -q "build timeout" --json         # JSON output
  python3 semantic_search.py -q "compilation failed" --top 5   # More results
        """
    )

    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Describe the build failure you want to analyze"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of similar failures to show (default: 3)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format for API integration"
    )

    parser.add_argument(
        "--load-data",
        type=str,
        default=DATA_FILE,
        help=f"JSON file containing failure logs (default: {DATA_FILE})"
    )

    args = parser.parse_args()

    # Initialize the system
    print("Initializing DevOps Build Failure Analyzer...")
    query_engine = QueryEngine()

    # Load knowledge base
    if os.path.exists(args.load_data):
        query_engine.load_knowledge_base(args.load_data)
    else:
        print(f"Error: Data file '{args.load_data}' not found!")
        return 1

    formatter = OutputFormatter()

    if args.query:
        # Single query mode
        result = query_engine.analyze_failure(args.query, args.top)

        if args.json:
            print(formatter.format_json_output(result))
        else:
            print(formatter.format_console_output(result))
    else:
        # Demo mode
        run_demo(query_engine, formatter)

    return 0

def run_demo(query_engine: QueryEngine, formatter: OutputFormatter):
    """Run interactive demo with sample queries"""
    print("""
DevOps Build Failure Analyzer - Demo Mode
============================================

This system helps you quickly find solutions to build failures by
searching through historical failure patterns and their resolutions.
""")

    # Demo queries representing common failure scenarios
    demo_queries = [
        "docker container out of memory error",
        "compilation failed missing dependency",
        "artifactory upload storage full",
        "kubernetes deployment imagepull failed",
        "unit tests timeout database connection"
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\nDemo Query {i}/{len(demo_queries)}")
        print("=" * 60)

        result = query_engine.analyze_failure(query, top_results=2)
        print(formatter.format_console_output(result))

        if i < len(demo_queries):
            input("\nPress Enter to continue to next demo...")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Try your own query!")
    print("=" * 60)

    try:
        while True:
            user_query = input("\nEnter your build failure description (or 'quit' to exit): ")

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Thanks for using the Build Failure Analyzer!")
                break

            if user_query.strip():
                result = query_engine.analyze_failure(user_query, top_results=3)
                print(formatter.format_console_output(result))
            else:
                print("Please enter a failure description")

    except (EOFError, KeyboardInterrupt):
        print("\nThanks for using the Build Failure Analyzer!")

# === Example Usage as Library ===
def example_library_usage():
    """Example of how to use this as a library in other applications"""

    # Initialize the analyzer
    analyzer = QueryEngine()
    analyzer.load_knowledge_base("pipeline-failure-log.json")

    # Analyze a failure
    result = analyzer.analyze_failure("docker build failed out of memory")

    # Access structured results
    if result['status'] == 'matches_found':
        print(f"Primary cause: {result['summary']['most_likely_cause']}")
        print(f"Confidence: {result['summary']['confidence']}")
        print(f"Action needed: {result['summary']['automated_action']}")

        # Get contact teams
        for failure in result['similar_failures']:
            teams = failure['solution']['contact_teams']
            print(f"Contact: {', '.join(teams)}")

    return result

# === Health Check Function ===
def health_check():
    """Basic health check to verify system components"""
    try:
        # Test embedding model
        model = SentenceTransformer(EMBEDDING_MODEL)
        test_embedding = model.encode(["test query"])

        # Test database
        client, collection = setup_database()

        print("All components working correctly")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    sys.exit(main())