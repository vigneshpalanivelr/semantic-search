import json
import os
import subprocess
import sys
import argparse

def install_requirements():
    """Install required packages if not available"""
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully!")
        from sentence_transformers import SentenceTransformer
        import chromadb
    return SentenceTransformer, chromadb

# Install and import dependencies
SentenceTransformer, chromadb = install_requirements()
from chromadb.config import Settings

# === Config ===
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Or try "all-mpnet-base-v2" for higher accuracy

# === Setup ===
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
if "errors" in [c.name for c in client.list_collections()]:
    client.delete_collection("errors")
collection = client.create_collection("errors")

model = SentenceTransformer(EMBEDDING_MODEL)

# === Document Ingestion ===
def ingest_documents(json_paths):
    for idx, path in enumerate(json_paths):
        with open(path) as f:
            data = json.load(f)
        text = json.dumps(data)  # Use specific fields if needed
        emb = model.encode([text])[0].tolist()
        collection.add(
            ids=[f"doc-{idx}"],
            embeddings=[emb],
            documents=[text],
            metadatas=[{"source": path}]
        )

# === Semantic Search ===
def semantic_search(query, top_k=5):
    emb = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results

# === Custom Error Logic ===
def analyze_error(doc_str):
    text = doc_str.lower()
    if "docker" in text and ("out of memory" in text or "oom" in text or "137" in text):
        return "Resource Issue: Contact DevOps team for memory/resource allocation."
    elif "artifactory" in text and ("storage" in text or "disk" in text or "507" in text):
        return "Storage Issue: Contact DevOps/IT for Artifactory disk space."
    elif "compilation" in text or "compile failed" in text or "syntax" in text:
        return "Build Issue: Check syntax errors and missing dependencies."
    elif "module" in text and "not found" in text:
        return "Dependency Issue: Check imports and package dependencies."
    return None

# === CLI Interface ===
def search_errors(query):
    """Simple function to search for errors given a query string"""
    results = semantic_search(query, top_k=3)
    matches = []
    
    for i, doc in enumerate(results['documents'][0]):
        match_info = {
            'similarity': 1-results['distances'][0][i],
            'source': results['metadatas'][0][i]['source']
        }
        
        try:
            doc_data = json.loads(doc)
            for error_type, errors in doc_data.items():
                if isinstance(errors, list):
                    for error in errors:
                        if isinstance(error, dict):
                            match_info.update({
                                'error_type': error_type,
                                'product': error.get('product', 'N/A'),
                                'stage': error.get('stage', 'N/A'),
                                'error_msg': error.get('error_msg', 'N/A'),
                                'details': error.get('details', 'N/A'),
                                'suggestion': error.get('suggestion', 'N/A'),
                                'contact_team': error.get('contact_team', []),
                                'pipeline_url': error.get('pipeline_url', '')
                            })
                            break
        except json.JSONDecodeError:
            match_info['raw_snippet'] = doc[:200]
        
        action = analyze_error(doc)
        if action:
            match_info['automated_action'] = action
            
        matches.append(match_info)
    
    return matches

# === CLI Interface ===
def main():
    parser = argparse.ArgumentParser(
        description="Semantic Search for Pipeline Failure Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 semantic_search.py                                    # Interactive demo
  python3 semantic_search.py -q "docker out of memory"         # Single query
  python3 semantic_search.py --query "compilation failed"      # Single query (long form)
  python3 semantic_search.py -q "build timeout" --top 5        # Show top 5 results
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Search query for pipeline failures"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of top results to show (default: 3)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    # Ingest documents
    docs = ["pipeline-failure-log.json"]
    ingest_documents(docs)
    
    if args.query:
        # Single query mode
        results = semantic_search(args.query, top_k=args.top)
        
        if args.json:
            # JSON output
            output = []
            for i, doc in enumerate(results['documents'][0]):
                match_info = {
                    'similarity': 1-results['distances'][0][i],
                    'source': results['metadatas'][0][i]['source']
                }
                
                try:
                    doc_data = json.loads(doc)
                    for error_type, errors in doc_data.items():
                        if isinstance(errors, list):
                            for error in errors:
                                if isinstance(error, dict):
                                    match_info.update({
                                        'error_type': error_type,
                                        'product': error.get('product', 'N/A'),
                                        'stage': error.get('stage', 'N/A'),
                                        'error_msg': error.get('error_msg', 'N/A'),
                                        'details': error.get('details', 'N/A'),
                                        'suggestion': error.get('suggestion', 'N/A'),
                                        'contact_team': error.get('contact_team', []),
                                        'pipeline_url': error.get('pipeline_url', '')
                                    })
                                    break
                except json.JSONDecodeError:
                    match_info['raw_snippet'] = doc[:200]
                
                action = analyze_error(doc)
                if action:
                    match_info['automated_action'] = action
                    
                output.append(match_info)
            
            print(json.dumps(output, indent=2))
        else:
            # Pretty output
            print(f"Query: '{args.query}'")
            print("=" * 50)
            
            if not results['documents'][0]:
                print("No results found.")
                return
            
            for i, doc in enumerate(results['documents'][0]):
                print(f"\nMatch {i+1} (Similarity: {1-results['distances'][0][i]:.4f})")
                print(f"Source: {results['metadatas'][0][i]['source']}")
                
                try:
                    doc_data = json.loads(doc)
                    for error_type, errors in doc_data.items():
                        if isinstance(errors, list):
                            for error in errors:
                                if isinstance(error, dict):
                                    print(f"\nError Type: {error_type}")
                                    print(f"Product: {error.get('product', 'N/A')}")
                                    print(f"Stage: {error.get('stage', 'N/A')}")
                                    print(f"Error: {error.get('error_msg', 'N/A')}")
                                    print(f"Details: {error.get('details', 'N/A')}")
                                    print(f"Suggestion: {error.get('suggestion', 'N/A')}")
                                    if 'contact_team' in error:
                                        teams = error['contact_team'] if isinstance(error['contact_team'], list) else [error['contact_team']]
                                        print(f"Contact: {', '.join(teams)}")
                                    if 'pipeline_url' in error:
                                        print(f"Pipeline: {error['pipeline_url']}")
                                    break
                except json.JSONDecodeError:
                    print(f"Raw snippet: {doc[:200]}...")
                
                action = analyze_error(doc)
                if action:
                    print(f"\nAutomated Action: {action}")
    else:
        # Demo mode
        run_demo()

def run_demo():
    """Run the interactive demo"""
    print("\n🔍 Semantic Search Demo - Pipeline Failure Analysis")
    print("=" * 55)
    
    # Test multiple queries
    queries = [
        "compilation error syntax",
        "docker out of memory",
        "artifactory storage full",
        "module not found error"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        results = semantic_search(query, top_k=2)
        
        if not results['documents'][0]:
            print("No results found.")
            continue
            
        for i, doc in enumerate(results['documents'][0]):
            print(f"\nMatch {i+1} (Similarity: {1-results['distances'][0][i]:.4f})")
            print(f"Source: {results['metadatas'][0][i]['source']}")
            
            # Parse and display structured data
            try:
                doc_data = json.loads(doc)
                for error_type, errors in doc_data.items():
                    if isinstance(errors, list):
                        for error in errors:
                            if isinstance(error, dict):
                                print(f"\nError Type: {error_type}")
                                print(f"Product: {error.get('product', 'N/A')}")
                                print(f"Stage: {error.get('stage', 'N/A')}")
                                print(f"Error: {error.get('error_msg', 'N/A')}")
                                print(f"Details: {error.get('details', 'N/A')}")
                                print(f"Suggestion: {error.get('suggestion', 'N/A')}")
                                if 'contact_team' in error:
                                    teams = error['contact_team'] if isinstance(error['contact_team'], list) else [error['contact_team']]
                                    print(f"Contact: {', '.join(teams)}")
                                if 'pipeline_url' in error:
                                    print(f"Pipeline: {error['pipeline_url']}")
                                break
            except json.JSONDecodeError:
                print(f"Raw snippet: {doc[:200]}...")
            
            action = analyze_error(doc)
            if action:
                print(f"\nAutomated Action: {action}")
        
        print("\n" + "=" * 55)
    
    # Interactive search option  
    print("\n🔍 Try your own search query (or press Enter to skip):")
    try:
        user_query = input("> ")
        if user_query.strip():
            print(f"\nQuery: '{user_query}'")
            print("-" * 50)
            results = semantic_search(user_query, top_k=3)
            
            for i, doc in enumerate(results['documents'][0]):
                print(f"\nMatch {i+1} (Similarity: {1-results['distances'][0][i]:.4f})")
                print(f"Source: {results['metadatas'][0][i]['source']}")
                
                try:
                    doc_data = json.loads(doc)
                    for error_type, errors in doc_data.items():
                        if isinstance(errors, list):
                            for error in errors:
                                if isinstance(error, dict):
                                    print(f"\nError Type: {error_type}")
                                    print(f"Product: {error.get('product', 'N/A')}")
                                    print(f"Stage: {error.get('stage', 'N/A')}")
                                    print(f"Error: {error.get('error_msg', 'N/A')}")
                                    print(f"Details: {error.get('details', 'N/A')}")
                                    print(f"Suggestion: {error.get('suggestion', 'N/A')}")
                                    if 'contact_team' in error:
                                        teams = error['contact_team'] if isinstance(error['contact_team'], list) else [error['contact_team']]
                                        print(f"Contact: {', '.join(teams)}")
                                    if 'pipeline_url' in error:
                                        print(f"Pipeline: {error['pipeline_url']}")
                                    break
                except json.JSONDecodeError:
                    print(f"Raw snippet: {doc[:200]}...")
                
                action = analyze_error(doc)
                if action:
                    print(f"\nAutomated Action: {action}")
    except (EOFError, KeyboardInterrupt):
        print("\nSearch demo completed.")

# === Example Usage ===
if __name__ == "__main__":
    main()
