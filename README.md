# DevOps Build Failure Analyzer - Semantic Search System

An enterprise-grade semantic search system that transforms CI/CD build failures into actionable insights using AI-powered analysis to find similar past failures and their proven solutions.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Command Line Interface](#command-line-interface)
   - [Programmatic Usage](#programmatic-usage)
   - [Demo Mode](#demo-mode)
5. [Configuration](#configuration)
6. [Data Model](#data-model)
7. [Error Categories](#error-categories)
8. [API Reference](#api-reference)
9. [Examples](#examples)
10. [Performance](#performance)
11. [Development](#development)
12. [Contributing](#contributing)
13. [Troubleshooting](#troubleshooting)

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DevOps Build Failure Analyzer                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   Log Ingestion │    │   ML Processing  │    │   Query Engine   │   │
│  │                 │    │                  │    │                  │   │
│  │ • JSON Files    │───▶│ • Embedding      │───▶│ • Semantic       │   │
│  │ • Error Parsing │    │ • Categorization │    │   Search         │   │
│  │ • Validation    │    │ • Vector Storage │    │ • Confidence     │   │
│  │                 │    │                  │    │   Scoring        │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────┘   │
│                                ▼                                        │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   Knowledge     │    │   Vector         │    │   Output         │   │
│  │   Base          │    │   Database       │    │   Formatter      │   │
│  │                 │    │                  │    │                  │   │
│  │ • Historical    │◀───│ • ChromaDB       │───▶│ • Console        │   │
│  │   Failures      │    │ • Embeddings     │    │ • JSON API       │   │
│  │ • Solutions     │    │ • Metadata       │    │ • Structured     │   │
│  │ • Metadata      │    │                  │    │   Output         │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Error Processor
- **Purpose**: Extracts and cleans key information from failure logs
- **Functions**: Text preprocessing, key information extraction, error categorization
- **Technology**: Python text processing with regex and NLP techniques

#### 2. Vector Database
- **Purpose**: Stores semantic embeddings and enables fast similarity search
- **Technology**: ChromaDB with persistent storage
- **Features**: Automatic embedding generation, metadata storage, similarity queries

#### 3. Query Engine
- **Purpose**: Main interface for analyzing build failures and generating recommendations
- **Features**: Confidence scoring, automated action suggestions, comprehensive analysis
- **Intelligence**: Pattern-based recommendations and team routing

#### 4. Output Formatter
- **Purpose**: Presents results in multiple formats for different use cases
- **Formats**: Pretty console output, structured JSON for API integration
- **Features**: Configurable verbosity, color-coded output, machine-readable formats

### Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embeddings** | Sentence Transformers | Convert text to semantic vectors |
| **Vector DB** | ChromaDB | Store and search embeddings |
| **ML Model** | all-MiniLM-L6-v2 | Fast, accurate semantic encoding |
| **Data Format** | JSON | Structured failure log storage |
| **Interface** | Python CLI | Command-line and library access |
| **Environment** | Conda | Dependency management |

## Features

- **Semantic Search Engine**: Vector similarity using Sentence Transformers for semantic matching with confidence scoring and ranked results
- **Intelligent Error Analysis**: Pattern recognition with automatic categorization of 7+ error types and automated response suggestions
- **Multi-format Output**: Console display and JSON API for integration with monitoring systems
- **Enterprise Integration**: CLI interface, library mode, batch processing, and health monitoring
- **Knowledge Base Management**: Historical learning that adapts to new failure types with data validation
- **Performance Optimization**: Persistent vector storage, configurable models, and efficient batch processing
- **Team Routing**: Smart assignment of failures to appropriate contact teams with resolution time estimation

## Installation

### Prerequisites
- **Python**: 3.8 or higher
- **Conda**: Miniconda or Anaconda (recommended for environment management)
- **System Memory**: 4GB RAM minimum (8GB+ recommended for large datasets)
- **Disk Space**: 2GB for dependencies and models

### Quick Setup

```bash
# Clone/download the project
cd semantic-search

# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh

# Activate environment
source activate_env.sh

# Test installation
python3 semantic_search.py --help
```

### Setup Options

```bash
# Clean installation (removes existing environment)
./setup.sh --clean

# Install with API dependencies (Flask)
./setup.sh --with-api

# Install with development tools
./setup.sh --dev

# View setup options
./setup.sh --help
```

### Manual Installation

If you prefer manual setup:

```bash
# Create conda environment
conda create -n semantic-search python=3.10 -y
conda activate semantic-search

# Install core dependencies
pip install sentence-transformers>=2.2.0 chromadb>=0.4.0

# Install utilities
pip install numpy>=1.21.0 pandas>=1.3.0

# Verify installation
python3 -c "import sentence_transformers, chromadb; print('Ready!')"
```

### Verification

```bash
# Health check
python3 semantic_search.py -q "test query" --json

# Component test
python3 -c "
from semantic_search import health_check
print('System healthy' if health_check() else 'Issues detected')
"
```

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Interactive demo with sample queries
python3 semantic_search.py

# Analyze specific failure
python3 semantic_search.py -q "docker build failed COPY"
python3 semantic_search.py -q "artifactory upload 507 storage"
python3 semantic_search.py -q "gitlab runner out of disk space"
python3 semantic_search.py -q "makefile missing separator"

# Get JSON output for integration
python3 semantic_search.py -q "build failed" --json

# Control number of results
python3 semantic_search.py -q "test timeout" --top 5

# Use custom data file
python3 semantic_search.py -q "error" --load-data custom-failures.json
```

#### Advanced Usage

```bash
# Batch analysis with shell script
#!/bin/bash
ERRORS=("docker memory" "compilation failed" "test timeout")
for error in "${ERRORS[@]}"; do
  echo "Analyzing: $error"
  python3 semantic_search.py -q "$error" --json | jq '.summary'
done

# Pipeline integration
FAILURE_ANALYSIS=$(python3 semantic_search.py -q "$CI_ERROR" --json)
TEAMS=$(echo $FAILURE_ANALYSIS | jq -r '.recommendations[] | select(contains("Contact"))')
echo "Notify: $TEAMS"
```

#### Help and Options

```bash
# View all options
python3 semantic_search.py --help

# Available arguments:
#   -q, --query          Failure description to analyze
#   --top               Number of results (default: 3)
#   --json              JSON output format
#   --load-data         Custom data file path
```

### Programmatic Usage

#### Basic Library Usage

```python
from semantic_search import QueryEngine, OutputFormatter

# Initialize analyzer
analyzer = QueryEngine()
analyzer.load_knowledge_base("pipeline-failure-log.json")

# Analyze failure
result = analyzer.analyze_failure("docker build out of memory")

# Process results
if result['status'] == 'matches_found':
    print(f"Cause: {result['summary']['most_likely_cause']}")
    print(f"Confidence: {result['summary']['confidence']}")
    print(f"Action: {result['summary']['automated_action']}")

    # Get teams to contact
    for failure in result['similar_failures']:
        teams = failure['solution']['contact_teams']
        print(f"Contact: {', '.join(teams)}")
```

#### Advanced Integration

```python
import json
from semantic_search import QueryEngine, OutputFormatter

class FailureAnalyzer:
    def __init__(self, data_path="pipeline-failure-log.json"):
        self.engine = QueryEngine()
        self.engine.load_knowledge_base(data_path)
        self.formatter = OutputFormatter()

    def analyze_batch(self, failures):
        """Analyze multiple failures"""
        results = []
        for failure_desc in failures:
            result = self.engine.analyze_failure(failure_desc, top_results=5)
            results.append(result)
        return results

    def get_team_assignments(self, failure_desc):
        """Get recommended teams for a failure"""
        result = self.engine.analyze_failure(failure_desc)
        teams = set()

        for failure in result.get('similar_failures', []):
            teams.update(failure['solution']['contact_teams'])

        return list(teams)

    def export_results(self, result, format='json'):
        """Export analysis results"""
        if format == 'json':
            return self.formatter.format_json_output(result)
        else:
            return self.formatter.format_console_output(result)

# Usage example
analyzer = FailureAnalyzer()
teams = analyzer.get_team_assignments("kubernetes deployment failed")
print(f"Recommended teams: {teams}")
```

#### Integration with CI/CD Systems

```python
# Jenkins Pipeline Integration
def analyze_build_failure(build_log, job_name):
    from semantic_search import QueryEngine

    analyzer = QueryEngine()
    analyzer.load_knowledge_base("failures.json")

    # Extract error from build log
    error_desc = extract_error_from_log(build_log)

    # Analyze failure
    result = analyzer.analyze_failure(error_desc)

    # Create JIRA ticket with analysis
    if result['status'] == 'matches_found':
        create_jira_ticket(
            summary=f"Build failure in {job_name}",
            description=result['summary']['most_likely_cause'],
            assignee=result['similar_failures'][0]['solution']['contact_teams'][0],
            priority=get_priority(result['summary']['confidence'])
        )

    return result
```

### Demo Mode

The system includes an interactive demo showcasing various failure scenarios:

```bash
python3 semantic_search.py
```

**Demo Features:**
- **Sample Queries**: Pre-configured examples covering common failure types
- **Interactive Mode**: Enter custom queries and see real-time analysis
- **Educational**: Shows system capabilities and output formats
- **Step-through**: Guided walkthrough of different failure categories

## Configuration

### Embedding Model Configuration

The system supports multiple Sentence Transformer models with different performance characteristics:

```python
# In semantic_search.py, modify EMBEDDING_MODEL:

# Fast, good quality (default)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"         # 384 dim, ~80MB

# Higher accuracy, slower
EMBEDDING_MODEL = "all-mpnet-base-v2"        # 768 dim, ~420MB

# Multilingual support
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dim, ~420MB

# Domain-specific (if available)
EMBEDDING_MODEL = "sentence-transformers/all-distilroberta-v1"  # 768 dim, ~290MB
```

### Database Configuration

```python
# Storage location
CHROMA_PATH = "chroma_db"                    # Local storage path

# Collection settings
COLLECTION_NAME = "build_failures"          # Vector collection name

# Performance tuning
BATCH_SIZE = 100                            # Batch processing size
MAX_RESULTS = 50                            # Maximum search results
```

### Error Analysis Customization

Extend the automated analysis system:

```python
class CustomErrorProcessor(ErrorProcessor):
    @staticmethod
    def categorize_error(error_text: str) -> str:
        """Custom error categorization logic"""
        text = error_text.lower()

        # Add custom patterns
        if "custom_framework" in text and "initialization" in text:
            return "framework_init"
        elif "database" in text and ("migration" in text or "schema" in text):
            return "database_migration"
        elif "ssl" in text or "certificate" in text:
            return "security_cert"

        # Fall back to default categorization
        return super().categorize_error(error_text)

# Custom automated actions
CUSTOM_ACTIONS = {
    'framework_init': "Framework Issue: Check initialization parameters and dependencies",
    'database_migration': "🗄️ Database Issue: Verify migration scripts and database connectivity",
    'security_cert': "Security Issue: Update certificates and check SSL configuration"
}
```

### Output Customization

```python
# Modify output verbosity
class CustomOutputFormatter(OutputFormatter):
    @staticmethod
    def format_console_output(result: Dict, verbose=True) -> str:
        """Custom formatting with verbosity control"""
        if not verbose:
            # Minimal output for scripts
            return f"Cause: {result['summary']['most_likely_cause']}\n" \
                   f"Action: {result['summary']['automated_action']}"

        # Full detailed output (existing implementation)
        return OutputFormatter.format_console_output(result)
```

## Data Model

### Error Record Schema

Each pipeline failure is stored as a structured JSON object with the following schema:

```json
{
  "timestamp": "2025-09-04T10:22:37Z",           // ISO 8601 timestamp
  "error_type": "docker_failure",                // Primary error category
  "product": "backend-app",                      // Affected product/service
  "stage": "build",                              // Pipeline stage
  "error_msg": "Container exited with code 137", // Brief error description
  "details": "The container was killed due to an OOM error...", // Detailed information
  "suggestion": "Investigate memory leakage...", // Recommended solution
  "host": "build-4",                            // Server/node identifier
  "environment": ["Jenkins", "GitLab CI"],       // CI/CD platforms
  "pipeline_url": "https://...",                // Link to failed pipeline
  "contact_team": ["DevOps", "IT Helpdesk"],    // Responsible teams
  "resolution_time": "30min",                   // Expected fix duration
  "severity": "high"                            // Impact level
}
```


### Data Collection Schema

The system expects JSON files with grouped error types:

```json
{
  "docker_failure": [
    { /* error record 1 */ },
    { /* error record 2 */ }
  ],
  "compilation_failure": [
    { /* error record 3 */ },
    { /* error record 4 */ }
  ],
  "test_failure": [
    { /* error record 5 */ }
  ]
}
```

### Metadata Enrichment

The system automatically enriches records with:

```json
{
  "category": "resource",              // Automated categorization
  "confidence_score": 0.85,           // Similarity confidence
  "indexed_at": "2025-09-05T...",     // Processing timestamp
  "embedding_model": "all-MiniLM-L6-v2", // Model version used
  "processing_version": "1.0.0"       // System version
}
```

## Error Categories

### Supported Categories

The system automatically categorizes failures into the following types:

#### 1. Resource Issues (`resource`)
**Patterns**: memory, OOM, disk space, CPU limits, resource constraints
```json
{
  "automated_action": "Resource Issue: Check memory/CPU allocation and contact DevOps team",
  "typical_resolution": "15-60 minutes",
  "primary_teams": ["DevOps", "Platform Team"]
}
```

**Common Examples**:
- Docker container OOM kills (exit code 137)
- Disk space exhaustion during builds
- CPU throttling in Kubernetes
- Memory leaks in application processes

#### 2. Build Issues (`build`)
**Patterns**: compile, syntax, module not found, dependency errors
```json
{
  "automated_action": "Build Issue: Review code changes and dependency configurations",
  "typical_resolution": "10-45 minutes",
  "primary_teams": ["Development", "DevOps"]
}
```

**Common Examples**:
- Compilation errors in Java/C++/Python
- Missing dependencies in package managers
- Syntax errors in source code
- Maven/Gradle build configuration issues

#### 3. Test Issues (`test`)
**Patterns**: test failed, assertion, timeout, mock service
```json
{
  "automated_action": "Test Issue: Check test environment and data setup",
  "typical_resolution": "20-90 minutes",
  "primary_teams": ["Development", "QA"]
}
```

**Common Examples**:
- Unit test failures due to environment changes
- Integration test timeouts
- Mock service unavailability
- Test data corruption or unavailability

#### 4. Deployment Issues (`deployment`)
**Patterns**: deploy, Kubernetes, ImagePull, Helm, rollout
```json
{
  "automated_action": "Deployment Issue: Verify deployment configuration and resources",
  "typical_resolution": "25-120 minutes",
  "primary_teams": ["DevOps", "Platform Team"]
}
```

**Common Examples**:
- Kubernetes ImagePullBackOff errors
- Helm chart deployment failures
- Resource quota exceeded in namespaces
- Service mesh configuration issues

#### 5. Network Issues (`network`)
**Patterns**: network, connection, timeout, DNS, connectivity
```json
{
  "automated_action": "Network Issue: Check connectivity and firewall rules",
  "typical_resolution": "15-90 minutes",
  "primary_teams": ["DevOps", "Network Team"]
}
```

**Common Examples**:
- DNS resolution failures
- Firewall blocking connections
- Service discovery issues
- API endpoint timeouts

#### 6. Security Issues (`security`)
**Patterns**: security, vulnerability, CVE, certificate, SSL
```json
{
  "automated_action": "Security Issue: Address vulnerabilities before proceeding",
  "typical_resolution": "60-240 minutes",
  "primary_teams": ["Security Team", "Development"]
}
```

**Common Examples**:
- Vulnerable dependencies detected
- Certificate expiration
- Security scan failures
- SAST/DAST tool violations

#### 7. Storage Issues (`storage`)
**Patterns**: artifactory, storage, disk, registry, upload
```json
{
  "automated_action": "Storage Issue: Check disk space and clean up if needed",
  "typical_resolution": "20-90 minutes",
  "primary_teams": ["DevOps", "IT Infrastructure"]
}
```

**Common Examples**:
- Artifactory storage full
- Container registry upload failures
- Database storage exhaustion
- Log retention issues

#### 8. General Issues (`general`)
**Default category for unmatched patterns**
```json
{
  "automated_action": "Manual investigation required",
  "typical_resolution": "varies",
  "primary_teams": ["DevOps"]
}
```

### Category Confidence Levels

The system assigns confidence levels based on pattern matching strength:

| Confidence | Score Range | Description | Action |
|------------|-------------|-------------|---------|
| **High** | 0.8 - 1.0 | Strong pattern match | Auto-route to team |
| **Medium** | 0.6 - 0.79 | Moderate confidence | Review and route |
| **Low** | 0.0 - 0.59 | Weak match | Manual investigation |

### Custom Category Extension

Add new categories by extending the ErrorProcessor:

```python
class ExtendedErrorProcessor(ErrorProcessor):
    @staticmethod
    def categorize_error(error_text: str) -> str:
        text = error_text.lower()

        # Custom categories
        if "blockchain" in text or "smart contract" in text:
            return "blockchain"
        elif "machine learning" in text or "model training" in text:
            return "ml_training"
        elif "data pipeline" in text or "etl" in text:
            return "data_processing"

        # Fall back to base categories
        return ErrorProcessor.categorize_error(error_text)

# Custom automated actions
EXTENDED_ACTIONS = {
    'blockchain': "⛓️ Blockchain Issue: Check network connectivity and smart contract deployment",
    'ml_training': "🧠 ML Issue: Verify training data and model parameters",
    'data_processing': "Data Issue: Check ETL pipeline configuration and data sources"
}
```

## API Reference

### Core Classes

#### QueryEngine

Main interface for analyzing build failures.

```python
class QueryEngine:
    def __init__(self):
        """Initialize the query engine with vector database and models"""

    def load_knowledge_base(self, json_file: str) -> None:
        """Load historical failure data from JSON file"""

    def analyze_failure(self, query: str, top_results: int = 3) -> Dict:
        """
        Analyze a build failure and return recommendations

        Args:
            query (str): Failure description or error message
            top_results (int): Number of similar failures to return

        Returns:
            Dict: Analysis results with summary, matches, and recommendations
        """
```

#### VectorDatabase

Handles embedding generation and semantic search.

```python
class VectorDatabase:
    def __init__(self):
        """Initialize ChromaDB and embedding model"""

    def ingest_errors(self, json_file_path: str) -> None:
        """Process and store error logs in vector database"""

    def search_similar_failures(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Find similar past failures using semantic search

        Args:
            query (str): Search query text
            top_k (int): Number of results to return

        Returns:
            List[Dict]: Ranked similar failures with metadata
        """
```

#### ErrorProcessor

Processes and categorizes error messages.

```python
class ErrorProcessor:
    @staticmethod
    def extract_key_info(error_record: Dict) -> str:
        """Extract relevant text for semantic search"""

    @staticmethod
    def categorize_error(error_text: str) -> str:
        """Categorize error into predefined types"""
```

#### OutputFormatter

Formats analysis results for different output modes.

```python
class OutputFormatter:
    @staticmethod
    def format_console_output(result: Dict) -> str:
        """Format results for human-readable console display"""

    @staticmethod
    def format_json_output(result: Dict) -> str:
        """Format results as structured JSON"""
```

### CLI Arguments Reference

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--query` | `-q` | string | None | Build failure description to analyze |
| `--top` | | integer | 3 | Number of similar failures to return |
| `--json` | | flag | False | Output in JSON format for API integration |
| `--load-data` | | string | pipeline-failure-log.json | JSON file containing failure logs |
| `--help` | `-h` | flag | - | Show help message and usage examples |

### Response Schema

#### Analysis Result Structure

```json
{
  "query": "docker out of memory",
  "status": "matches_found | no_matches",
  "summary": {
    "most_likely_cause": "Docker Failure",
    "confidence": "high | medium | low",
    "automated_action": "Resource Issue: Check memory/CPU allocation...",
    "estimated_resolution": "30min"
  },
  "similar_failures": [
    {
      "similarity_score": 0.8567,
      "confidence": "high",
      "category": "resource",
      "error_info": {
        "type": "docker_failure",
        "product": "backend-app",
        "stage": "build",
        "message": "Container exited with code 137",
        "details": "The container was killed due to OOM...",
        "severity": "high"
      },
      "solution": {
        "suggestion": "Investigate memory leakage or allocate more resources",
        "contact_teams": ["DevOps", "IT Helpdesk"],
        "estimated_resolution_time": "30min"
      },
      "references": {
        "pipeline_url": "https://jenkins.company.com/job/backend/123",
        "timestamp": "2025-09-04T10:22:37Z",
        "host": "build-4"
      }
    }
  ],
  "recommendations": [
    "Investigate memory leakage or allocate more resources",
    "Contact teams: DevOps, IT Helpdesk"
  ]
}
```

### Health Check API

```python
def health_check() -> bool:
    """
    Verify system components are working correctly

    Returns:
        bool: True if all components are healthy

    Raises:
        Exception: If critical components fail
    """
```

### Library Integration Examples

#### Webhook Integration

```python
from flask import Flask, request, jsonify
from semantic_search import QueryEngine

app = Flask(__name__)
analyzer = QueryEngine()
analyzer.load_knowledge_base("failures.json")

@app.route('/analyze', methods=['POST'])
def analyze_failure():
    data = request.get_json()
    error_description = data.get('error', '')

    if not error_description:
        return jsonify({'error': 'No error description provided'}), 400

    try:
        result = analyzer.analyze_failure(error_description)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Monitoring Integration

```python
import requests
from semantic_search import QueryEngine

def handle_alert(alert_data):
    """Process monitoring alert through failure analyzer"""

    # Extract error information from alert
    error_desc = f"{alert_data['service']} {alert_data['error_message']}"

    # Analyze the failure
    analyzer = QueryEngine()
    analyzer.load_knowledge_base("monitoring-failures.json")
    result = analyzer.analyze_failure(error_desc)

    # Route to appropriate team
    if result['status'] == 'matches_found':
        teams = result['similar_failures'][0]['solution']['contact_teams']

        # Send notification
        for team in teams:
            send_team_notification(team, alert_data, result['summary'])

    return result
```

## Examples

### Common Query Patterns

| Query Type | Example Query | Expected Results | Use Case |
|------------|---------------|-----------------|----------|
| **Error Codes** | "exit code 137", "HTTP 507" | Specific error conditions | Exact error matching |
| **Technology Stack** | "docker memory", "maven build failed" | Technology-specific issues | Platform problems |
| **Symptoms** | "build timeout", "out of space" | Symptom-based matching | Resource issues |
| **Components** | "artifactory upload", "kubernetes deploy" | Component failures | Service problems |
| **Processes** | "unit test failed", "security scan error" | Process-specific issues | Pipeline stage failures |

### Detailed Usage Examples

#### Example 1: Docker Memory Issue

```bash
# Query
python3 semantic_search.py -q "docker container out of memory" --json

# Response (abbreviated)
{
  "query": "docker container out of memory",
  "summary": {
    "most_likely_cause": "Docker Failure",
    "confidence": "high",
    "automated_action": "Resource Issue: Check memory/CPU allocation and contact DevOps team",
    "estimated_resolution": "30min"
  },
  "similar_failures": [
    {
      "similarity_score": 0.8567,
      "error_info": {
        "message": "Container exited with code 137",
        "details": "OOM error. Memory usage exceeded allocated limit of 2GB"
      },
      "solution": {
        "suggestion": "Investigate memory leakage or allocate more resources",
        "contact_teams": ["DevOps", "IT Helpdesk"]
      }
    }
  ]
}
```

#### Example 2: Compilation Failure

```bash
# Query
python3 semantic_search.py -q "java compilation failed missing dependency"

# Console Output
Build Failure Analysis
==================================================
Query: "java compilation failed missing dependency"

SUMMARY
--------------------
Most Likely Cause: Compilation Failure
Confidence: MEDIUM
Automated Action: Build Issue: Review code changes and dependency configurations
Est. Resolution Time: 25min

RECOMMENDATIONS
--------------------
Add 'com.stripe.api' dependency to pom.xml
Update Maven repository settings or check if Spark 3.4.0 is available
Contact teams: Development, DevOps

SIMILAR PAST FAILURES
------------------------------

#1 - Similarity: 0.7234 (medium confidence)
├─ Product: frontend-app
├─ Stage: build
├─ Error: Compilation failed: ModuleNotFoundError
├─ Solution: Add 'com.stripe.api' dependency to pom.xml
└─ Teams: Development, DevOps
   Pipeline: https://git.internal.com/rnd-public/devops/mr-proper/-/jobs/8340535
```

#### Example 3: Batch Analysis Script

```bash
#!/bin/bash
# analyze_failures.sh - Process multiple failures from log file

LOGFILE="ci_errors.log"
OUTPUT_DIR="analysis_results"

mkdir -p $OUTPUT_DIR

# Read errors from log file and analyze each
while IFS= read -r error_line; do
    # Extract timestamp and error message
    timestamp=$(echo "$error_line" | cut -d'|' -f1)
    error_msg=$(echo "$error_line" | cut -d'|' -f2-)

    echo "Analyzing: $error_msg"

    # Run analysis
    result=$(python3 semantic_search.py -q "$error_msg" --json)

    # Extract recommended teams
    teams=$(echo "$result" | jq -r '.recommendations[] | select(contains("Contact")) | sub("Contact teams: "; "")')

    # Save detailed results
    echo "$result" > "$OUTPUT_DIR/analysis_${timestamp}.json"

    # Log summary
    echo "$timestamp|$teams|$(echo "$result" | jq -r '.summary.automated_action')" >> "$OUTPUT_DIR/summary.log"

done < "$LOGFILE"

echo "Analysis complete. Results in $OUTPUT_DIR/"
```

### Integration Examples

#### GitHub Actions Integration

```yaml
name: Analyze Build Failures
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  analyze-failure:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Analyzer
        run: |
          pip install sentence-transformers chromadb

      - name: Download Failure Logs
        uses: actions/download-artifact@v3
        with:
          name: build-logs

      - name: Analyze Failure
        id: analyze
        run: |
          ERROR_MSG=$(grep "ERROR" build.log | head -1)
          RESULT=$(python3 semantic_search.py -q "$ERROR_MSG" --json)
          echo "result<<EOF" >> $GITHUB_OUTPUT
          echo "$RESULT" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      - name: Create Issue
        uses: actions/github-script@v6
        with:
          script: |
            const result = JSON.parse(`${{ steps.analyze.outputs.result }}`);
            if (result.status === 'matches_found') {
              await github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: `Build Failure: ${result.summary.most_likely_cause}`,
                body: `
                **Automated Analysis Result**

                **Cause**: ${result.summary.most_likely_cause}
                **Confidence**: ${result.summary.confidence}
                **Estimated Resolution**: ${result.summary.estimated_resolution}

                **Recommended Action**: ${result.summary.automated_action}

                **Teams to Contact**: ${result.recommendations.filter(r => r.includes('Contact')).join(', ')}

                **Similar Past Failures**: ${result.similar_failures.length} found
                `,
                labels: ['bug', 'automated-analysis', result.summary.confidence + '-confidence']
              });
            }
```

#### Jenkins Pipeline Integration

```groovy
pipeline {
    agent any

    post {
        failure {
            script {
                // Capture build failure details
                def buildLog = currentBuild.rawBuild.getLog(100).join('\n')
                def errorPattern = /ERROR.*|FAILED.*|Exception.*/
                def errorLines = buildLog.findAll(errorPattern)

                if (errorLines) {
                    def errorDescription = errorLines[0]

                    // Run semantic analysis
                    def analysisResult = sh(
                        script: """
                        python3 semantic_search.py -q "${errorDescription}" --json
                        """,
                        returnStdout: true
                    ).trim()

                    def result = readJSON text: analysisResult

                    if (result.status == 'matches_found') {
                        // Create JIRA ticket
                        def teams = result.recommendations
                            .findAll { it.contains('Contact') }
                            .collect { it.replaceAll('Contact teams: ', '') }
                            .join(', ')

                        def ticketData = [
                            project: 'DEVOPS',
                            summary: "Build Failure: ${result.summary.most_likely_cause}",
                            description: """
                            Automated analysis of build failure in job ${env.JOB_NAME} #${env.BUILD_NUMBER}

                            Cause: ${result.summary.most_likely_cause}
                            Confidence: ${result.summary.confidence}
                            Action Required: ${result.summary.automated_action}

                            Teams to notify: ${teams}

                            Build URL: ${env.BUILD_URL}
                            """,
                            assignee: teams.split(',')[0].trim()
                        ]

                        // Send to JIRA (pseudo-code)
                        createJiraTicket(ticketData)

                        // Send Slack notification
                        slackSend(
                            channel: '#devops-alerts',
                            message: """
                            🚨 Build Failure Analysis
                            Job: ${env.JOB_NAME} #${env.BUILD_NUMBER}
                            Cause: ${result.summary.most_likely_cause} (${result.summary.confidence} confidence)
                            Teams: ${teams}
                            JIRA: Created ticket for investigation
                            """
                        )
                    }
                }
            }
        }
    }
}
```

#### Slack Bot Integration

```python
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from semantic_search import QueryEngine

class FailureAnalyzerBot:
    def __init__(self, slack_token):
        self.client = WebClient(token=slack_token)
        self.analyzer = QueryEngine()
        self.analyzer.load_knowledge_base("failures.json")

    def handle_failure_report(self, channel, text):
        """Handle failure reports from Slack users"""
        try:
            # Analyze the failure
            result = self.analyzer.analyze_failure(text)

            if result['status'] == 'matches_found':
                # Format response
                response = f"""
**Build Failure Analysis**

**Query**: {text}
**Most Likely Cause**: {result['summary']['most_likely_cause']}
**Confidence**: {result['summary']['confidence'].upper()}
**Estimated Resolution**: {result['summary']['estimated_resolution']}

**Recommended Action**: {result['summary']['automated_action']}

**Teams to Contact**: {', '.join(set(
    team for failure in result['similar_failures']
    for team in failure['solution']['contact_teams']
))}

**Top Similar Failure**:
• Product: {result['similar_failures'][0]['error_info']['product']}
• Solution: {result['similar_failures'][0]['solution']['suggestion']}
• Pipeline: {result['similar_failures'][0]['references'].get('pipeline_url', 'N/A')}
"""
            else:
                response = f"No similar failures found for: {text}\nConsider adding this to the knowledge base."

            # Send response
            self.client.chat_postMessage(
                channel=channel,
                text=response,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": response}
                    }
                ]
            )

        except SlackApiError as e:
            print(f"Error sending message: {e}")

# Usage
bot = FailureAnalyzerBot("xoxb-your-slack-token")
bot.handle_failure_report("#devops", "docker build failed with exit code 137")
```

### Performance Examples

#### Batch Processing Performance

```python
import time
from semantic_search import QueryEngine

def benchmark_batch_analysis():
    """Benchmark batch processing performance"""

    analyzer = QueryEngine()
    analyzer.load_knowledge_base("large-failures.json")

    # Test queries of varying complexity
    test_queries = [
        "docker memory",
        "compilation failed with multiple dependency errors in maven build",
        "kubernetes deployment timeout with ImagePullBackOff status",
        "network connection refused",
        "security scan found high severity vulnerabilities"
    ]

    # Warm up
    analyzer.analyze_failure("test query")

    # Benchmark single queries
    single_times = []
    for query in test_queries:
        start = time.time()
        result = analyzer.analyze_failure(query, top_results=5)
        end = time.time()
        single_times.append(end - start)
        print(f"Query '{query[:30]}...': {end-start:.3f}s")

    print(f"\nAverage single query time: {sum(single_times)/len(single_times):.3f}s")

    # Benchmark batch processing
    start = time.time()
    for query in test_queries * 10:  # 50 queries total
        analyzer.analyze_failure(query, top_results=3)
    end = time.time()

    print(f"Batch processing (50 queries): {end-start:.3f}s")
    print(f"Average per query in batch: {(end-start)/50:.3f}s")

if __name__ == "__main__":
    benchmark_batch_analysis()
```

#### Memory Usage Monitoring

```python
import psutil
import os
from semantic_search import QueryEngine

def monitor_memory_usage():
    """Monitor memory usage during analysis"""

    process = psutil.Process(os.getpid())

    print("Initial memory:", process.memory_info().rss / 1024 / 1024, "MB")

    # Initialize analyzer
    analyzer = QueryEngine()
    print("After init:", process.memory_info().rss / 1024 / 1024, "MB")

    # Load knowledge base
    analyzer.load_knowledge_base("pipeline-failure-log.json")
    print("After loading:", process.memory_info().rss / 1024 / 1024, "MB")

    # Run multiple analyses
    for i in range(100):
        result = analyzer.analyze_failure(f"test query {i}")
        if i % 10 == 0:
            print(f"After {i} queries:", process.memory_info().rss / 1024 / 1024, "MB")

if __name__ == "__main__":
    monitor_memory_usage()
```

## Performance

### Benchmarks

| Metric | Typical Performance | Enterprise Scale |
|--------|-------------------|------------------|
| **Search Latency** | 100-500ms | 200-800ms |
| **Memory Usage** | 2-4GB | 4-8GB |
| **Throughput** | 50-100 queries/min | 200-500 queries/min |
| **Startup Time** | 5-15 seconds | 30-60 seconds |
| **Index Size** | 100MB per 1000 failures | 1GB per 10000 failures |

### Performance Characteristics

#### Search Performance
- **Cold Start**: 5-15 seconds for model loading and database initialization
- **Warm Queries**: 100-300ms average response time
- **Complex Queries**: 300-800ms for detailed multi-failure analysis
- **Batch Processing**: 20-50ms per query when processing multiple failures

#### Memory Usage
- **Base System**: ~1.5GB (Python runtime + libraries)
- **Embedding Model**: ~500MB (all-MiniLM-L6-v2) to ~2GB (larger models)
- **Vector Database**: ~100MB per 1000 indexed failures
- **Peak Usage**: 4-8GB for large-scale deployments

#### Scalability Factors
- **Dataset Size**: Linear scaling with number of indexed failures
- **Query Complexity**: Longer queries take more time for embedding generation
- **Result Count**: Minimal impact from changing top_k results
- **Concurrent Users**: Shared model reduces per-user memory overhead

### Optimization Strategies

#### Model Selection Trade-offs

```python
# Performance vs Accuracy trade-offs
MODELS = {
    # Fast, lower accuracy
    "all-MiniLM-L6-v2": {
        "size": "80MB",
        "dimensions": 384,
        "speed": "fast",
        "accuracy": "good"
    },

    # Balanced option
    "all-distilroberta-v1": {
        "size": "290MB",
        "dimensions": 768,
        "speed": "medium",
        "accuracy": "very good"
    },

    # High accuracy, slower
    "all-mpnet-base-v2": {
        "size": "420MB",
        "dimensions": 768,
        "speed": "slow",
        "accuracy": "excellent"
    }
}
```

#### Database Optimization

```python
# Optimize ChromaDB performance
import chromadb
from chromadb.config import Settings

# Performance-tuned settings
settings = Settings(
    chroma_db_impl="duckdb+parquet",  # Faster than SQLite
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False,       # Disable telemetry
    chroma_server_http_port=None      # Embedded mode
)

# Batch operations for better performance
def optimize_ingestion(errors, batch_size=100):
    """Ingest errors in batches for better performance"""
    for i in range(0, len(errors), batch_size):
        batch = errors[i:i+batch_size]
        # Process batch...
```

#### Memory Management

```python
import gc
from contextlib import contextmanager

@contextmanager
def memory_efficient_analysis():
    """Context manager for memory-efficient analysis"""
    try:
        # Your analysis code here
        yield
    finally:
        # Force garbage collection
        gc.collect()

# Usage
with memory_efficient_analysis():
    result = analyzer.analyze_failure(query)
```

#### Caching Strategies

```python
from functools import lru_cache
import hashlib

class CachedQueryEngine(QueryEngine):
    def __init__(self, cache_size=1000):
        super().__init__()
        self._cache_size = cache_size

    @lru_cache(maxsize=1000)
    def _cached_embedding(self, text):
        """Cache embeddings for repeated queries"""
        return self.vector_db.model.encode([text])[0].tolist()

    def analyze_failure(self, query: str, top_results: int = 3) -> Dict:
        """Use cached embeddings when possible"""
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # Check if we've seen this query before
        cached_result = self._get_cached_result(query_hash)
        if cached_result:
            return cached_result

        # Perform analysis and cache result
        result = super().analyze_failure(query, top_results)
        self._cache_result(query_hash, result)
        return result
```

### Performance Monitoring

```python
import time
import logging
from contextlib import contextmanager

# Performance logging
logging.basicConfig(level=logging.INFO)
performance_logger = logging.getLogger('performance')

@contextmanager
def performance_timer(operation_name):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        performance_logger.info(f"{operation_name}: {duration:.3f}s")

# Usage in analyzer
class MonitoredQueryEngine(QueryEngine):
    def analyze_failure(self, query: str, top_results: int = 3) -> Dict:
        with performance_timer(f"analyze_failure(top_k={top_results})"):
            with performance_timer("embedding_generation"):
                # Generate query embedding
                query_embedding = self.vector_db.model.encode([query])[0]

            with performance_timer("vector_search"):
                # Search similar failures
                results = self.vector_db.search_similar_failures(query, top_results)

            with performance_timer("result_processing"):
                # Process and format results
                return self._format_results(results)
```

### Production Deployment Recommendations

#### Hardware Requirements

```yaml
# Minimum Requirements
cpu: 2 cores
memory: 4GB RAM
storage: 10GB SSD

# Recommended Production
cpu: 4-8 cores
memory: 8-16GB RAM
storage: 50GB SSD (NVMe preferred)
network: 1Gbps

# Enterprise Scale
cpu: 8-16 cores
memory: 16-32GB RAM
storage: 100GB+ NVMe SSD
network: 10Gbps
```

#### Container Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set resource limits
ENV MALLOC_ARENA_MAX=2
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python3 -c "from semantic_search import health_check; exit(0 if health_check() else 1)"

# Run application
CMD ["python3", "semantic_search.py"]
```

## Development

### Development Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd semantic-search

# Setup development environment with all tools
./setup.sh --dev

# Activate environment
source activate_env.sh

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Structure

```
semantic-search/
├── semantic_search.py          # Main application
├── setup.sh                    # Installation script
├── activate_env.sh             # Environment activation (auto-generated)
├── requirements.txt            # Python dependencies
├── pipeline-failure-log.json  # Sample failure data
├── chroma_db/                  # Vector database (auto-created)
├── README.md                   # Complete documentation
└── tests/                      # Test suite (optional)
    ├── test_error_processor.py
    ├── test_vector_database.py
    ├── test_query_engine.py
    └── test_integration.py
```

### Testing

#### Unit Tests

```python
import pytest
from semantic_search import ErrorProcessor, VectorDatabase, QueryEngine

class TestErrorProcessor:
    def test_extract_key_info(self):
        error_record = {
            'error_msg': 'Docker build failed',
            'details': 'Out of memory error',
            'error_type': 'docker_failure'
        }
        result = ErrorProcessor.extract_key_info(error_record)
        assert 'Docker build failed' in result
        assert 'Out of memory error' in result

    def test_categorize_error(self):
        # Test resource categorization
        text = "container killed due to out of memory"
        category = ErrorProcessor.categorize_error(text)
        assert category == 'resource'

        # Test build categorization
        text = "compilation failed syntax error"
        category = ErrorProcessor.categorize_error(text)
        assert category == 'build'

class TestVectorDatabase:
    @pytest.fixture
    def vector_db(self):
        return VectorDatabase()

    def test_ingest_errors(self, vector_db, tmp_path):
        # Create test data file
        test_data = {
            "test_failure": [{
                "error_msg": "Test failed",
                "details": "Assertion error",
                "suggestion": "Fix the test"
            }]
        }

        test_file = tmp_path / "test_data.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)

        # Test ingestion
        vector_db.ingest_errors(str(test_file))

        # Verify data was ingested
        results = vector_db.search_similar_failures("test error")
        assert len(results) > 0

class TestQueryEngine:
    @pytest.fixture
    def query_engine(self, tmp_path):
        # Setup test data
        test_data = {
            "docker_failure": [{
                "error_msg": "Container exited with code 137",
                "details": "Out of memory error",
                "suggestion": "Increase memory allocation",
                "contact_team": ["DevOps"]
            }]
        }

        test_file = tmp_path / "test_failures.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)

        engine = QueryEngine()
        engine.load_knowledge_base(str(test_file))
        return engine

    def test_analyze_failure_success(self, query_engine):
        result = query_engine.analyze_failure("docker memory error")

        assert result['status'] == 'matches_found'
        assert 'summary' in result
        assert 'similar_failures' in result
        assert len(result['similar_failures']) > 0

    def test_analyze_failure_no_matches(self, query_engine):
        result = query_engine.analyze_failure("completely unknown error type xyz")

        # Should still return a result structure
        assert 'status' in result
        assert 'query' in result
```

#### Integration Tests

```python
import subprocess
import json
import tempfile
import os

class TestCLIIntegration:
    def test_cli_json_output(self):
        """Test CLI JSON output format"""
        result = subprocess.run([
            'python3', 'semantic_search.py',
            '-q', 'docker memory error',
            '--json'
        ], capture_output=True, text=True)

        assert result.returncode == 0

        # Parse JSON output
        output_data = json.loads(result.stdout)
        assert 'query' in output_data
        assert 'status' in output_data

    def test_cli_console_output(self):
        """Test CLI console output format"""
        result = subprocess.run([
            'python3', 'semantic_search.py',
            '-q', 'compilation failed'
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert 'Build Failure Analysis' in result.stdout
        assert 'SUMMARY' in result.stdout

    def test_custom_data_file(self):
        """Test loading custom data file"""
        # Create temporary data file
        test_data = {
            "custom_failure": [{
                "error_msg": "Custom error",
                "details": "Custom details",
                "suggestion": "Custom solution"
            }]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            result = subprocess.run([
                'python3', 'semantic_search.py',
                '-q', 'custom error',
                '--load-data', temp_file,
                '--json'
            ], capture_output=True, text=True)

            assert result.returncode == 0
            output_data = json.loads(result.stdout)
            assert output_data['status'] == 'matches_found'

        finally:
            os.unlink(temp_file)

# Performance tests
class TestPerformance:
    def test_query_performance(self):
        """Test that queries complete within reasonable time"""
        import time
        from semantic_search import QueryEngine

        engine = QueryEngine()
        engine.load_knowledge_base("pipeline-failure-log.json")

        start_time = time.time()
        result = engine.analyze_failure("docker build failed")
        end_time = time.time()

        # Should complete within 2 seconds
        assert end_time - start_time < 2.0
        assert result['status'] in ['matches_found', 'no_matches']

    def test_memory_usage(self):
        """Test memory usage stays within bounds"""
        import psutil
        import os
        from semantic_search import QueryEngine

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        engine = QueryEngine()
        engine.load_knowledge_base("pipeline-failure-log.json")

        # Run multiple queries
        for i in range(10):
            engine.analyze_failure(f"test query {i}")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory growth should be reasonable (less than 1GB increase)
        assert final_memory - initial_memory < 1024
```

#### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=semantic_search tests/

# Run specific test
pytest tests/test_query_engine.py::TestQueryEngine::test_analyze_failure_success

# Run performance tests
pytest tests/test_performance.py -v
```

### Code Quality Tools

```bash
# Code formatting
black semantic_search.py tests/

# Type checking (if using type hints)
mypy semantic_search.py

# Linting
flake8 semantic_search.py

# Security scanning
bandit semantic_search.py
```

### Contributing Workflow

```bash
# Create feature branch
git checkout -b feature/new-error-category

# Make changes and test
# ... code changes ...
pytest tests/

# Format code
black semantic_search.py

# Commit changes
git add .
git commit -m "Add support for new error category: database_migration"

# Push and create PR
git push origin feature/new-error-category
```

## Contributing

### How to Contribute

We welcome contributions to improve the DevOps Build Failure Analyzer! Here are ways you can help:

#### 1. Error Type Extensions
Add support for new failure categories:

```python
# In ErrorProcessor.categorize_error()
elif "database" in text and ("migration" in text or "schema" in text):
    return "database_migration"
elif "cache" in text and ("redis" in text or "memcached" in text):
    return "cache_failure"
elif "queue" in text and ("rabbitmq" in text or "kafka" in text):
    return "message_queue"
```

#### 2. Data Collection
Contribute real-world failure data:

```json
{
  "database_migration": [{
    "timestamp": "2025-09-05T...",
    "error_type": "database_migration",
    "product": "user-service",
    "stage": "deploy",
    "error_msg": "Database migration failed: duplicate column",
    "details": "ALTER TABLE users ADD COLUMN email VARCHAR(255); ERROR: column 'email' already exists",
    "suggestion": "Check existing database schema before running migrations",
    "contact_team": ["DBA", "Development"],
    "resolution_time": "45min",
    "severity": "high"
  }]
}
```

#### 3. Model Improvements
Experiment with different embedding models:

```python
# Test new models and report performance
EXPERIMENTAL_MODELS = [
    "sentence-transformers/paraphrase-distilroberta-base-v2",
    "sentence-transformers/all-roberta-large-v1",
    "sentence-transformers/gtr-t5-base",
    "custom-domain-specific-model"
]
```

#### 4. Integration Connectors
Create connectors for popular tools:

```python
# Example: Splunk connector
class SplunkConnector:
    def fetch_failure_logs(self, query, time_range):
        """Fetch logs from Splunk for analysis"""
        pass

    def send_analysis_results(self, results):
        """Send analysis back to Splunk"""
        pass

# Example: PagerDuty connector
class PagerDutyConnector:
    def create_incident(self, analysis_result):
        """Create PD incident with analysis"""
        pass
```

#### 5. Performance Optimizations
Contribute performance improvements:

```python
# Parallel processing for batch analysis
from concurrent.futures import ThreadPoolExecutor

def analyze_failures_parallel(queries, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_failure, query) for query in queries]
        return [future.result() for future in futures]
```

### Contribution Guidelines

#### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add type hints where appropriate
- Include docstrings for public functions

```python
def analyze_failure(self, query: str, top_results: int = 3) -> Dict[str, Any]:
    """
    Analyze a build failure and return recommendations.

    Args:
        query: Description of the failure to analyze
        top_results: Number of similar failures to return

    Returns:
        Dictionary containing analysis results with summary and recommendations

    Raises:
        ValueError: If query is empty or invalid
        RuntimeError: If analysis engine is not initialized
    """
```

#### Testing Requirements
- Add unit tests for new features
- Ensure 80%+ test coverage
- Include integration tests for CLI changes
- Add performance tests for optimization contributions

#### Documentation
- Update README.md for new features
- Add inline code comments for complex logic
- Include usage examples for new functionality
- Update API reference for interface changes

#### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch with descriptive name
3. **Implement** changes with tests
4. **Run** full test suite: `pytest tests/`
5. **Format** code: `black semantic_search.py`
6. **Submit** pull request with clear description

#### Review Criteria
- **Functionality**: Does it solve a real problem?
- **Quality**: Is the code well-written and tested?
- **Performance**: Does it maintain or improve system performance?
- **Documentation**: Are changes properly documented?
- **Backwards Compatibility**: Does it break existing functionality?

### Feature Requests

Priority areas for contributions:

#### High Priority
- **Real-time Integration**: WebSocket/HTTP API for live failure analysis
- **Advanced ML Models**: Fine-tuned models for specific error domains
- **Workflow Integration**: Native Jenkins/GitHub Actions/GitLab CI plugins
- **Trend Analysis**: Historical failure pattern analysis and prediction

#### Medium Priority
- **Web Interface**: Browser-based UI for non-CLI users
- **Multi-language Support**: Support for logs in different languages
- **Alert Integration**: PagerDuty, OpsGenie, VictorOps connectors
- **Metrics and Monitoring**: Prometheus/Grafana integration

#### Nice to Have
- **Mobile App**: Smartphone interface for on-call engineers
- **Machine Learning**: Auto-improvement based on resolution feedback
- **Custom Models**: Industry-specific or company-specific trained models
- **Advanced Visualization**: Failure pattern graphs and network diagrams

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Wiki**: Community-maintained documentation and examples
- **Slack/Discord**: Real-time community chat (if established)

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: `ModuleNotFoundError` when importing dependencies
```bash
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution**: Run the setup script or install dependencies manually
```bash
# Option 1: Use setup script
./setup.sh --clean

# Option 2: Manual installation
conda activate semantic-search
pip install sentence-transformers>=2.2.0 chromadb>=0.4.0
```

---

**Issue**: Conda environment activation fails
```bash
CondaError: Run 'conda init' before 'conda activate'
```

**Solution**: Initialize conda shell integration
```bash
# Initialize conda
conda init bash  # or zsh/fish depending on your shell

# Restart terminal or source the config
source ~/.bashrc  # or ~/.zshrc

# Try activation again
conda activate semantic-search
```

---

**Issue**: ChromaDB database creation fails
```bash
Failed to initialize ChromaDB: [Errno 13] Permission denied
```

**Solution**: Check directory permissions and disk space
```bash
# Check current directory permissions
ls -la

# Check disk space
df -h .

# Create database directory manually if needed
mkdir -p chroma_db
chmod 755 chroma_db
```

#### Runtime Issues

**Issue**: "No results found" for valid queries
```bash
No similar failures found in knowledge base
```

**Solutions**:
```bash
# 1. Verify data file exists and is valid
ls -la pipeline-failure-log.json
python3 -c "import json; json.load(open('pipeline-failure-log.json'))"

# 2. Check if database was populated
ls -la chroma_db/
python3 -c "
from semantic_search import VectorDatabase
db = VectorDatabase()
db.ingest_errors('pipeline-failure-log.json')
"

# 3. Try broader query terms
python3 semantic_search.py -q "error" --top 10
```

---

**Issue**: High memory usage or Out of Memory errors
```bash
MemoryError: Unable to allocate array
```

**Solutions**:
```bash
# 1. Use smaller embedding model
# Edit semantic_search.py:
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Instead of larger models

# 2. Reduce batch size for large datasets
# Edit the ingestion code to process smaller batches

# 3. Monitor memory usage
python3 -c "
import psutil, os
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

---

**Issue**: Slow query performance
```bash
# Queries taking >5 seconds
```

**Solutions**:
```bash
# 1. Check system resources
htop  # or Activity Monitor on macOS

# 2. Use faster model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fastest option

# 3. Reduce result count
python3 semantic_search.py -q "your query" --top 3  # Instead of --top 10

# 4. Check database size
du -sh chroma_db/
```

#### Data Issues

**Issue**: JSON parsing errors
```bash
JSONDecodeError: Expecting ',' delimiter: line 45 column 3
```

**Solution**: Validate and fix JSON format
```bash
# Validate JSON
python3 -c "
import json
try:
    with open('pipeline-failure-log.json') as f:
        data = json.load(f)
    print('JSON is valid')
except json.JSONDecodeError as e:
    print(f'JSON error: {e}')
    print(f'   Line {e.lineno}, Column {e.colno}')
"

# Use online JSON validator or:
jq . pipeline-failure-log.json  # If jq is installed
```

---

**Issue**: Missing required fields in data
```bash
KeyError: 'error_msg'
```

**Solution**: Ensure all records have required fields
```python
# Validation script
import json

required_fields = ['error_msg', 'details', 'error_type']

with open('pipeline-failure-log.json') as f:
    data = json.load(f)

for error_type, errors in data.items():
    for i, error in enumerate(errors):
        missing = [field for field in required_fields if field not in error]
        if missing:
            print(f"{error_type}[{i}] missing fields: {missing}")
```

#### Performance Issues

**Issue**: System hangs during initialization
```bash
# Application starts but never responds
```

**Diagnostic steps**:
```bash
# 1. Check system resources
top
free -h  # Linux
vm_stat  # macOS

# 2. Test minimal functionality
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model loaded')
"

# 3. Test database connection
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='test_db')
print('ChromaDB working')
"
```

#### Network Issues

**Issue**: Model download fails
```bash
HTTPSConnectionPool: Max retries exceeded
```

**Solutions**:
```bash
# 1. Check internet connectivity
ping huggingface.co

# 2. Set HTTP proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# 3. Pre-download models manually
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models/')
"
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
# Add to semantic_search.py for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or run with debug environment
PYTHONPATH=. LOGGING_LEVEL=DEBUG python3 semantic_search.py -q "test"
```

### Health Check Script

```python
#!/usr/bin/env python3
"""
Comprehensive health check script for the analyzer system
"""

import sys
import json
import os
import traceback
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")

    try:
        import sentence_transformers
        print("sentence-transformers available")
    except ImportError:
        print("sentence-transformers not found")
        return False

    try:
        import chromadb
        print("chromadb available")
    except ImportError:
        print("chromadb not found")
        return False

    return True

def check_data_file():
    """Check if data file exists and is valid"""
    print("\nChecking data file...")

    data_file = "pipeline-failure-log.json"

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return False

    try:
        with open(data_file) as f:
            data = json.load(f)

        error_count = sum(len(errors) for errors in data.values())
        print(f"Data file valid: {len(data)} error types, {error_count} total errors")
        return True

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return False

def check_database():
    """Check if database can be created and used"""
    print("\nChecking database...")

    try:
        import chromadb
        from chromadb.config import Settings

        # Test database creation
        client = chromadb.PersistentClient(
            path="health_check_db",
            settings=Settings(allow_reset=True)
        )

        # Clean up test database
        if "test" in [c.name for c in client.list_collections()]:
            client.delete_collection("test")

        collection = client.create_collection("test")

        # Test basic operations
        collection.add(
            ids=["test1"],
            documents=["test document"],
            embeddings=[[0.1, 0.2, 0.3]]
        )

        results = collection.query(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=1
        )

        print("Database operations working")

        # Cleanup
        client.delete_collection("test")
        import shutil
        shutil.rmtree("health_check_db", ignore_errors=True)

        return True

    except Exception as e:
        print(f"Database error: {e}")
        return False

def check_model():
    """Check if embedding model can be loaded"""
    print("\nChecking embedding model...")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Test encoding
        test_text = "test error message"
        embedding = model.encode([test_text])

        print(f"Model loaded: {embedding[0].shape} dimensions")
        return True

    except Exception as e:
        print(f"Model error: {e}")
        return False

def check_full_system():
    """Test the complete system end-to-end"""
    print("\nChecking full system...")

    try:
        from semantic_search import QueryEngine

        # Initialize analyzer
        analyzer = QueryEngine()
        analyzer.load_knowledge_base("pipeline-failure-log.json")

        # Test analysis
        result = analyzer.analyze_failure("test docker error")

        if result and 'status' in result:
            print("Full system working")
            print(f"   Status: {result['status']}")
            return True
        else:
            print("System returned invalid result")
            return False

    except Exception as e:
        print(f"System error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all health checks"""
    print("DevOps Build Failure Analyzer - Health Check")
    print("=" * 50)

    checks = [
        check_dependencies,
        check_data_file,
        check_database,
        check_model,
        check_full_system
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"Check failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Health Check Summary")
    print("=" * 50)

    if all(results):
        print("🎉 All checks passed! System is healthy.")
        return 0
    else:
        failed_count = sum(1 for r in results if not r)
        print(f"{failed_count}/{len(results)} checks failed.")
        print("\nRecommendations:")
        print("1. Run ./setup.sh --clean to reinstall")
        print("2. Check system resources (memory, disk)")
        print("3. Verify network connectivity for model downloads")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Getting Help

If you're still experiencing issues:

1. **Check system requirements**: Ensure you have Python 3.8+, adequate memory (4GB+), and disk space (2GB+)

2. **Review logs**: Enable debug logging and check for specific error messages

3. **Minimal test**: Try with a simple query and small dataset first

4. **Community support**:
   - GitHub Issues: Report bugs with full error traces
   - Documentation: Check README for configuration options
   - Examples: Review working examples in the documentation

5. **System information**: When reporting issues, include:
   ```bash
   # System info
   python3 --version
   conda --version
   cat /etc/os-release  # Linux
   system_profiler SPSoftwareDataType  # macOS

   # Memory and disk
   free -h  # Linux
   df -h

   # Python environment
   pip list | grep -E "(sentence|chroma)"
   ```

---

**Ready to automate your support activities? Run `python3 semantic_search.py` and start searching!**