#!/bin/bash
# DevOps Build Failure Analyzer - Setup Script
# ============================================

set -e  # Exit on any error

echo "Setting up DevOps Build Failure Analyzer..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_status "Please install Miniconda or Anaconda first:"
        print_status "https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda found: $(conda --version)"
}

# Create or activate conda environment
setup_environment() {
    local env_name="semantic-search"

    print_status "Checking for conda environment: $env_name"

    if conda env list | grep -q "^$env_name "; then
        print_warning "Environment '$env_name' already exists"
        print_status "Activating existing environment..."
        eval "$(conda shell.bash hook)"
        conda activate $env_name
    else
        print_status "Creating new conda environment: $env_name"
        conda create -n $env_name python=3.10 -y
        eval "$(conda shell.bash hook)"
        conda activate $env_name
        print_success "Environment '$env_name' created and activated"
    fi
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."

    # Core dependencies
    print_status "Installing core ML dependencies..."
    pip install sentence-transformers>=2.2.0 chromadb>=0.4.0

    # Utility dependencies
    print_status "Installing utility dependencies..."
    pip install numpy>=1.21.0 pandas>=1.3.0

    # Optional API dependencies
    if [[ "$1" == "--with-api" ]]; then
        print_status "Installing API dependencies..."
        pip install flask>=2.3.0 flask-cors>=4.0.0
    fi

    # Development dependencies
    if [[ "$1" == "--dev" ]]; then
        print_status "Installing development dependencies..."
        pip install pytest>=7.0.0 black>=22.0.0
    fi

    print_success "All dependencies installed successfully"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."

    # Test Python imports
    python3 -c "
import sentence_transformers
import chromadb
import numpy
print('Core dependencies working')
" 2>/dev/null || {
        print_error "Dependency verification failed"
        exit 1
    }

    # Test the main script
    if [[ -f "semantic_search.py" ]]; then
        print_status "Testing main application..."
        python3 -c "
import sys
sys.path.insert(0, '.')
from semantic_search import health_check
if health_check():
    print('Application components working')
else:
    sys.exit(1)
" 2>/dev/null || {
            print_warning "Application test failed - may need data file"
        }
    fi

    print_success "Installation verification completed"
}

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."

    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Quick activation script for semantic-search environment

echo "Activating DevOps Build Failure Analyzer environment..."
eval "$(conda shell.bash hook)"
conda activate semantic-search

echo "Environment activated!"
echo "Usage examples:"
echo "   python3 semantic_search.py                     # Demo mode"
echo "   python3 semantic_search.py -q \"build failed\"   # Query mode"
echo "   python3 semantic_search.py --help              # Full help"
EOF

    chmod +x activate_env.sh
    print_success "Created activation script: activate_env.sh"
}

# Main setup process
main() {
    echo "========================================"
    echo "DevOps Build Failure Analyzer Setup"
    echo "========================================"
    echo

    check_conda
    setup_environment
    install_dependencies "$1"
    verify_installation
    create_activation_script

    echo
    print_success "🎉 Setup completed successfully!"
    echo
    print_status "Next steps:"
    print_status "1. Run: source activate_env.sh"
    print_status "2. Test: python3 semantic_search.py"
    print_status "3. Query: python3 semantic_search.py -q \"docker memory error\""
    echo
    print_status "For help: python3 semantic_search.py --help"
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "DevOps Build Failure Analyzer Setup Script"
        echo
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --help, -h      Show this help message"
        echo "  --with-api      Install optional API dependencies (Flask)"
        echo "  --dev           Install development dependencies (pytest, black)"
        echo "  --clean         Clean existing environment before setup"
        echo
        exit 0
        ;;
    --clean)
        print_status "Cleaning existing environment..."
        conda env remove -n semantic-search -y 2>/dev/null || true
        main
        ;;
    *)
        main "$1"
        ;;
esac