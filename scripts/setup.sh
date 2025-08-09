#!/bin/bash

# MLOps Pipeline Setup Script

set -e

echo "🚀 Setting up MLOps Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [ "$(echo "$PYTHON_VERSION >= 3.9" | bc)" -eq 1 ]; then
            print_status "Python $PYTHON_VERSION found ✓"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created ✓"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_status "Virtual environment activated ✓"
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Dependencies installed ✓"
}

# Install pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    print_status "Pre-commit hooks installed ✓"
}

# Initialize DVC
setup_dvc() {
    print_status "Initializing DVC..."
    if [ ! -d ".dvc" ]; then
        dvc init
        print_status "DVC initialized ✓"
    else
        print_warning "DVC already initialized"
    fi
}

# Create necessary directories
create_dirs() {
    print_status "Creating necessary directories..."
    mkdir -p data logs checkpoints models metrics plots
    
    # Create .gitkeep files to track empty directories
    touch data/.gitkeep logs/.gitkeep checkpoints/.gitkeep
    touch models/.gitkeep metrics/.gitkeep plots/.gitkeep
    
    print_status "Directories created ✓"
}

# Setup Weights & Biases
setup_wandb() {
    print_status "Setting up Weights & Biases..."
    if [ -z "$WANDB_API_KEY" ]; then
        print_warning "WANDB_API_KEY not set. Run 'wandb login' manually later."
    else
        echo "$WANDB_API_KEY" | wandb login --relogin
        print_status "Weights & Biases configured ✓"
    fi
}

# Install package in development mode
install_package() {
    print_status "Installing package in development mode..."
    pip install -e .
    print_status "Package installed ✓"
}

# Run tests to verify setup
run_tests() {
    print_status "Running tests to verify setup..."
    pytest tests/ -v --tb=short
    if [ $? -eq 0 ]; then
        print_status "All tests passed ✓"
    else
        print_warning "Some tests failed. Check the output above."
    fi
}

# Main setup function
main() {
    echo "================================================"
    echo "       MLOps Pipeline Setup"
    echo "================================================"
    echo
    
    check_python
    create_venv
    activate_venv
    install_deps
    install_package
    setup_precommit
    setup_dvc
    create_dirs
    setup_wandb
    run_tests
    
    echo
    echo "================================================"
    print_status "Setup completed successfully! 🎉"
    echo "================================================"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Configure your AWS credentials: aws configure"
    echo "3. Set up your S3 bucket for DVC: dvc remote add -d myremote s3://your-bucket/dvc-store"
    echo "4. Start training: python src/training/train.py"
    echo "5. Check the documentation: README.md"
    echo
}

# Run main function
main "$@"
