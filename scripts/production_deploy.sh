#!/bin/bash

# EXPLAINIUM Production Deployment Script
# This script prepares and deploys EXPLAINIUM for production use

set -e  # Exit on any error

echo "🚀 EXPLAINIUM Production Deployment"
echo "==================================="

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success "Created .env from template. Please configure it before proceeding."
        exit 1
    else
        print_error ".env.example not found. Cannot proceed."
        exit 1
    fi
fi

print_status "Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d" " -f2)
required_version="3.8"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi
print_success "Python version: $python_version"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed"
    exit 1
fi
print_success "Docker is available"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is required but not installed"
    exit 1
fi
print_success "Docker Compose is available"

print_status "Setting up production environment..."

# Create necessary directories
mkdir -p logs
mkdir -p uploaded_files
mkdir -p backups
mkdir -p monitoring

print_success "Created necessary directories"

# Set proper permissions
chmod 755 logs uploaded_files backups monitoring
print_success "Set directory permissions"

print_status "Installing Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install production requirements
if [ -f "requirements-prod.txt" ]; then
    pip install -r requirements-prod.txt
    print_success "Installed production requirements"
else
    pip install -r requirements.txt
    print_warning "Using development requirements (requirements-prod.txt not found)"
fi

print_status "Setting up database..."

# Run database migrations
python -m alembic upgrade head
print_success "Database migrations completed"

print_status "Building Docker containers..."

# Build and start services
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d db redis tika

print_success "Docker services started"

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Health check
print_status "Performing health checks..."

# Check if services are responding
if curl -f http://localhost:9998/tika > /dev/null 2>&1; then
    print_success "Tika service is healthy"
else
    print_warning "Tika service may not be ready yet"
fi

print_status "Starting application services..."

# Start the application
if [ "$1" = "--background" ]; then
    nohup python run_app.py > logs/app.log 2>&1 &
    print_success "Application started in background"
    print_status "Check logs/app.log for application logs"
else
    print_success "Production deployment completed!"
    print_status "Run the following command to start the application:"
    echo "python run_app.py"
fi

print_success "🎉 EXPLAINIUM is ready for production!"
echo ""
echo "Access points:"
echo "  Frontend: http://localhost:8501"
echo "  API:      http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/health"
echo ""
echo "Monitoring:"
echo "  Logs:     tail -f logs/app.log"
echo "  Docker:   docker-compose logs -f"
echo ""