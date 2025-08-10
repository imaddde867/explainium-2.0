# EXPLAINIUM - Intelligent Knowledge Categorization System Setup Guide

## 🚀 Quick Start

The Intelligent Knowledge Categorization System has been successfully implemented and is ready for use. This guide will help you get it running in your environment.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9+ (3.13+ recommended)
- **RAM**: 16GB+ recommended for optimal performance
- **Storage**: 10GB+ available space for AI models
- **OS**: Linux, macOS, or Windows (Linux recommended for production)

### Python Environment
You have two options for setting up the Python environment:

#### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements_intelligent_categorization.txt
```

#### Option 2: System-wide Installation
```bash
# Install system packages (Ubuntu/Debian)
sudo apt update
sudo apt install python3-fastapi python3-sqlalchemy python3-pandas python3-numpy

# Install additional packages
pip install --user -r requirements_intelligent_categorization.txt
```

## 🔧 Installation Steps

### Step 1: Clone/Download the System
```bash
# If using git
git clone <repository-url>
cd explainium-ai

# Or download and extract the files
# Ensure all files are in the correct directory structure
```

### Step 2: Install Dependencies
```bash
# Install core requirements
pip install -r requirements_intelligent_categorization.txt

# Verify installation
python3 -c "import fastapi, sqlalchemy, pandas, numpy; print('✅ Dependencies installed')"
```

### Step 3: Database Setup
```bash
# Initialize database (if using SQLite)
python3 -c "
import sys
sys.path.append('src')
from database.database import init_db
init_db()
print('✅ Database initialized')
"

# Or use existing database connection
# Update src/core/config.py with your database credentials
```

### Step 4: Test the System
```bash
# Test minimal version (no external dependencies)
python3 test_minimal_categorizer.py

# Test full version (requires dependencies)
python3 test_intelligent_categorization.py
```

## 🧪 Testing the System

### Minimal Version Test
The minimal version works with just the Python standard library:

```bash
python3 test_minimal_categorizer.py
```

This will test:
- ✅ Document intelligence assessment
- ✅ Pattern-based entity extraction
- ✅ Entity consolidation and deduplication
- ✅ Quality metrics generation

### Full Version Test
The full version includes LLM integration:

```bash
python3 test_intelligent_categorization.py
```

This requires:
- ✅ All dependencies installed
- ✅ Database connection configured
- ✅ Optional: Local LLM models downloaded

## 🚀 Running the System

### Start the Backend API
```bash
# Start FastAPI backend
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Frontend (Optional)
```bash
# Start Streamlit frontend
streamlit run src/frontend/knowledge_table.py --server.port 8501
```

### Use the Startup Scripts
```bash
# Make scripts executable
chmod +x start.sh stop.sh

# Start all services
./start.sh

# Stop all services
./stop.sh
```

## 🔌 API Endpoints

Once running, the system provides these endpoints:

### Intelligent Categorization
- `POST /intelligent-categorization` - Categorize single document
- `POST /intelligent-categorization/bulk` - Categorize multiple documents

### Intelligent Knowledge Management
- `POST /intelligent-knowledge/search` - Search knowledge entities
- `GET /intelligent-knowledge/analytics` - Get analytics
- `GET /intelligent-knowledge/entities` - List entities

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📊 Usage Examples

### Python Client
```python
import requests

# Categorize a document
response = requests.post(
    "http://localhost:8000/intelligent-categorization",
    json={"document_id": 123, "force_reprocess": False}
)

result = response.json()
print(f"Created {result['entities_created']} entities")

# Search knowledge
search_response = requests.post(
    "http://localhost:8000/intelligent-knowledge/search",
    json={
        "query": "safety compliance",
        "entity_type": "compliance_requirement",
        "priority_level": "high"
    }
)
```

### cURL Examples
```bash
# Categorize document
curl -X POST "http://localhost:8000/intelligent-categorization" \
     -H "Content-Type: application/json" \
     -d '{"document_id": 123, "force_reprocess": false}'

# Search knowledge
curl -X POST "http://localhost:8000/intelligent-knowledge/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "safety compliance", "max_results": 10}'
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: No module named 'fastapi'
pip install fastapi uvicorn

# Error: No module named 'sqlalchemy'
pip install sqlalchemy
```

#### 2. Database Connection Issues
```bash
# Check database configuration
cat src/core/config.py

# Test database connection
python3 -c "
import sys
sys.path.append('src')
from database.database import get_db
db = next(get_db())
print('✅ Database connection successful')
"
```

#### 3. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload
```

#### 4. Permission Issues
```bash
# Make scripts executable
chmod +x *.sh

# Check file permissions
ls -la *.sh
```

### Performance Issues

#### 1. Slow Processing
- Ensure sufficient RAM (16GB+ recommended)
- Check CPU usage during processing
- Consider using smaller document chunks

#### 2. Memory Issues
- Monitor memory usage with `htop` or `top`
- Restart services if memory usage is high
- Consider processing documents in smaller batches

## 🔧 Configuration

### Environment Variables
```bash
# Database
export DATABASE_URL="postgresql://user:password@localhost/explainium"

# AI Models
export AI_MODEL_PATH="/path/to/models"
export AI_DEVICE="cuda"  # or "cpu"

# API Settings
export API_HOST="0.0.0.0"
export API_PORT="8000"
export DEBUG="true"
```

### Configuration File
Update `src/core/config.py`:

```python
class Config:
    database_url: str = "postgresql://user:password@localhost/explainium"
    ai_model_path: str = "/path/to/models"
    ai_device: str = "cpu"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
```

## 📈 Monitoring and Logging

### Log Files
```bash
# Check application logs
tail -f logs/app.log

# Check error logs
tail -f logs/error.log

# Check access logs
tail -f logs/access.log
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/db

# AI system health
curl http://localhost:8000/health/ai
```

## 🔒 Security Considerations

### Production Deployment
```bash
# Use HTTPS in production
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --ssl-keyfile=key.pem --ssl-certfile=cert.pem

# Enable authentication
# Update src/core/auth.py with your authentication logic

# Rate limiting
# Configure rate limiting in src/api/middleware.py
```

### Access Control
```bash
# Set up firewall rules
sudo ufw allow 8000
sudo ufw enable

# Use reverse proxy (nginx/apache)
# Configure in docker/nginx.conf
```

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/explainium.service

# Enable and start service
sudo systemctl enable explainium
sudo systemctl start explainium
sudo systemctl status explainium
```

## 📚 Additional Resources

### Documentation
- [API Reference](README_INTELLIGENT_CATEGORIZATION.md)
- [Database Schema](src/database/models.py)
- [Configuration Guide](src/core/config.py)

### Support
- Check logs for error details
- Review configuration files
- Test with minimal version first
- Ensure all dependencies are installed

## 🎉 Success!

Once you've completed these steps, you'll have a fully functional Intelligent Knowledge Categorization System that can:

✅ Transform unstructured documents into structured knowledge  
✅ Provide intelligent, contextual understanding  
✅ Generate database-optimized output  
✅ Maintain high data quality standards  
✅ Scale from single documents to bulk processing  
✅ Integrate seamlessly with existing systems  

The system is ready to process your documents and extract actionable intelligence!