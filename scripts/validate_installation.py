#!/usr/bin/env python3
"""
EXPLAINIUM Installation Validation Script

Validates that all components are properly installed and configured.
"""

import sys
import os
import importlib
from pathlib import Path

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_status(message):
    print(f"{BLUE}[INFO]{NC} {message}")

def print_success(message):
    print(f"{GREEN}[✓]{NC} {message}")

def print_warning(message):
    print(f"{YELLOW}[!]{NC} {message}")

def print_error(message):
    print(f"{RED}[✗]{NC} {message}")

def validate_python_version():
    """Validate Python version"""
    print_status("Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def validate_required_packages():
    """Validate that required packages are installed"""
    print_status("Checking required packages...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'celery', 'redis',
        'sqlalchemy', 'alembic', 'pandas', 'numpy', 'plotly',
        'requests', 'pillow', 'opencv-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print_success(f"Package: {package}")
        except ImportError:
            print_error(f"Missing package: {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    return True

def validate_project_structure():
    """Validate project structure"""
    print_status("Checking project structure...")
    
    required_files = [
        'src/api/app.py',
        'src/frontend/knowledge_table.py',
        'src/frontend/progress_tracker.py',
        'src/processors/processor.py',
        'src/database/models.py',
        'requirements.txt',
        'requirements-prod.txt',
        'docker-compose.yml',
        '.env.example',
        'scripts/production_deploy.sh',
        'scripts/health_check.py'
    ]
    
    required_dirs = [
        'src/api',
        'src/frontend', 
        'src/processors',
        'src/database',
        'src/ai',
        'scripts',
        'alembic'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"File: {file_path}")
        else:
            print_error(f"Missing file: {file_path}")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"Directory: {dir_path}")
        else:
            print_error(f"Missing directory: {dir_path}")
            missing_dirs.append(dir_path)
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def validate_progress_tracker():
    """Validate progress tracker components"""
    print_status("Checking progress tracking components...")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        # Test imports
        from frontend.progress_tracker import ProgressTracker, create_professional_upload_interface
        print_success("Progress tracker imports successful")
        
        # Test class instantiation
        tracker = ProgressTracker()
        print_success("Progress tracker instantiation successful")
        
        return True
        
    except Exception as e:
        print_error(f"Progress tracker validation failed: {e}")
        return False

def validate_api_structure():
    """Validate API structure and endpoints"""
    print_status("Checking API structure...")
    
    try:
        sys.path.insert(0, 'src')
        
        # Test API imports
        from api.app import app
        print_success("API app import successful")
        
        # Check if required endpoints exist
        routes = [route.path for route in app.routes]
        required_endpoints = ['/health', '/upload', '/tasks/{task_id}', '/workers/stats']
        
        for endpoint in required_endpoints:
            # Handle path parameters
            endpoint_check = endpoint.replace('{task_id}', 'test')
            if any(endpoint.replace('{task_id}', '') in route for route in routes):
                print_success(f"Endpoint: {endpoint}")
            else:
                print_error(f"Missing endpoint: {endpoint}")
                return False
        
        return True
        
    except Exception as e:
        print_error(f"API validation failed: {e}")
        return False

def validate_database_models():
    """Validate database models"""
    print_status("Checking database models...")
    
    try:
        sys.path.insert(0, 'src')
        
        from database.models import ProcessingTask, Document
        print_success("Database models import successful")
        
        # Check if progress tracking fields exist
        if hasattr(ProcessingTask, 'progress_percentage'):
            print_success("Progress tracking fields present")
        else:
            print_error("Progress tracking fields missing")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Database validation failed: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🔍 EXPLAINIUM Installation Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", validate_python_version),
        ("Required Packages", validate_required_packages),
        ("Project Structure", validate_project_structure),
        ("Progress Tracker", validate_progress_tracker),
        ("API Structure", validate_api_structure),
        ("Database Models", validate_database_models)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        if check_func():
            passed += 1
        else:
            print_error(f"{check_name} validation failed")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print_success("🎉 All validations passed! EXPLAINIUM is ready to use.")
        print("\nNext steps:")
        print("1. Start the application: ./start.sh")
        print("2. Open http://localhost:8501 in your browser")
        print("3. Upload a file and test the progress tracking!")
        return True
    else:
        print_error(f"❌ {total - passed} validation(s) failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Check project structure")
        print("3. Review error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)