# 📁 EXPLAINIUM Project Structure

This document outlines the organized structure of the EXPLAINIUM v2.0 codebase.

## 🏗️ Directory Structure

```
explainium-2.0/
├── 📁 src/                          # Main source code
│   ├── 📁 api/                      # FastAPI backend
│   │   ├── app.py                   # Main API application
│   │   └── celery_worker.py         # Background task processing
│   ├── 📁 frontend/                 # Streamlit frontend
│   │   ├── knowledge_table.py       # Main frontend application
│   │   ├── structured_knowledge_display.py  # Enhanced knowledge display
│   │   └── progress_tracker.py      # Progress tracking components (NEW)
│   ├── 📁 processors/               # Document processing
│   │   └── processor.py             # Main document processor
│   ├── 📁 ai/                       # AI and machine learning
│   │   ├── advanced_knowledge_engine.py    # AI knowledge engine
│   │   └── knowledge_analyst.py     # AI knowledge analyst
│   ├── 📁 database/                 # Database layer
│   │   ├── database.py              # Database configuration
│   │   ├── models.py                # SQLAlchemy models
│   │   └── crud.py                  # Database operations
│   ├── 📁 core/                     # Core utilities
│   │   ├── config.py                # Configuration management
│   │   └── optimization.py          # Performance optimizations
│   ├── 📁 export/                   # Export functionality
│   │   └── knowledge_export.py      # Knowledge export utilities
│   ├── exceptions.py                # Custom exceptions
│   ├── logging_config.py            # Logging configuration
│   └── middleware.py                # API middleware
├── 📁 scripts/                      # Utility scripts
│   ├── production_deploy.sh         # Production deployment (NEW)
│   ├── health_check.py              # Health monitoring (NEW)
│   └── model_manager.py             # AI model management
├── 📁 docker/                       # Docker configuration
├── 📁 alembic/                      # Database migrations
├── 📁 models/                       # AI model storage
├── 📁 documents_samples/            # Sample documents
├── 📁 logs/                         # Application logs
├── 📁 uploaded_files/               # File upload storage
├── 📁 backups/                      # Database backups
├── 📁 monitoring/                   # Monitoring data
├── requirements.txt                 # Development dependencies
├── requirements-prod.txt            # Production dependencies (NEW)
├── .env.example                     # Environment template (UPDATED)
├── docker-compose.yml               # Docker services
├── run_app.py                       # Application runner
├── README.md                        # Project documentation (UPDATED)
├── PRODUCTION_CHECKLIST.md          # Production checklist (NEW)
└── PROJECT_STRUCTURE.md             # This file (NEW)
```

## 🔧 Component Overview

### Backend Components

#### 🚀 API Layer (`src/api/`)
- **app.py**: Main FastAPI application with enhanced endpoints
- **celery_worker.py**: Background task processing with detailed progress tracking

#### 🔄 Processing Layer (`src/processors/`)
- **processor.py**: Multi-format document processing with progress callbacks

#### 🧠 AI Layer (`src/ai/`)
- **advanced_knowledge_engine.py**: Core AI processing engine
- **knowledge_analyst.py**: 3-phase knowledge analysis framework

#### 🗄️ Data Layer (`src/database/`)
- **models.py**: SQLAlchemy models with progress tracking fields
- **crud.py**: Database operations with task management
- **database.py**: Database configuration and connection management

### Frontend Components

#### 🎨 User Interface (`src/frontend/`)
- **knowledge_table.py**: Main Streamlit application with enhanced UI
- **structured_knowledge_display.py**: Advanced knowledge visualization
- **progress_tracker.py**: Real-time progress tracking components (NEW)

### Infrastructure

#### 🐳 Containerization
- **docker-compose.yml**: Multi-service Docker configuration
- **docker/**: Additional Docker configurations

#### 🔧 Scripts (`scripts/`)
- **production_deploy.sh**: Automated production deployment (NEW)
- **health_check.py**: Comprehensive health monitoring (NEW)
- **model_manager.py**: AI model management utilities

## 📊 New Features in v2.0

### Progress Tracking System
1. **Backend Progress Updates**: Celery tasks with percentage-based progress
2. **Real-time API**: WebSocket-style polling for live updates
3. **Professional UI**: Modern progress bars and status indicators
4. **Task Management**: Database tracking of all processing tasks

### Enhanced UI/UX
1. **Modern Design**: Gradient backgrounds and professional styling
2. **Responsive Layout**: Clean, organized interface with proper spacing
3. **Interactive Elements**: Enhanced buttons, cards, and animations
4. **Status Indicators**: Clear system status with color-coded health checks

### Production Features
1. **Health Monitoring**: Comprehensive system health checks
2. **Environment Management**: Proper configuration templates
3. **Deployment Automation**: One-click production deployment
4. **Security Enhancements**: Production-ready security configurations

## 🔄 Data Flow

### File Processing Pipeline
1. **Upload**: File uploaded via Streamlit interface
2. **Queue**: Task queued in Celery with Redis backend
3. **Process**: Document processed with progress updates (10% → 100%)
4. **Store**: Results stored in PostgreSQL database
5. **Display**: Real-time progress shown to user
6. **Complete**: Final results displayed with analytics

### Progress Tracking Flow
1. **Task Creation**: Celery task created with unique ID
2. **Progress Updates**: Regular progress updates (10%, 20%, 40%, etc.)
3. **Database Sync**: Progress stored in database for persistence
4. **Frontend Polling**: Frontend polls API for real-time updates
5. **Completion**: Final results displayed with statistics

## 🔧 Configuration

### Environment Variables
- **Development**: Use `.env` with development settings
- **Production**: Use `.env` with production settings from `.env.example`

### Key Configuration Files
- **alembic.ini**: Database migration configuration
- **docker-compose.yml**: Service orchestration
- **requirements-prod.txt**: Pinned production dependencies

## 🚀 Deployment Options

### Development
```bash
./start.sh
```

### Production
```bash
./scripts/production_deploy.sh
```

### Health Monitoring
```bash
python scripts/health_check.py --monitor
```

---

This structure ensures maintainability, scalability, and production readiness while providing clear separation of concerns and easy navigation for developers.