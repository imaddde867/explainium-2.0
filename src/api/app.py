"""
EXPLAINIUM - Consolidated Knowledge Extraction API

A clean, professional implementation consolidating all functionality
from the previous main.py and enhanced_main.py files.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Internal imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database.database import get_db, init_db
    from database import models, crud
    from processors.processor import DocumentProcessor
    from api.celery_worker import process_document_task, get_task_status
    from middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
    from logging_config import get_logger
    from core.config import config as config_manager
    from exceptions import ProcessingError, ValidationError
except ImportError as e:
    print(f"Import error: {e}")
    # Create minimal fallbacks
    def get_db(): pass
    def init_db(): pass
    models = None
    crud = None
    DocumentProcessor = None
    process_document_task = None
    get_task_status = None
    ErrorHandlingMiddleware = None
    RequestLoggingMiddleware = None
    def get_logger(name): return logging.getLogger(name)
    config_manager = None
    ProcessingError = Exception
    ValidationError = Exception

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EXPLAINIUM - Knowledge Extraction System",
    description="Professional AI-powered system for extracting structured knowledge from documents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware, log_request_body=False, log_response_body=False)
origins = getattr(config_manager, 'get_cors_origins', lambda: ['*'])()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounting intentionally disabled; Streamlit is the only frontend

# Configuration
UPLOAD_DIRECTORY = config_manager.get_upload_directory()
MAX_FILE_SIZE = config_manager.get_max_file_size()
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Initialize document processor
document_processor = DocumentProcessor()


# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    task_id: str
    message: str
    filename: str


class ProcessSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    hierarchy_level: Optional[str] = None
    confidence_threshold: float = 0.7
    max_results: int = 50


class IntelligentCategorizationRequest(BaseModel):
    document_id: int
    force_reprocess: bool = False


class BulkCategorizationRequest(BaseModel):
    document_ids: List[int]
    force_reprocess: bool = False


class IntelligentKnowledgeSearchRequest(BaseModel):
    query: str
    entity_type: Optional[str] = None
    priority_level: Optional[str] = None
    confidence_threshold: float = 0.7
    max_results: int = 50


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database: str
    services: Dict[str, str]


# API Endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "EXPLAINIUM - Advanced AI-Powered Knowledge Extraction System",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Deep Knowledge Extraction",
            "AI-Powered Analysis", 
            "Knowledge Graph Building",
            "Multi-format Export",
            "Real-time Processing"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    services = {
        "database": db_status,
        "document_processor": "healthy" if document_processor else "unhealthy",
        "celery": "healthy"  # TODO: Add actual celery health check
    }
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=datetime.now(),
        version="2.0.0",
        database=db_status,
        services=services
    )


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        # Save uploaded file
        file_path = Path(UPLOAD_DIRECTORY) / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create database record
        document = crud.create_document(
            db=db,
            filename=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Queue processing task
        task = process_document_task.delay(str(file_path), document.id)
        
        logger.info(f"Document uploaded and queued for processing: {file.filename}")
        
        return DocumentUploadResponse(
            task_id=task.id,
            message="Document uploaded successfully and queued for processing",
            filename=file.filename
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload document")


@app.get("/documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all documents"""
    documents = crud.get_documents(db, skip=skip, limit=limit)
    return {"documents": documents, "total": len(documents)}


@app.get("/documents/{document_id}")
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get document details"""
    document = crud.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.get("/processes")
async def list_processes(
    domain: Optional[str] = Query(None),
    hierarchy_level: Optional[str] = Query(None),
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List processes with filtering"""
    processes = crud.get_processes(
        db=db,
        domain=domain,
        hierarchy_level=hierarchy_level,
        confidence_threshold=confidence_threshold,
        skip=skip,
        limit=limit
    )
    return {"processes": processes, "total": len(processes)}


@app.post("/knowledge/search")
async def search_knowledge(
    request: ProcessSearchRequest,
    db: Session = Depends(get_db)
):
    """Advanced knowledge search"""
    try:
        results = crud.search_knowledge(
            db=db,
            query=request.query,
            domain=request.domain,
            hierarchy_level=request.hierarchy_level,
            confidence_threshold=request.confidence_threshold,
            max_results=request.max_results
        )
        return {"results": results, "query": request.query}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """Get processing task status"""
    try:
        status = get_task_status(task_id)
        return {"task_id": task_id, "status": status}
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task status")


@app.post("/intelligent-categorization")
async def categorize_document_intelligently(
    request: IntelligentCategorizationRequest,
    db: Session = Depends(get_db)
):
    """
    Apply intelligent knowledge categorization to a specific document.
    
    This endpoint implements the three-phase intelligent knowledge categorization framework:
    1. Document Intelligence Assessment
    2. Intelligent Knowledge Categorization  
    3. Database-Optimized Output Generation
    """
    try:
        # Check if document exists
        document = crud.get_document(db, request.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if intelligent categorization already exists
        if not request.force_reprocess:
            existing_intelligence = crud.get_document_intelligence(db, request.document_id)
            if existing_intelligence:
                return {
                    "message": "Document already has intelligent categorization",
                    "document_id": request.document_id,
                    "existing_intelligence_id": existing_intelligence.id
                }
        
        # Prepare document for categorization
        doc_for_categorization = {
            'id': document.id,
            'content': document.content,
            'metadata': {
                'filename': document.filename,
                'file_type': document.file_type,
                'uploaded_at': document.uploaded_at.isoformat() if document.uploaded_at else None,
                'file_size': document.file_size
            },
            'type': document.file_type,
            'filename': document.filename,
            'uploaded_at': document.uploaded_at
        }
        
        # Apply intelligent categorization
        categorization_result = await document_processor.categorize_document_intelligently(doc_for_categorization)
        
        # Store results in database
        try:
            # Create document intelligence record
            doc_intelligence = crud.create_document_intelligence(
                db=db,
                document_id=request.document_id,
                document_type=categorization_result['document_intelligence']['document_type'],
                target_audience=categorization_result['document_intelligence']['target_audience'],
                information_architecture=categorization_result['document_intelligence']['information_architecture'],
                priority_contexts=categorization_result['document_intelligence']['priority_contexts'],
                confidence_score=categorization_result['document_intelligence']['confidence_score'],
                analysis_method=categorization_result['document_intelligence']['analysis_method']
            )
            
            # Prepare entities for bulk creation
            entities_data = []
            for entity in categorization_result['intelligent_entities']:
                entity_data = {
                    'document_id': request.document_id,
                    'entity_type': entity['entity_type'],
                    'key_identifier': entity['key_identifier'],
                    'core_content': entity['core_content'],
                    'context_tags': entity['context_tags'],
                    'priority_level': entity['priority_level'],
                    'confidence': entity['confidence'],
                    'summary': entity['summary'],
                    'source_text': entity['source_text'],
                    'source_page': entity['source_page'],
                    'source_section': entity['source_section'],
                    'extraction_method': entity['extraction_method']
                }
                entities_data.append(entity_data)
            
            # Bulk create intelligent knowledge entities
            created_entities = crud.bulk_create_intelligent_knowledge_entities(db, entities_data)
            
            return {
                "message": "Intelligent categorization completed successfully",
                "document_id": request.document_id,
                "document_intelligence_id": doc_intelligence.id,
                "entities_created": len(created_entities),
                "quality_metrics": categorization_result['quality_metrics'],
                "processing_timestamp": categorization_result['processing_timestamp']
            }
            
        except Exception as db_error:
            logger.error(f"Database error during intelligent categorization: {db_error}")
            raise HTTPException(status_code=500, detail="Failed to store categorization results")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligent categorization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligent categorization failed: {str(e)}")


@app.post("/intelligent-categorization/bulk")
async def bulk_categorize_documents(
    request: BulkCategorizationRequest,
    db: Session = Depends(get_db)
):
    """
    Apply intelligent knowledge categorization to multiple documents.
    
    This endpoint processes multiple documents in parallel for efficiency.
    """
    try:
        # Validate all documents exist
        documents = []
        for doc_id in request.document_ids:
            doc = crud.get_document(db, doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            documents.append(doc)
        
        # Prepare documents for categorization
        docs_for_categorization = []
        for doc in documents:
            doc_data = {
                'id': doc.id,
                'content': doc.content,
                'metadata': {
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'uploaded_at': doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    'file_size': doc.file_size
                },
                'type': doc.file_type,
                'filename': doc.filename,
                'uploaded_at': doc.uploaded_at
            }
            docs_for_categorization.append(doc_data)
        
        # Apply bulk categorization
        results = await document_processor.bulk_categorize_documents(docs_for_categorization)
        
        # Store successful results in database
        successful_categorizations = []
        failed_categorizations = []
        
        for i, result in enumerate(results):
            if 'error' in result:
                failed_categorizations.append({
                    'document_id': request.document_ids[i],
                    'error': result['error']
                })
            else:
                try:
                    # Create document intelligence record
                    doc_intelligence = crud.create_document_intelligence(
                        db=db,
                        document_id=result['document_id'],
                        document_type=result['document_intelligence']['document_type'],
                        target_audience=result['document_intelligence']['target_audience'],
                        information_architecture=result['document_intelligence']['information_architecture'],
                        priority_contexts=result['document_intelligence']['priority_contexts'],
                        confidence_score=result['document_intelligence']['confidence_score'],
                        analysis_method=result['document_intelligence']['analysis_method']
                    )
                    
                    # Prepare entities for bulk creation
                    entities_data = []
                    for entity in result['intelligent_entities']:
                        entity_data = {
                            'document_id': result['document_id'],
                            'entity_type': entity['entity_type'],
                            'key_identifier': entity['key_identifier'],
                            'core_content': entity['core_content'],
                            'context_tags': entity['context_tags'],
                            'priority_level': entity['priority_level'],
                            'confidence': entity['confidence'],
                            'summary': entity['summary'],
                            'source_text': entity['source_text'],
                            'source_page': entity['source_page'],
                            'source_section': entity['source_section'],
                            'extraction_method': entity['extraction_method']
                        }
                        entities_data.append(entity_data)
                    
                    # Bulk create intelligent knowledge entities
                    created_entities = crud.bulk_create_intelligent_knowledge_entities(db, entities_data)
                    
                    successful_categorizations.append({
                        'document_id': result['document_id'],
                        'document_intelligence_id': doc_intelligence.id,
                        'entities_created': len(created_entities)
                    })
                    
                except Exception as db_error:
                    logger.error(f"Database error for document {result['document_id']}: {db_error}")
                    failed_categorizations.append({
                        'document_id': result['document_id'],
                        'error': f"Database error: {str(db_error)}"
                    })
        
        return {
            "message": "Bulk intelligent categorization completed",
            "total_documents": len(request.document_ids),
            "successful": len(successful_categorizations),
            "failed": len(failed_categorizations),
            "successful_categorizations": successful_categorizations,
            "failed_categorizations": failed_categorizations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk intelligent categorization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk intelligent categorization failed: {str(e)}")


@app.post("/intelligent-knowledge/search")
async def search_intelligent_knowledge(
    request: IntelligentKnowledgeSearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search intelligent knowledge entities with advanced filtering.
    
    This endpoint provides semantic search across all intelligent knowledge entities
    with filtering by entity type, priority level, and confidence threshold.
    """
    try:
        results = crud.search_intelligent_knowledge(
            db=db,
            query=request.query,
            entity_type=request.entity_type,
            priority_level=request.priority_level,
            confidence_threshold=request.confidence_threshold,
            max_results=request.max_results
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_results": len(results),
            "filters_applied": {
                "entity_type": request.entity_type,
                "priority_level": request.priority_level,
                "confidence_threshold": request.confidence_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Intelligent knowledge search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/intelligent-knowledge/analytics")
async def get_intelligent_knowledge_analytics(db: Session = Depends(get_db)):
    """
    Get analytics and insights about intelligent knowledge entities.
    
    This endpoint provides comprehensive analytics including:
    - Total entity counts
    - Distribution by entity type and priority level
    - Average confidence scores
    - Recent activity metrics
    """
    try:
        analytics = crud.get_intelligent_knowledge_analytics(db)
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get intelligent knowledge analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@app.get("/intelligent-knowledge/entities")
async def list_intelligent_knowledge_entities(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    document_id: Optional[int] = Query(None),
    entity_type: Optional[str] = Query(None),
    priority_level: Optional[str] = Query(None),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """
    List intelligent knowledge entities with optional filtering.
    
    This endpoint provides access to all intelligent knowledge entities
    with filtering by document, entity type, priority level, and confidence.
    """
    try:
        entities = crud.get_intelligent_knowledge_entities(
            db=db,
            skip=skip,
            limit=limit,
            document_id=document_id,
            entity_type=entity_type,
            priority_level=priority_level,
            confidence_threshold=confidence_threshold
        )
        
        # Format entities for response
        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                "id": entity.id,
                "entity_type": entity.entity_type.value,
                "key_identifier": entity.key_identifier,
                "core_content": entity.core_content[:200] + "..." if len(entity.core_content) > 200 else entity.core_content,
                "context_tags": entity.context_tags,
                "priority_level": entity.priority_level.value,
                "summary": entity.summary,
                "confidence": entity.confidence,
                "document_id": entity.document_id,
                "extracted_at": entity.extracted_at.isoformat() if entity.extracted_at else None,
                "last_updated": entity.last_updated.isoformat() if entity.last_updated else None
            })
        
        return {
            "entities": formatted_entities,
            "total": len(formatted_entities),
            "filters_applied": {
                "document_id": document_id,
                "entity_type": entity_type,
                "priority_level": priority_level,
                "confidence_threshold": confidence_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list intelligent knowledge entities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list entities: {str(e)}")


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting EXPLAINIUM application")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)