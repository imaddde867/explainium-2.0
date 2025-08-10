"""
EXPLAINIUM - Intelligent Knowledge Categorization Engine

Implements the three-phase intelligent knowledge categorization framework:
1. Document Intelligence Assessment
2. Intelligent Knowledge Categorization  
3. Database-Optimized Output Generation

This system transforms unstructured documents into structured, actionable knowledge
databases with intelligent, contextual understanding.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from pathlib import Path

# AI and NLP libraries
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Internal imports
from src.logging_config import get_logger
from src.core.config import AIConfig
from src.database.models import (
    EntityType, PriorityLevel, DocumentType, TargetAudience,
    KnowledgeDomain, HierarchyLevel, CriticalityLevel
)

logger = get_logger(__name__)


@dataclass
class DocumentIntelligenceAssessment:
    """Results of Phase 1: Document Intelligence Assessment"""
    document_type: DocumentType
    target_audience: TargetAudience
    information_architecture: Dict[str, Any]
    priority_contexts: List[str]
    confidence_score: float
    analysis_timestamp: datetime
    analysis_method: str


@dataclass
class IntelligentKnowledgeEntity:
    """Results of Phase 2: Intelligent Knowledge Categorization"""
    entity_type: EntityType
    key_identifier: str
    core_content: str
    context_tags: List[str]
    priority_level: PriorityLevel
    summary: Optional[str]
    confidence: float
    source_text: str
    source_page: Optional[int]
    source_section: Optional[str]
    extraction_method: str


@dataclass
class DatabaseOptimizedOutput:
    """Results of Phase 3: Database-Optimized Output Generation"""
    entities: List[IntelligentKnowledgeEntity]
    document_intelligence: DocumentIntelligenceAssessment
    quality_metrics: Dict[str, Any]
    processing_timestamp: datetime


class IntelligentKnowledgeCategorizer:
    """
    Intelligent Knowledge Categorization Engine
    
    Transforms unstructured documents into structured, actionable knowledge
    databases with intelligent, contextual understanding.
    """
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.initialized = False
        self.llm = None
        self.tokenizer = None
        self.model = None
        
        # Initialize categorization patterns
        self._init_categorization_patterns()
    
    def _init_categorization_patterns(self):
        """Initialize patterns for intelligent categorization"""
        self.process_patterns = {
            'workflow': r'(?:workflow|process|procedure|step|stage|phase)',
            'decision': r'(?:decision|choice|option|alternative|if|when|unless)',
            'dependency': r'(?:depends on|requires|prerequisite|before|after|following)',
            'completion': r'(?:complete|finish|end|result|outcome|deliverable)'
        }
        
        self.compliance_patterns = {
            'requirement': r'(?:must|shall|required|mandatory|obligatory)',
            'regulation': r'(?:regulation|standard|compliance|policy|rule)',
            'audit': r'(?:audit|review|inspection|verification|validation)',
            'threshold': r'(?:limit|maximum|minimum|threshold|boundary)'
        }
        
        self.quantitative_patterns = {
            'metric': r'(?:metric|measure|indicator|kpi|performance)',
            'number': r'(?:\d+(?:\.\d+)?(?:%|ppm|mg|kg|m|ft|hr|min))',
            'time': r'(?:schedule|timeline|deadline|duration|frequency)',
            'specification': r'(?:spec|specification|parameter|setting|configuration)'
        }
        
        self.organizational_patterns = {
            'role': r'(?:role|responsibility|duty|authority|permission)',
            'team': r'(?:team|department|division|unit|group)',
            'escalation': r'(?:escalate|escalation|supervisor|manager|director)',
            'communication': r'(?:contact|notify|inform|report|update)'
        }
        
        self.knowledge_patterns = {
            'definition': r'(?:define|definition|means|refers to|is a)',
            'concept': r'(?:concept|principle|theory|method|approach)',
            'terminology': r'(?:term|vocabulary|jargon|nomenclature)',
            'context': r'(?:context|background|scope|purpose|objective)'
        }
        
        self.risk_patterns = {
            'hazard': r'(?:hazard|risk|danger|threat|vulnerability)',
            'mitigation': r'(?:mitigate|prevent|control|reduce|eliminate)',
            'warning': r'(?:warning|caution|attention|alert|notice)',
            'contingency': r'(?:contingency|backup|alternative|fallback|emergency)'
        }
    
    async def initialize(self):
        """Initialize the categorizer with AI models"""
        try:
            # Initialize local LLM if available
            if hasattr(self.config, 'llm_path') and self.config.llm_path:
                await self._initialize_local_llm()
            
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            self.initialized = True
            logger.info("Intelligent Knowledge Categorizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize categorizer: {e}")
            raise
    
    async def _initialize_local_llm(self):
        """Initialize local LLM for intelligent analysis"""
        try:
            from llama_cpp import Llama
            
            llm_path = Path(self.config.llm_path)
            if llm_path.exists():
                # Find the first .gguf file
                gguf_files = list(llm_path.glob("*.gguf"))
                if gguf_files:
                    model_path = str(gguf_files[0])
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_threads=4,
                        n_gpu_layers=0  # CPU only for now
                    )
                    logger.info(f"Local LLM loaded: {model_path}")
                else:
                    logger.warning("No .gguf files found in LLM directory")
            else:
                logger.warning("LLM directory not found")
                
        except ImportError:
            logger.warning("llama-cpp-python not available, using fallback methods")
        except Exception as e:
            logger.warning(f"Failed to initialize local LLM: {e}")
    
    async def _initialize_embedding_model(self):
        """Initialize embedding model for semantic analysis"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = getattr(self.config, 'embedding_model', 'BAAI/bge-small-en-v1.5')
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
    
    async def categorize_document(self, document: Dict[str, Any]) -> DatabaseOptimizedOutput:
        """
        Main entry point for intelligent knowledge categorization
        
        Implements the three-phase framework:
        1. Document Intelligence Assessment
        2. Intelligent Knowledge Categorization
        3. Database-Optimized Output Generation
        """
        if not self.initialized:
            await self.initialize()
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        logger.info(f"Starting intelligent categorization for document: {document.get('id', 'unknown')}")
        
        # Phase 1: Document Intelligence Assessment
        logger.info("Phase 1: Document Intelligence Assessment")
        document_intelligence = await self._assess_document_intelligence(content, metadata)
        
        # Phase 2: Intelligent Knowledge Categorization
        logger.info("Phase 2: Intelligent Knowledge Categorization")
        entities = await self._categorize_knowledge_intelligently(content, metadata, document_intelligence)
        
        # Phase 3: Database-Optimized Output Generation
        logger.info("Phase 3: Database-Optimized Output Generation")
        quality_metrics = self._assess_output_quality(entities, document_intelligence)
        
        output = DatabaseOptimizedOutput(
            entities=entities,
            document_intelligence=document_intelligence,
            quality_metrics=quality_metrics,
            processing_timestamp=datetime.now()
        )
        
        logger.info(f"Intelligent categorization completed. Generated {len(entities)} entities.")
        return output
    
    async def _assess_document_intelligence(self, content: str, metadata: Dict[str, Any]) -> DocumentIntelligenceAssessment:
        """Phase 1: Rapidly analyze document to determine type, audience, and architecture"""
        try:
            # Use LLM for intelligent assessment if available
            if self.llm:
                assessment = await self._llm_assess_document(content, metadata)
            else:
                assessment = await self._pattern_based_assessment(content, metadata)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in document intelligence assessment: {e}")
            # Return default assessment
            return DocumentIntelligenceAssessment(
                document_type=DocumentType.REPORT,
                target_audience=TargetAudience.TECHNICAL_STAFF,
                information_architecture={"structure": "unknown", "sections": []},
                priority_contexts=["general"],
                confidence_score=0.5,
                analysis_timestamp=datetime.now(),
                analysis_method="fallback"
            )
    
    async def _llm_assess_document(self, content: str, metadata: Dict[str, Any]) -> DocumentIntelligenceAssessment:
        """Use LLM for intelligent document assessment"""
        prompt = f"""
        Analyze this document and provide intelligence assessment in JSON format:
        
        Document content (first 2000 characters):
        {content[:2000]}
        
        Metadata: {metadata}
        
        Provide assessment in this exact JSON format:
        {{
            "document_type": "manual|contract|report|policy|specification|procedure|guideline|standard|form|template",
            "target_audience": "technical_staff|management|end_users|compliance_officers|training_personnel|maintenance_team|quality_assurance|regulatory_bodies",
            "information_architecture": {{
                "structure": "description of how knowledge is organized",
                "sections": ["list", "of", "main", "sections"],
                "interconnections": "description of how sections relate"
            }},
            "priority_contexts": ["list", "of", "critical", "information", "types"],
            "confidence_score": 0.95
        }}
        """
        
        try:
            response = self.llm(
                prompt,
                max_tokens=1024,
                temperature=0.1,
                stop=["</s>", "\n\n"]
            )
            result_text = response['choices'][0]['text']
            
            # Parse JSON response
            assessment_data = json.loads(result_text)
            
            return DocumentIntelligenceAssessment(
                document_type=DocumentType(assessment_data.get('document_type', 'report')),
                target_audience=TargetAudience(assessment_data.get('target_audience', 'technical_staff')),
                information_architecture=assessment_data.get('information_architecture', {}),
                priority_contexts=assessment_data.get('priority_contexts', []),
                confidence_score=assessment_data.get('confidence_score', 0.8),
                analysis_timestamp=datetime.now(),
                analysis_method="llm_analysis"
            )
            
        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}, falling back to pattern-based")
            return await self._pattern_based_assessment(content, metadata)
    
    async def _pattern_based_assessment(self, content: str, metadata: Dict[str, Any]) -> DocumentIntelligenceAssessment:
        """Fallback pattern-based document assessment"""
        content_lower = content.lower()
        
        # Determine document type
        if any(word in content_lower for word in ['manual', 'guide', 'instruction']):
            doc_type = DocumentType.MANUAL
        elif any(word in content_lower for word in ['contract', 'agreement', 'terms']):
            doc_type = DocumentType.CONTRACT
        elif any(word in content_lower for word in ['policy', 'procedure', 'standard']):
            doc_type = DocumentType.POLICY
        elif any(word in content_lower for word in ['specification', 'spec', 'requirement']):
            doc_type = DocumentType.SPECIFICATION
        else:
            doc_type = DocumentType.REPORT
        
        # Determine target audience
        if any(word in content_lower for word in ['technical', 'engineer', 'developer']):
            audience = TargetAudience.TECHNICAL_STAFF
        elif any(word in content_lower for word in ['management', 'executive', 'director']):
            audience = TargetAudience.MANAGEMENT
        elif any(word in content_lower for word in ['compliance', 'regulatory', 'audit']):
            audience = TargetAudience.COMPLIANCE_OFFICERS
        else:
            audience = TargetAudience.END_USERS
        
        # Analyze information architecture
        sections = re.findall(r'^[A-Z][A-Z\s]+$', content, re.MULTILINE)
        structure = "hierarchical" if len(sections) > 3 else "linear"
        
        return DocumentIntelligenceAssessment(
            document_type=doc_type,
            target_audience=audience,
            information_architecture={
                "structure": structure,
                "sections": sections[:10],  # Limit to first 10 sections
                "interconnections": "pattern-based inference"
            },
            priority_contexts=["general"],
            confidence_score=0.6,
            analysis_timestamp=datetime.now(),
            analysis_method="pattern_analysis"
        )
    
    async def _categorize_knowledge_intelligently(self, content: str, metadata: Dict[str, Any], 
                                                document_intelligence: DocumentIntelligenceAssessment) -> List[IntelligentKnowledgeEntity]:
        """Phase 2: Systematically identify and classify information into structured database entities"""
        entities = []
        
        # Split content into manageable chunks for analysis
        chunks = self._split_content_into_chunks(content, max_chunk_size=2000)
        
        for i, chunk in enumerate(chunks):
            chunk_entities = await self._analyze_chunk_intelligently(
                chunk, metadata, document_intelligence, chunk_index=i
            )
            entities.extend(chunk_entities)
        
        # Consolidate and deduplicate entities
        entities = self._consolidate_entities(entities)
        
        return entities
    
    def _split_content_into_chunks(self, content: str, max_chunk_size: int = 2000) -> List[str]:
        """Split content into manageable chunks while preserving context"""
        chunks = []
        
        # Try to split on paragraph boundaries first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If we have too few chunks, split on sentences
        if len(chunks) < 2:
            sentences = re.split(r'[.!?]+', content)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += sentence + "."
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "."
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _analyze_chunk_intelligently(self, chunk: str, metadata: Dict[str, Any], 
                                         document_intelligence: DocumentIntelligenceAssessment, 
                                         chunk_index: int) -> List[IntelligentKnowledgeEntity]:
        """Analyze a content chunk to extract intelligent knowledge entities"""
        entities = []
        
        # Use LLM for intelligent extraction if available
        if self.llm:
            llm_entities = await self._llm_extract_entities(chunk, metadata, document_intelligence)
            entities.extend(llm_entities)
        
        # Fallback to pattern-based extraction
        pattern_entities = self._pattern_based_extraction(chunk, metadata, document_intelligence)
        entities.extend(pattern_entities)
        
        # Add chunk context
        for entity in entities:
            entity.source_text = chunk
            entity.source_section = f"chunk_{chunk_index + 1}"
        
        return entities
    
    async def _llm_extract_entities(self, chunk: str, metadata: Dict[str, Any], 
                                   document_intelligence: DocumentIntelligenceAssessment) -> List[IntelligentKnowledgeEntity]:
        """Use LLM to extract intelligent knowledge entities"""
        prompt = f"""
        Analyze this document chunk and extract intelligent knowledge entities.
        
        Document type: {document_intelligence.document_type.value}
        Target audience: {document_intelligence.target_audience.value}
        
        Chunk content:
        {chunk}
        
        Extract entities in this exact JSON format:
        {{
            "entities": [
                {{
                    "entity_type": "process|policy|metric|role|compliance_requirement|risk_assessment|workflow|decision_point|technical_specification|organizational_structure|knowledge_concept|mitigation_strategy",
                    "key_identifier": "unique descriptive label",
                    "core_content": "complete synthesized information unit",
                    "context_tags": ["relevant", "categories"],
                    "priority_level": "high|medium|low",
                    "summary": "2-3 sentence human-readable overview",
                    "confidence": 0.95
                }}
            ]
        }}
        
        Focus on actionable, business-relevant information that drives decisions or actions.
        Ensure each entity is self-contained and unambiguous.
        """
        
        try:
            response = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.1,
                stop=["</s>", "\n\n"]
            )
            result_text = response['choices'][0]['text']
            
            # Parse JSON response
            extraction_data = json.loads(result_text)
            
            entities = []
            for entity_data in extraction_data.get('entities', []):
                try:
                    entity = IntelligentKnowledgeEntity(
                        entity_type=EntityType(entity_data['entity_type']),
                        key_identifier=entity_data['key_identifier'],
                        core_content=entity_data['core_content'],
                        context_tags=entity_data.get('context_tags', []),
                        priority_level=PriorityLevel(entity_data['priority_level']),
                        summary=entity_data.get('summary'),
                        confidence=entity_data.get('confidence', 0.8),
                        source_text="",  # Will be set by caller
                        source_page=None,
                        source_section="",
                        extraction_method="llm_extraction"
                    )
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to create entity from LLM data: {e}")
                    continue
            
            return entities
            
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return []
    
    def _pattern_based_extraction(self, chunk: str, metadata: Dict[str, Any], 
                                 document_intelligence: DocumentIntelligenceAssessment) -> List[IntelligentKnowledgeEntity]:
        """Fallback pattern-based entity extraction"""
        entities = []
        chunk_lower = chunk.lower()
        
        # Extract process intelligence
        if re.search(self.process_patterns['workflow'], chunk_lower):
            entities.append(self._create_process_entity(chunk, "Process Workflow"))
        
        # Extract compliance & governance
        if re.search(self.compliance_patterns['requirement'], chunk_lower):
            entities.append(self._create_compliance_entity(chunk, "Compliance Requirement"))
        
        # Extract quantitative intelligence
        if re.search(self.quantitative_patterns['metric'], chunk_lower):
            entities.append(self._create_metric_entity(chunk, "Performance Metric"))
        
        # Extract organizational intelligence
        if re.search(self.organizational_patterns['role'], chunk_lower):
            entities.append(self._create_role_entity(chunk, "Organizational Role"))
        
        # Extract knowledge definitions
        if re.search(self.knowledge_patterns['definition'], chunk_lower):
            entities.append(self._create_knowledge_entity(chunk, "Knowledge Definition"))
        
        # Extract risk & mitigation
        if re.search(self.risk_patterns['hazard'], chunk_lower):
            entities.append(self._create_risk_entity(chunk, "Risk Assessment"))
        
        return entities
    
    def _create_process_entity(self, chunk: str, identifier: str) -> IntelligentKnowledgeEntity:
        """Create a process intelligence entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.PROCESS,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["workflow", "procedure", "operational"],
            priority_level=PriorityLevel.MEDIUM,
            summary=f"Process workflow identified in document section",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_compliance_entity(self, chunk: str, identifier: str) -> IntelligentKnowledgeEntity:
        """Create a compliance & governance entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.COMPLIANCE_REQUIREMENT,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["compliance", "regulation", "requirement"],
            priority_level=PriorityLevel.HIGH,
            summary=f"Compliance requirement identified in document section",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_metric_entity(self, chunk: str, identifier: str) -> IntelligentKnowledgeEntity:
        """Create a quantitative intelligence entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.METRIC,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["metric", "performance", "measurement"],
            priority_level=PriorityLevel.MEDIUM,
            summary=f"Performance metric identified in document section",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_role_entity(self, chunk: str, identifier: str) -> IntelligentKnowledgeEntity:
        """Create an organizational intelligence entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.ROLE,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["role", "responsibility", "organization"],
            priority_level=PriorityLevel.MEDIUM,
            summary=f"Organizational role identified in document section",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_knowledge_entity(self, chunk: str, identifier: str) -> IntelligentKnowledgeEntity:
        """Create a knowledge definition entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.KNOWLEDGE_CONCEPT,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["definition", "concept", "knowledge"],
            priority_level=PriorityLevel.LOW,
            summary=f"Knowledge definition identified in document section",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_risk_entity(self, chunk: str, identifier: str) -> IntelligentKnowledgeEntity:
        """Create a risk & mitigation entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.RISK_ASSESSMENT,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["risk", "hazard", "mitigation"],
            priority_level=PriorityLevel.HIGH,
            summary=f"Risk assessment identified in document section",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _consolidate_entities(self, entities: List[IntelligentKnowledgeEntity]) -> List[IntelligentKnowledgeEntity]:
        """Consolidate and deduplicate entities"""
        # Group entities by type and identifier
        entity_groups = {}
        
        for entity in entities:
            key = (entity.entity_type, entity.key_identifier)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Consolidate each group
        consolidated = []
        for key, group in entity_groups.items():
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # Merge multiple entities of the same type and identifier
                merged = self._merge_entity_group(group)
                consolidated.append(merged)
        
        return consolidated
    
    def _merge_entity_group(self, group: List[IntelligentKnowledgeEntity]) -> IntelligentKnowledgeEntity:
        """Merge multiple entities into one comprehensive entity"""
        if not group:
            return group[0]
        
        # Use the entity with highest confidence as base
        base_entity = max(group, key=lambda x: x.confidence)
        
        # Merge content and context tags
        all_content = []
        all_tags = set()
        all_summaries = []
        
        for entity in group:
            if entity.core_content:
                all_content.append(entity.core_content)
            if entity.context_tags:
                all_tags.update(entity.context_tags)
            if entity.summary:
                all_summaries.append(entity.summary)
        
        # Create merged entity
        merged = IntelligentKnowledgeEntity(
            entity_type=base_entity.entity_type,
            key_identifier=base_entity.key_identifier,
            core_content="\n\n".join(all_content),
            context_tags=list(all_tags),
            priority_level=max(group, key=lambda x: x.priority_level.value).priority_level,
            summary="; ".join(all_summaries) if all_summaries else base_entity.summary,
            confidence=sum(e.confidence for e in group) / len(group),
            source_text=base_entity.source_text,
            source_page=base_entity.source_page,
            source_section=base_entity.source_section,
            extraction_method="merged_extraction"
        )
        
        return merged
    
    def _assess_output_quality(self, entities: List[IntelligentKnowledgeEntity], 
                              document_intelligence: DocumentIntelligenceAssessment) -> Dict[str, Any]:
        """Assess the quality of the generated output"""
        if not entities:
            return {
                "overall_quality": 0.0,
                "entity_count": 0,
                "quality_issues": ["No entities generated"],
                "recommendations": ["Check document content and extraction methods"]
            }
        
        # Calculate quality metrics
        total_confidence = sum(e.confidence for e in entities)
        avg_confidence = total_confidence / len(entities)
        
        # Check for quality issues
        quality_issues = []
        recommendations = []
        
        # Check for fragmentation
        if len(entities) > 20:  # Arbitrary threshold
            quality_issues.append("High entity count may indicate fragmentation")
            recommendations.append("Consider consolidating similar entities")
        
        # Check for low confidence
        low_confidence_count = sum(1 for e in entities if e.confidence < 0.5)
        if low_confidence_count > len(entities) * 0.3:
            quality_issues.append("High percentage of low-confidence entities")
            recommendations.append("Review extraction methods and improve confidence scoring")
        
        # Check for missing summaries
        missing_summaries = sum(1 for e in entities if not e.summary)
        if missing_summaries > len(entities) * 0.5:
            quality_issues.append("Many entities lack human-readable summaries")
            recommendations.append("Ensure all entities have concise summaries")
        
        # Calculate overall quality score
        quality_score = avg_confidence * 0.6 + (1 - len(quality_issues) * 0.1) + 0.3
        
        return {
            "overall_quality": min(quality_score, 1.0),
            "entity_count": len(entities),
            "average_confidence": avg_confidence,
            "quality_issues": quality_issues,
            "recommendations": recommendations,
            "entity_types_distribution": self._get_entity_type_distribution(entities),
            "priority_levels_distribution": self._get_priority_level_distribution(entities)
        }
    
    def _get_entity_type_distribution(self, entities: List[IntelligentKnowledgeEntity]) -> Dict[str, int]:
        """Get distribution of entity types"""
        distribution = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        return distribution
    
    def _get_priority_level_distribution(self, entities: List[IntelligentKnowledgeEntity]) -> Dict[str, int]:
        """Get distribution of priority levels"""
        distribution = {}
        for entity in entities:
            priority = entity.priority_level.value
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution