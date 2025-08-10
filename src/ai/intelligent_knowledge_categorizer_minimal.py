#!/usr/bin/env python3
"""
Minimal Intelligent Knowledge Categorizer
This version works with just the Python standard library for testing purposes.
"""

import re
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


# Minimal enum definitions
class EntityType(Enum):
    PROCESS = "process"
    POLICY = "policy"
    METRIC = "metric"
    ROLE = "role"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"
    RISK_ASSESSMENT = "risk_assessment"
    WORKFLOW = "workflow"
    DECISION_POINT = "decision_point"
    TECHNICAL_SPECIFICATION = "technical_specification"
    ORGANIZATIONAL_STRUCTURE = "organizational_structure"
    KNOWLEDGE_CONCEPT = "knowledge_concept"
    MITIGATION_STRATEGY = "mitigation_strategy"


class PriorityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DocumentType(Enum):
    MANUAL = "manual"
    CONTRACT = "contract"
    REPORT = "report"
    POLICY = "policy"
    SPECIFICATION = "specification"
    PROCEDURE = "procedure"
    GUIDELINE = "guideline"
    STANDARD = "standard"
    FORM = "form"
    TEMPLATE = "template"


class TargetAudience(Enum):
    TECHNICAL_STAFF = "technical_staff"
    MANAGEMENT = "management"
    END_USERS = "end_users"
    COMPLIANCE_OFFICERS = "compliance_officers"
    TRAINING_PERSONNEL = "training_personnel"
    MAINTENANCE_TEAM = "maintenance_team"
    QUALITY_ASSURANCE = "quality_assurance"
    REGULATORY_BODIES = "regulatory_bodies"


@dataclass
class DocumentIntelligenceAssessment:
    document_type: DocumentType
    target_audience: TargetAudience
    information_architecture: Dict[str, Any]
    priority_contexts: List[str]
    confidence_score: float
    analysis_timestamp: datetime
    analysis_method: str


@dataclass
class IntelligentKnowledgeEntity:
    entity_type: EntityType
    key_identifier: str
    core_content: str
    context_tags: List[str]
    priority_level: PriorityLevel
    summary: Optional[str]
    confidence: float
    source_text: str
    source_page: Optional[int]
    source_section: str
    extraction_method: str


@dataclass
class IntelligentCategorizationResult:
    document_intelligence: DocumentIntelligenceAssessment
    entities: List[IntelligentKnowledgeEntity]
    quality_metrics: Dict[str, Any]


class MinimalIntelligentKnowledgeCategorizer:
    """
    Minimal version of the intelligent knowledge categorizer
    that works with just the Python standard library.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.llm = None  # No LLM available in minimal version
        
        # Pattern definitions for extraction
        self.process_patterns = {
            'workflow': r'\b(procedure|process|workflow|protocol|method|step|sequence)\b',
            'decision': r'\b(if|when|then|else|decision|condition|criteria)\b',
            'trigger': r'\b(trigger|initiate|start|begin|activate)\b'
        }
        
        self.compliance_patterns = {
            'requirement': r'\b(must|shall|required|mandatory|compliance|regulation|standard)\b',
            'audit': r'\b(audit|inspection|verification|validation|check|review)\b',
            'certification': r'\b(certified|approved|accredited|licensed|permitted)\b'
        }
        
        self.quantitative_patterns = {
            'metric': r'\b(\d+%|\d+\.\d+|\d+\s*(?:hours?|days?|weeks?|months?))\b',
            'limit': r'\b(maximum|minimum|threshold|limit|range|between)\b',
            'specification': r'\b(spec|specification|requirement|parameter|value)\b'
        }
        
        self.organizational_patterns = {
            'role': r'\b(responsible|accountable|consulted|informed|role|duty|responsibility)\b',
            'authority': r'\b(authority|power|decision|approval|signature|authorization)\b',
            'escalation': r'\b(escalate|escalation|supervisor|manager|director|executive)\b'
        }
        
        self.knowledge_patterns = {
            'definition': r'\b(defined as|means|refers to|is|are|consists of)\b',
            'concept': r'\b(concept|principle|theory|methodology|approach|strategy)\b',
            'terminology': r'\b(term|vocabulary|jargon|language|nomenclature)\b'
        }
        
        self.risk_patterns = {
            'hazard': r'\b(hazard|risk|danger|threat|vulnerability|exposure)\b',
            'mitigation': r'\b(prevent|avoid|reduce|minimize|control|mitigate)\b',
            'contingency': r'\b(emergency|backup|alternative|fallback|contingency)\b'
        }
    
    async def initialize(self):
        """Initialize the categorizer (minimal version)"""
        print("Minimal Intelligent Knowledge Categorizer initialized")
        return True
    
    async def categorize_document(self, document: Dict[str, Any]) -> IntelligentCategorizationResult:
        """
        Main method to categorize a document using the three-phase approach
        """
        try:
            # Phase 1: Document Intelligence Assessment
            document_intelligence = await self._assess_document_intelligence(
                document.get('content', ''),
                document.get('metadata', {})
            )
            
            # Phase 2: Intelligent Knowledge Categorization
            entities = await self._categorize_knowledge_intelligently(
                document.get('content', ''),
                document.get('metadata', {}),
                document_intelligence
            )
            
            # Phase 3: Quality Assessment
            quality_metrics = self._assess_output_quality(entities, document_intelligence)
            
            return IntelligentCategorizationResult(
                document_intelligence=document_intelligence,
                entities=entities,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            print(f"Error in document categorization: {e}")
            # Return default result
            return IntelligentCategorizationResult(
                document_intelligence=DocumentIntelligenceAssessment(
                    document_type=DocumentType.REPORT,
                    target_audience=TargetAudience.TECHNICAL_STAFF,
                    information_architecture={"structure": "unknown", "sections": []},
                    priority_contexts=["general"],
                    confidence_score=0.5,
                    analysis_timestamp=datetime.now(),
                    analysis_method="fallback"
                ),
                entities=[],
                quality_metrics={"overall_quality": 0.0, "entity_count": 0}
            )
    
    async def _assess_document_intelligence(self, content: str, metadata: Dict[str, Any]) -> DocumentIntelligenceAssessment:
        """Phase 1: Document intelligence assessment using pattern analysis"""
        content_lower = content.lower()
        
        # Determine document type
        if any(word in content_lower for word in ['manual', 'guide', 'instruction', 'procedure']):
            doc_type = DocumentType.MANUAL
        elif any(word in content_lower for word in ['contract', 'agreement', 'terms', 'policy']):
            doc_type = DocumentType.POLICY
        elif any(word in content_lower for word in ['specification', 'spec', 'requirement', 'standard']):
            doc_type = DocumentType.SPECIFICATION
        else:
            doc_type = DocumentType.REPORT
        
        # Determine target audience
        if any(word in content_lower for word in ['technical', 'engineer', 'developer', 'specialist']):
            audience = TargetAudience.TECHNICAL_STAFF
        elif any(word in content_lower for word in ['management', 'executive', 'director', 'supervisor']):
            audience = TargetAudience.MANAGEMENT
        elif any(word in content_lower for word in ['compliance', 'regulatory', 'audit', 'legal']):
            audience = TargetAudience.COMPLIANCE_OFFICERS
        else:
            audience = TargetAudience.END_USERS
        
        # Analyze information architecture
        sections = re.findall(r'^[A-Z][A-Z\s]+$', content, re.MULTILINE)
        structure = "hierarchical" if len(sections) > 3 else "linear"
        
        # Determine priority contexts
        priority_contexts = []
        if re.search(r'\b(safety|security|compliance)\b', content_lower):
            priority_contexts.append("safety")
        if re.search(r'\b(quality|performance|efficiency)\b', content_lower):
            priority_contexts.append("quality")
        if re.search(r'\b(risk|hazard|emergency)\b', content_lower):
            priority_contexts.append("risk")
        if not priority_contexts:
            priority_contexts.append("general")
        
        return DocumentIntelligenceAssessment(
            document_type=doc_type,
            target_audience=audience,
            information_architecture={
                "structure": structure,
                "sections": sections[:10],
                "interconnections": "pattern-based inference"
            },
            priority_contexts=priority_contexts,
            confidence_score=0.7,
            analysis_timestamp=datetime.now(),
            analysis_method="pattern_analysis"
        )
    
    async def _categorize_knowledge_intelligently(self, content: str, metadata: Dict[str, Any], 
                                                document_intelligence: DocumentIntelligenceAssessment) -> List[IntelligentKnowledgeEntity]:
        """Phase 2: Extract intelligent knowledge entities using pattern matching"""
        entities = []
        
        # Split content into manageable chunks
        chunks = self._split_content_into_chunks(content, max_chunk_size=2000)
        
        for i, chunk in enumerate(chunks):
            chunk_entities = self._analyze_chunk_patterns(chunk, metadata, document_intelligence, chunk_index=i)
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
    
    def _analyze_chunk_patterns(self, chunk: str, metadata: Dict[str, Any], 
                               document_intelligence: DocumentIntelligenceAssessment, 
                               chunk_index: int) -> List[IntelligentKnowledgeEntity]:
        """Analyze a content chunk using pattern matching"""
        entities = []
        chunk_lower = chunk.lower()
        
        # Extract process intelligence
        if re.search(self.process_patterns['workflow'], chunk_lower):
            entities.append(self._create_process_entity(chunk, "Process Workflow", chunk_index))
        
        # Extract compliance & governance
        if re.search(self.compliance_patterns['requirement'], chunk_lower):
            entities.append(self._create_compliance_entity(chunk, "Compliance Requirement", chunk_index))
        
        # Extract quantitative intelligence
        if re.search(self.quantitative_patterns['metric'], chunk_lower):
            entities.append(self._create_metric_entity(chunk, "Performance Metric", chunk_index))
        
        # Extract organizational intelligence
        if re.search(self.organizational_patterns['role'], chunk_lower):
            entities.append(self._create_role_entity(chunk, "Organizational Role", chunk_index))
        
        # Extract knowledge definitions
        if re.search(self.knowledge_patterns['definition'], chunk_lower):
            entities.append(self._create_knowledge_entity(chunk, "Knowledge Definition", chunk_index))
        
        # Extract risk & mitigation
        if re.search(self.risk_patterns['hazard'], chunk_lower):
            entities.append(self._create_risk_entity(chunk, "Risk Assessment", chunk_index))
        
        # Add chunk context
        for entity in entities:
            entity.source_text = chunk
            entity.source_section = f"chunk_{chunk_index + 1}"
        
        return entities
    
    def _create_process_entity(self, chunk: str, identifier: str, chunk_index: int) -> IntelligentKnowledgeEntity:
        """Create a process intelligence entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.PROCESS,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["workflow", "procedure", "operational"],
            priority_level=PriorityLevel.MEDIUM,
            summary=f"Process workflow identified in document section {chunk_index + 1}",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_compliance_entity(self, chunk: str, identifier: str, chunk_index: int) -> IntelligentKnowledgeEntity:
        """Create a compliance & governance entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.COMPLIANCE_REQUIREMENT,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["compliance", "regulation", "requirement"],
            priority_level=PriorityLevel.HIGH,
            summary=f"Compliance requirement identified in document section {chunk_index + 1}",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_metric_entity(self, chunk: str, identifier: str, chunk_index: int) -> IntelligentKnowledgeEntity:
        """Create a quantitative intelligence entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.METRIC,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["metric", "performance", "measurement"],
            priority_level=PriorityLevel.MEDIUM,
            summary=f"Performance metric identified in document section {chunk_index + 1}",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_role_entity(self, chunk: str, identifier: str, chunk_index: int) -> IntelligentKnowledgeEntity:
        """Create an organizational intelligence entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.ROLE,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["role", "responsibility", "organization"],
            priority_level=PriorityLevel.MEDIUM,
            summary=f"Organizational role identified in document section {chunk_index + 1}",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_knowledge_entity(self, chunk: str, identifier: str, chunk_index: int) -> IntelligentKnowledgeEntity:
        """Create a knowledge definition entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.KNOWLEDGE_CONCEPT,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["definition", "concept", "knowledge"],
            priority_level=PriorityLevel.LOW,
            summary=f"Knowledge definition identified in document section {chunk_index + 1}",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _create_risk_entity(self, chunk: str, identifier: str, chunk_index: int) -> IntelligentKnowledgeEntity:
        """Create a risk & mitigation entity"""
        return IntelligentKnowledgeEntity(
            entity_type=EntityType.RISK_ASSESSMENT,
            key_identifier=identifier,
            core_content=chunk[:500] + "..." if len(chunk) > 500 else chunk,
            context_tags=["risk", "hazard", "mitigation"],
            priority_level=PriorityLevel.HIGH,
            summary=f"Risk assessment identified in document section {chunk_index + 1}",
            confidence=0.7,
            source_text="",
            source_page=None,
            source_section="",
            extraction_method="pattern_extraction"
        )
    
    def _consolidate_entities(self, entities: List[IntelligentKnowledgeEntity]) -> List[IntelligentKnowledgeEntity]:
        """Consolidate and deduplicate entities"""
        if not entities:
            return entities
        
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