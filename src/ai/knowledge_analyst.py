"""
EXPLAINIUM - AI Knowledge Analyst

Transforms any unstructured document into a structured, actionable, and synthesized 
knowledge base using a three-phase framework:

Phase 1: Holistic Comprehension
Phase 2: Thematic Abstraction  
Phase 3: Synthesis & Structured Output
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

# AI libraries
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

from src.logging_config import get_logger
from src.core.config import AIConfig

logger = get_logger(__name__)


class DocumentType(Enum):
    """Document type classifications"""
    MANUAL = "manual"
    CONTRACT = "contract"
    REPORT = "report"
    POLICY = "policy"
    PROCEDURE = "procedure"
    SPECIFICATION = "specification"
    TRAINING = "training"
    REFERENCE = "reference"
    COMMUNICATION = "communication"
    OTHER = "other"


class InformationType(Enum):
    """Information type categories for Phase 2"""
    PROCESSES_WORKFLOWS = "processes_workflows"
    POLICIES_RULES_REQUIREMENTS = "policies_rules_requirements"
    KEY_DATA_METRICS = "key_data_metrics"
    ROLES_RESPONSIBILITIES = "roles_responsibilities"
    DEFINITIONS = "definitions"
    RISKS_CORRECTIVE_ACTIONS = "risks_corrective_actions"


@dataclass
class DocumentContext:
    """Results from Phase 1: Holistic Comprehension"""
    primary_purpose: str
    document_type: DocumentType
    intended_audience: str
    structure_analysis: Dict[str, Any]
    key_sections: List[str]
    complexity_level: str  # "basic", "intermediate", "advanced"
    domain: str  # e.g., "technical", "legal", "operational"
    confidence: float


@dataclass
class ThematicBucket:
    """Container for categorized information from Phase 2"""
    information_type: InformationType
    items: List[Dict[str, Any]]
    priority: str  # "critical", "important", "informational"
    relationships: List[str]  # relationships to other buckets


@dataclass
class SynthesizedKnowledge:
    """Final structured output from Phase 3"""
    document_context: DocumentContext
    thematic_buckets: Dict[InformationType, ThematicBucket]
    synthesized_summary: str
    actionable_insights: List[str]
    key_takeaways: List[str]
    structured_markdown: str
    metadata: Dict[str, Any]


class AIKnowledgeAnalyst:
    """
    AI Knowledge Analyst that transforms unstructured documents into 
    structured, actionable knowledge bases using a three-phase framework.
    """
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.llm = None
        self.initialized = False
        
        # Phase 1: Document type detection patterns
        self.document_patterns = {
            DocumentType.MANUAL: [
                r"manual", r"guide", r"instructions", r"how to", r"step by step",
                r"procedure", r"handbook", r"tutorial"
            ],
            DocumentType.CONTRACT: [
                r"agreement", r"contract", r"terms", r"conditions", r"parties",
                r"whereas", r"therefore", r"liability", r"indemnification"
            ],
            DocumentType.REPORT: [
                r"report", r"analysis", r"findings", r"results", r"summary",
                r"conclusion", r"recommendation", r"executive summary"
            ],
            DocumentType.POLICY: [
                r"policy", r"standard", r"guideline", r"regulation", r"compliance",
                r"requirement", r"shall", r"must", r"mandatory"
            ],
            DocumentType.PROCEDURE: [
                r"procedure", r"process", r"workflow", r"steps", r"sequence",
                r"protocol", r"methodology", r"approach"
            ]
        }
        
        # Phase 2: Information extraction patterns
        self.extraction_patterns = {
            InformationType.PROCESSES_WORKFLOWS: [
                r"step \d+", r"process", r"workflow", r"procedure", r"sequence",
                r"first.*then.*finally", r"begin.*complete", r"start.*finish"
            ],
            InformationType.POLICIES_RULES_REQUIREMENTS: [
                r"must", r"shall", r"required", r"mandatory", r"policy",
                r"rule", r"regulation", r"standard", r"guideline", r"compliance"
            ],
            InformationType.KEY_DATA_METRICS: [
                r"\d+%", r"\$\d+", r"\d+\s*(days|hours|minutes)", r"temperature.*\d+",
                r"pressure.*\d+", r"\d+\s*(kg|lbs|tons)", r"deadline.*\d+"
            ],
            InformationType.ROLES_RESPONSIBILITIES: [
                r"responsible for", r"accountable", r"role", r"position",
                r"manager", r"supervisor", r"coordinator", r"lead", r"team"
            ],
            InformationType.DEFINITIONS: [
                r"defined as", r"means", r"refers to", r"definition",
                r"terminology", r"glossary", r"acronym", r"abbreviation"
            ],
            InformationType.RISKS_CORRECTIVE_ACTIONS: [
                r"risk", r"hazard", r"danger", r"warning", r"caution",
                r"corrective action", r"mitigation", r"prevention", r"emergency"
            ]
        }

    async def initialize(self):
        """Initialize the AI Knowledge Analyst"""
        try:
            if LLAMA_AVAILABLE and self.config.llm_path:
                model_path = f"{self.config.llm_path}/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    n_batch=512,
                    n_threads=8,
                    verbose=False
                )
            self.initialized = True
            logger.info("AI Knowledge Analyst initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Knowledge Analyst: {e}")
            raise

    async def analyze_document(self, content: str, metadata: Dict[str, Any] = None) -> SynthesizedKnowledge:
        """
        Transform unstructured document into structured knowledge base
        using the three-phase framework.
        """
        if not self.initialized:
            await self.initialize()
        
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty")
        
        metadata = metadata or {}
        
        logger.info("Starting AI Knowledge Analysis with three-phase framework")
        
        # Phase 1: Holistic Comprehension
        logger.info("Phase 1: Holistic Comprehension")
        document_context = await self._phase1_holistic_comprehension(content, metadata)
        
        # Phase 2: Thematic Abstraction
        logger.info("Phase 2: Thematic Abstraction")
        thematic_buckets = await self._phase2_thematic_abstraction(content, document_context)
        
        # Phase 3: Synthesis & Structured Output
        logger.info("Phase 3: Synthesis & Structured Output")
        synthesized_knowledge = await self._phase3_synthesis_output(
            content, document_context, thematic_buckets, metadata
        )
        
        logger.info("AI Knowledge Analysis completed successfully")
        return synthesized_knowledge

    async def _phase1_holistic_comprehension(self, content: str, metadata: Dict[str, Any]) -> DocumentContext:
        """
        Phase 1: Analyze the entire document to understand its primary purpose,
        intended audience, and overall structure.
        """
        logger.info("Analyzing document holistically...")
        
        # Detect document type
        document_type = self._detect_document_type(content)
        
        # Analyze structure
        structure_analysis = self._analyze_document_structure(content)
        
        # Determine primary purpose using AI
        primary_purpose = await self._determine_primary_purpose(content, document_type)
        
        # Identify intended audience
        intended_audience = await self._identify_intended_audience(content, document_type)
        
        # Extract key sections
        key_sections = self._extract_key_sections(content, structure_analysis)
        
        # Assess complexity and domain
        complexity_level = self._assess_complexity(content)
        domain = self._identify_domain(content)
        
        return DocumentContext(
            primary_purpose=primary_purpose,
            document_type=document_type,
            intended_audience=intended_audience,
            structure_analysis=structure_analysis,
            key_sections=key_sections,
            complexity_level=complexity_level,
            domain=domain,
            confidence=0.85  # Base confidence, can be improved with more analysis
        )

    async def _phase2_thematic_abstraction(self, content: str, context: DocumentContext) -> Dict[InformationType, ThematicBucket]:
        """
        Phase 2: Identify and categorize information into thematic buckets:
        - Processes & Workflows
        - Policies, Rules & Requirements  
        - Key Data & Metrics
        - Roles & Responsibilities
        - Definitions
        - Risks & Corrective Actions
        """
        logger.info("Performing thematic abstraction...")
        
        buckets = {}
        
        for info_type in InformationType:
            logger.info(f"Extracting {info_type.value}...")
            
            # Extract information for this category
            items = await self._extract_information_by_type(content, info_type, context)
            
            # Determine priority
            priority = self._determine_priority(items, info_type, context)
            
            # Find relationships
            relationships = self._find_bucket_relationships(items, info_type)
            
            buckets[info_type] = ThematicBucket(
                information_type=info_type,
                items=items,
                priority=priority,
                relationships=relationships
            )
        
        return buckets

    async def _phase3_synthesis_output(self, content: str, context: DocumentContext, 
                                     buckets: Dict[InformationType, ThematicBucket],
                                     metadata: Dict[str, Any]) -> SynthesizedKnowledge:
        """
        Phase 3: Synthesize extracted information into coherent, structured output
        with clear Markdown formatting.
        """
        logger.info("Synthesizing knowledge into structured output...")
        
        # Generate synthesized summary
        synthesized_summary = await self._generate_synthesized_summary(context, buckets)
        
        # Extract actionable insights
        actionable_insights = await self._extract_actionable_insights(buckets, context)
        
        # Generate key takeaways
        key_takeaways = await self._generate_key_takeaways(buckets, context)
        
        # Create structured markdown output
        structured_markdown = self._create_structured_markdown(
            context, buckets, synthesized_summary, actionable_insights, key_takeaways
        )
        
        return SynthesizedKnowledge(
            document_context=context,
            thematic_buckets=buckets,
            synthesized_summary=synthesized_summary,
            actionable_insights=actionable_insights,
            key_takeaways=key_takeaways,
            structured_markdown=structured_markdown,
            metadata={
                **metadata,
                'analysis_timestamp': datetime.now().isoformat(),
                'analyst_version': '2.0',
                'framework_version': '3-phase'
            }
        )

    def _detect_document_type(self, content: str) -> DocumentType:
        """Detect document type based on content patterns"""
        content_lower = content.lower()
        
        scores = {}
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            scores[doc_type] = score
        
        # Return the type with highest score, or OTHER if no clear match
        if not scores or max(scores.values()) == 0:
            return DocumentType.OTHER
        
        return max(scores, key=scores.get)

    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure and organization"""
        lines = content.split('\n')
        
        structure = {
            'total_lines': len(lines),
            'total_chars': len(content),
            'paragraphs': len([line for line in lines if line.strip()]),
            'headings': [],
            'lists': 0,
            'tables': 0,
            'sections': []
        }
        
        # Detect headings (lines that look like titles)
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headings
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headings
            r'^\d+\.\s+[A-Z]',   # Numbered sections
            r'^[A-Z][^.!?]*:$'   # Title with colon
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for headings
            for pattern in heading_patterns:
                if re.match(pattern, line_stripped):
                    structure['headings'].append({
                        'line': i + 1,
                        'text': line_stripped,
                        'level': self._determine_heading_level(line_stripped)
                    })
                    break
            
            # Count lists
            if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                structure['lists'] += 1
            
            # Detect tables (simple heuristic)
            if '|' in line and line.count('|') >= 2:
                structure['tables'] += 1
        
        return structure

    def _determine_heading_level(self, heading: str) -> int:
        """Determine heading level (1-6)"""
        if heading.startswith('#'):
            return min(heading.count('#'), 6)
        elif heading.isupper():
            return 1
        elif re.match(r'^\d+\.\s+', heading):
            return 2
        else:
            return 3

    async def _determine_primary_purpose(self, content: str, doc_type: DocumentType) -> str:
        """Use AI to determine the document's primary purpose"""
        if not self.llm:
            # Fallback to pattern-based analysis
            return self._fallback_purpose_detection(content, doc_type)
        
        prompt = f"""
        Analyze this document and determine its PRIMARY PURPOSE in one clear sentence.
        
        Document type: {doc_type.value}
        Content preview: {content[:1000]}...
        
        Purpose (one sentence):"""
        
        try:
            response = self.llm(prompt, max_tokens=100, temperature=0.1)
            purpose = response['choices'][0]['text'].strip()
            return purpose
        except Exception as e:
            logger.warning(f"AI purpose detection failed: {e}")
            return self._fallback_purpose_detection(content, doc_type)

    def _fallback_purpose_detection(self, content: str, doc_type: DocumentType) -> str:
        """Fallback purpose detection using patterns"""
        content_lower = content.lower()
        
        purpose_indicators = {
            DocumentType.MANUAL: "to provide instructions and guidance",
            DocumentType.CONTRACT: "to establish legal terms and obligations",
            DocumentType.REPORT: "to present findings and analysis",
            DocumentType.POLICY: "to establish rules and standards",
            DocumentType.PROCEDURE: "to define step-by-step processes"
        }
        
        return purpose_indicators.get(doc_type, "to convey information")

    async def _identify_intended_audience(self, content: str, doc_type: DocumentType) -> str:
        """Identify the intended audience for the document"""
        content_lower = content.lower()
        
        # Pattern-based audience detection
        audience_patterns = {
            "technical staff": [r"technical", r"engineer", r"developer", r"system", r"API"],
            "management": [r"manager", r"executive", r"director", r"leadership", r"strategy"],
            "general employees": [r"employee", r"staff", r"team member", r"personnel"],
            "customers": [r"customer", r"client", r"user", r"end user"],
            "legal/compliance": [r"legal", r"compliance", r"audit", r"regulatory"],
            "operations": [r"operation", r"production", r"maintenance", r"support"]
        }
        
        scores = {}
        for audience, patterns in audience_patterns.items():
            score = sum(len(re.findall(pattern, content_lower)) for pattern in patterns)
            scores[audience] = score
        
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return "general audience"

    async def _extract_information_by_type(self, content: str, info_type: InformationType, 
                                         context: DocumentContext) -> List[Dict[str, Any]]:
        """Extract information for a specific thematic category"""
        
        patterns = self.extraction_patterns.get(info_type, [])
        items = []
        
        if info_type == InformationType.PROCESSES_WORKFLOWS:
            items = await self._extract_processes_workflows(content, context)
        elif info_type == InformationType.POLICIES_RULES_REQUIREMENTS:
            items = await self._extract_policies_rules(content, context)
        elif info_type == InformationType.KEY_DATA_METRICS:
            items = await self._extract_key_data_metrics(content, context)
        elif info_type == InformationType.ROLES_RESPONSIBILITIES:
            items = await self._extract_roles_responsibilities(content, context)
        elif info_type == InformationType.DEFINITIONS:
            items = await self._extract_definitions(content, context)
        elif info_type == InformationType.RISKS_CORRECTIVE_ACTIONS:
            items = await self._extract_risks_actions(content, context)
        
        return items

    async def _extract_processes_workflows(self, content: str, context: DocumentContext) -> List[Dict[str, Any]]:
        """Extract step-by-step instructions or procedures"""
        processes = []
        
        # Pattern-based extraction for numbered steps
        step_pattern = r'(?:step\s+)?(\d+)[\.\)]\s*([^.\n]+(?:\.[^.\n]*)*)'
        matches = re.finditer(step_pattern, content, re.IGNORECASE)
        
        current_process = None
        for match in matches:
            step_num = match.group(1)
            step_text = match.group(2).strip()
            
            if current_process is None or int(step_num) == 1:
                # Start new process
                current_process = {
                    'type': 'workflow',
                    'title': f"Process starting at step {step_num}",
                    'steps': [],
                    'confidence': 0.8
                }
                processes.append(current_process)
            
            current_process['steps'].append({
                'number': int(step_num),
                'description': step_text,
                'actionable': True
            })
        
        # Also look for workflow keywords
        workflow_sections = self._extract_workflow_sections(content)
        processes.extend(workflow_sections)
        
        return processes

    async def _extract_policies_rules(self, content: str, context: DocumentContext) -> List[Dict[str, Any]]:
        """Extract mandatory guidelines, standards, or constraints"""
        policies = []
        
        # Look for mandatory language
        mandatory_pattern = r'(must|shall|required|mandatory|obligatory)[^.!?]*[.!?]'
        matches = re.finditer(mandatory_pattern, content, re.IGNORECASE)
        
        for match in matches:
            rule_text = match.group(0).strip()
            policies.append({
                'type': 'requirement',
                'text': rule_text,
                'mandatory': True,
                'confidence': 0.9
            })
        
        # Look for policy statements
        policy_pattern = r'policy\s*:\s*([^.\n]+(?:\.[^.\n]*)*)'
        matches = re.finditer(policy_pattern, content, re.IGNORECASE)
        
        for match in matches:
            policy_text = match.group(1).strip()
            policies.append({
                'type': 'policy',
                'text': policy_text,
                'mandatory': False,
                'confidence': 0.8
            })
        
        return policies

    async def _extract_key_data_metrics(self, content: str, context: DocumentContext) -> List[Dict[str, Any]]:
        """Extract specific, quantifiable data points"""
        metrics = []
        
        # Numerical patterns
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage'),
            (r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', 'currency'),
            (r'(\d+(?:\.\d+)?)\s*(kg|lbs|tons|grams)', 'weight'),
            (r'(\d+(?:\.\d+)?)\s*(°C|°F|celsius|fahrenheit)', 'temperature'),
            (r'(\d+(?:\.\d+)?)\s*(hours?|minutes?|seconds?|days?)', 'time'),
            (r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'date'),
            (r'(\d+(?:\.\d+)?)\s*(psi|bar|pascal)', 'pressure')
        ]
        
        for pattern, metric_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context_text = content[start:end].strip()
                
                metrics.append({
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else '',
                    'type': metric_type,
                    'context': context_text,
                    'confidence': 0.95
                })
        
        return metrics

    async def _extract_roles_responsibilities(self, content: str, context: DocumentContext) -> List[Dict[str, Any]]:
        """Extract roles and responsibilities"""
        roles = []
        
        # Pattern for role assignments
        role_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?(?:responsible\s+for|accountable\s+for|shall|must)\s+([^.\n]+)'
        matches = re.finditer(role_pattern, content)
        
        for match in matches:
            role = match.group(1).strip()
            responsibility = match.group(2).strip()
            
            roles.append({
                'role': role,
                'responsibility': responsibility,
                'type': 'assignment',
                'confidence': 0.8
            })
        
        return roles

    async def _extract_definitions(self, content: str, context: DocumentContext) -> List[Dict[str, Any]]:
        """Extract definitions and explanations of key terms"""
        definitions = []
        
        # Pattern for definitions
        definition_patterns = [
            r'([A-Z][a-zA-Z\s]+)\s+(?:means|is defined as|refers to)\s+([^.\n]+)',
            r'([A-Z][A-Z\s]+)\s*:\s*([^.\n]+)',  # ACRONYM: definition
            r'\"([^\"]+)\"\s+means\s+([^.\n]+)'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                definitions.append({
                    'term': term,
                    'definition': definition,
                    'type': 'definition',
                    'confidence': 0.85
                })
        
        return definitions

    async def _extract_risks_actions(self, content: str, context: DocumentContext) -> List[Dict[str, Any]]:
        """Extract potential issues and their prescribed solutions"""
        risks = []
        
        # Risk identification patterns
        risk_pattern = r'(?:risk|hazard|danger|warning|caution)[^.!?]*[.!?]'
        matches = re.finditer(risk_pattern, content, re.IGNORECASE)
        
        for match in matches:
            risk_text = match.group(0).strip()
            
            # Look for corrective actions nearby
            risk_start = match.start()
            surrounding_text = content[max(0, risk_start-200):risk_start+400]
            
            corrective_actions = self._find_corrective_actions(surrounding_text)
            
            risks.append({
                'risk': risk_text,
                'corrective_actions': corrective_actions,
                'type': 'risk_assessment',
                'confidence': 0.8
            })
        
        return risks

    def _find_corrective_actions(self, text: str) -> List[str]:
        """Find corrective actions in text"""
        action_patterns = [
            r'(?:to\s+(?:prevent|avoid|mitigate|address|resolve))[^.!?]*[.!?]',
            r'(?:corrective\s+action|mitigation|prevention)[^.!?]*[.!?]',
            r'(?:should|must|shall)\s+(?:immediately|promptly)[^.!?]*[.!?]'
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                actions.append(match.group(0).strip())
        
        return actions

    def _extract_workflow_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract workflow sections that aren't numbered"""
        workflows = []
        
        # Look for sequential language
        sequential_patterns = [
            r'first[^.]*then[^.]*(?:finally|last)[^.]*',
            r'begin[^.]*(?:next|then)[^.]*(?:complete|finish)[^.]*',
            r'start[^.]*(?:proceed|continue)[^.]*(?:end|conclude)[^.]*'
        ]
        
        for pattern in sequential_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                workflow_text = match.group(0).strip()
                workflows.append({
                    'type': 'workflow',
                    'title': 'Sequential Process',
                    'description': workflow_text,
                    'confidence': 0.7
                })
        
        return workflows

    def _extract_key_sections(self, content: str, structure: Dict[str, Any]) -> List[str]:
        """Extract key sections from document structure"""
        sections = []
        
        for heading in structure.get('headings', []):
            sections.append(heading['text'])
        
        # If no headings found, create logical sections
        if not sections:
            lines = content.split('\n')
            content_lines = [line for line in lines if line.strip()]
            
            # Create sections based on content length
            if len(content_lines) > 50:
                sections = ['Introduction', 'Main Content', 'Conclusion']
            elif len(content_lines) > 20:
                sections = ['Overview', 'Details']
            else:
                sections = ['Content']
        
        return sections

    def _assess_complexity(self, content: str) -> str:
        """Assess document complexity level"""
        # Simple heuristics for complexity
        word_count = len(content.split())
        avg_sentence_length = len(content.split()) / max(1, content.count('.'))
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', content))  # Acronyms
        
        if word_count > 5000 or avg_sentence_length > 25 or technical_terms > 20:
            return "advanced"
        elif word_count > 1000 or avg_sentence_length > 15 or technical_terms > 5:
            return "intermediate"
        else:
            return "basic"

    def _identify_domain(self, content: str) -> str:
        """Identify the domain/field of the document"""
        content_lower = content.lower()
        
        domain_keywords = {
            "technical": ["software", "system", "code", "API", "database", "server"],
            "legal": ["contract", "agreement", "liability", "clause", "jurisdiction"],
            "medical": ["patient", "medical", "diagnosis", "treatment", "clinical"],
            "financial": ["budget", "cost", "revenue", "financial", "accounting"],
            "operational": ["process", "procedure", "workflow", "operations"],
            "hr": ["employee", "personnel", "hiring", "performance", "training"],
            "safety": ["safety", "hazard", "risk", "emergency", "protection"]
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            scores[domain] = score
        
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return "general"

    def _determine_priority(self, items: List[Dict[str, Any]], info_type: InformationType, 
                          context: DocumentContext) -> str:
        """Determine priority level for a thematic bucket"""
        if not items:
            return "informational"
        
        # High priority types
        high_priority_types = [
            InformationType.POLICIES_RULES_REQUIREMENTS,
            InformationType.RISKS_CORRECTIVE_ACTIONS
        ]
        
        if info_type in high_priority_types:
            return "critical"
        
        # Check for mandatory language
        mandatory_count = sum(1 for item in items 
                            if item.get('mandatory', False) or 
                               any(word in str(item).lower() 
                                   for word in ['must', 'shall', 'required']))
        
        if mandatory_count > 0:
            return "critical"
        elif len(items) > 5:
            return "important"
        else:
            return "informational"

    def _find_bucket_relationships(self, items: List[Dict[str, Any]], 
                                 info_type: InformationType) -> List[str]:
        """Find relationships between thematic buckets"""
        relationships = []
        
        # Simple relationship detection based on content overlap
        if info_type == InformationType.PROCESSES_WORKFLOWS:
            relationships.append("implements_policies")
            relationships.append("requires_roles")
        elif info_type == InformationType.POLICIES_RULES_REQUIREMENTS:
            relationships.append("governs_processes")
            relationships.append("defines_roles")
        elif info_type == InformationType.RISKS_CORRECTIVE_ACTIONS:
            relationships.append("mitigated_by_processes")
            relationships.append("requires_policies")
        
        return relationships

    async def _generate_synthesized_summary(self, context: DocumentContext, 
                                          buckets: Dict[InformationType, ThematicBucket]) -> str:
        """Generate a synthesized summary of the document"""
        
        summary_parts = [
            f"This {context.document_type.value} document serves {context.primary_purpose}.",
            f"It is intended for {context.intended_audience} and has {context.complexity_level} complexity level."
        ]
        
        # Add bucket summaries
        for info_type, bucket in buckets.items():
            if bucket.items:
                count = len(bucket.items)
                summary_parts.append(
                    f"Contains {count} {info_type.value.replace('_', ' ')} items of {bucket.priority} priority."
                )
        
        return " ".join(summary_parts)

    async def _extract_actionable_insights(self, buckets: Dict[InformationType, ThematicBucket],
                                         context: DocumentContext) -> List[str]:
        """Extract actionable insights from the analysis"""
        insights = []
        
        # Critical items need immediate attention
        critical_buckets = [bucket for bucket in buckets.values() if bucket.priority == "critical"]
        if critical_buckets:
            insights.append("Critical compliance and safety requirements identified that need immediate attention")
        
        # Process optimization opportunities
        process_bucket = buckets.get(InformationType.PROCESSES_WORKFLOWS)
        if process_bucket and len(process_bucket.items) > 3:
            insights.append("Multiple processes identified - consider creating process optimization roadmap")
        
        # Risk mitigation
        risk_bucket = buckets.get(InformationType.RISKS_CORRECTIVE_ACTIONS)
        if risk_bucket and risk_bucket.items:
            insights.append("Risk factors identified with corresponding mitigation strategies")
        
        return insights

    async def _generate_key_takeaways(self, buckets: Dict[InformationType, ThematicBucket],
                                    context: DocumentContext) -> List[str]:
        """Generate key takeaways from the analysis"""
        takeaways = []
        
        # Document-type specific takeaways
        if context.document_type == DocumentType.MANUAL:
            takeaways.append("Comprehensive operational guidance provided")
        elif context.document_type == DocumentType.POLICY:
            takeaways.append("Regulatory compliance framework established")
        elif context.document_type == DocumentType.REPORT:
            takeaways.append("Data-driven insights and recommendations presented")
        
        # Content-based takeaways
        total_items = sum(len(bucket.items) for bucket in buckets.values())
        takeaways.append(f"Total of {total_items} structured knowledge items extracted")
        
        # Priority-based takeaways
        critical_count = sum(1 for bucket in buckets.values() if bucket.priority == "critical")
        if critical_count > 0:
            takeaways.append(f"{critical_count} critical knowledge areas require immediate attention")
        
        return takeaways

    def _create_structured_markdown(self, context: DocumentContext, 
                                  buckets: Dict[InformationType, ThematicBucket],
                                  summary: str, insights: List[str], 
                                  takeaways: List[str]) -> str:
        """Create structured Markdown output with clear formatting"""
        
        markdown_parts = []
        
        # Header
        markdown_parts.append("# Knowledge Analysis Report")
        markdown_parts.append("")
        
        # Document Overview
        markdown_parts.append("## Document Overview")
        markdown_parts.append(f"**Type:** {context.document_type.value.title()}")
        markdown_parts.append(f"**Primary Purpose:** {context.primary_purpose}")
        markdown_parts.append(f"**Intended Audience:** {context.intended_audience}")
        markdown_parts.append(f"**Complexity Level:** {context.complexity_level.title()}")
        markdown_parts.append(f"**Domain:** {context.domain.title()}")
        markdown_parts.append("")
        
        # Executive Summary
        markdown_parts.append("## Executive Summary")
        markdown_parts.append(summary)
        markdown_parts.append("")
        
        # Key Takeaways
        if takeaways:
            markdown_parts.append("## Key Takeaways")
            for takeaway in takeaways:
                markdown_parts.append(f"- {takeaway}")
            markdown_parts.append("")
        
        # Actionable Insights
        if insights:
            markdown_parts.append("## Actionable Insights")
            for insight in insights:
                markdown_parts.append(f"- **{insight}**")
            markdown_parts.append("")
        
        # Detailed Knowledge Categories
        markdown_parts.append("## Detailed Knowledge Analysis")
        markdown_parts.append("")
        
        # Sort buckets by priority
        priority_order = {"critical": 0, "important": 1, "informational": 2}
        sorted_buckets = sorted(buckets.items(), 
                              key=lambda x: priority_order.get(x[1].priority, 3))
        
        for info_type, bucket in sorted_buckets:
            if not bucket.items:
                continue
                
            # Section header with priority indicator
            priority_emoji = {"critical": "🔴", "important": "🟡", "informational": "🔵"}
            emoji = priority_emoji.get(bucket.priority, "⚪")
            
            section_title = info_type.value.replace('_', ' ').title()
            markdown_parts.append(f"### {emoji} {section_title} ({bucket.priority.title()})")
            markdown_parts.append("")
            
            # Add items based on type
            if info_type == InformationType.PROCESSES_WORKFLOWS:
                markdown_parts.extend(self._format_processes(bucket.items))
            elif info_type == InformationType.POLICIES_RULES_REQUIREMENTS:
                markdown_parts.extend(self._format_policies(bucket.items))
            elif info_type == InformationType.KEY_DATA_METRICS:
                markdown_parts.extend(self._format_metrics(bucket.items))
            elif info_type == InformationType.ROLES_RESPONSIBILITIES:
                markdown_parts.extend(self._format_roles(bucket.items))
            elif info_type == InformationType.DEFINITIONS:
                markdown_parts.extend(self._format_definitions(bucket.items))
            elif info_type == InformationType.RISKS_CORRECTIVE_ACTIONS:
                markdown_parts.extend(self._format_risks(bucket.items))
            
            markdown_parts.append("")
        
        return "\n".join(markdown_parts)

    def _format_processes(self, items: List[Dict[str, Any]]) -> List[str]:
        """Format process and workflow items"""
        formatted = []
        
        for item in items:
            if item.get('type') == 'workflow' and 'steps' in item:
                formatted.append(f"**{item.get('title', 'Process')}**")
                for step in item['steps']:
                    formatted.append(f"{step['number']}. {step['description']}")
                formatted.append("")
            else:
                formatted.append(f"- {item.get('description', str(item))}")
        
        return formatted

    def _format_policies(self, items: List[Dict[str, Any]]) -> List[str]:
        """Format policy and requirements items"""
        formatted = []
        
        for item in items:
            text = item.get('text', str(item))
            if item.get('mandatory', False):
                formatted.append(f"- **MANDATORY:** {text}")
            else:
                formatted.append(f"- {text}")
        
        return formatted

    def _format_metrics(self, items: List[Dict[str, Any]]) -> List[str]:
        """Format key data and metrics"""
        formatted = []
        
        # Group by type
        by_type = {}
        for item in items:
            metric_type = item.get('type', 'general')
            if metric_type not in by_type:
                by_type[metric_type] = []
            by_type[metric_type].append(item)
        
        for metric_type, metrics in by_type.items():
            formatted.append(f"**{metric_type.title()}:**")
            for metric in metrics:
                value = metric.get('value', '')
                unit = metric.get('unit', '')
                context = metric.get('context', '')[:100] + "..." if len(metric.get('context', '')) > 100 else metric.get('context', '')
                formatted.append(f"- {value} {unit} - {context}")
            formatted.append("")
        
        return formatted

    def _format_roles(self, items: List[Dict[str, Any]]) -> List[str]:
        """Format roles and responsibilities"""
        formatted = []
        
        for item in items:
            role = item.get('role', 'Unknown Role')
            responsibility = item.get('responsibility', 'No description')
            formatted.append(f"- **{role}:** {responsibility}")
        
        return formatted

    def _format_definitions(self, items: List[Dict[str, Any]]) -> List[str]:
        """Format definitions and terms"""
        formatted = []
        
        for item in items:
            term = item.get('term', 'Unknown Term')
            definition = item.get('definition', 'No definition')
            formatted.append(f"- **{term}:** {definition}")
        
        return formatted

    def _format_risks(self, items: List[Dict[str, Any]]) -> List[str]:
        """Format risks and corrective actions"""
        formatted = []
        
        for item in items:
            risk = item.get('risk', 'Unknown Risk')
            actions = item.get('corrective_actions', [])
            
            formatted.append(f"**Risk:** {risk}")
            if actions:
                formatted.append("**Corrective Actions:**")
                for action in actions:
                    formatted.append(f"  - {action}")
            formatted.append("")
        
        return formatted