"""
EXPLAINIUM - Intelligent Knowledge Extractor

A sophisticated knowledge extraction system that produces structured,
categorized knowledge output in the expected format with emojis and
proper categorization for all document types.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# AI and NLP libraries
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy

# Internal imports
from src.logging_config import get_logger

logger = get_logger(__name__)


class IntelligentKnowledgeExtractor:
    """
    Intelligent knowledge extractor that produces structured, categorized output
    in the expected format with emojis and proper knowledge categorization.
    """
    
    def __init__(self):
        self.initialized = False
        self.nlp = None
        self.llm = None
        self.embedding_model = None
        
        # Knowledge categories with emojis and descriptions
        self.knowledge_categories = {
            "concepts": {
                "emoji": "💡",
                "description": "Key ideas, terminology, and fundamental concepts",
                "keywords": ["concept", "definition", "term", "principle", "theory", "methodology"]
            },
            "processes": {
                "emoji": "⚙️",
                "description": "Procedures, workflows, and operational processes",
                "keywords": ["process", "procedure", "workflow", "step", "method", "protocol", "routine"]
            },
            "systems": {
                "emoji": "🖥️",
                "description": "Tools, equipment, software, and technological systems",
                "keywords": ["system", "tool", "equipment", "software", "technology", "platform", "device"]
            },
            "requirements": {
                "emoji": "📋",
                "description": "Regulatory requirements, standards, and compliance needs",
                "keywords": ["requirement", "regulation", "standard", "compliance", "must", "shall", "required"]
            },
            "people": {
                "emoji": "👥",
                "description": "Roles, responsibilities, and organizational structures",
                "keywords": ["role", "responsibility", "person", "team", "department", "position", "authority"]
            },
            "risks": {
                "emoji": "⚠️",
                "description": "Hazards, risks, and safety considerations",
                "keywords": ["risk", "hazard", "danger", "threat", "safety", "warning", "caution"]
            }
        }
    
    async def initialize(self):
        """Initialize the knowledge extractor with required models"""
        try:
            # Initialize spaCy for NLP
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize text classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            
            self.initialized = True
            logger.info("Intelligent Knowledge Extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge extractor: {e}")
            self.initialized = False
    
    async def extract_knowledge(self, content: str, document_type: str = "general") -> Dict[str, Any]:
        """
        Extract structured knowledge from document content
        
        Args:
            content: The document content to analyze
            document_type: Type of document (pdf, image, video, etc.)
            
        Returns:
            Structured knowledge in the expected format
        """
        if not self.initialized:
            await self.initialize()
        
        if not content or len(content.strip()) < 50:
            return {"error": "Insufficient content for analysis"}
        
        try:
            # Clean and preprocess content
            cleaned_content = self._preprocess_content(content)
            
            # Extract knowledge using multiple strategies
            knowledge = {
                "extraction_timestamp": datetime.now().isoformat(),
                "document_type": document_type,
                "content_length": len(cleaned_content),
                "categories": {}
            }
            
            # Extract knowledge for each category
            for category, config in self.knowledge_categories.items():
                category_knowledge = await self._extract_category_knowledge(
                    cleaned_content, category, config
                )
                knowledge["categories"][category] = category_knowledge
            
            # Generate the formatted output
            formatted_output = self._format_knowledge_output(knowledge)
            
            return {
                "raw_knowledge": knowledge,
                "formatted_output": formatted_output,
                "extraction_confidence": self._calculate_overall_confidence(knowledge)
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction: {e}")
            return {"error": f"Knowledge extraction failed: {str(e)}"}
    
    def _preprocess_content(self, content: str) -> str:
        """Clean and preprocess document content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere with analysis
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', ' ', content)
        
        # Normalize spacing around punctuation
        content = re.sub(r'\s+([\.\,\;\:\!\?])', r'\1', content)
        
        return content.strip()
    
    async def _extract_category_knowledge(self, content: str, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge for a specific category"""
        try:
            # Find sentences that contain category keywords
            relevant_sentences = self._find_relevant_sentences(content, config["keywords"])
            
            if not relevant_sentences:
                return {
                    "items": [],
                    "count": 0,
                    "confidence": 0.0
                }
            
            # Extract structured knowledge from relevant sentences
            extracted_items = []
            for sentence in relevant_sentences[:20]:  # Limit to top 20 most relevant
                item = await self._extract_knowledge_item(sentence, category, config)
                if item:
                    extracted_items.append(item)
            
            # Sort by confidence and relevance
            extracted_items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            return {
                "items": extracted_items,
                "count": len(extracted_items),
                "confidence": sum(item.get("confidence", 0) for item in extracted_items) / max(len(extracted_items), 1)
            }
            
        except Exception as e:
            logger.error(f"Error extracting {category} knowledge: {e}")
            return {"items": [], "count": 0, "confidence": 0.0}
    
    def _find_relevant_sentences(self, content: str, keywords: List[str]) -> List[str]:
        """Find sentences that contain relevant keywords"""
        if not self.nlp:
            return []
        
        doc = self.nlp(content)
        relevant_sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if any(keyword.lower() in sent_text.lower() for keyword in keywords):
                relevant_sentences.append(sent_text)
        
        return relevant_sentences
    
    async def _extract_knowledge_item(self, sentence: str, category: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a single knowledge item from a sentence"""
        try:
            # Basic extraction based on category
            if category == "concepts":
                return self._extract_concept_item(sentence, config)
            elif category == "processes":
                return self._extract_process_item(sentence, config)
            elif category == "systems":
                return self._extract_system_item(sentence, config)
            elif category == "requirements":
                return self._extract_requirement_item(sentence, config)
            elif category == "people":
                return self._extract_person_item(sentence, config)
            elif category == "risks":
                return self._extract_risk_item(sentence, config)
            else:
                return self._extract_general_item(sentence, category, config)
                
        except Exception as e:
            logger.error(f"Error extracting knowledge item: {e}")
            return None
    
    def _extract_concept_item(self, sentence: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract concept knowledge item"""
        # Look for definition patterns
        definition_patterns = [
            r'(\w+(?:\s+\w+)*)\s*:\s*(.+)',
            r'(\w+(?:\s+\w+)*)\s+is\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+refers\s+to\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+means\s+(.+)'
        ]
        
        for pattern in definition_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                concept_name = match.group(1).strip()
                description = match.group(2).strip()
                
                return {
                    "name": concept_name,
                    "description": description,
                    "confidence": 0.9,
                    "source_sentence": sentence,
                    "extraction_method": "pattern_matching"
                }
        
        # Fallback: extract noun phrases
        if self.nlp:
            doc = self.nlp(sentence)
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            if noun_chunks:
                return {
                    "name": noun_chunks[0],
                    "description": sentence,
                    "confidence": 0.6,
                    "source_sentence": sentence,
                    "extraction_method": "noun_phrase"
                }
        
        return None
    
    def _extract_process_item(self, sentence: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract process knowledge item"""
        # Look for process patterns
        process_patterns = [
            r'(\d+\.\s*)(.+)',
            r'(step\s+\d+[:\s]+)(.+)',
            r'(procedure\s*[:\s]+)(.+)',
            r'(process\s*[:\s]+)(.+)'
        ]
        
        for pattern in process_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                step_number = match.group(1).strip()
                description = match.group(2).strip()
                
                return {
                    "name": f"Step {step_number}",
                    "description": description,
                    "confidence": 0.85,
                    "source_sentence": sentence,
                    "extraction_method": "pattern_matching"
                }
        
        # Look for action verbs
        action_verbs = ["check", "verify", "ensure", "apply", "use", "place", "conduct", "perform"]
        words = sentence.lower().split()
        for verb in action_verbs:
            if verb in words:
                return {
                    "name": f"Action: {verb.title()}",
                    "description": sentence,
                    "confidence": 0.7,
                    "source_sentence": sentence,
                    "extraction_method": "action_verb"
                }
        
        return None
    
    def _extract_system_item(self, sentence: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract system knowledge item"""
        # Look for equipment and system mentions
        system_keywords = ["equipment", "tool", "system", "device", "machine", "instrument", "apparatus"]
        
        for keyword in system_keywords:
            if keyword.lower() in sentence.lower():
                # Extract the system name
                doc = self.nlp(sentence)
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
                        return {
                            "name": ent.text,
                            "description": sentence,
                            "confidence": 0.8,
                            "source_sentence": sentence,
                            "extraction_method": "named_entity"
                        }
                
                # Fallback: extract around the keyword
                words = sentence.split()
                try:
                    keyword_index = [w.lower() for w in words].index(keyword.lower())
                    if keyword_index > 0:
                        system_name = words[keyword_index - 1]
                        return {
                            "name": system_name,
                            "description": sentence,
                            "confidence": 0.7,
                            "source_sentence": sentence,
                            "extraction_method": "keyword_context"
                        }
                except ValueError:
                    pass
        
        return None
    
    def _extract_requirement_item(self, sentence: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract requirement knowledge item"""
        # Look for requirement patterns
        requirement_patterns = [
            r'(must|shall|required|essential|necessary)\s+(.+)',
            r'(.+)\s+(must|shall|required|essential|necessary)',
            r'(regulation|standard|requirement|compliance)\s*[:\s]+(.+)'
        ]
        
        for pattern in requirement_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                if match.group(1).lower() in ["must", "shall", "required", "essential", "necessary"]:
                    requirement = match.group(2).strip()
                else:
                    requirement = sentence
                
                return {
                    "name": "Regulatory Requirement",
                    "description": requirement,
                    "confidence": 0.9,
                    "source_sentence": sentence,
                    "extraction_method": "requirement_pattern"
                }
        
        return None
    
    def _extract_person_item(self, sentence: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract person/role knowledge item"""
        if not self.nlp:
            return None
        
        doc = self.nlp(sentence)
        
        # Look for person entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return {
                    "name": ent.text,
                    "description": sentence,
                    "confidence": 0.8,
                    "source_sentence": sentence,
                    "extraction_method": "named_entity"
                }
        
        # Look for role patterns
        role_patterns = [
            r'(inspector|supervisor|manager|operator|technician|worker|employee)',
            r'(role|position|title|responsibility)'
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return {
                    "name": match.group(1).title(),
                    "description": sentence,
                    "confidence": 0.7,
                    "source_sentence": sentence,
                    "extraction_method": "role_pattern"
                }
        
        return None
    
    def _extract_risk_item(self, sentence: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk knowledge item"""
        # Look for risk patterns
        risk_patterns = [
            r'(risk|hazard|danger|threat|warning|caution)\s*[:\s]+(.+)',
            r'(.+)\s+(risk|hazard|danger|threat)',
            r'(safety|protective|preventive)\s+(.+)'
        ]
        
        for pattern in risk_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                if match.group(1).lower() in ["risk", "hazard", "danger", "threat", "warning", "caution"]:
                    risk_desc = match.group(2).strip()
                else:
                    risk_desc = sentence
                
                return {
                    "name": "Safety Risk",
                    "description": risk_desc,
                    "confidence": 0.85,
                    "source_sentence": sentence,
                    "extraction_method": "risk_pattern"
                }
        
        return None
    
    def _extract_general_item(self, sentence: str, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract general knowledge item"""
        return {
            "name": f"{category.title()} Item",
            "description": sentence,
            "confidence": 0.5,
            "source_sentence": sentence,
            "extraction_method": "general"
        }
    
    def _format_knowledge_output(self, knowledge: Dict[str, Any]) -> str:
        """Format knowledge into the expected output format with emojis"""
        output_parts = []
        
        for category, config in self.knowledge_categories.items():
            category_data = knowledge["categories"].get(category, {})
            items = category_data.get("items", [])
            
            if items:
                # Add category header with emoji
                output_parts.append(f"{config['emoji']} {config['description'].title()}")
                
                # Add items
                for item in items[:10]:  # Limit to top 10 items per category
                    name = item.get("name", "Unknown")
                    description = item.get("description", "")
                    
                    # Format the item
                    if len(description) > 200:
                        description = description[:200] + "..."
                    
                    output_parts.append(f"{name}: {description}")
                
                output_parts.append("")  # Empty line between categories
        
        return "\n".join(output_parts)
    
    def _calculate_overall_confidence(self, knowledge: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the extraction"""
        total_confidence = 0.0
        total_categories = 0
        
        for category_data in knowledge["categories"].values():
            if category_data.get("count", 0) > 0:
                total_confidence += category_data.get("confidence", 0.0)
                total_categories += 1
        
        if total_categories == 0:
            return 0.0
        
        return total_confidence / total_categories
    
    async def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract knowledge from image content (placeholder for OCR integration)"""
        # This would integrate with OCR to extract text first
        # For now, return a placeholder
        return {
            "error": "Image processing not yet implemented",
            "suggestion": "Use OCR to extract text first, then pass to extract_knowledge"
        }
    
    async def extract_from_video(self, video_path: str) -> Dict[str, Any]:
        """Extract knowledge from video content (placeholder for video analysis)"""
        # This would integrate with video analysis to extract audio/text first
        # For now, return a placeholder
        return {
            "error": "Video processing not yet implemented",
            "suggestion": "Extract audio/text from video first, then pass to extract_knowledge"
        }