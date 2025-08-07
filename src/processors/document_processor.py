import os
import requests
import json
import ffmpeg
import whisper
import re
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.ai.ner_extractor import ner_extractor
from src.ai.classifier import classifier
from src.ai.keyphrase_extractor import keyphrase_extractor
from src.exceptions import ProcessingError, AIError, ServiceUnavailableError
from src.logging_config import get_logger, log_processing_step, log_error
from src.config import config_manager

logger = get_logger(__name__)

# Get Tika URL from configuration
TIKA_SERVER_URL = config_manager.get_tika_url()

# Define the candidate labels for document classification
CANDIDATE_LABELS = [
    "Operational Procedures",
    "Safety Documentation",
    "Training Materials",
    "Technical Specifications",
    "Maintenance Guides",
    "Quality Standards"
]

# Image processing components
class ImageFormat(Enum):
    """Supported image formats for OCR processing."""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    TIF = "tif"
    WEBP = "webp"

@dataclass
class OCRResult:
    """Structured result from OCR processing."""
    text: str
    confidence: float
    text_blocks: List[Dict]
    processing_time: float
    image_quality_score: float

class ImagePreprocessor:
    """Optimized image preprocessing for OCR accuracy."""
    
    def __init__(self):
        self.min_confidence = 30
        self.optimal_dpi = 300
        
    def _validate_image(self, image: np.ndarray) -> None:
        """Validate image quality and dimensions."""
        if image is None:
            raise ProcessingError("Invalid image: image is None")
        
        if image.size == 0:
            raise ProcessingError("Invalid image: image is empty")
        
        if len(image.shape) < 2:
            raise ProcessingError("Invalid image: not a valid image format")
    
    def _resize_for_ocr(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """Resize image to optimal DPI for OCR."""
        height, width = image.shape[:2]
        
        # Calculate optimal size (assuming 96 DPI baseline)
        scale_factor = target_dpi / 96.0
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        if new_width > 4000 or new_height > 4000:
            # Limit maximum size to prevent memory issues
            scale_factor = min(4000 / width, 4000 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges."""
        if len(image.shape) == 3:
            # For color images, denoise each channel
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # For grayscale images
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Create binary image for better OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use adaptive thresholding for better results
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Clean up small artifacts
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Complete image preprocessing pipeline."""
        self._validate_image(image)
        
        # Resize to optimal DPI
        resized = self._resize_for_ocr(image)
        
        # Enhance contrast
        enhanced = self._enhance_contrast(resized)
        
        # Denoise
        denoised = self._denoise_image(enhanced)
        
        # Binarize
        binary = self._binarize_image(denoised)
        
        return binary

class OptimizedOCRProcessor:
    """High-performance OCR processor with multiple strategies."""
    
    def __init__(self, tesseract_path: str = "/usr/bin/tesseract"):
        self.tesseract_path = tesseract_path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Optimized OCR configurations for different scenarios
        self.ocr_configs = {
            'default': '--oem 3 --psm 6',
            'single_line': '--oem 3 --psm 7',
            'single_word': '--oem 3 --psm 8',
            'sparse_text': '--oem 3 --psm 6',
            'uniform_block': '--oem 3 --psm 6'
        }
        
        self.preprocessor = ImagePreprocessor()
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score for OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast
        contrast = gray.std()
        
        # Normalize scores (0-1 range)
        sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize to reasonable range
        contrast_score = min(contrast / 100, 1.0)
        
        # Combined quality score
        quality_score = (sharpness_score * 0.6 + contrast_score * 0.4)
        
        return max(0.0, min(1.0, quality_score))
    
    def _extract_with_multiple_configs(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Extract text using multiple OCR configurations for better accuracy."""
        best_result = {"text": "", "confidence": 0.0, "blocks": []}
        
        for config_name, config in self.ocr_configs.items():
            try:
                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT,
                    config=config
                )
                
                # Extract text and confidence
                text_parts = []
                confidence_scores = []
                text_blocks = []
                
                for i, conf in enumerate(ocr_data['conf']):
                    if conf > self.preprocessor.min_confidence:
                        text = ocr_data['text'][i].strip()
                        if text:
                            text_parts.append(text)
                            confidence_scores.append(conf)
                            text_blocks.append({
                                'text': text,
                                'confidence': conf,
                                'bbox': {
                                    'x': ocr_data['left'][i],
                                    'y': ocr_data['top'][i],
                                    'width': ocr_data['width'][i],
                                    'height': ocr_data['height'][i]
                                }
                            })
                
                if text_parts:
                    combined_text = ' '.join(text_parts)
                    avg_confidence = np.mean(confidence_scores)
                    
                    # Keep the best result
                    if avg_confidence > best_result["confidence"]:
                        best_result = {
                            "text": combined_text,
                            "confidence": avg_confidence,
                            "blocks": text_blocks
                        }
                        
            except Exception as e:
                logger.warning(f"OCR config {config_name} failed: {e}")
                continue
        
        return best_result["text"], best_result["confidence"], best_result["blocks"]
    
    def extract_text(self, image_path: str) -> OCRResult:
        """Extract text from image with comprehensive processing."""
        import time
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ProcessingError(f"Could not read image: {image_path}")
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(image)
            
            # Preprocess image
            preprocessed = self.preprocessor.preprocess(image)
            
            # Extract text with multiple configurations
            text, confidence, text_blocks = self._extract_with_multiple_configs(preprocessed)
            
            # Clean up text
            cleaned_text = self._clean_text(text)
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=cleaned_text,
                confidence=confidence,
                text_blocks=text_blocks,
                processing_time=processing_time,
                image_quality_score=quality_score
            )
            
        except Exception as e:
            log_error(logger, e, f"OCR extraction failed for {image_path}")
            raise ProcessingError(
                f"OCR extraction failed for {image_path}",
                file_path=image_path,
                processing_stage="ocr_extraction"
            ) from e
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'[|]', 'I', text)  # Fix common OCR confusion
        text = re.sub(r'[0]', 'O', text)  # Fix common OCR confusion
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()

# Initialize OCR processor
ocr_processor = OptimizedOCRProcessor()

def get_file_type(file_path: str) -> str:
    # Basic file type detection based on extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
        return "document"
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
        return "image"
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return "video"
    else:
        return "other"

# --- Helper for Section Extraction ---
def extract_sections(text: str) -> dict:
    sections = {}
    # Simple regex to find common section headers (e.g., 1. Introduction, 2. Procedure, etc.)
    # This can be greatly improved with more sophisticated parsing or ML models
    section_pattern = re.compile(r'^\s*(\d+\.?\d*\s*[A-Z][a-zA-Z0-9\s\-]+)\s*$\n', re.MULTILINE)
    matches = list(section_pattern.finditer(text))

    if not matches:
        sections["Full Document"] = text
        return sections

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_index = match.end()
        end_index = matches[i+1].start() if i + 1 < len(matches) else len(text)
        sections[section_title] = text[start_index:end_index].strip()
    
    # Add any leading text before the first section
    if matches[0].start() > 0:
        sections["Introduction/Preamble"] = text[:matches[0].start()].strip()

    return sections

# --- Structured Data Extraction Functions (Enhanced for Demonstration) ---

def _extract_equipment_data(text: str, entities: list) -> list[dict]:
    equipment_list = []
    # Look for common equipment names and associated specs using regex and NER
    equipment_patterns = {
        "Pump": r"\b(pump|centrifugal pump|gear pump)\b",
        "Motor": r"\b(motor|electric motor|induction motor)\b",
        "Valve": r"\b(valve|ball valve|gate valve|check valve)\b",
        "Sensor": r"\b(sensor|temperature sensor|pressure sensor)\b"
    }
    spec_patterns = {
        "power": r"(\d+\.?\d*)\s*(HP|kW|W)\b",
        "voltage": r"(\d+)\s*(V|volt)\b",
        "flow_rate": r"(\d+\.?\d*)\s*(gpm|l/s|m3/hr)\b",
        "pressure": r"(\d+\.?\d*)\s*(psi|bar|kPa)\b"
    }

    for eq_type, pattern in equipment_patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(0).strip()
            specs = {}
            # Search for specs in a window around the equipment name
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context_text = text[context_start:context_end]

            for spec_name, spec_re in spec_patterns.items():
                spec_match = re.search(spec_re, context_text, re.IGNORECASE)
                if spec_match:
                    specs[spec_name] = spec_match.group(0).strip()
            
            equipment_list.append({"name": name, "type": eq_type, "specifications": specs, "location": None, "confidence": 0.8}) # Assign a default confidence

    # Refine with NER entities if available and relevant
    for entity in entities:
        if entity['entity_group'] == 'ORG' or entity['entity_group'] == 'PRODUCT': # Assuming these might be equipment names/models
            if any(kw in entity['word'].lower() for kw in equipment_patterns.keys()):
                # Avoid duplicates, merge specs if already found
                found = False
                for eq in equipment_list:
                    if eq["name"].lower() == entity['word'].lower():
                        found = True
                        break
                if not found:
                    equipment_list.append({"name": entity['word'], "type": "Generic", "specifications": {}, "location": None, "confidence": entity['score']}) # Use NER score as confidence

    return equipment_list

def _extract_procedure_data(text: str, entities: list) -> list[dict]:
    procedures = []
    # Look for common procedure titles and numbered/bulleted steps
    procedure_titles = re.findall(r'\b(Procedure|Operating Instructions|Maintenance Steps|Installation Guide)\b[\s\S]{0,50}\n', text, re.IGNORECASE)
    
    for title_match in procedure_titles:
        title = title_match.strip()
        steps = []
        # Attempt to find steps following the title
        # This is a very basic step extraction, needs improvement for complex documents
        step_matches = re.findall(r'^\s*(\d+\.\s*|\*\s*|\-\s*)(.*?)(?=\n\s*(\d+\.\s*|\*\s*|\-\s*)|$)', text[text.find(title):], re.MULTILINE | re.DOTALL)
        
        for i, step_tuple in enumerate(step_matches):
            step_text = step_tuple[1].strip()
            if step_text:
                steps.append({"step_number": i + 1, "description": step_text})
        
        if steps:
            procedures.append({"title": title, "steps": steps, "category": None, "confidence": 0.7}) # Assign a default confidence

    # If no specific titles, try to extract any numbered/bulleted lists as a generic procedure
    if not procedures:
        steps = []
        step_matches = re.findall(r'^\s*(\d+\.\s*|\*\s*|\-\s*)(.*?)(?=\n\s*(\d+\.\s*|\*\s*|\-\s*)|$)', text, re.MULTILINE | re.DOTALL)
        for i, step_tuple in enumerate(step_matches):
            step_text = step_tuple[1].strip()
            if step_text:
                steps.append({"step_number": i + 1, "description": step_text})
        if steps:
            procedures.append({"title": "General Procedure", "steps": steps, "category": None, "confidence": 0.6}) # Assign a lower confidence

    return procedures

def _extract_safety_information(text: str, entities: list) -> list[dict]:
    safety_info_list = []
    # Look for hazard/precaution keywords and associated PPE
    hazard_patterns = [
        r"\b(danger|warning|caution|hazard|risk)\b[\s\S]{0,100}?(?:\b(avoid|do not|ensure|wear)\b[\s\S]{0,100}?(?:\b(gloves|helmet|goggles|respirator|ear protection|safety glasses)\b)?)",
        r"\b(PPE required|Personal Protective Equipment)\b[\s\S]{0,100}?(?:\b(gloves|helmet|goggles|respirator|ear protection|safety glasses)\b)"
    ]
    ppe_keywords = ["gloves", "helmet", "goggles", "respirator", "ear protection", "safety glasses"]

    for pattern in hazard_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            hazard_text = match.group(0).strip()
            ppe_found = [ppe for ppe in ppe_keywords if ppe in hazard_text.lower()]
            safety_info_list.append({"hazard": hazard_text, "precaution": "See context", "ppe_required": ', '.join(ppe_found), "severity": None, "confidence": 0.8}) # Assign a default confidence
    
    # Further extraction could use NER entities for specific chemicals (e.g., CHEM) or conditions
    return safety_info_list

def _extract_technical_specifications(text: str, entities: list) -> list[dict]:
    tech_specs = []
    # Look for common measurement patterns with units
    measurement_patterns = [
        r"(\d+\.?\d*)\s*(C|F|K)\b", # Temperature
        r"(\d+\.?\d*)\s*(psi|bar|kPa)\b", # Pressure
        r"(\d+\.?\d*)\s*(mm|cm|m|inch)\b", # Length/Dimension
        r"(\d+\.?\d*)\s*(kg|g|lb)\b", # Weight
        r"(\d+\.?\d*)\s*(Hz|kHz|MHz)\b", # Frequency
        r"(\d+\.?\d*)\s*(A|amp)\b", # Current
        r"(\d+\.?\d*)\s*(V|volt)\b" # Voltage
    ]

    for pattern in measurement_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(1)
            unit = match.group(2)
            parameter = "Unknown"
            if unit.lower() in ["c", "f", "k"]: parameter = "Temperature"
            elif unit.lower() in ["psi", "bar", "kpa"]: parameter = "Pressure"
            elif unit.lower() in ["mm", "cm", "m", "inch"]: parameter = "Dimension"
            elif unit.lower() in ["kg", "g", "lb"]: parameter = "Weight"
            elif unit.lower() in ["hz", "khz", "mhz"]: parameter = "Frequency"
            elif unit.lower() in ["a", "amp"]: parameter = "Current"
            elif unit.lower() in ["v", "volt"]: parameter = "Voltage"

            tech_specs.append({"parameter": parameter, "value": value, "unit": unit, "tolerance": None, "confidence": 0.9}) # Assign a default confidence
    
    # Look for tolerance patterns (e.g., +/- 0.5mm)
    tolerance_pattern = r"\b[+-]\s*(\d+\.?\d*)\s*([a-zA-Z%]+)\b"
    for spec in tech_specs:
        context_start = max(0, text.find(spec["value"]) - 50)
        context_end = min(len(text), text.find(spec["value"]) + 50)
        context_text = text[context_start:context_end]
        tol_match = re.search(tolerance_pattern, context_text, re.IGNORECASE)
        if tol_match:
            spec["tolerance"] = tol_match.group(0)
            spec["confidence"] = 0.95 # Higher confidence if tolerance found

    return tech_specs

def _extract_personnel_data(text: str, entities: list) -> list[dict]:
    personnel_list = []
    # Look for NER entities classified as 'PERSON' and try to find roles/certifications nearby
    role_keywords = ["engineer", "technician", "operator", "manager", "supervisor", "specialist"]
    certification_patterns = [
        r"\b(OSHA\s*\d+)\b",
        r"\b(ISO\s*\d+)\b",
        r"\b(certified|licensed)\b[\s\S]{0,50}?(?:\b(welder|electrician|mechanic)\b)"
    ]

    for entity in entities:
        if entity['entity_group'] == 'PER':
            name = entity['word']
            role = "Unknown"
            certifications = []
            confidence = entity['score'] # Start with NER confidence

            # Search for role and certifications in a window around the person's name
            context_start = max(0, entity['start'] - 100)
            context_end = min(len(text), entity['end'] + 100)
            context_text = text[context_start:context_end]

            for r_kw in role_keywords:
                if re.search(r'\b' + re.escape(r_kw) + r'\b', context_text, re.IGNORECASE):
                    role = r_kw.capitalize()
                    confidence = max(confidence, 0.85) # Boost confidence if role found
                    break
            
            for cert_pattern in certification_patterns:
                for cert_match in re.finditer(cert_pattern, context_text, re.IGNORECASE):
                    certifications.append(cert_match.group(0).strip())
                    confidence = max(confidence, 0.9) # Boost confidence if certification found
            
            personnel_list.append({"name": name, "role": role, "responsibilities": None, "certifications": list(set(certifications)), "confidence": confidence})
    return personnel_list

# --- Main Processing Functions ---

def process_document(file_path: str):
    logger.info(f"Starting document processing with Tika: {file_path}")
    filename = os.path.basename(file_path)
    extracted_text = ""
    metadata = {}
    status = "failed"
    extracted_entities = []
    classification_result = {"category": "unclassified", "score": 0.0}
    key_phrases = []

    # Structured data placeholders
    equipment_data = []
    procedure_data = []
    safety_info_data = []
    technical_spec_data = []
    personnel_data = []
    document_sections = {}

    # Validate file exists
    if not os.path.exists(file_path):
        raise ProcessingError(
            f"File not found: {file_path}",
            file_path=file_path,
            processing_stage="file_validation"
        )

    try:
        log_processing_step(logger, "tika_extraction", "started", extra_data={'filename': filename})
        
        with open(file_path, 'rb') as file:
            headers = {
                'Accept': 'application/json',  # Request JSON output
                'X-Tika-OCRTesseractPath': '/usr/bin/tesseract',  # Assuming Tesseract is installed in Tika container
                'X-Tika-OCRTimeout': str(config_manager.get_config().processing.tika_ocr_timeout_seconds),  # OCR timeout from config
                'X-Tika-PDFextractInlineImages': 'true',  # Extract images from PDF
                'X-Tika-PDFOcrStrategy': 'ocr_and_text_extraction',  # OCR and text extraction
            }
            # Use /rmeta endpoint for rich metadata and content
            try:
                response = requests.put(f"{TIKA_SERVER_URL}/rmeta", data=file, headers=headers, timeout=config_manager.get_config().processing.tika_timeout_seconds)
                response.raise_for_status()  # Raise an exception for HTTP errors
            except requests.exceptions.Timeout:
                raise ServiceUnavailableError(
                    f"Tika server timeout processing {filename}",
                    service_name="Apache Tika",
                    service_url=TIKA_SERVER_URL
                )
            except requests.exceptions.ConnectionError:
                raise ServiceUnavailableError(
                    f"Cannot connect to Tika server",
                    service_name="Apache Tika",
                    service_url=TIKA_SERVER_URL
                )
            
            tika_output = response.json()
            
            if tika_output and isinstance(tika_output, list) and len(tika_output) > 0:
                # Tika /rmeta returns a list of JSON objects, one for each embedded document/part
                main_content = tika_output[0]
                extracted_text = main_content.get('X-TIKA:content', '')
                metadata = main_content.get('metadata', {})
                status = "processed"

                # Extract document sections
                document_sections = extract_sections(extracted_text)

                log_processing_step(logger, "tika_extraction", "completed", extra_data={'text_length': len(extracted_text)})

                # Perform AI processing with error handling
                try:
                    log_processing_step(logger, "ner_extraction", "started")
                    extracted_entities = ner_extractor.extract_entities(extracted_text)
                    log_processing_step(logger, "ner_extraction", "completed", extra_data={'entities_count': len(extracted_entities)})
                except Exception as e:
                    log_error(logger, e, "NER extraction failed - continuing with empty entities")
                    extracted_entities = []

                try:
                    log_processing_step(logger, "document_classification", "started")
                    classification_result = classifier.classify_document(extracted_text, CANDIDATE_LABELS)
                    log_processing_step(logger, "document_classification", "completed", extra_data={'category': classification_result.get('category')})
                except Exception as e:
                    log_error(logger, e, "Document classification failed - using default")
                    classification_result = {"category": "unclassified", "score": 0.0}

                try:
                    log_processing_step(logger, "keyphrase_extraction", "started")
                    key_phrases = keyphrase_extractor.extract_keyphrases(extracted_text)
                    log_processing_step(logger, "keyphrase_extraction", "completed", extra_data={'keyphrases_count': len(key_phrases)})
                except Exception as e:
                    log_error(logger, e, "Keyphrase extraction failed - continuing with empty list")
                    key_phrases = []

                # Extract structured data
                equipment_data = _extract_equipment_data(extracted_text, extracted_entities)
                procedure_data = _extract_procedure_data(extracted_text, extracted_entities)
                safety_info_data = _extract_safety_information(extracted_text, extracted_entities)
                technical_spec_data = _extract_technical_specifications(extracted_text, extracted_entities)
                personnel_data = _extract_personnel_data(extracted_text, extracted_entities)
                
            else:
                extracted_text = "Tika returned empty or unexpected output."
                status = "failed"

    except requests.exceptions.RequestException as e:
        raise ServiceUnavailableError(
            f"Error communicating with Tika server: {str(e)}",
            service_name="Apache Tika",
            service_url=TIKA_SERVER_URL,
            details={'filename': filename}
        ) from e
    except json.JSONDecodeError as e:
        raise ProcessingError(
            f"Error decoding Tika JSON response for {filename}",
            file_path=file_path,
            processing_stage="tika_response_parsing",
            details={'json_error': str(e)}
        ) from e
    except Exception as e:
        raise ProcessingError(
            f"Unexpected error processing {filename}",
            file_path=file_path,
            processing_stage="document_processing",
            details={'error': str(e)}
        ) from e

    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "metadata": metadata,
        "extracted_entities": extracted_entities,
        "classification": classification_result,
        "key_phrases": key_phrases,
        "equipment_data": equipment_data,
        "procedure_data": procedure_data,
        "safety_info_data": safety_info_data,
        "technical_spec_data": technical_spec_data,
        "personnel_data": personnel_data,
        "document_sections": document_sections,
        "status": status
    }

def process_video(file_path: str):
    print(f"Processing video with FFmpeg and Whisper: {file_path}")
    filename = os.path.basename(file_path)
    extracted_text = ""
    metadata = {"source": "video_processor", "original_path": file_path}
    status = "failed"
    extracted_entities = []
    classification_result = {"category": "unclassified", "score": 0.0}
    key_phrases = []
    audio_output_path = os.path.join(os.path.dirname(file_path), f"{os.path.splitext(filename)[0]}.mp3")

    # Structured data placeholders
    equipment_data = []
    procedure_data = []
    safety_info_data = []
    technical_spec_data = []
    personnel_data = []
    document_sections = {}

    try:
        # 1. Extract audio using ffmpeg
        ffmpeg.input(file_path).output(audio_output_path, acodec='libmp3lame').run(overwrite_output=True)
        print(f"Audio extracted to: {audio_output_path}")

        # 2. Transcribe audio using Whisper
        model = whisper.load_model("base") 
        result = model.transcribe(audio_output_path)
        extracted_text = result["text"]
        status = "processed"

        # Extract document sections (from transcript)
        document_sections = extract_sections(extracted_text)

        # Perform NER on the extracted text
        extracted_entities = ner_extractor.extract_entities(extracted_text)

        # Classify the document
        classification_result = classifier.classify_document(extracted_text, CANDIDATE_LABELS)

        # Extract key phrases
        key_phrases = keyphrase_extractor.extract_keyphrases(extracted_text)

        # Extract structured data
        equipment_data = _extract_equipment_data(extracted_text, extracted_entities)
        procedure_data = _extract_procedure_data(extracted_text, extracted_entities)
        safety_info_data = _extract_safety_information(extracted_text, extracted_entities)
        technical_spec_data = _extract_technical_specifications(extracted_text, extracted_entities)
        personnel_data = _extract_personnel_data(extracted_text, extracted_entities)

    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        extracted_text = f"FFmpeg error processing {filename}: {e.stderr.decode()}"
        metadata["error"] = str(e)
    except Exception as e:
        print(f"Whisper or other error: {e}")
        extracted_text = f"Error transcribing {filename}: {e}"
        metadata["error"] = str(e)
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)
            print(f"Cleaned up temporary audio file: {audio_output_path}")

    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "metadata": metadata,
        "extracted_entities": extracted_entities,
        "classification": classification_result,
        "key_phrases": key_phrases,
        "equipment_data": equipment_data,
        "procedure_data": procedure_data,
        "safety_info_data": safety_info_data,
        "technical_spec_data": technical_spec_data,
        "personnel_data": personnel_data,
        "document_sections": document_sections,
        "status": status
    }

def process_image(file_path: str) -> Dict:
    """
    Process standalone images with optimized OCR and AI integration.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary containing extracted text and structured data
    """
    logger.info(f"Processing image with optimized OCR: {file_path}")
    filename = os.path.basename(file_path)
    
    # Initialize result structure
    result = {
        "filename": filename,
        "extracted_text": "",
        "metadata": {
            "source": "optimized_image_processor",
            "original_path": file_path,
            "processor_version": "2.0"
        },
        "extracted_entities": [],
        "classification": {"category": "unclassified", "score": 0.0},
        "key_phrases": [],
        "equipment_data": [],
        "procedure_data": [],
        "safety_info_data": [],
        "technical_spec_data": [],
        "personnel_data": [],
        "document_sections": {},
        "status": "failed",
        "ocr_metadata": {}
    }
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise ProcessingError(
            f"Image file not found: {file_path}",
            file_path=file_path,
            processing_stage="file_validation"
        )
    
    # Check if image format is supported
    supported_formats = {fmt.value for fmt in ImageFormat}
    ext = Path(file_path).suffix.lower().lstrip('.')
    if ext not in supported_formats:
        raise ProcessingError(
            f"Unsupported image format: {ext}",
            file_path=file_path,
            processing_stage="format_validation"
        )
    
    try:
        # Extract text using optimized OCR
        log_processing_step(logger, "ocr_extraction", "started", extra_data={'filename': filename})
        ocr_result = ocr_processor.extract_text(file_path)
        log_processing_step(logger, "ocr_extraction", "completed", 
                          extra_data={'text_length': len(ocr_result.text), 'confidence': ocr_result.confidence})
        
        result["extracted_text"] = ocr_result.text
        result["ocr_metadata"] = {
            'confidence': ocr_result.confidence,
            'processing_time': ocr_result.processing_time,
            'image_quality_score': ocr_result.image_quality_score,
            'text_blocks_count': len(ocr_result.text_blocks),
            'ocr_engine': 'tesseract_optimized'
        }
        
        if not ocr_result.text.strip():
            logger.warning(f"No text extracted from image: {filename}")
            result["status"] = "no_text_found"
            return result
        
        result["status"] = "processed"
        
        # Extract document sections
        result["document_sections"] = extract_sections(ocr_result.text)
        
        # Perform AI processing with error handling
        try:
            log_processing_step(logger, "ner_extraction", "started")
            result["extracted_entities"] = ner_extractor.extract_entities(ocr_result.text)
            log_processing_step(logger, "ner_extraction", "completed", 
                              extra_data={'entities_count': len(result["extracted_entities"])})
        except Exception as e:
            log_error(logger, e, "NER extraction failed - continuing with empty entities")
            result["extracted_entities"] = []
        
        try:
            log_processing_step(logger, "document_classification", "started")
            result["classification"] = classifier.classify_document(ocr_result.text, CANDIDATE_LABELS)
            log_processing_step(logger, "document_classification", "completed", 
                              extra_data={'category': result["classification"].get('category')})
        except Exception as e:
            log_error(logger, e, "Document classification failed - using default")
            result["classification"] = {"category": "unclassified", "score": 0.0}
        
        try:
            log_processing_step(logger, "keyphrase_extraction", "started")
            result["key_phrases"] = keyphrase_extractor.extract_keyphrases(ocr_result.text)
            log_processing_step(logger, "keyphrase_extraction", "completed", 
                              extra_data={'keyphrases_count': len(result["key_phrases"])})
        except Exception as e:
            log_error(logger, e, "Keyphrase extraction failed - continuing with empty list")
            result["key_phrases"] = []
        
        # Extract structured data
        result["equipment_data"] = _extract_equipment_data(ocr_result.text, result["extracted_entities"])
        result["procedure_data"] = _extract_procedure_data(ocr_result.text, result["extracted_entities"])
        result["safety_info_data"] = _extract_safety_information(ocr_result.text, result["extracted_entities"])
        result["technical_spec_data"] = _extract_technical_specifications(ocr_result.text, result["extracted_entities"])
        result["personnel_data"] = _extract_personnel_data(ocr_result.text, result["extracted_entities"])
        
        logger.info(f"Image processing completed successfully: {filename}")
        
    except Exception as e:
        log_error(logger, e, f"Image processing failed for {filename}")
        result["status"] = "failed"
        result["metadata"]["error"] = str(e)
        raise ProcessingError(
            f"Image processing failed for {filename}",
            file_path=file_path,
            processing_stage="image_processing"
        ) from e
    
    return result