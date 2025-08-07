"""
Comprehensive tests for the integrated document processor.
Tests document processing, image OCR, and video processing functionality.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.processors.document_processor import (
    process_document,
    process_image, 
    process_video,
    ImageFormat,
    OCRResult,
    ImagePreprocessor,
    OptimizedOCRProcessor
)
from src.exceptions import ProcessingError, UnsupportedFileTypeError

@pytest.fixture
def image_preprocessor():
    return ImagePreprocessor()

@pytest.fixture
def ocr_processor():
    return OptimizedOCRProcessor()

@pytest.fixture
def mock_image():
    """Create a mock image array for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

class TestImageProcessing:
    """Test image processing functionality integrated into document processor."""
    
    def test_supported_image_formats(self):
        """Test that all expected image formats are supported."""
        expected_formats = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp'}
        actual_formats = {fmt.value for fmt in ImageFormat}
        assert actual_formats == expected_formats
    
    def test_image_preprocessor_initialization(self, image_preprocessor):
        """Test preprocessor initialization."""
        assert image_preprocessor.min_confidence == 30
        assert image_preprocessor.optimal_dpi == 300
    
    def test_validate_image_valid(self, image_preprocessor, mock_image):
        """Test image validation with valid image."""
        image_preprocessor._validate_image(mock_image)
        # Should not raise any exception
    
    def test_validate_image_none(self, image_preprocessor):
        """Test image validation with None."""
        with pytest.raises(ProcessingError, match="Invalid image: image is None"):
            image_preprocessor._validate_image(None)
    
    @patch('src.processors.document_processor.cv2.resize')
    def test_resize_for_ocr(self, mock_resize, image_preprocessor, mock_image):
        """Test image resizing for OCR."""
        mock_resize.return_value = mock_image
        result = image_preprocessor._resize_for_ocr(mock_image)
        mock_resize.assert_called_once()
        assert result is not None
    
    @patch('src.processors.document_processor.cv2.cvtColor')
    @patch('src.processors.document_processor.cv2.split')
    @patch('src.processors.document_processor.cv2.createCLAHE')
    @patch('src.processors.document_processor.cv2.merge')
    def test_enhance_contrast_color(self, mock_merge, mock_clahe, mock_split, mock_cvt, image_preprocessor, mock_image):
        """Test contrast enhancement for color images."""
        mock_cvt.return_value = mock_image
        mock_split.return_value = [mock_image, mock_image, mock_image]
        mock_clahe_instance = MagicMock()
        mock_clahe.return_value = mock_clahe_instance
        mock_merge.return_value = mock_image
        
        result = image_preprocessor._enhance_contrast(mock_image)
        
        mock_cvt.assert_called()
        mock_split.assert_called()
        mock_clahe.assert_called()
        assert result is not None

class TestOCRProcessing:
    """Test OCR processing functionality."""
    
    def test_ocr_processor_initialization(self, ocr_processor):
        """Test OCR processor initialization."""
        assert ocr_processor.tesseract_path == "/usr/bin/tesseract"
        assert len(ocr_processor.ocr_configs) > 0
        assert ocr_processor.preprocessor is not None
    
    def test_calculate_quality_score(self, ocr_processor, mock_image):
        """Test image quality score calculation."""
        score = ocr_processor._calculate_quality_score(mock_image)
        assert 0.0 <= score <= 1.0
    
    @patch('src.processors.document_processor.pytesseract.image_to_data')
    def test_extract_with_multiple_configs(self, mock_ocr, ocr_processor, mock_image):
        """Test OCR extraction with multiple configurations."""
        # Mock OCR output
        mock_ocr.return_value = {
            'text': ['Hello', 'World'],
            'conf': [90, 85],
            'left': [10, 20],
            'top': [30, 40],
            'width': [50, 60],
            'height': [70, 80]
        }
        
        text, confidence, blocks = ocr_processor._extract_with_multiple_configs(mock_image)
        
        assert "Hello World" in text
        assert confidence > 0
        assert len(blocks) == 2
        assert blocks[0]['text'] == 'Hello'
        assert blocks[0]['confidence'] == 90
    
    @patch('src.processors.document_processor.cv2.imread')
    @patch.object(OptimizedOCRProcessor, '_calculate_quality_score')
    @patch.object(ImagePreprocessor, 'preprocess')
    @patch.object(OptimizedOCRProcessor, '_extract_with_multiple_configs')
    @patch.object(OptimizedOCRProcessor, '_clean_text')
    def test_extract_text_success(self, mock_clean, mock_extract, mock_preprocess, mock_quality, mock_imread, ocr_processor):
        """Test successful text extraction."""
        mock_imread.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_quality.return_value = 0.8
        mock_preprocess.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_extract.return_value = ("Hello World", 85.0, [])
        mock_clean.return_value = "Hello World"
        
        result = ocr_processor.extract_text("/tmp/test.jpg")
        
        assert isinstance(result, OCRResult)
        assert result.text == "Hello World"
        assert result.confidence == 85.0
        assert result.image_quality_score == 0.8
        assert result.processing_time > 0
    
    def test_clean_text(self, ocr_processor):
        """Test text cleaning functionality."""
        # Test whitespace normalization
        assert ocr_processor._clean_text("  Hello   World  ") == "Hello World"
        
        # Test OCR error correction
        assert ocr_processor._clean_text("He||o W0rld") == "HeIIo WOrld"
        
        # Test empty text
        assert ocr_processor._clean_text("") == ""
        assert ocr_processor._clean_text(None) == ""

class TestIntegratedProcessing:
    """Test integrated processing functions."""
    
    @patch('src.processors.document_processor.ocr_processor.extract_text')
    @patch('src.processors.document_processor.ner_extractor.extract_entities')
    @patch('src.processors.document_processor.classifier.classify_document')
    @patch('src.processors.document_processor.keyphrase_extractor.extract_keyphrases')
    def test_process_image_success(self, mock_keyphrase, mock_classify, mock_ner, mock_ocr):
        """Test successful image processing."""
        # Mock OCR extraction
        mock_ocr.return_value = OCRResult(
            text="Sample text from image",
            confidence=85.0,
            text_blocks=[],
            processing_time=1.5,
            image_quality_score=0.8
        )
        
        # Mock AI components
        mock_ner.return_value = []
        mock_classify.return_value = {"category": "Technical Specifications", "score": 0.8}
        mock_keyphrase.return_value = ["sample", "text"]
        
        with patch('os.path.exists', return_value=True):
            result = process_image("/tmp/test.jpg")
        
        assert result["status"] == "processed"
        assert result["extracted_text"] == "Sample text from image"
        assert result["classification"]["category"] == "Technical Specifications"
        assert result["ocr_metadata"]["confidence"] == 85.0
        assert result["ocr_metadata"]["processing_time"] == 1.5
        assert result["ocr_metadata"]["image_quality_score"] == 0.8
        assert result["metadata"]["processor_version"] == "2.0"
    
    def test_process_image_file_not_found(self):
        """Test image processing with non-existent file."""
        with pytest.raises(ProcessingError, match="Image file not found"):
            process_image("/nonexistent/file.jpg")
    
    def test_process_image_unsupported_format(self):
        """Test image processing with unsupported format."""
        with patch('os.path.exists', return_value=True):
            with pytest.raises(ProcessingError, match="Unsupported image format"):
                process_image("/tmp/test.txt")
    
    @patch('src.processors.document_processor.ocr_processor.extract_text')
    def test_process_image_no_text_found(self, mock_ocr):
        """Test image processing when no text is found."""
        mock_ocr.return_value = OCRResult(
            text="",
            confidence=0.0,
            text_blocks=[],
            processing_time=1.0,
            image_quality_score=0.1
        )
        
        with patch('os.path.exists', return_value=True):
            result = process_image("/tmp/test.jpg")
        
        assert result["status"] == "no_text_found"
    
    @patch('src.processors.document_processor.requests.put')
    def test_process_document_success(self, mock_request):
        """Test successful document processing."""
        # Mock Tika response
        mock_response = MagicMock()
        mock_response.json.return_value = [{
            'X-TIKA:content': 'Sample document text',
            'metadata': {'Content-Type': 'application/pdf'}
        }]
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        # Mock AI components
        with patch('src.processors.document_processor.ner_extractor.extract_entities', return_value=[]), \
             patch('src.processors.document_processor.classifier.classify_document', return_value={"category": "Technical Specifications", "score": 0.8}), \
             patch('src.processors.document_processor.keyphrase_extractor.extract_keyphrases', return_value=["sample", "text"]), \
             patch('os.path.exists', return_value=True):
            
            result = process_document("/tmp/test.pdf")
        
        assert result["status"] == "processed"
        assert result["extracted_text"] == "Sample document text"
        assert result["classification"]["category"] == "Technical Specifications"
    
    @patch('src.processors.document_processor.ffmpeg.input')
    @patch('src.processors.document_processor.whisper.load_model')
    def test_process_video_success(self, mock_whisper, mock_ffmpeg):
        """Test successful video processing."""
        # Mock FFmpeg
        mock_ffmpeg_instance = MagicMock()
        mock_ffmpeg.return_value.output.return_value.run.return_value = mock_ffmpeg_instance
        
        # Mock Whisper
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Sample video transcript"}
        mock_whisper.return_value = mock_model
        
        # Mock AI components
        with patch('src.processors.document_processor.ner_extractor.extract_entities', return_value=[]), \
             patch('src.processors.document_processor.classifier.classify_document', return_value={"category": "Training Materials", "score": 0.8}), \
             patch('src.processors.document_processor.keyphrase_extractor.extract_keyphrases', return_value=["sample", "video"]), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove'):  # Mock file cleanup
            
            result = process_video("/tmp/test.mp4")
        
        assert result["status"] == "processed"
        assert result["extracted_text"] == "Sample video transcript"
        assert result["classification"]["category"] == "Training Materials"

class TestErrorHandling:
    """Test error handling in integrated processor."""
    
    @patch('src.processors.document_processor.ocr_processor.extract_text')
    def test_process_image_ocr_failure(self, mock_ocr):
        """Test image processing when OCR fails."""
        mock_ocr.side_effect = Exception("OCR failed")
        
        with patch('os.path.exists', return_value=True):
            with pytest.raises(ProcessingError, match="Image processing failed"):
                process_image("/tmp/test.jpg")
    
    @patch('src.processors.document_processor.requests.put')
    def test_process_document_tika_failure(self, mock_request):
        """Test document processing when Tika fails."""
        mock_request.side_effect = Exception("Tika connection failed")
        
        with patch('os.path.exists', return_value=True):
            with pytest.raises(ProcessingError, match="Unexpected error processing"):
                process_document("/tmp/test.pdf")
    
    @patch('src.processors.document_processor.ffmpeg.input')
    def test_process_video_ffmpeg_failure(self, mock_ffmpeg):
        """Test video processing when FFmpeg fails."""
        mock_ffmpeg.side_effect = Exception("FFmpeg failed")
        
        with patch('os.path.exists', return_value=True):
            result = process_video("/tmp/test.mp4")
        
        assert result["status"] == "failed"
        assert "FFmpeg failed" in result["extracted_text"] 