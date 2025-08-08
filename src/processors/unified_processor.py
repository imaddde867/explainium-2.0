import os
from typing import Dict, List, Optional
from src.processors.document_processor import process_document, process_image, process_video, get_file_type
from src.exceptions import ProcessingError, UnsupportedFileTypeError
from src.logging_config import get_logger, log_processing_step, log_error

logger = get_logger(__name__)

class UnifiedProcessor:
    """
    Unified processor that routes different file types to appropriate processors.
    Provides a single interface for processing documents, images, and videos.
    """
    
    def __init__(self):
        # Supported file extensions for each processor
        self.supported_extensions = {
            'document': ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.txt', '.rtf'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
        }
    
    def get_processor_type(self, file_path: str) -> str:
        """
        Determine which processor to use based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processor type: 'document', 'image', 'video', or 'audio'
            
        Raises:
            UnsupportedFileTypeError: If file type is not supported
        """
        _, ext = os.path.splitext(file_path.lower())
        
        for processor_type, extensions in self.supported_extensions.items():
            if ext in extensions:
                return processor_type
        
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {ext}",
            file_path=file_path,
            supported_types=list(self.supported_extensions.keys())
        )
    
    def process_file(self, file_path: str) -> Dict:
        """
        Process a file using the appropriate processor based on file type.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing extracted text and structured data
            
        Raises:
            ProcessingError: If processing fails
            UnsupportedFileTypeError: If file type is not supported
        """
        logger.info(f"Starting unified processing: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise ProcessingError(
                f"File not found: {file_path}",
                file_path=file_path,
                processing_stage="file_validation"
            )
        
        try:
            # Determine processor type
            processor_type = self.get_processor_type(file_path)
            log_processing_step(logger, "processor_selection", "completed", 
                              extra_data={'processor_type': processor_type, 'file_path': file_path})
            
            # Route to appropriate processor (all in document_processor.py now)
            if processor_type == 'document':
                return process_document(file_path)
            elif processor_type == 'image':
                return process_image(file_path)
            elif processor_type == 'video':
                return process_video(file_path)
            elif processor_type == 'audio':
                # For now, treat audio like video (extract audio and use Whisper)
                return process_video(file_path)
            else:
                raise UnsupportedFileTypeError(
                    f"Unknown processor type: {processor_type}",
                    file_path=file_path
                )
                
        except UnsupportedFileTypeError:
            # Re-raise as-is
            raise
        except Exception as e:
            log_error(logger, e, f"Unified processing failed for {file_path}")
            raise ProcessingError(
                f"Unified processing failed for {file_path}",
                file_path=file_path,
                processing_stage="unified_processing"
            ) from e
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get list of supported file formats for each processor type.
        
        Returns:
            Dictionary mapping processor types to supported file extensions
        """
        return self.supported_extensions.copy()
    
    def is_supported(self, file_path: str) -> bool:
        """
        Check if a file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file type is supported, False otherwise
        """
        try:
            self.get_processor_type(file_path)
            return True
        except UnsupportedFileTypeError:
            return False

# Initialize the unified processor
unified_processor = UnifiedProcessor()

def process_any_file(file_path: str) -> Dict:
    """
    Convenience function to process any supported file type.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Dictionary containing extracted text and structured data
    """
    return unified_processor.process_file(file_path)

def get_supported_file_types() -> Dict[str, List[str]]:
    """
    Get all supported file types.
    
    Returns:
        Dictionary mapping processor types to supported file extensions
    """
    return unified_processor.get_supported_formats() 