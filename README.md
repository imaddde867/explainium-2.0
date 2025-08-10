# EXPLAINIUM - Intelligent Knowledge Extraction System

A sophisticated document processing and knowledge extraction system that produces structured, categorized knowledge output in a specific format with emojis and proper categorization.

## Features

- **Unified Document Processing**: Handles all document types (PDF, DOC, DOCX, images, videos, audio, spreadsheets, presentations)
- **Intelligent Knowledge Extraction**: Uses advanced AI to extract structured knowledge
- **Categorized Output**: Produces knowledge in organized categories (💡 Concepts, ⚙️ Processes, 🖥️ Systems)
- **Multi-format Support**: Processes text, images, audio, and video documents
- **High-Quality OCR**: Advanced image processing and text extraction
- **Async Processing**: Built with async/await for high performance

## Expected Output Format

The system produces knowledge in this structured format:

```
💡 Concepts
Integrated Pest Management (IPM): A system integrating chemical, physical, cultural, and biological controls...

⚙️ Processes
Pest Management Framework: A three-step approach to pest management in food facilities...

🖥️ Systems
Pest Monitoring Tools: Various tools and methods for monitoring pest activity...
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd explainium
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies** (for OCR and audio processing):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr ffmpeg
   
   # macOS
   brew install tesseract ffmpeg
   
   # Windows
   # Download and install Tesseract and FFmpeg manually
   ```

4. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage

```python
import asyncio
from src.processors.unified_document_processor import UnifiedDocumentProcessor

async def process_document():
    processor = UnifiedDocumentProcessor()
    result = await processor.process_document("path/to/document.pdf", 1)
    
    # Access extracted knowledge
    knowledge = result['knowledge']
    print("Extracted concepts:", knowledge.get('concepts', []))
    print("Extracted processes:", knowledge.get('processes', []))

# Run
asyncio.run(process_document())
```

### Test the System

Run the test script to see the system in action:

```bash
python test_knowledge_extraction.py
```

This will process the `documents_samples/AG1157.pdf` file and display the extracted knowledge.

## Supported Document Types

| Type | Extensions | Processing Method |
|------|------------|-------------------|
| **Text** | `.pdf`, `.doc`, `.docx`, `.txt`, `.rtf` | Text extraction + AI analysis |
| **Images** | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff` | OCR + AI analysis |
| **Spreadsheets** | `.xls`, `.xlsx`, `.csv` | Data extraction + AI analysis |
| **Presentations** | `.ppt`, `.pptx` | Text extraction + AI analysis |
| **Audio** | `.mp3`, `.wav`, `.flac`, `.aac` | Transcription + AI analysis |
| **Video** | `.mp4`, `.avi`, `.mov`, `.mkv` | Audio extraction + transcription + AI analysis |

## Architecture

- **`UnifiedDocumentProcessor`**: Main processor that handles all document types
- **`IntelligentKnowledgeExtractor`**: AI-powered knowledge extraction engine
- **`celery_worker.py`**: Background task processing for large documents
- **`exceptions.py`**: Custom exception handling
- **`logging_config.py`**: Centralized logging configuration

## Configuration

Set environment variables for customization:

```bash
export LOG_LEVEL=INFO
export CREATE_LOGS_DIR=true
```

## API Integration

The system integrates with FastAPI and Celery for web-based document processing:

```python
from fastapi import FastAPI, UploadFile
from src.processors.unified_document_processor import UnifiedDocumentProcessor

app = FastAPI()
processor = UnifiedDocumentProcessor()

@app.post("/process-document/")
async def process_document(file: UploadFile):
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Process document
    result = await processor.process_document(file_path, 1)
    return result
```

## Performance

- **Text documents**: ~1-5 seconds per page
- **Images**: ~2-10 seconds depending on complexity
- **Audio/Video**: ~1-3x real-time duration
- **Large documents**: Processed asynchronously via Celery

## Troubleshooting

### Common Issues

1. **OCR not working**: Ensure Tesseract is installed and in PATH
2. **Audio processing fails**: Install FFmpeg and ensure Whisper is available
3. **Memory issues**: Large documents may require more RAM, use Celery for background processing

### Logs

Check logs for detailed error information:
```bash
tail -f logs/explainium.log
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
flake8 src/
```

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub or contact the development team.

