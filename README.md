# EXPLAINIUM - Intelligent Knowledge Extraction Demo

🚀 **Advanced AI-powered document processing and knowledge extraction system**

## Quick Demo Start

```bash
# Clone and run the demo
git clone <repository-url>
cd explainium
./demo.sh
```

Then open your browser to `http://localhost:8501` and start uploading files!

## 🌟 Demo Features

### Multi-Format Processing
- **📄 Documents**: PDF, DOC, DOCX, TXT with intelligent text extraction
- **🖼️ Images**: JPG, PNG, GIF with advanced OCR and content analysis
- **🎥 Videos**: MP4, AVI with audio transcription and frame OCR
- **🎵 Audio**: MP3, WAV with Whisper AI transcription
- **📊 Spreadsheets**: CSV, XLS, XLSX with data analysis

### AI-Powered Intelligence
- **🧠 LLM-First Processing**: Primary method using advanced language models
- **🔧 Advanced Fallback**: Secondary processing with enhanced extraction
- **📈 Real-time Analytics**: Confidence scoring and knowledge categorization
- **🎯 Smart Filtering**: Filter by type, confidence, and search terms

## 📁 Sample Files

The `documents_samples/` directory contains various file types to demonstrate the system:

- **Images**: `safe-use-of-MEWP.png`, `71147668_2237079513081416_8142274033388355584_n.jpg`
- **Videos**: `Safe Lifting - PhoenixParks (360p, h264).mp4`
- **PDFs**: Multiple technical documents and manuals
- **Text**: `test_sample.txt`

## 🔧 Demo Workflow

1. **Start the Application**: Run `./demo.sh`
2. **Upload a File**: Use the sidebar file uploader
3. **View Results**: See extracted knowledge in the main table
4. **Explore Analytics**: Check the sidebar charts and statistics
5. **Filter & Search**: Use the filtering options to explore data

## 🎯 What Gets Extracted

The system intelligently extracts and categorizes:

- **Concepts**: Definitions, technical terms, structured content
- **Processes**: Procedures, workflows, step-by-step instructions
- **Systems**: Tools, equipment, technical specifications
- **Requirements**: Compliance, regulations, standards
- **Risks**: Safety concerns, hazards, warnings
- **People**: Roles, responsibilities, organizational information

## 🛠️ System Requirements

- **Python 3.8+**
- **Tesseract OCR** (for image processing)
- **FFmpeg** (for video audio extraction)
- **2GB+ RAM** (for AI models)

### Optional Dependencies
- **Whisper AI** (for high-quality audio transcription)
- **OpenCV** (for advanced image/video processing)
- **spaCy models** (for enhanced NLP)

## 📊 Processing Methods

The system uses a hierarchical processing approach:

1. **🧠 LLM-First Processing** (Primary)
   - Advanced language model analysis
   - High-quality entity extraction
   - Intelligent categorization

2. **🔧 Advanced Engine** (Fallback)
   - Enhanced pattern recognition
   - Structured data extraction
   - Quality validation

3. **📝 Text Analysis** (Final Fallback)
   - Pattern-based extraction
   - Basic categorization
   - Reliability baseline

## 🚀 Architecture

```
Frontend (Streamlit) 
    ↓
Document Processors
    ↓
AI Engines (LLM + Advanced)
    ↓
Knowledge Extraction
    ↓
Structured Output
```

## 📈 Performance

- **Image Processing**: ~5-10 seconds per image
- **Video Processing**: ~30-60 seconds per video (depending on length)
- **Document Processing**: ~10-20 seconds per document
- **Audio Processing**: ~20-40 seconds per audio file

## 🔍 Demo Tips

### Best Results
- **High-quality images** with clear text
- **Videos with visible text** or clear audio
- **Well-structured documents** (PDFs, Word docs)
- **Files under 50MB** for optimal performance

### Troubleshooting
- If OCR fails, check Tesseract installation
- If video processing fails, check FFmpeg installation
- Large files may take longer to process
- AI models download automatically on first use

## 🛡️ Security & Privacy

- **Local Processing**: All files processed locally
- **No Cloud Dependencies**: No data sent to external services
- **Temporary Files**: Automatically cleaned up after processing
- **Privacy First**: Your data never leaves your machine

## 🎨 User Interface

- **Clean Dashboard**: Modern, intuitive interface
- **Real-time Updates**: Live processing status
- **Interactive Charts**: Visual analytics and insights
- **Export Capabilities**: Download results as CSV
- **Responsive Design**: Works on desktop and tablet

---

**Ready to explore? Run `./demo.sh` and start extracting knowledge!** 🚀
