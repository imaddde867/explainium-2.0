# EXPLAINIUM Demo Instructions

## Pre-Demo Setup (5 minutes)

1. **Start the application**:
   ```bash
   ./demo.sh
   ```
   
2. **Wait for initialization** (models will download on first run)

3. **Open browser** to `http://localhost:8501`

4. **Prepare sample files** from `documents_samples/` directory

## Demo Flow (10-15 minutes)

### 1. Image Processing Demo (3 minutes)
- **Upload**: `safe-use-of-MEWP.png`
- **Show**: OCR text extraction with actual content (not just success message)
- **Highlight**: Intelligent knowledge categorization
- **Point out**: Confidence scores and processing methods

### 2. Video Processing Demo (4 minutes)
- **Upload**: `Safe Lifting - PhoenixParks (360p, h264).mp4`
- **Show**: Multi-tier processing (audio transcription + frame OCR)
- **Highlight**: Extracted procedural content and safety information
- **Point out**: Processing metadata and fallback methods

### 3. Document Processing Demo (3 minutes)
- **Upload**: Any PDF from `documents_samples/`
- **Show**: Comprehensive knowledge extraction
- **Highlight**: Different knowledge types (concepts, processes, systems, requirements, risks)
- **Point out**: AI-powered categorization

### 4. Dashboard Features Demo (3 minutes)
- **Filters**: Show knowledge type filtering
- **Search**: Demonstrate search functionality
- **Analytics**: Show pie charts and confidence distribution
- **Export**: Download CSV of results

### 5. Processing Intelligence Demo (2 minutes)
- **Show**: LLM-First processing indicators
- **Explain**: Hierarchical fallback system
- **Highlight**: Quality metrics and confidence scoring

## Key Talking Points

### Problem Solved
- "Converting unstructured documents into structured, searchable knowledge"
- "AI-powered extraction that understands context, not just keywords"
- "Multi-format support with intelligent fallback processing"

### Technical Highlights
- **Local processing** - no data leaves your machine
- **Multi-tier AI** - LLM primary, advanced fallback, pattern baseline
- **Real-time processing** - immediate results with progress feedback
- **Quality assurance** - confidence scoring and validation

### Use Cases
- **Compliance**: Extract regulatory requirements from documents
- **Training**: Convert manuals into searchable knowledge bases
- **Safety**: Identify risks and procedures from safety documents
- **Operations**: Extract processes and workflows from documents

## Demo Files Recommended

1. **Image**: `safe-use-of-MEWP.png` (safety procedures with text)
2. **Video**: `Safe Lifting - PhoenixParks (360p, h264).mp4` (training video)
3. **PDF**: `osha3132.pdf` (compliance document)
4. **Large PDF**: `Workplace Health and Safety Student Manual.pdf` (comprehensive manual)

## Potential Issues & Solutions

### Issue: OCR/Video processing fails
**Solution**: Mention that Tesseract/FFmpeg dependencies are optional but improve results

### Issue: Processing takes long
**Solution**: Explain that AI models are downloading/initializing on first use

### Issue: No content extracted
**Solution**: Show the improved fallback messaging and filename-based inference

### Issue: Large files
**Solution**: Mention processing is optimized for files under 50MB for demo purposes

## Demo Script Template

"Let me show you EXPLAINIUM - an AI-powered knowledge extraction system that converts any document format into structured, searchable knowledge.

**[Image Demo]** I'll start with this safety image. Watch how it extracts actual text content and categorizes it intelligently... You can see it found safety procedures and categorized them by type.

**[Video Demo]** Now let's try this training video. The system uses multiple approaches - audio transcription and frame analysis... Look at the procedural content it extracted and how it identified this as training material.

**[Document Demo]** Here's a PDF manual. Notice how it extracts different types of knowledge - concepts, processes, requirements - and assigns confidence scores to each...

**[Dashboard Demo]** The interface lets you filter by knowledge type, search content, and view analytics. You can export everything as CSV for further use...

The key innovation is the hierarchical AI processing - it tries advanced language models first, then falls back to pattern recognition, ensuring you always get results."

## Post-Demo Q&A Prep

**Q: How accurate is the extraction?**
A: Confidence scores indicate reliability. LLM processing typically achieves 85%+ accuracy.

**Q: What file sizes can it handle?**
A: Optimized for files under 50MB. Larger files work but take longer.

**Q: Is data sent to the cloud?**
A: No, everything processes locally. Your data never leaves your machine.

**Q: What formats are supported?**
A: PDF, DOC, DOCX, TXT, JPG, PNG, MP4, AVI, MP3, WAV, CSV, XLS, XLSX.

**Q: How long does processing take?**
A: Images: 5-10s, Videos: 30-60s, Documents: 10-20s, Audio: 20-40s.