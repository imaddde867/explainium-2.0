#!/bin/bash

echo "🚀 EXPLAINIUM Demo Startup"
echo "============================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📚 Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if Tesseract is available
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Warning: Tesseract OCR not found. Image processing will be limited."
    echo "   Install with: sudo apt-get install tesseract-ocr (Ubuntu/Debian)"
    echo "                brew install tesseract (macOS)"
else
    echo "✅ Tesseract OCR found"
fi

# Check if FFmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  Warning: FFmpeg not found. Video audio extraction will be limited."
    echo "   Install with: sudo apt-get install ffmpeg (Ubuntu/Debian)"
    echo "                brew install ffmpeg (macOS)"
else
    echo "✅ FFmpeg found"
fi

echo ""
echo "🌟 Starting EXPLAINIUM Demo..."
echo "📊 Access the dashboard at: http://localhost:8501"
echo "📁 Sample files available in: documents_samples/"
echo ""
echo "🔧 Demo Features:"
echo "   ✓ Image OCR with intelligent text extraction"
echo "   ✓ Video processing with audio transcription and frame analysis"
echo "   ✓ Audio transcription with Whisper AI"
echo "   ✓ PDF, document, and spreadsheet processing"
echo "   ✓ AI-powered knowledge extraction and categorization"
echo ""

# Run the application
streamlit run src/frontend/knowledge_table.py --server.port 8501 --server.address 0.0.0.0