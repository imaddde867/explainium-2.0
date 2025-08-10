#!/usr/bin/env python3
"""
Test script for the new knowledge extraction system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.processors.unified_document_processor import UnifiedDocumentProcessor
from src.logging_config import setup_root_logging


async def test_knowledge_extraction():
    """Test the knowledge extraction system with AG1157.pdf"""
    
    # Setup logging
    setup_root_logging('INFO')
    
    # Initialize processor
    processor = UnifiedDocumentProcessor()
    
    # Test file path
    test_file = "documents_samples/AG1157.pdf"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    print(f"Testing knowledge extraction with: {test_file}")
    print("=" * 60)
    
    try:
        # Process the document
        result = await processor.process_document(test_file, 1)
        
        print("✅ Document processed successfully!")
        print(f"File type: {result['file_type']}")
        print(f"Processing status: {result['processing_status']}")
        
        # Display extracted knowledge
        knowledge = result['knowledge']
        print("\n📚 EXTRACTED KNOWLEDGE:")
        print("=" * 60)
        
        # Check if there was an error in knowledge extraction
        if 'error' in knowledge:
            print(f"❌ Knowledge extraction error: {knowledge['error']}")
            return
        
        # Display the formatted output if available
        if 'formatted_output' in knowledge and knowledge['formatted_output']:
            print("\n🎯 STRUCTURED KNOWLEDGE:")
            print(knowledge['formatted_output'])
        else:
            print("⚠️ No formatted knowledge output available")
        
        # Display raw knowledge structure for debugging
        if 'raw_knowledge' in knowledge:
            raw = knowledge['raw_knowledge']
            print(f"\n🔍 RAW KNOWLEDGE STRUCTURE:")
            print(f"  • Document type: {raw.get('document_type', 'Unknown')}")
            print(f"  • Content length: {raw.get('content_length', 'Unknown')}")
            print(f"  • Extraction timestamp: {raw.get('extraction_timestamp', 'Unknown')}")
            
            # Display categories
            categories = raw.get('categories', {})
            if categories:
                print(f"\n📂 KNOWLEDGE CATEGORIES:")
                for category_name, category_data in categories.items():
                    items = category_data.get('items', [])
                    count = category_data.get('count', 0)
                    confidence = category_data.get('confidence', 0.0)
                    print(f"  • {category_name.title()}: {count} items (confidence: {confidence:.2f})")
                    
                    # Show first few items in each category
                    if items:
                        for i, item in enumerate(items[:3]):  # Show first 3 items
                            name = item.get('name', 'Unknown')
                            desc = item.get('description', '')[:100]  # Truncate description
                            if desc and len(desc) == 100:
                                desc += "..."
                            print(f"    {i+1}. {name}: {desc}")
            else:
                print("  • No categories found")
        
        # Display confidence score
        if 'extraction_confidence' in knowledge:
            confidence = knowledge['extraction_confidence']
            print(f"\n🎯 EXTRACTION CONFIDENCE: {confidence:.2f}")
        
        # Display content summary
        content = result['content']
        print(f"\n📄 CONTENT SUMMARY:")
        print(f"  • Pages: {content.get('page_count', 'Unknown')}")
        print(f"  • Text length: {len(content.get('text', ''))} characters")
        print(f"  • Extraction method: {content.get('extraction_method', 'Unknown')}")
        
        # Display processing stats
        stats = processor.get_processing_stats()
        print(f"\n🔧 PROCESSING CAPABILITIES:")
        print(f"  • OCR available: {stats['ocr_available']}")
        print(f"  • Audio processing: {stats['audio_processing_available']}")
        print(f"  • Supported formats: {stats['supported_formats']}")
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_knowledge_extraction())