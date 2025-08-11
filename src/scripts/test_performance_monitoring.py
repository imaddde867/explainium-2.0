#!/usr/bin/env python3
"""
Test Performance Monitoring System
==================================

This script demonstrates the performance monitoring capabilities
by processing sample documents and showing real-time metrics.

Usage:
    python src/scripts/test_performance_monitoring.py
"""

import sys
import os
import time
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.performance_monitor import get_performance_monitor
from src.processors.processor import DocumentProcessor

def create_sample_document(content: str, filename: str) -> Path:
    """Create a sample text document for testing"""
    temp_dir = Path(tempfile.gettempdir())
    file_path = temp_dir / filename
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return file_path

def create_sample_pdf_content() -> str:
    """Create sample PDF-like content"""
    return """
    TECHNICAL SPECIFICATIONS DOCUMENT
    
    Equipment: Industrial Pump System
    Model: XP-2000
    Serial Number: SN-2024-001
    
    Technical Specifications:
    - Power: 150 HP
    - Voltage: 480V
    - Flow Rate: 500 GPM
    - Pressure: 150 PSI
    - Temperature Range: 32°F to 212°F
    
    Safety Requirements:
    - WARNING: High voltage equipment
    - CAUTION: Hot surfaces during operation
    - DANGER: Rotating parts
    
    Maintenance Procedures:
    1. Daily inspection of pressure gauges
    2. Weekly lubrication of bearings
    3. Monthly filter replacement
    4. Quarterly pump alignment check
    
    Personnel Requirements:
    - Certified electrician for electrical work
    - Trained operator for daily operations
    - Maintenance technician for repairs
    
    This document contains critical information for safe operation.
    """

def create_sample_manual_content() -> str:
    """Create sample manual content"""
    return """
    OPERATION MANUAL - CONVEYOR SYSTEM
    
    System Overview:
    The conveyor system transports materials between processing stations.
    
    Operating Parameters:
    - Speed: 60 feet per minute
    - Capacity: 2000 lbs per hour
    - Motor: 25 HP, 3-phase
    - Voltage: 460V AC
    
    Safety Procedures:
    - WARNING: Keep hands clear of moving parts
    - CAUTION: Check emergency stop before operation
    - DANGER: Lock out power before maintenance
    
    Operating Instructions:
    1. Verify all safety guards are in place
    2. Check emergency stop button functionality
    3. Start system using control panel
    4. Monitor operation for unusual sounds
    5. Stop system using emergency stop if needed
    
    Troubleshooting:
    - Belt slipping: Check tension and alignment
    - Motor overheating: Check ventilation and load
    - Unusual vibration: Check bearings and alignment
    """

async def test_document_processing(processor: DocumentProcessor, monitor, content: str, filename: str, doc_id: int):
    """Test document processing with performance monitoring"""
    print(f"\n📄 Processing: {filename}")
    print("-" * 50)
    
    try:
        # Create temporary document
        file_path = create_sample_document(content, filename)
        
        # Process document
        start_time = time.time()
        result = await processor.process_document_async(str(file_path), doc_id)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 File type: {result.get('file_type', 'unknown')}")
        print(f"🧠 Knowledge entities: {len(result.get('knowledge', {}).get('entities', []))}")
        print(f"🎯 Processing method: {result.get('processing_method', 'unknown')}")
        
        # Clean up temporary file
        file_path.unlink()
        
        return result
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return None

async def main():
    """Main test function"""
    print("🧪 EXPLAINIUM PERFORMANCE MONITORING TEST")
    print("=" * 60)
    
    try:
        # Initialize systems
        print("🔄 Initializing performance monitor...")
        monitor = get_performance_monitor()
        
        print("🔄 Initializing document processor...")
        processor = DocumentProcessor()
        
        print("✅ Systems initialized successfully!")
        
        # Test documents
        test_documents = [
            ("Sample Technical Specs.pdf", create_sample_pdf_content(), 1001),
            ("Sample Operation Manual.pdf", create_sample_manual_content(), 1002),
            ("Sample Technical Specs.pdf", create_sample_pdf_content(), 1003),  # Test caching
        ]
        
        # Process documents
        results = []
        for filename, content, doc_id in test_documents:
            result = await test_document_processing(processor, monitor, content, filename, doc_id)
            if result:
                results.append(result)
            
            # Wait a bit between documents
            await asyncio.sleep(2)
        
        # Show performance summary
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE MONITORING RESULTS")
        print("=" * 60)
        
        summary = monitor.get_performance_summary()
        
        if 'error' not in summary:
            overview = summary.get('overview', {})
            print(f"📄 Total Documents Processed: {overview.get('total_documents_processed', 0)}")
            print(f"⏱️  Average Processing Time: {overview.get('average_processing_time', 0):.2f}s")
            print(f"🔄 Currently Processing: {overview.get('current_processing_count', 0)}")
            
            # Show recent documents
            recent_docs = summary.get('recent_performance', {}).get('last_10_documents', [])
            if recent_docs:
                print(f"\n📋 Recent Processing Results:")
                for doc in recent_docs:
                    print(f"  - {doc.get('document_id', 'N/A')}: {doc.get('duration', 0):.2f}s, Score: {doc.get('performance_score', 0):.0f}")
            
            # Show optimization recommendations
            recommendations = summary.get('optimization_recommendations', [])
            if recommendations:
                print(f"\n💡 Optimization Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec}")
        
        # Test detailed document performance
        if results:
            print(f"\n🔍 Detailed Performance Analysis:")
            for result in results:
                doc_id = result.get('document_id')
                if doc_id:
                    doc_perf = monitor.get_document_performance(str(doc_id))
                    if doc_perf:
                        print(f"\n📄 Document {doc_id}:")
                        print(f"  - Duration: {doc_perf.get('total_duration', 0):.2f}s")
                        print(f"  - Performance Score: {doc_perf.get('performance_score', 0):.0f}")
                        print(f"  - Memory Peak: {doc_perf.get('memory_peak', 0) / (1024*1024):.1f} MB")
                        
                        steps = doc_perf.get('steps', [])
                        if steps:
                            print(f"  - Processing Steps:")
                            for step in steps:
                                print(f"    * {step.get('name', 'Unknown')}: {step.get('duration', 0):.2f}s")
        
        # Export performance data
        export_path = f"performance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if monitor.export_performance_data(export_path):
            print(f"\n📊 Performance data exported to: {export_path}")
        
        print("\n✅ Performance monitoring test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            if 'processor' in locals():
                processor.cleanup()
            if 'monitor' in locals():
                monitor.cleanup()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())