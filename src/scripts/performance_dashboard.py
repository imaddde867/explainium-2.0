#!/usr/bin/env python3
"""
Performance Dashboard for Explainium
====================================

Real-time performance monitoring and analysis dashboard.
Shows the results of our performance optimizations.

Usage:
    python src/scripts/performance_dashboard.py
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.performance_monitor import get_performance_monitor, cleanup_performance_monitor
from src.processors.processor import DocumentProcessor

def print_header():
    """Print dashboard header"""
    print("=" * 80)
    print("🚀 EXPLAINIUM PERFORMANCE DASHBOARD")
    print("=" * 80)
    print(f"📊 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def print_performance_summary(monitor):
    """Print performance summary"""
    print("\n📈 PERFORMANCE OVERVIEW")
    print("-" * 40)
    
    try:
        summary = monitor.get_performance_summary()
        
        if 'error' in summary:
            print(f"❌ Error: {summary['error']}")
            return
        
        overview = summary.get('overview', {})
        print(f"📄 Total Documents Processed: {overview.get('total_documents_processed', 0)}")
        print(f"⏱️  Average Processing Time: {overview.get('average_processing_time', 0):.2f}s")
        print(f"🔄 Currently Processing: {overview.get('current_processing_count', 0)}")
        print(f"⚠️  Performance Issues: {overview.get('performance_issues_count', 0)}")
        
        # Recent performance
        recent = summary.get('recent_performance', {})
        print(f"\n📊 Recent Performance Trend: {recent.get('performance_trend', 'stable').upper()}")
        
        # System health
        system = summary.get('system_health', {})
        print(f"💻 Current CPU: {system.get('current_cpu', 0):.1f}%")
        print(f"🧠 Current Memory: {system.get('current_memory', 0):.1f}%")
        
        if system.get('system_warnings'):
            print(f"⚠️  System Warnings: {', '.join(system['system_warnings'])}")
        
    except Exception as e:
        print(f"❌ Error getting performance summary: {e}")

def print_optimization_recommendations(monitor):
    """Print optimization recommendations"""
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    try:
        summary = monitor.get_performance_summary()
        
        if 'error' in summary:
            return
        
        recommendations = summary.get('optimization_recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ No optimization recommendations at this time")
            
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")

def print_recent_documents(monitor):
    """Print recent document processing results"""
    print("\n📋 RECENT DOCUMENTS")
    print("-" * 40)
    
    try:
        summary = monitor.get_performance_summary()
        
        if 'error' in summary:
            return
        
        recent_docs = summary.get('recent_performance', {}).get('last_10_documents', [])
        
        if recent_docs:
            print(f"{'ID':<15} {'Duration':<12} {'Score':<8} {'Type':<15}")
            print("-" * 60)
            
            for doc in recent_docs[-5:]:  # Show last 5
                doc_id = doc.get('document_id', 'N/A')[:14]
                duration = f"{doc.get('duration', 0):.1f}s"
                score = f"{doc.get('performance_score', 0):.0f}"
                file_type = doc.get('file_type', 'N/A')[:14]
                
                print(f"{doc_id:<15} {duration:<12} {score:<8} {file_type:<15}")
        else:
            print("📝 No documents processed yet")
            
    except Exception as e:
        print(f"❌ Error getting recent documents: {e}")

def print_optimization_status():
    """Print current optimization status"""
    print("\n⚡ OPTIMIZATION STATUS")
    print("-" * 40)
    
    optimizations = [
        "✅ Async Document Processing Pipeline",
        "✅ Parallel Content Extraction", 
        "✅ Smart File Type Detection with Caching",
        "✅ Intelligent Fallback Routing",
        "✅ M4 Chip Optimizations (MLX)",
        "✅ Content Chunking for LLM Processing",
        "✅ Parallel Entity Extraction",
        "✅ Intelligent Processing Decisions",
        "✅ Performance Monitoring System",
        "✅ Memory-Efficient Processing",
        "✅ Pre-compiled Regex Patterns",
        "✅ Lazy Loading for Heavy Dependencies"
    ]
    
    for opt in optimizations:
        print(opt)

def print_target_metrics():
    """Print target performance metrics"""
    print("\n🎯 TARGET PERFORMANCE METRICS")
    print("-" * 40)
    
    targets = [
        "⏱️  Target Processing Time: < 2 minutes per document",
        "🚀 Speed Improvement Target: 5x faster than current",
        "🧠 Memory Usage: < 80% of available RAM",
        "💻 CPU Usage: < 90% during processing",
        "📊 Quality Score: > 75% confidence",
        "🔄 Cache Hit Rate: > 60% for repeated content"
    ]
    
    for target in targets:
        print(target)

def print_usage_instructions():
    """Print usage instructions"""
    print("\n📖 USAGE INSTRUCTIONS")
    print("-" * 40)
    
    instructions = [
        "1. Start the performance monitor: monitor = get_performance_monitor()",
        "2. Process documents: processor.process_document_async()",
        "3. View real-time metrics: monitor.get_performance_summary()",
        "4. Export data: monitor.export_performance_data('output.json')",
        "5. Cleanup: monitor.cleanup()"
    ]
    
    for instruction in instructions:
        print(instruction)

def print_footer():
    """Print dashboard footer"""
    print("\n" + "=" * 80)
    print("🔍 For detailed analysis, use monitor.get_document_performance(doc_id)")
    print("📊 Export data with monitor.export_performance_data('filename.json')")
    print("🧹 Cleanup with monitor.cleanup() when done")
    print("=" * 80)

def main():
    """Main dashboard function"""
    try:
        print_header()
        
        # Initialize performance monitor
        print("🔄 Initializing performance monitor...")
        monitor = get_performance_monitor()
        
        # Initialize processor for demo
        print("🔄 Initializing document processor...")
        processor = DocumentProcessor()
        
        print("✅ Systems initialized successfully!")
        
        # Display dashboard sections
        print_performance_summary(monitor)
        print_optimization_recommendations(monitor)
        print_recent_documents(monitor)
        print_optimization_status()
        print_target_metrics()
        print_usage_instructions()
        print_footer()
        
        # Interactive mode
        print("\n🔄 Dashboard will refresh every 10 seconds. Press Ctrl+C to exit.")
        
        try:
            while True:
                time.sleep(10)
                print("\n" + "="*80)
                print(f"🔄 Refreshing at {datetime.now().strftime('%H:%M:%S')}")
                print_performance_summary(monitor)
                print_recent_documents(monitor)
                
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down dashboard...")
            
    except Exception as e:
        print(f"❌ Error in dashboard: {e}")
        
    finally:
        # Cleanup
        try:
            if 'processor' in locals():
                processor.cleanup()
            if 'monitor' in locals():
                monitor.cleanup()
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")

if __name__ == "__main__":
    main()