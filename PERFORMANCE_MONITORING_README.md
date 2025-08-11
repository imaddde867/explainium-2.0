# Explainium Performance Monitoring System

## Overview

The performance monitoring system provides real-time insights into document processing performance, helping identify bottlenecks and optimization opportunities. It tracks:

- **Document-level metrics**: Processing time, memory usage, CPU usage
- **Step-level metrics**: Individual processing step performance
- **System metrics**: Overall system health and resource usage
- **Quality metrics**: Confidence scores and entity extraction counts
- **Optimization recommendations**: AI-powered suggestions for improvement

## Quick Start

### 1. Basic Usage

```python
from src.core.performance_monitor import get_performance_monitor
from src.processors.processor import DocumentProcessor

# Initialize systems
monitor = get_performance_monitor()
processor = DocumentProcessor()

# Process documents (monitoring happens automatically)
result = await processor.process_document_async("document.pdf", 123)

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Average processing time: {summary['overview']['average_processing_time']:.2f}s")

# Cleanup when done
processor.cleanup()
monitor.cleanup()
```

### 2. View Real-time Dashboard

```bash
# Run the interactive dashboard
python src/scripts/performance_dashboard.py
```

### 3. Test the System

```bash
# Run performance tests with sample documents
python src/scripts/test_performance_monitoring.py
```

## Key Features

### Automatic Monitoring

The system automatically tracks:
- **Document processing**: Start/end times, file metadata, processing method
- **Processing steps**: Content extraction, LLM processing, knowledge extraction
- **Resource usage**: Memory consumption, CPU usage, processing duration
- **Quality metrics**: Confidence scores, entity counts, success rates

### Performance Analysis

```python
# Get comprehensive performance summary
summary = monitor.get_performance_summary()

# Get specific document performance
doc_perf = monitor.get_document_performance("document_123")

# Export data for analysis
monitor.export_performance_data("performance_report.json")
```

### Optimization Recommendations

The system automatically generates recommendations based on:
- Processing time thresholds
- Memory usage patterns
- Step-specific bottlenecks
- Quality vs. speed trade-offs

## Performance Metrics

### Document Processing Stats

```python
{
    "document_id": "123",
    "file_path": "/path/to/document.pdf",
    "file_size": 1024000,
    "file_type": "pdf",
    "total_duration": 45.2,
    "performance_score": 85.0,
    "total_memory_peak": 512000000,
    "total_cpu_peak": 75.5,
    "steps": [...],
    "quality_metrics": {...},
    "optimization_recommendations": [...]
}
```

### Step-level Metrics

```python
{
    "step_name": "content_extraction",
    "duration": 12.3,
    "memory_delta": 25600000,
    "success": True,
    "quality_score": 0.92,
    "entities_extracted": 45,
    "confidence_score": 0.88
}
```

### System Performance Stats

```python
{
    "timestamp": "2024-01-15T10:30:00",
    "cpu_percent": 45.2,
    "memory_percent": 68.5,
    "memory_available": 8589934592,
    "memory_used": 18522046464,
    "disk_io_read": 1073741824,
    "disk_io_write": 536870912
}
```

## Configuration

### Performance Thresholds

```python
# Default thresholds (can be customized)
performance_thresholds = {
    'slow_processing': 120.0,      # 2 minutes
    'memory_warning': 0.8,         # 80% of RAM
    'cpu_warning': 0.9,            # 90% CPU
    'quality_threshold': 0.75      # 75% confidence
}
```

### Monitoring Intervals

- **System monitoring**: Every 5 seconds
- **Document processing**: Real-time (start/end events)
- **Step monitoring**: Real-time (start/end events)

## Integration Points

### Document Processor

The `DocumentProcessor` automatically integrates with performance monitoring:

```python
# In process_document_async method:
document_id_str = str(document_id)
self.performance_monitor.start_document_monitoring(
    document_id_str, str(file_path), file_size, file_type
)

# ... processing ...

self.performance_monitor.end_document_monitoring(document_id_str, quality_metrics)
```

### Processing Steps

Individual processing steps are monitored:

```python
# Start monitoring a step
step_id = self.performance_monitor.start_step_monitoring(
    document_id, "content_extraction"
)

# ... processing ...

# End monitoring with results
self.performance_monitor.end_step_monitoring(
    step_id,
    success=True,
    entities_extracted=len(entities),
    quality_score=confidence
)
```

## Best Practices

### 1. Resource Management

```python
# Always cleanup when done
try:
    # Process documents
    result = await processor.process_document_async(file_path, doc_id)
finally:
    processor.cleanup()
    monitor.cleanup()
```

### 2. Error Handling

```python
try:
    step_id = monitor.start_step_monitoring(doc_id, "llm_processing")
    # ... processing ...
    monitor.end_step_monitoring(step_id, success=True)
except Exception as e:
    # End monitoring with error
    if 'step_id' in locals():
        monitor.end_step_monitoring(step_id, success=False, error_message=str(e))
```

### 3. Performance Analysis

```python
# Regular performance reviews
summary = monitor.get_performance_summary()
if summary['overview']['performance_issues_count'] > 5:
    print("⚠️  Performance issues detected - review recommendations")

# Export data for external analysis
monitor.export_performance_data(f"performance_{datetime.now().strftime('%Y%m%d')}.json")
```

## Troubleshooting

### Common Issues

1. **Memory leaks**: Check if `cleanup()` is called properly
2. **High CPU usage**: Review processing steps for optimization opportunities
3. **Slow processing**: Check optimization recommendations
4. **Missing metrics**: Ensure all steps are properly wrapped with monitoring

### Debug Mode

```python
import logging
logging.getLogger('src.core.performance_monitor').setLevel(logging.DEBUG)
```

## Performance Targets

Based on the optimization goals:

- **Target processing time**: < 2 minutes per document
- **Speed improvement**: 5x faster than current baseline
- **Memory usage**: < 80% of available RAM
- **CPU usage**: < 90% during processing
- **Quality score**: > 75% confidence
- **Cache hit rate**: > 60% for repeated content

## Next Steps

The performance monitoring system provides the foundation for:

1. **Phase 3 Advanced Optimizations**:
   - Model quantization
   - Advanced caching strategies
   - Streaming processing

2. **Continuous Improvement**:
   - Performance trend analysis
   - Automated optimization suggestions
   - Resource usage optimization

3. **Production Monitoring**:
   - Real-time alerting
   - Performance dashboards
   - Historical trend analysis

## API Reference

### PerformanceMonitor Class

```python
class PerformanceMonitor:
    def start_document_monitoring(self, document_id: str, file_path: str, 
                                file_size: int, file_type: str) -> None
    
    def end_document_monitoring(self, document_id: str, 
                               quality_metrics: Optional[Dict[str, Any]] = None) -> DocumentProcessingStats
    
    def start_step_monitoring(self, document_id: str, step_name: str) -> str
    
    def end_step_monitoring(self, step_id: str, success: bool = True, 
                           error_message: Optional[str] = None,
                           quality_score: Optional[float] = None,
                           entities_extracted: int = 0,
                           confidence_score: Optional[float] = None) -> None
    
    def get_performance_summary(self) -> Dict[str, Any]
    
    def get_document_performance(self, document_id: str) -> Optional[Dict[str, Any]]
    
    def export_performance_data(self, output_path: str) -> bool
    
    def cleanup(self) -> None
```

For more detailed information, see the source code in `src/core/performance_monitor.py`.