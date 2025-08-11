"""
Performance Monitoring System for Explainium
OPTIMIZED FOR SPEED - Real-time metrics and performance analysis

This module provides comprehensive performance monitoring including:
- Real-time processing metrics
- Memory usage tracking  
- Processing time breakdown
- Quality vs speed trade-off analysis
- Performance optimization recommendations
"""

import time
import psutil
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import json
import gc
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Individual processing step metrics"""
    step_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int
    memory_after: int
    memory_delta: int
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    quality_score: Optional[float] = None
    entities_extracted: int = 0
    confidence_score: Optional[float] = None

@dataclass
class DocumentProcessingStats:
    """Complete document processing statistics"""
    document_id: str
    file_path: str
    file_size: int
    file_type: str
    total_duration: float
    total_memory_peak: int
    total_cpu_peak: float
    steps: List[ProcessingMetrics] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    optimization_recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SystemPerformanceStats:
    """System-wide performance statistics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    memory_used: int
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int

class PerformanceMonitor:
    """
    High-performance monitoring system for Explainium
    
    Features:
    - Real-time metrics collection
    - Memory and CPU tracking
    - Performance bottleneck identification
    - Quality vs speed analysis
    - Optimization recommendations
    """
    
    def __init__(self, max_history: int = 1000, enable_system_monitoring: bool = True):
        self.max_history = max_history
        self.enable_system_monitoring = enable_system_monitoring
        
        # Performance tracking
        self.document_stats: Dict[str, DocumentProcessingStats] = {}
        self.system_stats: deque = deque(maxlen=max_history)
        self.performance_history: deque = deque(maxlen=max_history)
        
        # Real-time monitoring
        self.current_processing: Dict[str, DocumentProcessingStats] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.performance_thresholds = {
            'slow_processing': 120.0,  # 2 minutes
            'memory_warning': 0.8,     # 80% memory usage
            'cpu_warning': 0.9,        # 90% CPU usage
            'quality_threshold': 0.75   # Minimum quality score
        }
        
        # Statistics
        self.total_documents_processed = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        self.performance_issues = []
        
        # Start system monitoring if enabled
        if self.enable_system_monitoring:
            self._start_system_monitoring()
    
    def start_document_monitoring(self, document_id: str, file_path: str, file_size: int, file_type: str) -> str:
        """Start monitoring a document processing session"""
        try:
            stats = DocumentProcessingStats(
                document_id=document_id,
                file_path=str(file_path),
                file_size=file_size,
                file_type=file_type,
                total_duration=0.0,
                total_memory_peak=0,
                total_cpu_peak=0.0
            )
            
            self.current_processing[document_id] = stats
            logger.info(f"Started monitoring document: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to start document monitoring: {e}")
            return document_id
    
    def start_step_monitoring(self, document_id: str, step_name: str) -> str:
        """Start monitoring a processing step"""
        try:
            if document_id not in self.current_processing:
                logger.warning(f"Document {document_id} not found in current processing")
                return step_name
            
            # Get current system metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            step_id = f"{document_id}_{step_name}_{int(time.time())}"
            
            metrics = ProcessingMetrics(
                step_name=step_name,
                start_time=time.time(),
                end_time=0.0,
                duration=0.0,
                memory_before=memory_info.rss,
                memory_after=0,
                memory_delta=0,
                cpu_percent=cpu_percent,
                success=False
            )
            
            # Store step metrics for later completion
            if not hasattr(self, '_pending_steps'):
                self._pending_steps = {}
            self._pending_steps[step_id] = metrics
            
            return step_id
            
        except Exception as e:
            logger.error(f"Failed to start step monitoring: {e}")
            return step_name
    
    def end_step_monitoring(self, step_id: str, success: bool = True, 
                           error_message: Optional[str] = None,
                           quality_score: Optional[float] = None,
                           entities_extracted: int = 0,
                           confidence_score: Optional[float] = None) -> None:
        """Complete monitoring for a processing step"""
        try:
            if not hasattr(self, '_pending_steps') or step_id not in self._pending_steps:
                logger.warning(f"Step {step_id} not found in pending steps")
                return
            
            metrics = self._pending_steps[step_id]
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            metrics.success = success
            metrics.error_message = error_message
            metrics.quality_score = quality_score
            metrics.entities_extracted = entities_extracted
            metrics.confidence_score = confidence_score
            
            # Get final system metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics.memory_after = memory_info.rss
            metrics.memory_delta = metrics.memory_after - metrics.memory_before
            
            # Find the document this step belongs to
            document_id = step_id.split('_')[0]
            if document_id in self.current_processing:
                self.current_processing[document_id].steps.append(metrics)
                
                # Update peak metrics
                if metrics.memory_after > self.current_processing[document_id].total_memory_peak:
                    self.current_processing[document_id].total_memory_peak = metrics.memory_after
                
                if metrics.cpu_percent > self.current_processing[document_id].total_cpu_peak:
                    self.current_processing[document_id].total_cpu_peak = metrics.cpu_percent
            
            # Remove from pending
            del self._pending_steps[step_id]
            
        except Exception as e:
            logger.error(f"Failed to end step monitoring: {e}")
    
    def end_document_monitoring(self, document_id: str, quality_metrics: Optional[Dict[str, Any]] = None) -> DocumentProcessingStats:
        """Complete monitoring for a document processing session"""
        try:
            if document_id not in self.current_processing:
                logger.warning(f"Document {document_id} not found in current processing")
                return None
            
            stats = self.current_processing[document_id]
            
            # Calculate total duration
            if stats.steps:
                start_time = min(step.start_time for step in stats.steps)
                end_time = max(step.end_time for step in stats.steps)
                stats.total_duration = end_time - start_time
            
            # Add quality metrics
            if quality_metrics:
                stats.quality_metrics = quality_metrics
            
            # Calculate performance score
            stats.performance_score = self._calculate_performance_score(stats)
            
            # Generate optimization recommendations
            stats.optimization_recommendations = self._generate_optimization_recommendations(stats)
            
            # Store completed stats
            self.document_stats[document_id] = stats
            
            # Update global statistics
            self._update_global_stats(stats)
            
            # Remove from current processing
            del self.current_processing[document_id]
            
            logger.info(f"Completed monitoring document: {document_id} in {stats.total_duration:.2f}s")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to end document monitoring: {e}")
            return None
    
    def _calculate_performance_score(self, stats: DocumentProcessingStats) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            score = 100.0
            
            # Time penalty
            if stats.total_duration > self.performance_thresholds['slow_processing']:
                time_penalty = (stats.total_duration - self.performance_thresholds['slow_processing']) / 60.0 * 10
                score -= min(time_penalty, 40)  # Max 40 point penalty for time
            
            # Memory penalty
            memory_usage = stats.total_memory_peak / psutil.virtual_memory().total
            if memory_usage > self.performance_thresholds['memory_warning']:
                memory_penalty = (memory_usage - self.performance_thresholds['memory_warning']) * 100
                score -= min(memory_penalty, 20)  # Max 20 point penalty for memory
            
            # Quality bonus
            if stats.quality_metrics:
                quality_score = stats.quality_metrics.get('overall_confidence', 0.0)
                if quality_score > self.performance_thresholds['quality_threshold']:
                    score += min(quality_score * 10, 20)  # Max 20 point bonus for quality
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate performance score: {e}")
            return 50.0
    
    def _generate_optimization_recommendations(self, stats: DocumentProcessingStats) -> List[str]:
        """Generate optimization recommendations based on performance analysis"""
        recommendations = []
        
        try:
            # Time-based recommendations
            if stats.total_duration > self.performance_thresholds['slow_processing']:
                recommendations.append("Consider implementing content chunking for large documents")
                recommendations.append("Enable parallel processing for independent extraction steps")
                recommendations.append("Review LLM model parameters for M4 optimization")
            
            # Memory-based recommendations
            memory_usage = stats.total_memory_peak / psutil.virtual_memory().total
            if memory_usage > self.performance_thresholds['memory_warning']:
                recommendations.append("Implement memory-efficient content processing")
                recommendations.append("Consider streaming large document content")
                recommendations.append("Review and optimize image/video processing")
            
            # Step-specific recommendations
            slow_steps = [step for step in stats.steps if step.duration > 30.0]
            for step in slow_steps:
                if 'llm' in step.step_name.lower():
                    recommendations.append(f"Optimize LLM processing in {step.step_name}")
                elif 'extraction' in step.step_name.lower():
                    recommendations.append(f"Consider pattern-based extraction for {step.step_name}")
            
            # Quality-based recommendations
            if stats.quality_metrics:
                confidence = stats.quality_metrics.get('overall_confidence', 0.0)
                if confidence < self.performance_thresholds['quality_threshold']:
                    recommendations.append("Review extraction patterns for better accuracy")
                    recommendations.append("Consider adjusting LLM parameters for quality")
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
            return ["Review performance logs for optimization opportunities"]
    
    def _update_global_stats(self, stats: DocumentProcessingStats) -> None:
        """Update global performance statistics"""
        try:
            self.total_documents_processed += 1
            self.total_processing_time += stats.total_duration
            self.average_processing_time = self.total_processing_time / self.total_documents_processed
            
            # Track performance issues
            if stats.performance_score < 70:
                self.performance_issues.append({
                    'document_id': stats.document_id,
                    'performance_score': stats.performance_score,
                    'total_duration': stats.total_duration,
                    'timestamp': stats.timestamp
                })
            
            # Keep only recent issues
            if len(self.performance_issues) > 100:
                self.performance_issues = self.performance_issues[-100:]
                
        except Exception as e:
            logger.error(f"Failed to update global stats: {e}")
    
    def _start_system_monitoring(self) -> None:
        """Start background system monitoring"""
        try:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Started system performance monitoring")
        except Exception as e:
            logger.error(f"Failed to start system monitoring: {e}")
    
    def _system_monitor_loop(self) -> None:
        """Background system monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                system_stats = SystemPerformanceStats(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available=memory.available,
                    memory_used=memory.used,
                    disk_io_read=disk_io.read_bytes if disk_io else 0,
                    disk_io_write=disk_io.write_bytes if disk_io else 0,
                    network_io_sent=network_io.bytes_sent if network_io else 0,
                    network_io_recv=network_io.bytes_recv if network_io else 0
                )
                
                self.system_stats.append(system_stats)
                
                # Check for system warnings
                if memory.percent > self.performance_thresholds['memory_warning'] * 100:
                    logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                if cpu_percent > self.performance_thresholds['cpu_warning'] * 100:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                # Sleep for monitoring interval
                time.sleep(5)  # 5 second intervals
                
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                'overview': {
                    'total_documents_processed': self.total_documents_processed,
                    'average_processing_time': self.average_processing_time,
                    'current_processing_count': len(self.current_processing),
                    'performance_issues_count': len(self.performance_issues)
                },
                'recent_performance': {
                    'last_10_documents': [],
                    'performance_trend': 'stable'
                },
                'system_health': {
                    'current_cpu': 0.0,
                    'current_memory': 0.0,
                    'system_warnings': []
                },
                'optimization_recommendations': []
            }
            
            # Recent performance
            recent_docs = list(self.document_stats.values())[-10:]
            summary['recent_performance']['last_10_documents'] = [
                {
                    'document_id': doc.document_id,
                    'duration': doc.total_duration,
                    'performance_score': doc.performance_score,
                    'file_type': doc.file_type
                }
                for doc in recent_docs
            ]
            
            # Performance trend
            if len(recent_docs) >= 2:
                recent_avg = sum(doc.total_duration for doc in recent_docs[-5:]) / 5
                older_avg = sum(doc.total_duration for doc in recent_docs[-10:-5]) / 5
                if recent_avg < older_avg * 0.9:
                    summary['recent_performance']['performance_trend'] = 'improving'
                elif recent_avg > older_avg * 1.1:
                    summary['recent_performance']['performance_trend'] = 'degrading'
            
            # System health
            if self.system_stats:
                latest_system = self.system_stats[-1]
                summary['system_health']['current_cpu'] = latest_system.cpu_percent
                summary['system_health']['current_memory'] = latest_system.memory_percent
                
                if latest_system.memory_percent > 80:
                    summary['system_health']['system_warnings'].append("High memory usage")
                if latest_system.cpu_percent > 90:
                    summary['system_health']['system_warnings'].append("High CPU usage")
            
            # Top optimization recommendations
            all_recommendations = []
            for doc_stats in self.document_stats.values():
                all_recommendations.extend(doc_stats.optimization_recommendations)
            
            # Count and sort recommendations
            rec_count = defaultdict(int)
            for rec in all_recommendations:
                rec_count[rec] += 1
            
            top_recommendations = sorted(rec_count.items(), key=lambda x: x[1], reverse=True)[:5]
            summary['optimization_recommendations'] = [rec for rec, count in top_recommendations]
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {'error': str(e)}
    
    def get_document_performance(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed performance data for a specific document"""
        try:
            if document_id not in self.document_stats:
                return None
            
            stats = self.document_stats[document_id]
            
            return {
                'document_id': stats.document_id,
                'file_path': stats.file_path,
                'file_size': stats.file_size,
                'file_type': stats.file_type,
                'total_duration': stats.total_duration,
                'performance_score': stats.performance_score,
                'memory_peak': stats.total_memory_peak,
                'cpu_peak': stats.total_cpu_peak,
                'steps': [
                    {
                        'name': step.step_name,
                        'duration': step.duration,
                        'memory_delta': step.memory_delta,
                        'success': step.success,
                        'quality_score': step.quality_score,
                        'entities_extracted': step.entities_extracted
                    }
                    for step in stats.steps
                ],
                'quality_metrics': stats.quality_metrics,
                'optimization_recommendations': stats.optimization_recommendations,
                'timestamp': stats.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get document performance: {e}")
            return None
    
    def export_performance_data(self, output_path: str) -> bool:
        """Export performance data to JSON file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'document_stats': {
                    doc_id: {
                        'file_path': stats.file_path,
                        'file_size': stats.file_size,
                        'file_type': stats.file_type,
                        'total_duration': stats.total_duration,
                        'performance_score': stats.performance_score,
                        'timestamp': stats.timestamp.isoformat()
                    }
                    for doc_id, stats in self.document_stats.items()
                },
                'system_stats': [
                    {
                        'timestamp': stats.timestamp.isoformat(),
                        'cpu_percent': stats.cpu_percent,
                        'memory_percent': stats.memory_percent,
                        'memory_available': stats.memory_available,
                        'memory_used': stats.memory_used
                    }
                    for stats in list(self.system_stats)[-100:]  # Last 100 system stats
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance data exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup monitoring resources"""
        try:
            self.monitoring_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            # Clear memory
            self.document_stats.clear()
            self.system_stats.clear()
            self.performance_history.clear()
            self.current_processing.clear()
            
            if hasattr(self, '_pending_steps'):
                self._pending_steps.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Performance monitor cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def cleanup_performance_monitor() -> None:
    """Cleanup global performance monitor"""
    global _performance_monitor
    if _performance_monitor:
        _performance_monitor.cleanup()
        _performance_monitor = None