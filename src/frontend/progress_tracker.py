"""
EXPLAINIUM - Progress Tracking Components

Professional progress tracking components for real-time file processing updates.
"""

import streamlit as st
import requests
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime


class ProgressTracker:
    """Professional progress tracking component for file processing"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        
    def upload_file_with_progress(self, uploaded_file, filename: str) -> Optional[str]:
        """
        Upload file to backend and return task_id for progress tracking
        
        Args:
            uploaded_file: Streamlit uploaded file object
            filename: Name of the file
            
        Returns:
            Task ID for tracking progress, or None if upload failed
        """
        try:
            # Prepare file for upload
            files = {
                'file': (filename, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            # Upload to backend
            response = requests.post(
                f"{self.api_base_url}/upload",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('task_id')
            else:
                st.error(f"Upload failed: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
            return None
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get current status of a processing task
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Dictionary with task status information
        """
        try:
            response = requests.get(
                f"{self.api_base_url}/tasks/{task_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'status': 'ERROR',
                    'message': f'Failed to get status: {response.text}',
                    'current': 0,
                    'total': 100
                }
                
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'Connection error: {str(e)}',
                'current': 0,
                'total': 100
            }
    
    def display_progress_bar(self, task_status: Dict[str, Any]) -> bool:
        """
        Display a professional progress bar with percentage and status
        
        Args:
            task_status: Task status dictionary
            
        Returns:
            True if task is complete, False if still processing
        """
        status = task_status.get('status', 'UNKNOWN')
        current = task_status.get('current', 0)
        total = task_status.get('total', 100)
        message = task_status.get('message', 'Processing...')
        
        # Calculate percentage
        if total > 0:
            percentage = int((current / total) * 100)
        else:
            percentage = 0
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Progress bar with custom styling
            if status == 'SUCCESS':
                st.success("✅ Processing completed successfully!")
                st.progress(1.0)
                return True
            elif status == 'FAILURE' or status == 'ERROR':
                st.error(f"❌ Processing failed: {message}")
                st.progress(0.0)
                return True
            elif status == 'PROGRESS' or status == 'RUNNING':
                # Show progress bar
                progress_value = percentage / 100.0
                st.progress(progress_value)
                
                # Status message with styling
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f0f2f6 0%, #e1e5eb 100%);
                    padding: 8px 12px;
                    border-radius: 4px;
                    border-left: 3px solid #1f77b4;
                    margin: 5px 0;
                ">
                    <small style="color: #555;">🔄 {message}</small>
                </div>
                """, unsafe_allow_html=True)
                
            else:  # PENDING
                st.info("⏳ Task is queued for processing...")
                st.progress(0.0)
        
        with col2:
            # Percentage display
            st.markdown(f"""
            <div style="
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                color: #1f77b4;
                background: #f8f9fa;
                padding: 10px;
                border-radius: 8px;
                border: 2px solid #e9ecef;
            ">
                {percentage}%
            </div>
            """, unsafe_allow_html=True)
        
        return False
    
    def track_processing_progress(self, task_id: str, container=None) -> Dict[str, Any]:
        """
        Track processing progress with real-time updates
        
        Args:
            task_id: Task ID to track
            container: Streamlit container for updates (optional)
            
        Returns:
            Final task result when complete
        """
        if container is None:
            container = st.empty()
        
        start_time = time.time()
        max_wait_time = 300  # 5 minutes max
        
        with container.container():
            st.markdown("### 📊 Processing Progress")
            
            # Create placeholders for dynamic updates
            progress_placeholder = st.empty()
            details_placeholder = st.empty()
            
            while True:
                # Check if we've exceeded max wait time
                if time.time() - start_time > max_wait_time:
                    with progress_placeholder.container():
                        st.error("⏰ Processing timeout - please check task status manually")
                    break
                
                # Get current status
                task_status = self.get_task_status(task_id)
                
                # Update progress display
                with progress_placeholder.container():
                    is_complete = self.display_progress_bar(task_status)
                
                # Show additional details
                with details_placeholder.container():
                    with st.expander("📋 Processing Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Task ID", task_id[:8] + "...")
                        with col2:
                            st.metric("Status", task_status.get('status', 'Unknown'))
                        with col3:
                            elapsed = int(time.time() - start_time)
                            st.metric("Elapsed Time", f"{elapsed}s")
                        
                        # Show result if available
                        if task_status.get('result'):
                            st.json(task_status['result'])
                
                # Check if complete
                if is_complete:
                    return task_status
                
                # Wait before next update
                time.sleep(2)
        
        return task_status


def create_professional_upload_interface():
    """Create a professional file upload interface with progress tracking"""
    
    st.markdown("""
    <style>
    .upload-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
    }
    .upload-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    .upload-subtitle {
        text-align: center;
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }
    .supported-formats {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-container">
        <div class="upload-title">🚀 EXPLAINIUM Knowledge Extraction</div>
        <div class="upload-subtitle">Upload your documents for AI-powered knowledge analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with enhanced styling
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=['pdf', 'txt', 'docx', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 
              'mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'flac'],
        help="Upload any supported media type for comprehensive knowledge extraction",
        label_visibility="collapsed"
    )
    
    # Show supported formats in a clean way
    with st.expander("📁 Supported File Formats", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**📄 Documents**\n- PDF\n- TXT\n- DOCX")
        with col2:
            st.markdown("**🖼️ Images**\n- JPG/JPEG\n- PNG\n- GIF/BMP/TIFF")
        with col3:
            st.markdown("**🎥 Videos**\n- MP4\n- AVI\n- MOV/MKV")
        with col4:
            st.markdown("**🎵 Audio**\n- MP3\n- WAV\n- FLAC")
    
    return uploaded_file


def display_processing_stats(task_result: Dict[str, Any]):
    """Display processing statistics in a professional format"""
    
    if not task_result or task_result.get('status') != 'SUCCESS':
        return
    
    result_data = task_result.get('result', {})
    
    st.markdown("### 📈 Processing Results")
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Knowledge Items",
            result_data.get('knowledge_extracted', 0),
            help="Total knowledge items extracted"
        )
    
    with col2:
        st.metric(
            "Processes",
            result_data.get('processes_created', 0),
            help="Business processes identified"
        )
    
    with col3:
        st.metric(
            "Decisions",
            result_data.get('decisions_created', 0),
            help="Decision points extracted"
        )
    
    with col4:
        st.metric(
            "Compliance",
            result_data.get('compliance_created', 0),
            help="Compliance requirements found"
        )
    
    # Show completion time
    if result_data.get('completed_at'):
        st.success(f"✅ Processing completed at {result_data['completed_at']}")


def show_processing_animation():
    """Show a professional processing animation"""
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div style="
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #1f77b4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    </div>
    """, unsafe_allow_html=True)