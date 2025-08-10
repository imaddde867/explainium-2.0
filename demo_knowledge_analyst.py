#!/usr/bin/env python3
"""
EXPLAINIUM - AI Knowledge Analyst Demo

Demo script to test the new AI Knowledge Analyst functionality
with sample documents to showcase the 3-phase framework.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from src.ai.knowledge_analyst import AIKnowledgeAnalyst, DocumentType, InformationType
from src.core.config import AIConfig


# Sample documents for testing
SAMPLE_DOCUMENTS = {
    "safety_manual": """
    WORKPLACE SAFETY MANUAL
    
    This manual provides essential safety guidelines for all employees working in our manufacturing facility.
    
    EMERGENCY PROCEDURES:
    
    1. In case of fire, immediately evacuate the building using the nearest exit
    2. Call emergency services at 911
    3. Report to the designated assembly point in the parking lot
    4. Wait for further instructions from safety personnel
    
    MANDATORY SAFETY REQUIREMENTS:
    
    All employees must wear protective equipment at all times while on the production floor.
    Hard hats are required in all designated areas.
    Safety glasses must be worn when operating machinery.
    
    ROLES AND RESPONSIBILITIES:
    
    Safety Manager is responsible for conducting weekly safety inspections.
    Supervisors must ensure all safety protocols are followed.
    Employees shall report any safety hazards immediately.
    
    RISK ASSESSMENT:
    
    Risk: Exposure to hazardous chemicals may cause respiratory issues.
    Corrective Action: Use proper ventilation systems and respiratory protection.
    
    Risk: Machinery malfunction could result in injury.
    Corrective Action: Perform daily equipment inspections and maintenance.
    
    DEFINITIONS:
    
    PPE: Personal Protective Equipment refers to specialized clothing or equipment worn for protection.
    MSDS: Material Safety Data Sheet means documentation containing chemical hazard information.
    
    KEY METRICS:
    
    Temperature in chemical storage areas must not exceed 75°F.
    Maximum noise level allowed is 85 decibels.
    Emergency response time should be under 3 minutes.
    """,
    
    "project_report": """
    PROJECT STATUS REPORT - Q4 2024
    
    This report presents the findings and recommendations for the digital transformation initiative.
    
    EXECUTIVE SUMMARY:
    
    The project has achieved 87% completion rate with a budget utilization of $2.4 million out of $3.0 million allocated.
    
    KEY FINDINGS:
    
    Implementation of the new CRM system has improved customer response time by 40%.
    Employee productivity has increased by 25% since the automation rollout.
    Cost savings of $500,000 annually have been realized through process optimization.
    
    RECOMMENDATIONS:
    
    1. Accelerate the deployment of remaining modules by January 15, 2025
    2. Conduct comprehensive user training for all departments
    3. Establish performance monitoring dashboards
    
    RISKS AND MITIGATION:
    
    Risk: Potential data migration issues during final deployment.
    Mitigation: Implement comprehensive backup procedures and staged rollout approach.
    
    PROJECT TEAM:
    
    Project Manager is responsible for overall project coordination and timeline management.
    Technical Lead must oversee system integration and technical implementation.
    Change Management Specialist shall facilitate user adoption and training programs.
    """,
    
    "contract_agreement": """
    SOFTWARE LICENSE AGREEMENT
    
    This agreement establishes the terms and conditions for software licensing between the parties.
    
    DEFINITIONS:
    
    Licensee means the entity obtaining rights to use the software.
    Licensor refers to the company granting the software usage rights.
    Software means the proprietary application and associated documentation.
    
    TERMS AND CONDITIONS:
    
    The Licensee must comply with all usage restrictions outlined in this agreement.
    Software shall not be redistributed or reverse-engineered without written consent.
    License fees are due within 30 days of invoice date.
    
    LIABILITY AND INDEMNIFICATION:
    
    Licensor shall not be liable for indirect damages exceeding $100,000.
    Licensee must indemnify Licensor against third-party claims.
    
    TERMINATION CONDITIONS:
    
    This agreement may be terminated with 60 days written notice.
    Upon termination, Licensee must cease all software usage and destroy all copies.
    """
}


async def demo_ai_knowledge_analyst():
    """Demo the AI Knowledge Analyst with sample documents"""
    print("=" * 70)
    print("EXPLAINIUM - AI Knowledge Analyst Demo")
    print("=" * 70)
    print()
    
    # Create a mock AI config
    class MockAIConfig:
        def __init__(self):
            self.llm_path = None  # Will use fallback methods
            self.quantization = "4bit"
            self.embedding_model = "bge-small"
    
    # Initialize the AI Knowledge Analyst
    config = MockAIConfig()
    analyst = AIKnowledgeAnalyst(config)
    
    try:
        await analyst.initialize()
        print("✅ AI Knowledge Analyst initialized successfully")
    except Exception as e:
        print(f"⚠️  AI models not available, using pattern-based analysis: {e}")
    
    print()
    
    # Process each sample document
    for doc_name, content in SAMPLE_DOCUMENTS.items():
        print(f"📄 Analyzing: {doc_name.replace('_', ' ').title()}")
        print("-" * 50)
        
        try:
            # Analyze the document
            result = await analyst.analyze_document(content, {'source': doc_name})
            
            # Display results
            print(f"📋 Document Type: {result.document_context.document_type.value}")
            print(f"🎯 Primary Purpose: {result.document_context.primary_purpose}")
            print(f"👥 Target Audience: {result.document_context.intended_audience}")
            print(f"🏷️  Domain: {result.document_context.domain}")
            print(f"📊 Complexity: {result.document_context.complexity_level}")
            print()
            
            print("📝 Executive Summary:")
            print(result.synthesized_summary)
            print()
            
            if result.actionable_insights:
                print("⚡ Actionable Insights:")
                for insight in result.actionable_insights:
                    print(f"  💡 {insight}")
                print()
            
            if result.key_takeaways:
                print("🔑 Key Takeaways:")
                for takeaway in result.key_takeaways:
                    print(f"  ✅ {takeaway}")
                print()
            
            print("🏷️ Thematic Analysis:")
            for info_type, bucket in result.thematic_buckets.items():
                if bucket.items:
                    priority_emoji = {'critical': '🔴', 'important': '🟡', 'informational': '🔵'}
                    emoji = priority_emoji.get(bucket.priority, '⚪')
                    category_name = info_type.value.replace('_', ' ').title()
                    print(f"  {emoji} {category_name}: {len(bucket.items)} items ({bucket.priority})")
            
            print()
            print("📄 Structured Markdown Report:")
            print("=" * 50)
            print(result.structured_markdown[:1000] + "..." if len(result.structured_markdown) > 1000 else result.structured_markdown)
            print("=" * 50)
            print()
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            print()
        
        print()


def test_document_type_detection():
    """Test document type detection with various samples"""
    print("🧪 Testing Document Type Detection")
    print("-" * 40)
    
    config = type('MockConfig', (), {'llm_path': None})()
    analyst = AIKnowledgeAnalyst(config)
    
    test_cases = [
        ("Policy Document", "This policy establishes mandatory guidelines for employee conduct. All staff must comply with these requirements."),
        ("Manual", "This manual provides step-by-step instructions for operating the equipment. Follow these procedures carefully."),
        ("Report", "This report presents our findings from the quarterly analysis. The results show significant improvement."),
        ("Contract", "This agreement establishes terms and conditions between the parties. Whereas the licensee agrees to the terms..."),
    ]
    
    for doc_name, content in test_cases:
        doc_type = analyst._detect_document_type(content)
        print(f"📄 {doc_name}: {doc_type.value}")
    
    print()


def test_information_extraction():
    """Test information extraction patterns"""
    print("🔍 Testing Information Extraction Patterns")
    print("-" * 40)
    
    test_content = """
    SAFETY PROCEDURE:
    
    Step 1. Check all equipment before starting
    Step 2. Wear appropriate protective gear
    Step 3. Follow the established workflow
    
    REQUIREMENTS:
    All operators must complete safety training.
    Temperature must not exceed 80°C.
    
    DEFINITIONS:
    SOP: Standard Operating Procedure means the documented process.
    """
    
    config = type('MockConfig', (), {'llm_path': None})()
    analyst = AIKnowledgeAnalyst(config)
    
    # Test processes extraction
    processes = asyncio.run(analyst._extract_processes_workflows(test_content, None))
    print(f"🔄 Processes found: {len(processes)}")
    
    # Test metrics extraction
    metrics = asyncio.run(analyst._extract_key_data_metrics(test_content, None))
    print(f"📊 Metrics found: {len(metrics)}")
    
    # Test definitions extraction
    definitions = asyncio.run(analyst._extract_definitions(test_content, None))
    print(f"📖 Definitions found: {len(definitions)}")
    
    print()


if __name__ == "__main__":
    print("Starting AI Knowledge Analyst Demo...")
    print()
    
    # Run document type detection test
    test_document_type_detection()
    
    # Run information extraction test
    test_information_extraction()
    
    # Run full analysis demo
    asyncio.run(demo_ai_knowledge_analyst())
    
    print("Demo completed! 🎉")
    print()
    print("To see the full system in action:")
    print("1. Run: ./start.sh")
    print("2. Open: http://localhost:8501")
    print("3. Upload a document to see the AI Knowledge Analyst in action!")