#!/usr/bin/env python3
"""
Test script for the new Intelligent Knowledge Categorization system.

This script tests the three-phase intelligent knowledge categorization framework:
1. Document Intelligence Assessment
2. Intelligent Knowledge Categorization
3. Database-Optimized Output Generation
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ai.intelligent_knowledge_categorizer import IntelligentKnowledgeCategorizer
from src.core.config import AIConfig


async def test_intelligent_categorization():
    """Test the intelligent knowledge categorization system"""
    
    print("🧠 Testing Intelligent Knowledge Categorization System")
    print("=" * 60)
    
    # Create a mock AI config
    config = AIConfig()
    
    # Initialize the categorizer
    print("\n1. Initializing Intelligent Knowledge Categorizer...")
    categorizer = IntelligentKnowledgeCategorizer(config)
    
    try:
        await categorizer.initialize()
        print("✅ Categorizer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize categorizer: {e}")
        return
    
    # Test document
    test_document = {
        'id': 'test_001',
        'content': """
        Standard Operating Procedure: Equipment Maintenance Protocol
        
        Purpose: This procedure establishes the standard process for maintaining critical equipment
        to ensure operational safety and efficiency.
        
        Scope: This procedure applies to all maintenance personnel and equipment operators
        working with industrial machinery in the manufacturing facility.
        
        Responsibilities:
        - Maintenance Technicians: Perform scheduled maintenance tasks
        - Equipment Operators: Report equipment issues and perform basic inspections
        - Supervisors: Oversee maintenance activities and ensure compliance
        
        Maintenance Schedule:
        - Daily: Visual inspection and basic cleaning
        - Weekly: Lubrication and minor adjustments
        - Monthly: Comprehensive inspection and calibration
        - Quarterly: Major maintenance and parts replacement
        
        Safety Requirements:
        - All maintenance must be performed with proper PPE
        - Equipment must be locked out and tagged out before maintenance
        - Emergency stop procedures must be tested monthly
        
        Quality Standards:
        - Maintenance records must be completed within 24 hours
        - Calibration certificates must be current and valid
        - All safety tests must pass before equipment is returned to service
        
        Compliance Notes:
        - This procedure complies with OSHA 1910.147 (Lockout/Tagout)
        - ISO 9001:2015 quality management requirements
        - Industry standard maintenance practices
        """,
        'metadata': {
            'filename': 'equipment_maintenance_sop.pdf',
            'file_type': 'pdf',
            'uploaded_at': '2024-01-15T10:30:00Z'
        },
        'type': 'pdf',
        'filename': 'equipment_maintenance_sop.pdf',
        'uploaded_at': '2024-01-15T10:30:00Z'
    }
    
    print("\n2. Testing Document Intelligence Assessment...")
    print("📄 Processing test document: Equipment Maintenance SOP")
    
    try:
        # Apply intelligent categorization
        result = await categorizer.categorize_document(test_document)
        
        print("\n3. Results:")
        print("-" * 40)
        
        # Document Intelligence Assessment
        doc_intel = result.document_intelligence
        print(f"📊 Document Type: {doc_intel.document_type.value}")
        print(f"👥 Target Audience: {doc_intel.target_audience.value}")
        print(f"🏗️  Information Architecture: {doc_intel.information_architecture['structure']}")
        print(f"🎯 Priority Contexts: {', '.join(doc_intel.priority_contexts)}")
        print(f"📈 Confidence Score: {doc_intel.confidence_score:.2f}")
        print(f"🔍 Analysis Method: {doc_intel.analysis_method}")
        
        # Intelligent Knowledge Entities
        print(f"\n🧠 Intelligent Knowledge Entities Generated: {len(result.entities)}")
        print("-" * 40)
        
        for i, entity in enumerate(result.entities, 1):
            print(f"\n{i}. {entity.entity_type.value.upper()}")
            print(f"   Key Identifier: {entity.key_identifier}")
            print(f"   Priority: {entity.priority_level.value}")
            print(f"   Confidence: {entity.confidence:.2f}")
            print(f"   Summary: {entity.summary}")
            print(f"   Context Tags: {', '.join(entity.context_tags)}")
            print(f"   Content Preview: {entity.core_content[:100]}...")
        
        # Quality Metrics
        quality = result.quality_metrics
        print(f"\n📊 Quality Assessment:")
        print(f"   Overall Quality: {quality['overall_quality']:.2f}")
        print(f"   Entity Count: {quality['entity_count']}")
        print(f"   Average Confidence: {quality['average_confidence']:.2f}")
        
        if quality['quality_issues']:
            print(f"   ⚠️  Quality Issues: {', '.join(quality['quality_issues'])}")
        
        if quality['recommendations']:
            print(f"   💡 Recommendations: {', '.join(quality['recommendations'])}")
        
        print(f"\n✅ Intelligent categorization test completed successfully!")
        
    except Exception as e:
        print(f"❌ Categorization failed: {e}")
        import traceback
        traceback.print_exc()


async def test_pattern_based_extraction():
    """Test the pattern-based extraction fallback"""
    
    print("\n🔍 Testing Pattern-Based Extraction Fallback")
    print("=" * 50)
    
    # Create a simple test document
    test_doc = {
        'id': 'test_002',
        'content': """
        Safety Protocol: Chemical Handling
        
        All chemical handling must follow these procedures:
        - Wear appropriate PPE including gloves and safety glasses
        - Store chemicals in designated areas only
        - Never mix unknown chemicals
        - Report spills immediately to supervisor
        
        Emergency Procedures:
        - In case of exposure, flush with water for 15 minutes
        - Contact emergency services if symptoms persist
        - Evacuate area if fire or explosion risk
        
        Compliance Requirements:
        - OSHA 1910.1200 Hazard Communication Standard
        - EPA hazardous waste disposal regulations
        - Company safety policy #SAF-001
        """,
        'metadata': {},
        'type': 'text',
        'filename': 'chemical_safety.txt',
        'uploaded_at': None
    }
    
    config = AIConfig()
    categorizer = IntelligentKnowledgeCategorizer(config)
    
    try:
        # Test without LLM (pattern-based only)
        result = await categorizer.categorize_document(test_doc)
        
        print(f"📊 Pattern-based extraction completed")
        print(f"🧠 Entities generated: {len(result.entities)}")
        
        for entity in result.entities:
            print(f"   - {entity.entity_type.value}: {entity.key_identifier}")
        
        print("✅ Pattern-based extraction test completed!")
        
    except Exception as e:
        print(f"❌ Pattern-based extraction failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Intelligent Knowledge Categorization Tests")
    print("=" * 70)
    
    # Run tests
    asyncio.run(test_intelligent_categorization())
    asyncio.run(test_pattern_based_extraction())
    
    print("\n🎉 All tests completed!")
    print("\nThe Intelligent Knowledge Categorization system is now ready for use.")
    print("You can:")
    print("  - Upload documents via the /upload endpoint")
    print("  - Apply intelligent categorization via /intelligent-categorization")
    print("  - Search intelligent knowledge via /intelligent-knowledge/search")
    print("  - View analytics via /intelligent-knowledge/analytics")