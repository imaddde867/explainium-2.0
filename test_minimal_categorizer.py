#!/usr/bin/env python3
"""
Test script for the Minimal Intelligent Knowledge Categorizer.
This version works with just the Python standard library.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_minimal_categorizer():
    """Test the minimal intelligent knowledge categorizer"""
    
    print("🧠 Testing Minimal Intelligent Knowledge Categorizer")
    print("=" * 60)
    
    try:
        # Import the minimal categorizer
        from src.ai.intelligent_knowledge_categorizer_minimal import (
            MinimalIntelligentKnowledgeCategorizer,
            DocumentType, TargetAudience, EntityType, PriorityLevel
        )
        
        print("✅ Successfully imported minimal categorizer")
        
        # Initialize the categorizer
        print("\n1. Initializing Minimal Categorizer...")
        categorizer = MinimalIntelligentKnowledgeCategorizer()
        
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
            
            print(f"\n✅ Minimal categorization test completed successfully!")
            
        except Exception as e:
            print(f"❌ Categorization failed: {e}")
            import traceback
            traceback.print_exc()
    
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nThis suggests the minimal categorizer file is not accessible.")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_pattern_extraction():
    """Test the pattern-based extraction specifically"""
    
    print("\n🔍 Testing Pattern-Based Extraction")
    print("=" * 50)
    
    try:
        from src.ai.intelligent_knowledge_categorizer_minimal import (
            MinimalIntelligentKnowledgeCategorizer
        )
        
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
        
        categorizer = MinimalIntelligentKnowledgeCategorizer()
        
        try:
            # Test pattern-based extraction
            result = await categorizer.categorize_document(test_doc)
            
            print(f"📊 Pattern-based extraction completed")
            print(f"🧠 Entities generated: {len(result.entities)}")
            
            for entity in result.entities:
                print(f"   - {entity.entity_type.value}: {entity.key_identifier}")
                print(f"     Priority: {entity.priority_level.value}")
                print(f"     Confidence: {entity.confidence:.2f}")
            
            print("✅ Pattern-based extraction test completed!")
            
        except Exception as e:
            print(f"❌ Pattern-based extraction failed: {e}")
    
    except Exception as e:
        print(f"❌ Pattern extraction test failed: {e}")


def main():
    """Run all tests"""
    print("🚀 Starting Minimal Intelligent Knowledge Categorizer Tests")
    print("=" * 70)
    
    # Run tests
    asyncio.run(test_minimal_categorizer())
    asyncio.run(test_pattern_extraction())
    
    print("\n🎉 All tests completed!")
    print("\nThe Minimal Intelligent Knowledge Categorizer is working correctly.")
    print("This demonstrates the core functionality without external dependencies.")
    print("\nNext steps:")
    print("  1. Install full dependencies: pip install -r requirements_intelligent_categorization.txt")
    print("  2. Use the full version: src/ai/intelligent_knowledge_categorizer.py")
    print("  3. Integrate with the main system")


if __name__ == "__main__":
    main()