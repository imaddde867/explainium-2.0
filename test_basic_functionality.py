#!/usr/bin/env python3
"""
Basic functionality test for the Intelligent Knowledge Categorization system.
This test focuses on the core logic and pattern-based extraction without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enum_definitions():
    """Test that all required enums are properly defined"""
    print("🧪 Testing Enum Definitions...")
    
    try:
        from src.database.models import (
            EntityType, PriorityLevel, DocumentType, TargetAudience,
            KnowledgeDomain
        )
        
        # Test EntityType
        assert EntityType.PROCESS.value == "process"
        assert EntityType.POLICY.value == "policy"
        assert EntityType.METRIC.value == "metric"
        assert EntityType.ROLE.value == "role"
        assert EntityType.COMPLIANCE_REQUIREMENT.value == "compliance_requirement"
        assert EntityType.RISK_ASSESSMENT.value == "risk_assessment"
        print("  ✅ EntityType enums defined correctly")
        
        # Test PriorityLevel
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.LOW.value == "low"
        print("  ✅ PriorityLevel enums defined correctly")
        
        # Test DocumentType
        assert DocumentType.MANUAL.value == "manual"
        assert DocumentType.CONTRACT.value == "contract"
        assert DocumentType.REPORT.value == "report"
        assert DocumentType.POLICY.value == "policy"
        print("  ✅ DocumentType enums defined correctly")
        
        # Test TargetAudience
        assert TargetAudience.TECHNICAL_STAFF.value == "technical_staff"
        assert TargetAudience.MANAGEMENT.value == "management"
        assert TargetAudience.END_USERS.value == "end_users"
        print("  ✅ TargetAudience enums defined correctly")
        
        print("✅ All enum definitions are correct!")
        return True
        
    except Exception as e:
        print(f"  ❌ Enum test failed: {e}")
        return False

def test_model_definitions():
    """Test that database models are properly defined"""
    print("\n🧪 Testing Database Model Definitions...")
    
    try:
        from src.database.models import (
            IntelligentKnowledgeEntity, DocumentIntelligence, Document
        )
        
        # Test that models have required attributes
        entity_attrs = dir(IntelligentKnowledgeEntity)
        required_entity_attrs = [
            'id', 'document_id', 'entity_type', 'key_identifier',
            'core_content', 'context_tags', 'priority_level', 'confidence'
        ]
        
        for attr in required_entity_attrs:
            assert hasattr(IntelligentKnowledgeEntity, attr), f"Missing attribute: {attr}"
        print("  ✅ IntelligentKnowledgeEntity model defined correctly")
        
        # Test DocumentIntelligence model
        intel_attrs = dir(DocumentIntelligence)
        required_intel_attrs = [
            'id', 'document_id', 'document_type', 'target_audience',
            'information_architecture', 'confidence_score'
        ]
        
        for attr in required_intel_attrs:
            assert hasattr(DocumentIntelligence, attr), f"Missing attribute: {attr}"
        print("  ✅ DocumentIntelligence model defined correctly")
        
        print("✅ All database models are defined correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def test_pattern_extraction():
    """Test the pattern-based extraction logic"""
    print("\n🧪 Testing Pattern-Based Extraction...")
    
    try:
        # Import the categorizer class
        from src.ai.intelligent_knowledge_categorizer import IntelligentKnowledgeCategorizer
        
        # Create a mock config
        class MockAIConfig:
            def __init__(self):
                self.llm = None  # No LLM available
        
        config = MockAIConfig()
        categorizer = IntelligentKnowledgeCategorizer(config)
        
        # Test document
        test_doc = {
            'id': 'test_001',
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
            'filename': 'chemical_safety.txt'
        }
        
        # Test pattern-based assessment
        import asyncio
        
        async def test_async():
            try:
                # Test without LLM (pattern-based only)
                result = await categorizer.categorize_document(test_doc)
                
                print(f"  ✅ Pattern-based extraction completed")
                print(f"  📊 Document Intelligence:")
                print(f"     - Type: {result.document_intelligence.document_type.value}")
                print(f"     - Audience: {result.document_intelligence.target_audience.value}")
                print(f"     - Confidence: {result.document_intelligence.confidence_score:.2f}")
                
                print(f"  🧠 Entities Generated: {len(result.entities)}")
                for i, entity in enumerate(result.entities[:3], 1):  # Show first 3
                    print(f"     {i}. {entity.entity_type.value}: {entity.key_identifier}")
                
                return True
                
            except Exception as e:
                print(f"  ❌ Pattern extraction failed: {e}")
                return False
        
        # Run the async test
        success = asyncio.run(test_async())
        return success
        
    except Exception as e:
        print(f"  ❌ Pattern extraction test failed: {e}")
        return False

def test_api_models():
    """Test that API Pydantic models are properly defined"""
    print("\n🧪 Testing API Model Definitions...")
    
    try:
        from src.api.app import (
            IntelligentCategorizationRequest,
            BulkCategorizationRequest,
            IntelligentKnowledgeSearchRequest
        )
        
        # Test IntelligentCategorizationRequest
        request = IntelligentCategorizationRequest(
            document_id=123,
            force_reprocess=False
        )
        assert request.document_id == 123
        assert request.force_reprocess == False
        print("  ✅ IntelligentCategorizationRequest model works")
        
        # Test BulkCategorizationRequest
        bulk_request = BulkCategorizationRequest(
            document_ids=[1, 2, 3],
            force_reprocess=True
        )
        assert bulk_request.document_ids == [1, 2, 3]
        assert bulk_request.force_reprocess == True
        print("  ✅ BulkCategorizationRequest model works")
        
        # Test IntelligentKnowledgeSearchRequest
        search_request = IntelligentKnowledgeSearchRequest(
            query="safety compliance",
            entity_type="compliance_requirement",
            priority_level="high",
            confidence_threshold=0.8,
            max_results=20
        )
        assert search_request.query == "safety compliance"
        assert search_request.entity_type == "compliance_requirement"
        assert search_request.priority_level == "high"
        print("  ✅ IntelligentKnowledgeSearchRequest model works")
        
        print("✅ All API models are defined correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ API model test failed: {e}")
        return False

def test_crud_functions():
    """Test that CRUD functions are properly defined"""
    print("\n🧪 Testing CRUD Function Definitions...")
    
    try:
        from src.database.crud import (
            create_intelligent_knowledge_entity,
            create_document_intelligence,
            get_intelligent_knowledge_entities,
            get_document_intelligence,
            search_intelligent_knowledge_entities,
            get_intelligent_knowledge_analytics
        )
        
        # Check that functions exist and are callable
        assert callable(create_intelligent_knowledge_entity)
        assert callable(create_document_intelligence)
        assert callable(get_intelligent_knowledge_entities)
        assert callable(get_document_intelligence)
        assert callable(search_intelligent_knowledge_entities)
        assert callable(get_intelligent_knowledge_analytics)
        
        print("  ✅ All CRUD functions are defined and callable")
        print("✅ CRUD function definitions are correct!")
        return True
        
    except Exception as e:
        print(f"  ❌ CRUD function test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        ("Enum Definitions", test_enum_definitions),
        ("Database Models", test_model_definitions),
        ("Pattern Extraction", test_pattern_extraction),
        ("API Models", test_api_models),
        ("CRUD Functions", test_crud_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The intelligent categorization system is ready.")
        print("\nNext steps:")
        print("  1. Install required dependencies: pip install -r requirements_intelligent_categorization.txt")
        print("  2. Run the full test: python test_intelligent_categorization.py")
        print("  3. Start using the new API endpoints")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nThe system may need additional setup or dependency installation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)