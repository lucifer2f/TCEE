#!/usr/bin/env python3
"""
Integration Test Script for Unified Data Processing Pipeline

This script tests the complete integration of AI Excel extraction with the existing
electrical design automation system, validating all components work together.
"""

import sys
import os
import traceback
import tempfile
import pandas as pd
from datetime import datetime

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("🔧 Testing module imports...")
    
    try:
        # Test model imports
        from models import Project, Load, Cable, Bus, Transformer, LoadType, InstallationMethod, Priority
        print("✅ Models imported successfully")
        
        # Test calculation engine imports
        from calculations import ElectricalCalculationEngine
        print("✅ Calculations engine imported successfully")
        
        # Test standards imports
        from standards import StandardsFactory
        print("✅ Standards imported successfully")
        
        # Test AI Excel extractor imports
        from ai_excel_extractor import AIExcelExtractor, ProcessingReport
        print("✅ AI Excel extractor imported successfully")
        
        # Test unified processor imports
        from unified_processor import UnifiedDataProcessor, ProcessingInterface, create_unified_processor
        print("✅ Unified processor imported successfully")
        
        # Test main app imports (Streamlit-specific)
        try:
            import streamlit as st
            print("✅ Streamlit imported successfully")
        except ImportError as e:
            print(f"⚠️ Streamlit not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_unified_processor_basic():
    """Test basic unified processor functionality"""
    print("\n🤖 Testing unified processor basics...")
    
    try:
        from unified_processor import UnifiedDataProcessor, create_unified_processor
        
        # Test factory function
        processor = create_unified_processor("IEC")
        assert processor is not None
        assert processor.standard == "IEC"
        print("✅ Factory function works correctly")
        
        # Test initialization
        processor = UnifiedDataProcessor("IEC")
        assert processor.ai_extractor is not None
        assert processor.calc_engine is not None
        assert processor.status is not None
        print("✅ Processor initialization successful")
        
        # Test status tracking
        processor.status.update_progress("Test", 50, "Testing progress")
        assert processor.status.current_step == "Test"
        assert processor.status.progress_percent == 50
        print("✅ Status tracking works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified processor test failed: {e}")
        traceback.print_exc()
        return False

def test_sample_excel_creation():
    """Create sample Excel files for testing"""
    print("\n📋 Creating sample Excel files...")
    
    try:
        # Create sample load schedule
        load_data = {
            'Load ID': ['L001', 'L002', 'L003'],
            'Load Name': ['Motor Pump', 'HVAC Unit', 'Lighting Panel'],
            'Power (kW)': [15.5, 25.0, 8.2],
            'Voltage (V)': [400, 400, 230],
            'Phases': [3, 3, 1],
            'Load Type': ['motor', 'hvac', 'lighting'],
            'Power Factor': [0.85, 0.80, 0.90],
            'Efficiency': [0.92, 0.88, 0.95],
            'Source Bus': ['B001', 'B001', 'B002'],
            'Priority': ['essential', 'critical', 'non-essential']
        }
        
        df_loads = pd.DataFrame(load_data)
        load_filename = "test_load_schedule.xlsx"
        df_loads.to_excel(load_filename, index=False)
        print(f"✅ Created sample load schedule: {load_filename}")
        
        # Create sample cable schedule
        cable_data = {
            'Cable ID': ['C001', 'C002', 'C003'],
            'From Equipment': ['B001', 'B001', 'B002'],
            'To Equipment': ['L001', 'L002', 'L003'],
            'Specification': ['4C x 4.0 sq.mm XLPE/PVC/SWA/PVC', 
                            '4C x 6.0 sq.mm XLPE/PVC/SWA/PVC', 
                            '3C x 2.5 sq.mm XLPE/PVC/PVC'],
            'Cores': [4, 4, 3],
            'Size (mm²)': [4.0, 6.0, 2.5],
            'Length (m)': [45, 38, 22],
            'Installation': ['tray', 'tray', 'conduit']
        }
        
        df_cables = pd.DataFrame(cable_data)
        cable_filename = "test_cable_schedule.xlsx"
        df_cables.to_excel(cable_filename, index=False)
        print(f"✅ Created sample cable schedule: {cable_filename}")
        
        return load_filename, cable_filename
        
    except Exception as e:
        print(f"❌ Sample Excel creation failed: {e}")
        traceback.print_exc()
        return None, None

def test_ai_extraction_pipeline():
    """Test the complete AI extraction pipeline"""
    print("\n🔍 Testing AI extraction pipeline...")
    
    try:
        from unified_processor import UnifiedDataProcessor
        
        # Create sample files
        load_file, cable_file = test_sample_excel_creation()
        if not load_file or not cable_file:
            return False
        
        # Test with load schedule
        processor = UnifiedDataProcessor("IEC")
        
        # Create a mock uploaded file
        class MockFile:
            def __init__(self, filename, size=1024):
                self.name = filename
                self.size = size
                self._data = open(filename, 'rb').read()
            
            def getvalue(self):
                return self._data
            
            def read(self, n=-1):
                if n == -1:
                    return self._data
                return self._data[:n]
            
            def seek(self, pos):
                pass
        
        # Test file validation
        mock_file = MockFile(load_file)
        is_valid = processor._validate_uploaded_file(mock_file)
        assert is_valid == True
        print("✅ File validation works correctly")
        
        # Test file saving
        saved_path = processor._save_uploaded_file(mock_file)
        assert os.path.exists(saved_path)
        print(f"✅ File saving works correctly: {saved_path}")
        
        # Clean up
        if os.path.exists(load_file):
            os.remove(load_file)
        if os.path.exists(cable_file):
            os.remove(cable_file)
        if os.path.exists(saved_path):
            os.remove(saved_path)
        
        return True
        
    except Exception as e:
        print(f"❌ AI extraction pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_data_models_integration():
    """Test data models work correctly with extracted data"""
    print("\n🏗️ Testing data models integration...")
    
    try:
        from models import Project, Load, Cable, Bus, LoadType, InstallationMethod, Priority
        from ai_excel_extractor import DataExtractor
        
        # Create a test project
        project = Project(
            project_name="Test Integration Project",
            standard="IEC",
            voltage_system="LV"
        )
        
        # Create test loads
        load1 = Load(
            load_id="L001",
            load_name="Test Motor",
            power_kw=15.5,
            voltage=400,
            phases=3,
            load_type=LoadType.MOTOR,
            installation_method=InstallationMethod.TRAY,
            priority=Priority.ESSENTIAL,
            cable_length=45.0  # Add valid cable length
        )
        
        load2 = Load(
            load_id="L002",
            load_name="Test HVAC",
            power_kw=25.0,
            voltage=400,
            phases=3,
            load_type=LoadType.HVAC,
            installation_method=InstallationMethod.TRAY,
            priority=Priority.CRITICAL,
            cable_length=38.0  # Add valid cable length
        )
        
        # Add loads to project
        project.add_load(load1)
        project.add_load(load2)
        
        # Create test buses
        bus1 = Bus(
            bus_id="B001",
            bus_name="Main Distribution Bus",
            voltage=400,
            phases=3,
            rated_current_a=630,
            short_circuit_rating_ka=50
        )
        
        project.buses.append(bus1)
        
        # Create test cables
        cable1 = Cable(
            cable_id="C001",
            from_equipment="B001",
            to_equipment="L001",
            cores=4,
            size_sqmm=4.0,
            cable_type="XLPE/PVC",
            insulation="PVC",
            length_m=45,
            installation_method=InstallationMethod.TRAY,
            armored=True
        )
        
        project.cables.append(cable1)
        
        # Validate project
        is_valid, errors = project.validate_project()
        assert is_valid == True
        print("✅ Data models integration successful")
        print(f"   Project has {len(project.loads)} loads, {len(project.buses)} buses, {len(project.cables)} cables")
        
        return True
        
    except Exception as e:
        print(f"❌ Data models integration test failed: {e}")
        traceback.print_exc()
        return False

def test_calculation_engine_integration():
    """Test calculation engine works with extracted data"""
    print("\n🧮 Testing calculation engine integration...")
    
    try:
        from models import Load, LoadType, InstallationMethod, Priority
        from calculations import ElectricalCalculationEngine
        
        # Create calculation engine
        calc_engine = ElectricalCalculationEngine("IEC")
        
        # Create test load
        load = Load(
            load_id="L001",
            load_name="Test Motor",
            power_kw=15.5,
            voltage=400,
            phases=3,
            load_type=LoadType.MOTOR,
            installation_method=InstallationMethod.TRAY,
            priority=Priority.ESSENTIAL,
            cable_length=45.0,
            power_factor=0.85,
            efficiency=0.92
        )
        
        # Calculate load parameters
        calculated_load = calc_engine.calculate_load(load)
        
        # Verify calculations
        assert calculated_load.current_a is not None
        assert calculated_load.current_a > 0
        assert calculated_load.apparent_power_kva is not None
        assert calculated_load.design_current_a is not None
        
        print(f"✅ Calculation engine integration successful")
        print(f"   Load current: {calculated_load.current_a:.2f} A")
        print(f"   Design current: {calculated_load.design_current_a:.2f} A")
        print(f"   Apparent power: {calculated_load.apparent_power_kva:.2f} kVA")
        
        return True
        
    except Exception as e:
        print(f"❌ Calculation engine integration test failed: {e}")
        traceback.print_exc()
        return False

def test_standards_compatibility():
    """Test standards work with different electrical standards"""
    print("\n📏 Testing standards compatibility...")
    
    try:
        from unified_processor import create_unified_processor
        from standards import StandardsFactory
        
        # Test different standards
        standards = ["IEC", "IS", "NEC"]
        
        for standard in standards:
            processor = create_unified_processor(standard)
            assert processor.standard == standard
            
            # Test standards factory
            std_instance = StandardsFactory.get_standard(standard)
            assert std_instance is not None
            
            print(f"✅ Standard {standard} works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Standards compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and graceful degradation"""
    print("\n⚠️ Testing error handling...")
    
    try:
        from unified_processor import UnifiedDataProcessor
        
        processor = UnifiedDataProcessor("IEC")
        
        # Test with invalid file
        class MockInvalidFile:
            def __init__(self):
                self.name = "invalid.txt"
                self.size = 1000
            
            def getvalue(self):
                return b"invalid data"
            
            def read(self, n=-1):
                return b"invalid data"
            
            def seek(self, pos):
                pass
        
        # Test file validation
        is_valid = processor._validate_uploaded_file(MockInvalidFile())
        assert is_valid == False
        print("✅ Invalid file rejection works correctly")
        
        # Test with None file
        is_valid = processor._validate_uploaded_file(None)
        assert is_valid == False
        print("✅ None file rejection works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_interface_integration():
    """Test Streamlit interface can import and initialize components"""
    print("\n🖥️ Testing Streamlit interface integration...")
    
    try:
        # Test imports that would be used in the Streamlit app
        from unified_processor import (
            ProcessingInterface, 
            initialize_processing_status, 
            get_processing_status
        )
        
        # Initialize processing status
        initialize_processing_status()
        print("✅ Processing status initialization works")
        
        # Test getting processing status (would return None without Streamlit session)
        status = get_processing_status()
        # This might return None without actual Streamlit context, which is expected
        print("✅ Processing status retrieval works")
        
        # Test UI component availability
        try:
            import streamlit as st
            print("✅ Streamlit components available")
        except ImportError:
            print("⚠️ Streamlit not available (expected in test environment)")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit interface integration test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all integration tests"""
    print("🚀 Starting Comprehensive Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Unified Processor Basics", test_unified_processor_basic),
        ("AI Extraction Pipeline", test_ai_extraction_pipeline),
        ("Data Models Integration", test_data_models_integration),
        ("Calculation Engine Integration", test_calculation_engine_integration),
        ("Standards Compatibility", test_standards_compatibility),
        ("Error Handling", test_error_handling),
        ("Streamlit Interface Integration", test_streamlit_interface_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(results)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Integration workflow is working correctly.")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the issues above.")
        return False

def main():
    """Main test execution"""
    try:
        success = run_comprehensive_test()
        return 0 if success else 1
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)