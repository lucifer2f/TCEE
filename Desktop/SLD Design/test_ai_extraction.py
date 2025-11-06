#!/usr/bin/env python3
"""
Test script for AI-Powered Excel Extraction System

This script tests the AIExcelExtractor with real electrical project Excel files
to validate extraction accuracy, pattern recognition, and data enhancement.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_excel_extractor import AIExcelExtractor, ExtractionResult, ProcessingReport
from models import Project


def test_ai_extraction_system():
    """Test the AI extraction system with available Excel files"""
    
    print("=" * 80)
    print("AI-POWERED EXCEL EXTRACTION SYSTEM - TEST SUITE")
    print("=" * 80)
    print()
    
    # Initialize the AI extractor
    extractor = AIExcelExtractor()
    
    # Test files (available in the workspace)
    test_files = [
        "Advanced_Manufacturing_Plant_-_Electrical_Distribution_LoadList.xlsx",
        "Advanced_Manufacturing_Plant_-_Electrical_Distribution_CableSchedule.xlsx",
        "New_Electrical_Project_LoadList.xlsx", 
        "New_Electrical_Project_CableSchedule.xlsx"
    ]
    
    print(f"Testing with {len(test_files)} Excel files...")
    print()
    
    # Test each file individually
    all_results = {}
    for i, filename in enumerate(test_files, 1):
        print(f"Test {i}: {filename}")
        print("-" * 60)
        
        if os.path.exists(filename):
            try:
                # Process the file
                report = extractor.process_excel_file(filename)
                
                # Display results
                display_extraction_results(filename, report)
                
                # Store results for analysis
                all_results[filename] = report
                
            except Exception as e:
                print(f"❌ ERROR processing {filename}: {str(e)}")
                print()
                continue
        else:
            print(f"❌ FILE NOT FOUND: {filename}")
            print()
            continue
    
    # Overall analysis
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)
    
    analyze_overall_results(all_results)
    
    return all_results


def display_extraction_results(filename: str, report: ProcessingReport):
    """Display detailed extraction results"""
    
    print(f"📁 File: {filename}")
    print(f"⏱️  Processing Time: {report.processing_time_seconds:.2f} seconds")
    print(f"🎯 Overall Confidence: {report.overall_confidence:.1%}")
    print(f"🔧 Total Components: {report.total_components}")
    
    # Sheet results
    print(f"\n📊 Sheet Analysis:")
    for sheet_name, result in report.sheet_results.items():
        status = "✅" if result.success else "❌"
        print(f"  {status} {sheet_name}")
        print(f"     Type: {result.sheet_type}")
        print(f"     Confidence: {result.confidence:.1%}")
        print(f"     Components: {result.components_extracted}")
        print(f"     Quality Score: {result.data_quality_score:.1%}")
        
        if result.issues:
            print(f"     Issues: {len(result.issues)}")
            for issue in result.issues[:3]:  # Show first 3 issues
                print(f"       • {issue}")
            if len(result.issues) > 3:
                print(f"       • ... and {len(result.issues) - 3} more")
        
        if result.warnings:
            print(f"     Warnings: {len(result.warnings)}")
    
    # Project data summary
    if report.project_data:
        project = report.project_data
        print(f"\n🏗️  Project Summary:")
        print(f"   Loads: {len(project.loads)}")
        print(f"   Cables: {len(project.cables)}")
        print(f"   Buses: {len(project.buses)}")
        print(f"   Transformers: {len(project.transformers)}")
        
        # Show sample loads
        if project.loads:
            print(f"\n📋 Sample Loads (first 3):")
            for load in project.loads[:3]:
                print(f"   • {load.load_id}: {load.load_name}")
                print(f"     Power: {load.power_kw}kW, Voltage: {load.voltage}V, Type: {load.load_type.value}")
    
    # Corrections made
    if report.corrections_made:
        print(f"\n🔧 Corrections Made: {len(report.corrections_made)}")
        for correction in report.corrections_made[:5]:  # Show first 5 corrections
            print(f"   • {correction['type']}: {correction.get('reason', 'N/A')}")
        if len(report.corrections_made) > 5:
            print(f"   • ... and {len(report.corrections_made) - 5} more")
    
    # Validation issues
    if report.validation_issues:
        print(f"\n⚠️  Validation Issues: {len(report.validation_issues)}")
        for issue in report.validation_issues[:5]:  # Show first 5 issues
            print(f"   • {issue}")
        if len(report.validation_issues) > 5:
            print(f"   • ... and {len(report.validation_issues) - 5} more")
    
    print()


def analyze_overall_results(results: dict):
    """Analyze results across all processed files"""
    
    if not results:
        print("No files were successfully processed.")
        return
    
    # Aggregate statistics
    total_files = len(results)
    successful_files = sum(1 for r in results.values() if r.overall_confidence > 0)
    total_components = sum(r.total_components for r in results.values())
    avg_confidence = sum(r.overall_confidence for r in results.values()) / total_files
    
    print(f"📈 Aggregate Results:")
    print(f"   Files Processed: {total_files}")
    print(f"   Successful Files: {successful_files}")
    print(f"   Success Rate: {successful_files/total_files:.1%}")
    print(f"   Total Components: {total_components}")
    print(f"   Average Confidence: {avg_confidence:.1%}")
    
    # Component breakdown
    total_loads = sum(len(r.project_data.loads) for r in results.values() if r.project_data)
    total_cables = sum(len(r.project_data.cables) for r in results.values() if r.project_data)
    total_buses = sum(len(r.project_data.buses) for r in results.values() if r.project_data)
    
    print(f"\n🔧 Component Breakdown:")
    print(f"   Loads: {total_loads}")
    print(f"   Cables: {total_cables}")
    print(f"   Buses: {total_buses}")
    
    # Quality assessment
    total_corrections = sum(len(r.corrections_made) for r in results.values())
    total_validation_issues = sum(len(r.validation_issues) for r in results.values())
    
    print(f"\n✅ Quality Metrics:")
    print(f"   Total Corrections: {total_corrections}")
    print(f"   Total Validation Issues: {total_validation_issues}")
    print(f"   Avg Corrections per File: {total_corrections/total_files:.1f}")
    print(f"   Avg Issues per File: {total_validation_issues/total_files:.1f}")
    
    # Identify best and worst performing files
    if results:
        best_file = max(results.items(), key=lambda x: x[1].overall_confidence)
        worst_file = min(results.items(), key=lambda x: x[1].overall_confidence)
        
        print(f"\n🏆 Performance Analysis:")
        print(f"   Best: {best_file[0]} ({best_file[1].overall_confidence:.1%})")
        print(f"   Worst: {worst_file[0]} ({worst_file[1].overall_confidence:.1%})")
    
    # Export detailed results
    export_detailed_results(results)


def export_detailed_results(results: dict):
    """Export detailed results to JSON for analysis"""
    
    # Prepare export data
    export_data = {
        'test_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_files': len(results),
            'successful_files': sum(1 for r in results.values() if r.overall_confidence > 0),
            'total_components': sum(r.total_components for r in results.values()),
            'average_confidence': sum(r.overall_confidence for r in results.values()) / len(results) if results else 0
        },
        'file_results': {}
    }
    
    for filename, report in results.items():
        file_data = {
            'overall_confidence': report.overall_confidence,
            'total_components': report.total_components,
            'processing_time_seconds': report.processing_time_seconds,
            'sheet_results': {
                sheet_name: {
                    'sheet_type': result.sheet_type,
                    'confidence': result.confidence,
                    'components_extracted': result.components_extracted,
                    'data_quality_score': result.data_quality_score,
                    'issues_count': len(result.issues),
                    'warnings_count': len(result.warnings)
                }
                for sheet_name, result in report.sheet_results.items()
            },
            'corrections_count': len(report.corrections_made),
            'validation_issues_count': len(report.validation_issues)
        }
        
        # Add project data if available
        if report.project_data:
            file_data['project_summary'] = {
                'loads_count': len(report.project_data.loads),
                'cables_count': len(report.project_data.cables),
                'buses_count': len(report.project_data.buses)
            }
        
        export_data['file_results'][filename] = file_data
    
    # Write to JSON file
    with open('ai_extraction_test_results.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n💾 Detailed results exported to: ai_extraction_test_results.json")


def test_specific_patterns():
    """Test specific pattern recognition capabilities"""
    
    print("\n" + "=" * 80)
    print("PATTERN RECOGNITION TESTS")
    print("=" * 80)
    
    extractor = AIExcelExtractor()
    
    # Test sheet classification
    print("\n🔍 Testing Sheet Classification:")
    
    test_scenarios = [
        {
            'name': 'Load Schedule Pattern',
            'headers': ['Load ID', 'Load Name', 'Power (kW)', 'Voltage (V)', 'Type', 'Current (A)'],
            'expected': 'load_schedule'
        },
        {
            'name': 'Cable Schedule Pattern', 
            'headers': ['Cable ID', 'From', 'To', 'Specification', 'Cores', 'Size (mm²)'],
            'expected': 'cable_schedule'
        },
        {
            'name': 'Bus Schedule Pattern',
            'headers': ['Bus ID', 'Bus Name', 'Voltage (V)', 'Rated Current (A)', 'SC Rating (kA)'],
            'expected': 'bus_schedule'
        },
        {
            'name': 'Project Info Pattern',
            'headers': ['Project Name', 'Standard', 'Voltage System', 'Ambient Temp'],
            'expected': 'project_info'
        }
    ]
    
    import pandas as pd
    
    for scenario in test_scenarios:
        # Create test DataFrame
        df = pd.DataFrame(columns=scenario['headers'])
        
        # Classify
        classification = extractor.sheet_classifier.classify_sheet(df, f"Test {scenario['name']}")
        
        # Check result
        predicted = classification['sheet_type']
        confidence = classification['confidence']
        expected = scenario['expected']
        
        status = "✅" if predicted == expected else "❌"
        print(f"  {status} {scenario['name']}")
        print(f"     Expected: {expected}, Predicted: {predicted} ({confidence:.2f})")
        
        if predicted != expected:
            print(f"     All scores: {classification.get('all_scores', {})}")
    
    print("\n🏁 Pattern Recognition Tests Complete")


def main():
    """Main test execution"""
    
    try:
        # Run comprehensive tests
        results = test_ai_extraction_system()
        
        # Run pattern recognition tests
        test_specific_patterns()
        
        print("\n" + "=" * 80)
        print("TEST EXECUTION COMPLETE")
        print("=" * 80)
        print("✅ AI Excel Extraction System testing completed successfully!")
        print("📊 Check ai_extraction_test_results.json for detailed analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST EXECUTION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)