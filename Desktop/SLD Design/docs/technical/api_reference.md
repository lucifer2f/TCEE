# API Reference - AI Excel Extraction System

## Overview

The AI Excel Extraction System provides a comprehensive API for intelligently extracting electrical distribution project data from Excel files. This reference covers all classes, methods, parameters, and usage patterns.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Structures](#data-structures)
3. [Main Orchestrator](#main-orchestrator)
4. [Configuration](#configuration)
5. [Error Handling](#error-handling)
6. [Usage Examples](#usage-examples)
7. [Integration Guide](#integration-guide)

---

## Core Classes

### SheetClassifier

**Purpose**: Identifies sheet types (Load, Cable, Bus, etc.) using pattern matching based on electrical engineering domain knowledge.

#### Constructor
```python
def __init__(self)
```
Initializes the classifier with electrical engineering patterns for different sheet types.

#### Methods

##### classify_sheet()
```python
def classify_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]
```

**Description**: Classifies Excel sheet based on content patterns

**Parameters**:
- `df` (pandas.DataFrame): DataFrame containing sheet data
- `sheet_name` (str): Name of the sheet being classified

**Returns**: Dictionary containing classification results:
```python
{
    'sheet_type': str,                    # Type: 'load_schedule', 'cable_schedule', etc.
    'confidence': float,                  # Confidence score 0.0-1.0
    'evidence': List[Tuple[str, str]],    # Pattern matches found
    'recommended_model_mapping': str,     # Model type: 'Load', 'Cable', 'Bus'
    'all_scores': Dict[str, float]        # Scores for all sheet types
}
```

**Example**:
```python
classifier = SheetClassifier()
df = pd.read_excel('load_schedule.xlsx')
result = classifier.classify_sheet(df, 'Load Schedule')
print(result['sheet_type'])  # 'load_schedule'
print(result['confidence'])  # 0.95
```

---

### ColumnMapper

**Purpose**: Intelligent column header mapping with fuzzy string matching for mapping Excel columns to electrical engineering data fields.

#### Constructor
```python
def __init__(self)
```
Initializes the mapper with field patterns for Load, Cable, and Bus models.

#### Methods

##### map_columns()
```python
def map_columns(self, columns: List[str], model_type: str, sheet_context: str = "") -> Dict[str, Any]
```

**Description**: Maps Excel columns to model fields with confidence scores

**Parameters**:
- `columns` (List[str]): List of Excel column headers
- `model_type` (str): Target model type ('Load', 'Cable', 'Bus')
- `sheet_context` (str, optional): Additional context about the sheet

**Returns**: Dictionary with mapping results:
```python
{
    'field_mappings': {
        'field_name': {
            'mapped_columns': List[str],    # Columns mapped to this field
            'confidence': float,            # Mapping confidence 0.0-1.0
            'data_type': str,               # Inferred data type
            'pattern_match': str            # Pattern that matched
        }
    },
    'overall_confidence': float,           # Average mapping confidence
    'unmapped_columns': List[str],         # Columns not mapped
    'mapping_quality': str                 # Quality assessment
}
```

**Example**:
```python
mapper = ColumnMapper()
columns = ['Load ID', 'Power (kW)', 'Voltage', 'Type']
mapping = mapper.map_columns(columns, 'Load')
print(mapping['field_mappings']['load_id']['mapped_columns'])  # ['Load ID']
print(mapping['mapping_quality'])  # 'excellent'
```

---

### DataExtractor

**Purpose**: Extracts and validates electrical components using existing Load, Cable, Bus, and Transformer models.

#### Constructor
```python
def __init__(self)
```
Initializes the extractor with calculation engine and pattern mappings.

#### Methods

##### extract_loads()
```python
def extract_loads(self, df: pd.DataFrame, field_mapping: Dict) -> Tuple[List[Load], ExtractionResult]
```

**Description**: Extracts Load objects from Excel DataFrame

**Parameters**:
- `df` (pandas.DataFrame): Load schedule data
- `field_mapping` (Dict): Column-to-field mapping from ColumnMapper

**Returns**: 
- `List[Load]`: List of extracted Load objects
- `ExtractionResult`: Result object with confidence and quality metrics

**Example**:
```python
extractor = DataExtractor()
loads, result = extractor.extract_loads(load_df, field_mapping)
print(f"Extracted {len(loads)} loads with {result.confidence:.1%} confidence")
```

##### extract_cables()
```python
def extract_cables(self, df: pd.DataFrame, field_mapping: Dict) -> Tuple[List[Cable], ExtractionResult]
```

**Description**: Extracts Cable objects from Excel DataFrame

**Parameters**:
- `df` (pandas.DataFrame): Cable schedule data
- `field_mapping` (Dict): Column-to-field mapping from ColumnMapper

**Returns**:
- `List[Cable]`: List of extracted Cable objects
- `ExtractionResult`: Result object with confidence and quality metrics

---

### DataEnhancer

**Purpose**: Automatically corrects common issues in extracted data (broken IDs, missing relationships, naming conventions).

#### Constructor
```python
def __init__(self)
```
Initializes the enhancer with ID patterns and correction rules.

#### Methods

##### enhance_project_data()
```python
def enhance_project_data(self, project: Project, extraction_results: List[ExtractionResult]) -> Dict[str, Any]
```

**Description**: Enhances extracted project data by fixing common issues

**Parameters**:
- `project` (Project): Project object to enhance
- `extraction_results` (List[ExtractionResult]): Results from extraction process

**Returns**: Dictionary with enhancement report:
```python
{
    'corrections_made': List[Dict],        # List of corrections applied
    'correction_count': int,               # Number of corrections
    'enhancement_success': bool,           # Whether enhancement succeeded
    'final_project': Project               # Enhanced project object
}
```

**Correction Types**:
- `load_id_fixed`: Generated missing or invalid load IDs
- `cable_connection_updated`: Updated cable connections when IDs change
- `bus_created`: Created missing bus systems
- `load_bus_assignment`: Assigned loads to buses
- `cable_created`: Generated missing cables for loads
- `load_name_standardized`: Standardized naming conventions

---

### ValidationEngine

**Purpose**: Cross-checks extracted data against electrical engineering rules and standards.

#### Constructor
```python
def __init__(self, standard: str = "IEC")
```

**Parameters**:
- `standard` (str): Electrical standard for validation ('IEC', 'IS', 'NEC')

#### Methods

##### validate_project()
```python
def validate_project(self, project: Project) -> Dict[str, Any]
```

**Description**: Validates entire project against electrical engineering rules

**Parameters**:
- `project` (Project): Project object to validate

**Returns**: Dictionary with validation results:
```python
{
    'is_valid': bool,                      # Overall validation status
    'errors': List[str],                   # Critical errors found
    'warnings': List[str],                 # Warnings identified
    'recommendations': List[str],          # Improvement suggestions
    'electrical_violations': List[str],    # Electrical safety violations
    'quality_score': float                 # Overall quality score 0.0-1.0
}
```

**Validation Categories**:
- **Basic Validation**: Missing IDs, invalid values
- **Electrical Validation**: Power balance, cable ratings, breaker coordination
- **Standards Compliance**: IEC/IS/NEC requirement checking
- **System Consistency**: Relationship integrity and load balance

---

## Data Structures

### ExtractionResult

**Purpose**: Container for extraction results with confidence scoring and quality metrics.

**Fields**:
```python
@dataclass
class ExtractionResult:
    success: bool                          # Whether extraction succeeded
    confidence: float                      # Confidence score 0.0-1.0
    sheet_type: str                        # Type of sheet processed
    components_extracted: int              # Number of components extracted
    data_quality_score: float             # Data quality score 0.0-1.0
    issues: List[str]                      # Issues encountered
    warnings: List[str]                    # Warnings generated
    extracted_data: Optional[Dict]         # Raw extracted data
```

### ProcessingReport

**Purpose**: Comprehensive processing report for the entire Excel file.

**Fields**:
```python
@dataclass
class ProcessingReport:
    overall_confidence: float               # Overall processing confidence
    total_components: int                   # Total components extracted
    processing_time_seconds: float          # Processing time
    sheet_results: Dict[str, ExtractionResult]  # Results per sheet
    project_data: Optional[Project]         # Created project object
    corrections_made: List[Dict]            # Auto-corrections applied
    validation_issues: List[str]            # Validation issues found
```

---

## Main Orchestrator

### AIExcelExtractor

**Purpose**: Main orchestrator for AI-powered Excel extraction, coordinating all components.

#### Constructor
```python
def __init__(self, standard: str = "IEC")
```

**Parameters**:
- `standard` (str): Electrical standard ('IEC', 'IS', 'NEC')

**Example**:
```python
# Initialize for IEC standard
extractor = AIExcelExtractor(standard="IEC")

# Initialize for Indian Standards
extractor = AIExcelExtractor(standard="IS")

# Initialize for NEC standard
extractor = AIExcelExtractor(standard="NEC")
```

#### Methods

##### process_excel_file()
```python
def process_excel_file(self, file_path: str) -> ProcessingReport
```

**Description**: Process Excel file and extract all electrical components

**Parameters**:
- `file_path` (str): Path to Excel file (.xlsx or .xls)

**Returns**: `ProcessingReport` with comprehensive results

**Processing Pipeline**:
1. **File Reading**: Read Excel file with all sheets
2. **Sheet Classification**: Classify each sheet using SheetClassifier
3. **Column Mapping**: Map columns using ColumnMapper
4. **Data Extraction**: Extract components using DataExtractor
5. **Data Enhancement**: Auto-correct issues using DataEnhancer
6. **Validation**: Validate against electrical rules using ValidationEngine
7. **Report Generation**: Create comprehensive ProcessingReport

**Example**:
```python
extractor = AIExcelExtractor()
report = extractor.process_excel_file("manufacturing_plant.xlsx")

print(f"Overall Confidence: {report.overall_confidence:.1%}")
print(f"Components Extracted: {report.total_components}")
print(f"Processing Time: {report.processing_time_seconds:.2f}s")

if report.project_data:
    print(f"Loads: {len(report.project_data.loads)}")
    print(f"Cables: {len(report.project_data.cables)}")
    print(f"Buses: {len(report.project_data.buses)}")

print(f"Corrections Made: {len(report.corrections_made)}")
print(f"Validation Issues: {len(report.validation_issues)}")
```

---

## Configuration

### Standard Options

The system supports three electrical standards:

#### IEC (International Electrotechnical Commission)
```python
extractor = AIExcelExtractor(standard="IEC")
```
- Follows IEC 60364 standards
- European electrical practices
- Cable sizing per IEC standards
- Protection coordination guidelines

#### IS (Indian Standards)
```python
extractor = AIExcelExtractor(standard="IS")
```
- Follows Indian Standards (IS)
- Indian electrical practices
- IS 732 cable sizing
- Local electrical codes

#### NEC (National Electrical Code)
```python
extractor = AIExcelExtractor(standard="NEC")
```
- Follows NEC standards
- US electrical practices
- NEC cable sizing rules
- American protection requirements

### Custom Configuration

#### Pattern Customization
```python
# Extend pattern recognition for custom headers
extractor = AIExcelExtractor()
extractor.sheet_classifier.load_patterns['custom'] = [
    r'equipment\s*id', r'custom\s*power\s*field'
]

# Add custom column mappings
extractor.column_mapper.field_mappings['Load']['custom_field'] = [
    'custom header 1', 'custom header 2'
]
```

#### Confidence Thresholds
```python
# High confidence for critical applications
high_confidence_extractor = AIExcelExtractor()
# Adjust internal thresholds (requires modifying source)
high_confidence_extractor.column_mapper.confidence_threshold = 0.8
```

---

## Error Handling

### Common Exceptions

#### FileNotFoundError
```python
try:
    report = extractor.process_excel_file("nonexistent.xlsx")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

#### ValueError
```python
try:
    report = extractor.process_excel_file("invalid.xlsx")
except ValueError as e:
    print(f"Invalid file format: {e}")
```

#### KeyError
```python
try:
    # Missing required columns
    mapping = mapper.map_columns(['Invalid', 'Columns'], 'Load')
except KeyError as e:
    print(f"Mapping error: {e}")
```

### Error Recovery

#### Graceful Degradation
```python
report = extractor.process_excel_file("problematic_file.xlsx")
if not report.overall_confidence > 0.5:
    print("Low confidence - manual review recommended")
    # Process individual sheets
    for sheet_name, result in report.sheet_results.items():
        if result.confidence > 0.8:
            print(f"Sheet '{sheet_name}' can be trusted")
        else:
            print(f"Sheet '{sheet_name}' needs review")
```

#### Partial Success Handling
```python
# Some sheets may succeed while others fail
for sheet_name, result in report.sheet_results.items():
    if result.success:
        print(f"✅ {sheet_name}: {result.components_extracted} components")
    else:
        print(f"❌ {sheet_name}: {', '.join(result.issues)}")
```

---

## Usage Examples

### Basic Usage
```python
from ai_excel_extractor import AIExcelExtractor

# Initialize extractor
extractor = AIExcelExtractor()

# Process file
report = extractor.process_excel_file("project.xlsx")

# Access results
if report.overall_confidence > 0.8:
    project = report.project_data
    print(f"Successfully extracted project: {project.project_name}")
else:
    print("Low confidence - manual review needed")
```

### Batch Processing
```python
import os
from ai_excel_extractor import AIExcelExtractor

def process_multiple_files(directory):
    extractor = AIExcelExtractor()
    results = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(directory, filename)
            try:
                report = extractor.process_excel_file(filepath)
                results.append({
                    'file': filename,
                    'confidence': report.overall_confidence,
                    'components': report.total_components
                })
            except Exception as e:
                results.append({
                    'file': filename,
                    'error': str(e)
                })
    
    return results

# Process all Excel files in directory
results = process_multiple_files("./excel_files")
for result in results:
    print(f"{result['file']}: {result.get('confidence', 'ERROR')}")
```

### Custom Processing Pipeline
```python
from ai_excel_extractor import SheetClassifier, ColumnMapper, DataExtractor
import pandas as pd

def custom_extraction_pipeline(excel_file):
    # Read Excel file
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    
    classifier = SheetClassifier()
    mapper = ColumnMapper()
    extractor = DataExtractor()
    
    results = {}
    
    for sheet_name, df in excel_data.items():
        # Classify sheet
        classification = classifier.classify_sheet(df, sheet_name)
        
        if classification['recommended_model_mapping']:
            # Map columns
            mapping = mapper.map_columns(
                df.columns.tolist(),
                classification['recommended_model_mapping']
            )
            
            # Extract data
            if classification['sheet_type'] == 'load_schedule':
                loads, result = extractor.extract_loads(df, mapping)
                results[sheet_name] = {'loads': loads, 'result': result}
            elif classification['sheet_type'] == 'cable_schedule':
                cables, result = extractor.extract_cables(df, mapping)
                results[sheet_name] = {'cables': cables, 'result': result}
    
    return results

# Use custom pipeline
results = custom_extraction_pipeline("custom_project.xlsx")
```

### Integration with Existing Models
```python
from ai_excel_extractor import AIExcelExtractor
from models import Load, Cable, Project
from calculations import ElectricalCalculationEngine

def enhanced_extraction_with_calculations(excel_file):
    # Standard extraction
    extractor = AIExcelExtractor()
    report = extractor.process_excel_file(excel_file)
    
    if report.project_data:
        # Perform additional calculations
        calc_engine = ElectricalCalculationEngine()
        
        # Recalculate all loads
        for load in report.project_data.loads:
            calc_engine.calculate_load(load)
            calc_engine.calculate_cable_for_load(load)
        
        # Validate system
        validation = calc_engine.validate_project_data(report.project_data)
        
        return {
            'extraction_report': report,
            'additional_validation': validation
        }
    
    return None

# Enhanced processing
result = enhanced_extraction_with_calculations("project.xlsx")
```

---

## Integration Guide

### Streamlit Integration
```python
import streamlit as st
from ai_excel_extractor import AIExcelExtractor

def ai_excel_import_page():
    st.header("🤖 AI Excel Import")
    
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file:
        # Save uploaded file temporarily
        with open("temp_upload.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process with AI
        extractor = AIExcelExtractor()
        report = extractor.process_excel_file("temp_upload.xlsx")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{report.overall_confidence:.1%}")
        with col2:
            st.metric("Components", report.total_components)
        with col3:
            st.metric("Time", f"{report.processing_time_seconds:.1f}s")
        
        if report.project_data:
            st.success(f"Created project: {report.project_data.project_name}")
            
            # Show correction summary
            if report.corrections_made:
                st.info(f"Made {len(report.corrections_made)} automatic corrections")
```

### REST API Integration
```python
from fastapi import FastAPI, UploadFile
from ai_excel_extractor import AIExcelExtractor
import tempfile

app = FastAPI()

@app.post("/extract")
async def extract_excel(file: UploadFile):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process with AI
        extractor = AIExcelExtractor()
        report = extractor.process_excel_file(tmp_path)
        
        return {
            "success": True,
            "confidence": report.overall_confidence,
            "components": report.total_components,
            "project_name": report.project_data.project_name if report.project_data else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Run with: uvicorn api_server:app --reload
```

### Database Integration
```python
import json
from ai_excel_extractor import AIExcelExtractor
import sqlite3

def save_extraction_to_database(excel_file, db_path):
    # Extract data
    extractor = AIExcelExtractor()
    report = extractor.process_excel_file(excel_file)
    
    if not report.project_data:
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Save project
        cursor.execute("""
            INSERT INTO projects (name, standard, extraction_confidence)
            VALUES (?, ?, ?)
        """, (report.project_data.project_name, 
              report.project_data.standard,
              report.overall_confidence))
        
        project_id = cursor.lastrowid
        
        # Save loads
        for load in report.project_data.loads:
            cursor.execute("""
                INSERT INTO loads (project_id, load_id, name, power_kw, voltage)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, load.load_id, load.load_name, 
                  load.power_kw, load.voltage))
        
        # Save cables
        for cable in report.project_data.cables:
            cursor.execute("""
                INSERT INTO cables (project_id, cable_id, from_equipment, to_equipment, size_sqmm)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, cable.cable_id, cable.from_equipment,
                  cable.to_equipment, cable.size_sqmm))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
        return False
    finally:
        conn.close()
```

---

## Performance Considerations

### File Size Limits
- **Recommended**: < 10MB per file
- **Maximum**: 50MB per file
- **Processing Time**: Approximately 1-5 seconds per MB

### Memory Usage
- **Small Files** (< 1MB): ~50MB memory
- **Medium Files** (1-10MB): ~100-200MB memory  
- **Large Files** (10-50MB): ~300-500MB memory

### Optimization Tips
```python
# For large files, process in chunks
def process_large_file_efficiently(file_path):
    # Read only specific sheets
    excel_data = pd.read_excel(file_path, sheet_name=['Load Schedule', 'Cable Schedule'])
    
    extractor = AIExcelExtractor()
    
    # Process each sheet separately
    results = {}
    for sheet_name, df in excel_data.items():
        classification = extractor.sheet_classifier.classify_sheet(df, sheet_name)
        if classification['confidence'] > 0.7:
            # Only process high-confidence sheets
            mapping = extractor.column_mapper.map_columns(
                df.columns.tolist(), 
                classification['recommended_model_mapping']
            )
            
            if classification['sheet_type'] == 'load_schedule':
                loads, result = extractor.data_extractor.extract_loads(df, mapping)
                results[sheet_name] = {'loads': loads, 'result': result}
    
    return results
```

---

This API reference provides comprehensive documentation for integrating the AI Excel Extraction System into your applications. For additional support and advanced usage patterns, refer to the integration examples and troubleshooting guides.