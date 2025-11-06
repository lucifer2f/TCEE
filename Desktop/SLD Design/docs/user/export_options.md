# Export Options - Saving and Using Extracted Data

## Overview

This guide covers all available export options for AI-extracted electrical project data, including formats, customization features, and integration methods with other systems.

## 📤 Export Formats Overview

The AI Excel extraction system provides multiple export formats to suit different needs and workflows:

### Available Export Formats

| Format | File Type | Primary Use | Integration | Features |
|--------|-----------|-------------|-------------|----------|
| **Excel Workbook** | .xlsx | Manual review | Excel, Sheets | ✅ Multiple sheets, ✅ Formatting |
| **JSON Data** | .json | API integration | Any system | ✅ Structured data, ✅ Programmatic |
| **CSV Files** | .csv | Data analysis | Excel, Analytics | ✅ Simple format, ✅ Universal |
| **PDF Reports** | .pdf | Documentation | Any viewer | ✅ Professional, ✅ Read-only |
| **XML** | .xml | Enterprise systems | Legacy systems | ✅ Standardized, ✅ Schema validation |

### Quick Export Selection

```
📤 Export Options
┌─────────────────────────────────────────────────────────────┐
│ Choose Export Format:                                        │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ │ 📊 Excel    │  │ 📄 JSON     │  │ 📋 CSV      │          │
│ │ Workbook    │  │ Data        │  │ Files       │          │
│ │             │  │             │  │             │          │
│ │ Best for:   │  │ Best for:   │  │ Best for:   │          │
│ │ • Review    │  │ • Integration│  │ • Analysis  │          │
│ │ • Editing   │  │ • Automation│  │ • Import    │          │
│ └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ │ 📑 PDF      │  │ 🔗 XML      │  │ ⚙️ Custom   │          │
│ │ Reports     │  │ Data        │  │ Format      │          │
│ │             │  │             │  │             │          │
│ │ Best for:   │  │ Best for:   │  │ Best for:   │          │
│ │ • Reports   │  │ • Legacy    │  │ • Specific  │          │
│ │ • Archival  │  │ Systems     │  │ Requirements│          │
│ └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Excel Workbook Export

### Complete Project Workbook

The most comprehensive export option creates a fully formatted Excel workbook with multiple sheets:

#### Workbook Structure
```
Manufacturing_Plant_Project.xlsx
├─ 📋 Project Summary
├─ 🔌 Load Schedule
├─ 🔌 Cable Schedule
├─ ⚡ Bus/Board Schedule
├─ 📈 Calculations
├─ 🔍 Validation Results
└─ 📊 Summary Charts
```

#### Sheet Details

**Project Summary Sheet**
- Project name, date, standard used
- Overall statistics (components, loads, etc.)
- Key performance metrics
- Validation summary

**Load Schedule Sheet**
```
| Load ID | Load Name    | Power(kW) | Voltage | Type    | Current(A) | Source Bus |
|---------|--------------|-----------|---------|---------|------------|------------|
| L001    | Main Pump    | 5.5       | 400     | Motor   | 8.9        | B001       |
| L002    | HVAC Unit 1  | 15.0      | 400     | HVAC    | 24.2       | B001       |
| L003    | Lighting 1   | 2.2       | 230     | Lighting| 9.6        | B002       |
```

**Cable Schedule Sheet**
```
| Cable ID | From   | To   | Size(mm²) | Cores | Length(m) | Current(A) | VD(%) |
|----------|--------|------|-----------|-------|-----------|------------|-------|
| C001     | DB-01  | L001 | 2.5       | 4     | 25        | 8.9        | 1.8   |
| C002     | DB-01  | L002 | 4.0       | 4     | 30        | 24.2       | 2.1   |
```

**Calculations Sheet**
- All electrical calculations performed
- Formulas included for verification
- Reference standards used
- Calculation methodology

**Validation Results Sheet**
- Standards compliance report
- Electrical safety checks
- Warning and error summary
- Recommended actions

### Customization Options

#### Content Selection
Choose which data to include:

```
Export Configuration
┌─────────────────────────────────────────────────────────────┐
│ Include in Export:                                           │
│                                                             │
│ ✅ Load Schedule (18 loads)                                 │
│ ✅ Cable Schedule (15 cables)                               │
│ ✅ Bus/Board Schedule (3 buses)                             │
│ ✅ Electrical Calculations                                  │
│ ✅ Validation Results                                       │
│ ✅ Project Summary                                          │
│ ☐ Raw Extraction Data                                       │
│ ☐ Processing Logs                                           │
│ ☐ Debug Information                                         │
└─────────────────────────────────────────────────────────────┘
```

#### Formatting Options
- **Professional Format**: Clean, print-ready layout
- **Detailed Format**: Include all calculations and references
- **Compact Format**: Essential data only
- **Custom Format**: User-defined layout

#### Sheet Naming
Customize sheet names:
- Default: "Load Schedule", "Cable Schedule"
- Custom: "Equipment List", "Wiring Details", "Distribution Boards"
- Language: Support for multiple languages
- Standards: Include standard references in names

### Excel Export Features

#### Automatic Formatting
- **Column Headers**: Bold, colored background
- **Data Types**: Appropriate number formatting
- **Conditional Formatting**: Highlight issues or warnings
- **Print Layout**: Page breaks and headers for printing

#### Calculated Columns
- **Current Calculations**: Automatically calculated load currents
- **Voltage Drop**: Cable voltage drop percentages
- **Diversity Factors**: Applied where appropriate
- **Safety Margins**: Highlighted in different colors

#### Interactive Elements
- **Drop-down Lists**: For standardized values
- **Data Validation**: Prevent invalid entries
- **Hyperlinks**: Navigate between sheets
- **Comments**: AI explanations for extracted data

## 📄 JSON Data Export

### Structured Data Export

JSON format provides structured, programmatic access to all extracted data:

#### Basic JSON Structure
```json
{
  "project_info": {
    "name": "Manufacturing Plant Project",
    "standard": "IEC",
    "created_date": "2025-11-02",
    "extraction_confidence": 0.94,
    "total_components": 31
  },
  "loads": [
    {
      "load_id": "L001",
      "load_name": "Main Pump Motor",
      "power_kw": 5.5,
      "voltage": 400,
      "phases": 3,
      "load_type": "motor",
      "current_a": 8.9,
      "source_bus": "B001",
      "confidence": 0.96
    }
  ],
  "cables": [
    {
      "cable_id": "C001",
      "from_equipment": "DB-01",
      "to_equipment": "L001",
      "size_sqmm": 2.5,
      "cores": 4,
      "length_m": 25,
      "current_rating_a": 24,
      "confidence": 0.92
    }
  ],
  "buses": [
    {
      "bus_id": "B001",
      "bus_name": "Main Distribution Board",
      "voltage": 400,
      "rated_current_a": 630,
      "connected_loads": ["L001", "L002", "L005"],
      "confidence": 0.98
    }
  ],
  "validation": {
    "overall_valid": true,
    "warnings": ["High voltage drop on cable C004"],
    "errors": [],
    "quality_score": 0.96
  }
}
```

#### Advanced JSON Features

**Metadata Inclusion**
```json
{
  "metadata": {
    "extraction_engine": "AI-Excel-Extractor v1.0",
    "processing_time_seconds": 1.8,
    "source_file": "manufacturing_plant.xlsx",
    "extraction_settings": {
      "confidence_threshold": 0.8,
      "auto_corrections": true,
      "standard": "IEC"
    }
  }
}
```

**Relationship Mapping**
```json
{
  "relationships": {
    "load_cable_connections": [
      {
        "load_id": "L001",
        "cable_id": "C001",
        "connection_type": "power_supply"
      }
    ],
    "bus_load_assignments": [
      {
        "bus_id": "B001",
        "load_ids": ["L001", "L002", "L005"],
        "total_power_kw": 28.2
      }
    ]
  }
}
```

### JSON Export Customization

#### Export Levels
- **Complete**: All data including metadata and relationships
- **Standard**: Core component data only
- **Minimal**: Essential fields for basic integration
- **Custom**: User-selected fields

#### Field Filtering
```
JSON Export Configuration
┌─────────────────────────────────────────────────────────────┐
│ Field Selection:                                             │
│                                                             │
│ ✅ Include confidence scores                                 │
│ ✅ Include calculated values                                 │
│ ✅ Include validation results                                │
│ ✅ Include relationships                                     │
│ ✅ Include metadata                                          │
│ ☐ Include processing logs                                   │
│ ☐ Include debug information                                 │
│                                                             │
│ Compression: [None ▼]                                       │
│ Indentation: [2 spaces ▼]                                   │
└─────────────────────────────────────────────────────────────┘
```

#### Integration Features
- **API-Ready**: Formatted for direct API consumption
- **Schema Validation**: JSON Schema included
- **Error Handling**: Consistent error reporting
- **Version Control**: API versioning support

## 📋 CSV File Export

### Tabular Data Export

CSV format provides simple, universal data exchange:

#### Multiple CSV Files
```
Export Files:
├─ loads.csv          (Load schedule data)
├─ cables.csv         (Cable schedule data)
├─ buses.csv          (Bus/board data)
├─ relationships.csv  (Component connections)
└─ validation.csv     (Validation results)
```

#### Load Schedule CSV
```csv
Load_ID,Load_Name,Power_kW,Voltage,Phases,Load_Type,Current_A,Source_Bus,Confidence
L001,Main Pump Motor,5.5,400,3,Motor,8.9,B001,0.96
L002,HVAC Unit 1,15.0,400,3,HVAC,24.2,B001,0.94
L003,Lighting Bank,2.2,230,1,Lighting,9.6,B002,0.89
```

#### Cable Schedule CSV
```csv
Cable_ID,From_Equipment,To_Equipment,Size_sqmm,Cores,Length_m,Installation_Method,Current_Rating_A,Voltage_Drop_Percent,Confidence
C001,DB-01,L001,2.5,4,25,Tray,24,1.8,0.92
C002,DB-01,L002,4.0,4,30,Tray,32,2.1,0.95
```

### CSV Customization Options

#### Field Selection
- **All Fields**: Complete data export
- **Standard Fields**: Essential data only
- **Custom Fields**: User-selected columns
- **Calculated Fields**: Include/exclude calculated values

#### Format Options
- **Delimiter**: Comma, semicolon, tab, pipe
- **Encoding**: UTF-8, ASCII, ISO-8859-1
- **Headers**: Include/exclude column headers
- **Date Format**: ISO, US, European formats

#### Data Filtering
```
CSV Export Options
┌─────────────────────────────────────────────────────────────┐
│ Data Filtering:                                              │
│                                                             │
│ ✅ Include header row                                        │
│ ✅ Include calculated columns                                │
│ ✅ Include validation status                                 │
│ ☐ Include confidence scores                                 │
│                                                             │
│ Format Options:                                              │
│ Delimiter: [Comma ▼]                                        │
│ Encoding: [UTF-8 ▼]                                         │
│ Quote Style: [Minimal ▼]                                    │
└─────────────────────────────────────────────────────────────┘
```

## 📑 PDF Report Export

### Professional Reports

PDF exports create documentation-ready reports:

#### Report Sections

**Executive Summary**
- Project overview and statistics
- Key findings and recommendations
- Overall quality assessment
- Implementation readiness

**Technical Details**
- Complete equipment schedules
- Electrical calculations
- Standards compliance status
- Safety assessments

**Validation Report**
- Detailed validation results
- Warnings and error explanations
- Compliance status for each standard
- Recommended actions

#### Report Templates

**Standard Report Template**
```
┌─────────────────────────────────────────────────────────────┐
│ Electrical Distribution Project Report                      │
│                                                             │
│ Project: Manufacturing Plant Extension                     │
│ Date: November 2, 2025                                      │
│ Standard: IEC 60364                                         │
│                                                             │
│ SUMMARY                                                     │
│ • Total Loads: 18 (95% confidence)                         │
│ • Total Cables: 15 (92% confidence)                        │
│ • Validation Status: ✅ PASSED                              │
│ • Ready for Implementation: ✅ YES                          │
│                                                             │
│ DETAILED SCHEDULES                                          │
│ [Load Schedule] [Cable Schedule] [Bus Schedule]            │
│                                                             │
│ VALIDATION RESULTS                                          │
│ ✅ Power Balance: PASSED                                    │
│ ✅ Cable Ratings: PASSED                                    │
│ ✅ Standards Compliance: PASSED                             │
│ ⚠️  Voltage Drop: 1 warning                                 │
└─────────────────────────────────────────────────────────────┘
```

**Compact Report Template**
- Essential information only
- Summary statistics
- Key validation results
- Implementation checklist

**Detailed Report Template**
- Complete technical data
- All calculations shown
- Full validation details
- Extensive appendices

### PDF Customization

#### Content Selection
```
PDF Report Configuration
┌─────────────────────────────────────────────────────────────┐
│ Include Sections:                                            │
│                                                             │
│ ✅ Executive Summary                                         │
│ ✅ Project Statistics                                        │
│ ✅ Load Schedule                                             │
│ ✅ Cable Schedule                                            │
│ ✅ Validation Results                                        │
│ ✅ Calculations (Detailed)                                   │
│ ✅ Standards Compliance                                      │
│ ☐ Raw Extraction Data                                       │
│ ☐ Processing Logs                                           │
│                                                             │
│ Report Style: [Professional ▼]                              │
│ Page Size: [A4 ▼]                                           │
│ Orientation: [Portrait ▼]                                   │
└─────────────────────────────────────────────────────────────┘
```

#### Branding Options
- **Company Logo**: Include organization branding
- **Color Scheme**: Custom color preferences
- **Footer Information**: Contact details, project references
- **Page Numbers**: Automatic page numbering
- **Watermark**: Draft/Confidential markings

## 🔗 XML Data Export

### Standardized Data Exchange

XML format for enterprise system integration:

#### XML Schema
```xml
<?xml version="1.0" encoding="UTF-8"?>
<electrical_project xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <project_info>
    <name>Manufacturing Plant Project</name>
    <standard>IEC</standard>
    <extraction_confidence>0.94</extraction_confidence>
  </project_info>
  <loads>
    <load id="L001">
      <name>Main Pump Motor</name>
      <power unit="kW">5.5</power>
      <voltage unit="V">400</voltage>
      <phases>3</phases>
      <type>motor</type>
      <current unit="A">8.9</current>
      <source_bus ref="B001"/>
      <confidence>0.96</confidence>
    </load>
  </loads>
  <cables>
    <cable id="C001">
      <from ref="DB-01"/>
      <to ref="L001"/>
      <size unit="mm2">2.5</size>
      <cores>4</cores>
      <length unit="m">25</length>
      <installation>tray</installation>
    </cable>
  </cables>
</electrical_project>
```

#### XML Features
- **Schema Validation**: XSD schema for data validation
- **Namespace Support**: Avoid naming conflicts
- **Attribute Usage**: Efficient data representation
- **Element Hierarchy**: Logical data organization

### Legacy System Integration

#### Supported XML Standards
- **IEC 61970**: Common Information Model (CIM)
- **IEC 61850**: Communication protocols
- **Custom Schemas**: User-defined XML structures
- **Industry Standards**: Utility industry formats

## ⚙️ Custom Format Export

### User-Defined Formats

For specific integration requirements:

#### Template-Based Export
```
Custom Format Template
┌─────────────────────────────────────────────────────────────┐
│ Template Configuration:                                      │
│                                                             │
│ Template Name: [Legacy_System_Format]                       │
│                                                             │
│ Field Mapping:                                               │
│ Component_ID → EquipmentID                                   │
│ Load_Name → EquipmentName                                   │
│ Power_kW → PowerRating                                      │
│ Voltage → SystemVoltage                                     │
│                                                             │
│ Output Format: [Custom_Delimited ▼]                         │
│ Field Separator: [|]                                        │
│ Text Qualifier: ["]                                         │
│ Include Headers: [✅]                                        │
│                                                             │
│ [Save Template] [Export] [Preview]                          │
└─────────────────────────────────────────────────────────────┘
```

#### Custom Templates
- **Field Mapping**: Map AI fields to custom fields
- **Format Rules**: Define output format specifications
- **Validation**: Ensure custom format integrity
- **Preview**: See export before generation

### API Integration Export

#### REST API Output
```json
{
  "api_version": "1.0",
  "timestamp": "2025-11-02T00:20:36Z",
  "request_id": "ext_20251102_001",
  "data": {
    "project": { ... },
    "loads": [ ... ],
    "cables": [ ... ]
  },
  "status": "success",
  "metadata": { ... }
}
```

#### Database Integration
- **Direct Database**: Export to SQL databases
- **ORM Mapping**: Object-relational mapping
- **Bulk Insert**: Efficient large dataset import
- **Transaction Support**: Rollback capability

## 🔄 Integration Workflows

### Common Integration Scenarios

#### 1. Excel Workflow
```
AI Extraction → Excel Workbook → Manual Review → Calculations → Documentation
```
**Best For**: Traditional engineering workflows, detailed review

#### 2. API Integration
```
AI Extraction → JSON → REST API → Database → Application
```
**Best For**: Automated systems, real-time integration

#### 3. Analysis Pipeline
```
AI Extraction → CSV → Analytics Tools → Reports
```
**Best For**: Data analysis, business intelligence

#### 4. Enterprise Integration
```
AI Extraction → XML → Legacy Systems → Documentation
```
**Best For**: Enterprise systems, compliance requirements

### Batch Processing

#### Multiple File Processing
```
📁 Input Folder: /projects/2025/
├─ project1.xlsx
├─ project2.xlsx
└─ project3.xlsx

┌─────────────────────────────────────────────────────────────┐
│ Batch Export Settings:                                       │
│                                                             │
│ ✅ Process all files                                         │
│ ✅ Create individual exports                                 │
│ ✅ Generate consolidated report                              │
│                                                             │
│ Output Format: [Excel Workbook ▼]                           │
│ Naming Convention: [ProjectName_Date ▼]                     │
│                                                             │
│ [Start Batch Export] [Cancel]                               │
└─────────────────────────────────────────────────────────────┘
```

#### Automated Workflows
- **Scheduled Exports**: Automatic periodic exports
- **Triggered Exports**: Export on specific events
- **Webhook Integration**: Real-time export notifications
- **Monitoring**: Export status and error tracking

## 📦 Export Management

### File Organization

#### Naming Conventions
```
Standard Naming:
{ProjectName}_{ComponentType}_{Date}_{Version}.{ext}

Examples:
Manufacturing_Plant_Loads_20251102_v1.xlsx
Manufacturing_Plant_Cables_20251102_v1.json
Manufacturing_Plant_Project_20251102_v1.pdf

Custom Naming:
{Prefix}_{CustomField}_{Timestamp}.{ext}
```

#### Folder Structure
```
Exports/
├── 2025/
│   ├── November/
│   │   ├── Manufacturing_Plant/
│   │   │   ├── Excel/
│   │   │   ├── JSON/
│   │   │   └── Reports/
│   │   └── Office_Building/
│   └── October/
└── Templates/
    ├── Standard_Loads.csv
    ├── Standard_Cables.json
    └── Standard_Report.pdf
```

### Export History

#### Version Management
```
Export History
┌─────────────────────────────────────────────────────────────┐
│ Recent Exports (Last 30 days)                               │
│                                                             │
│ Manufacturing_Plant_Loads_20251102.xlsx    [Download]      │
│ Manufacturing_Plant_Cables_20251102.xlsx   [Download]      │
│ Manufacturing_Plant_Project_20251102.pdf   [Download]      │
│                                                             │
│ Archive Exports (Older than 30 days)                       │
│                                                             │
│ Office_Building_Project_20251015.xlsx    [Download]       │
│                                                             │
│ [Clear History] [Export All]                                │
└─────────────────────────────────────────────────────────────┘
```

#### Metadata Tracking
- **Export Date**: When the export was created
- **Source File**: Original Excel file used
- **Settings Used**: Configuration for the export
- **Validation Status**: Export integrity verification

### Quality Assurance

#### Export Validation
```
Export Quality Check
┌─────────────────────────────────────────────────────────────┐
│ Validation Results:                                          │
│                                                             │
│ ✅ Data Integrity: All components exported                  │
│ ✅ Relationship Mapping: All connections preserved          │
│ ✅ Calculation Accuracy: All formulas verified              │
│ ✅ Format Compliance: Export format valid                   │
│                                                             │
│ Export Quality Score: 98%                                   │
│                                                             │
│ Export Ready: ✅ YES                                        │
│                                                             │
│ [Download Export] [Export Settings] [Re-export]            │
└─────────────────────────────────────────────────────────────┘
```

#### Verification Steps
- **Data Completeness**: Ensure all extracted data is included
- **Relationship Integrity**: Verify all connections are preserved
- **Calculation Accuracy**: Check all electrical calculations
- **Format Validity**: Validate exported format specifications

## 🎯 Best Practices for Export

### Choosing the Right Format

#### Format Selection Guide

**Use Excel When**:
- Need to review and edit extracted data
- Creating documentation or reports
- Sharing with stakeholders for review
- Performing manual calculations

**Use JSON When**:
- Integrating with other software systems
- Building automated workflows
- Need programmatic access to data
- Creating API interfaces

**Use CSV When**:
- Importing into analysis tools
- Simple data exchange required
- Working with business intelligence tools
- Creating custom reports

**Use PDF When**:
- Creating formal documentation
- Archiving project records
- Sharing with non-technical stakeholders
- Generating compliance reports

### Export Optimization

#### Performance Considerations
- **Large Datasets**: Use JSON or CSV for large projects
- **Real-time Processing**: Use API-compatible formats
- **Batch Operations**: Optimize for bulk processing
- **Memory Usage**: Consider format efficiency

#### Quality Assurance
- **Always Validate**: Check export integrity
- **Test Integration**: Verify with target systems
- **Document Settings**: Record export configurations
- **Monitor Performance**: Track export success rates

### Security and Privacy

#### Data Protection
- **Sensitive Data**: Anonymize personal information
- **Access Control**: Restrict export permissions
- **Audit Trail**: Log all export activities
- **Encryption**: Protect exported files in transit

#### Compliance
- **Standards Compliance**: Ensure exports meet regulatory requirements
- **Data Retention**: Follow organizational data policies
- **Privacy Regulations**: Comply with GDPR, CCPA, etc.
- **Audit Requirements**: Maintain export audit trails

---

**Next**: Explore the Technical Documentation for developers who need to integrate or extend the AI extraction system.