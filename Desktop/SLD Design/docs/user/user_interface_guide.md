# User Interface Guide - AI Excel Extraction

## Overview

This guide walks you through the AI Excel Extraction interface, explaining all features, navigation options, and how to get the most out of the system.

## 🖥️ Main Interface Layout

### Landing Page

When you select **"🤖 AI Excel Import"** from the main menu, you'll see the AI extraction dashboard:

```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 AI-Powered Excel Import                                   │
│ *Intelligent extraction for electrical distribution projects* │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ │ Success Rate│  │Avg Processing│  │ Components  │          │
│ │    95%      │  │    30s      │  │  Extracted  │          │
│ └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│ 📁 Upload Excel File                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │     Drag and drop your Excel file here                │ │
│ │              or click to browse                       │ │
│ │                                                         │ │
│ │           Supported: .xlsx, .xls                      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ⚙️ Advanced Options ▼                                      │
│                                                             │
│ [🔄 Process with AI]                                       │
└─────────────────────────────────────────────────────────────┘
```

### Header Section

#### System Status Indicators
- **Success Rate**: Historical accuracy of extractions (95% average)
- **Average Processing Time**: Typical time for file processing (30 seconds)
- **Components Extracted**: Total electrical components processed
- **AI Status**: Real-time system status (Ready, Processing, Complete)

#### Help and Support
- **❓ Help Icons**: Contextual help throughout the interface
- **📖 Documentation**: Links to detailed guides
- **🔄 Refresh**: Reset interface to initial state
- **⚙️ Settings**: Access advanced configuration

## 📁 File Upload Interface

### Upload Area

#### Drag-and-Drop Zone
- **Visual Feedback**: Dotted border when hovering files
- **File Preview**: Shows filename and size before upload
- **Format Validation**: Automatic format checking
- **Multiple Files**: Support for batch processing (coming soon)

#### Browse Button
- **File Dialog**: Opens system file browser
- **Format Filtering**: Shows only supported Excel formats
- **Size Checking**: Validates file size limits (50MB max)
- **Recent Files**: Quick access to recently used files

#### Upload Progress
- **Progress Bar**: Visual indication of upload progress
- **File Info**: Name, size, and type display
- **Status Messages**: Upload success or error messages
- **Cancel Option**: Ability to cancel upload in progress

### File Validation

The system automatically validates uploaded files:

#### Format Checks
- ✅ **Extension Validation**: .xlsx, .xls formats
- ✅ **File Integrity**: Checks for corruption
- ✅ **Readability**: Verifies file can be opened
- ✅ **Content Check**: Ensures data is present

#### Size and Content Validation
- **File Size**: Maximum 50MB recommended
- **Sheet Detection**: Automatic sheet counting
- **Data Detection**: Checks for tabular data
- **Header Validation**: Verifies column headers present

## ⚙️ Advanced Options Panel

### Configuration Settings

#### Confidence Threshold
```
Confidence Threshold: ████████░░ 80% (0.50 - 1.00)
```
- **Purpose**: Minimum confidence for automatic processing
- **Range**: 50% to 100%
- **Default**: 80%
- **Effect**: Higher values require more manual review

#### Auto-Corrections
- **Checkbox**: Enable/disable automatic data corrections
- **Scope**: Broken IDs, missing relationships, naming conventions
- **Default**: Enabled
- **User Control**: Can be toggled per session

#### Electrical Standards
```
Standards Selection: [IEC ▼]
```
- **IEC**: International Electrotechnical Commission
- **IS**: Indian Standards
- **NEC**: National Electrical Code
- **Custom**: User-defined standards (advanced)

#### Processing Options
- **Batch Mode**: Process multiple files simultaneously
- **Parallel Processing**: Use multiple CPU cores
- **Memory Optimization**: For large files
- **Background Processing**: Non-blocking operations

## 🔄 Processing Interface

### Real-Time Progress

When processing begins, the interface shows detailed progress:

```
🔄 AI Processing Status
┌─────────────────────────────────────────────────┐
│ Analyzing file structure...    [■■■■■■□□□□] 60% │
│                                                     │
│ Current Step: Column Mapping                       │
│ • Load Schedule: 95% confidence                   │
│ • Cable Schedule: 88% confidence                  │
│ • Bus Schedule: 92% confidence                    │
│                                                     │
│ Estimated Time Remaining: 15 seconds              │
│                                                     │
│ [⏸️ Pause] [⏹️ Cancel]                            │
└─────────────────────────────────────────────────┘
```

### Processing Steps

The AI follows a systematic processing pipeline:

#### 1. File Analysis
- **File Structure**: Identify sheets and layout
- **Data Detection**: Locate tabular data areas
- **Format Assessment**: Determine data quality

#### 2. Pattern Recognition
- **Sheet Classification**: Identify electrical data types
- **Column Mapping**: Map headers to data fields
- **Content Analysis**: Extract electrical components

#### 3. Data Extraction
- **Component Creation**: Generate Load, Cable, Bus objects
- **Relationship Building**: Establish connections
- **Calculation Integration**: Apply electrical formulas

#### 4. Quality Enhancement
- **ID Generation**: Create missing component IDs
- **Relationship Repair**: Fix broken connections
- **Data Standardization**: Apply naming conventions

#### 5. Validation
- **Electrical Rules**: Check engineering constraints
- **Standards Compliance**: Verify regulatory requirements
- **System Consistency**: Validate overall project integrity

### Processing Controls

#### Pause/Resume
- **Pause**: Temporarily stop processing
- **Resume**: Continue from last checkpoint
- **Use Cases**: Review intermediate results, system resource management

#### Cancel
- **Stop Processing**: Immediately halt all operations
- **Clean Up**: Remove temporary files and data
- **Reset Interface**: Return to upload state

#### Detailed Logs
- **Processing Log**: Real-time processing information
- **Error Reports**: Detailed error descriptions
- **Performance Metrics**: Timing and resource usage
- **Debug Information**: Technical details for support

## 📊 Results Dashboard

### Summary Metrics Panel

After processing completes, the interface displays comprehensive results:

```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Extraction Results Summary                               │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ │Confidence   │  │Components   │  │Processing   │          │
│ │   92%       │  │    23       │  │   1.8s      │          │
│ │ Excellent   │  │ Loads       │  │ Complete    │          │
│ └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ │Quality Score│  │Corrections  │  │Validation   │          │
│ │    96%      │  │    15       │  │    8        │          │
│ │ Very High   │  │ Auto-Fixed  │  │ Passed      │          │
│ └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Tabbed Results Interface

The detailed results are organized in tabs for easy navigation:

#### 🔌 Loads Tab
```
Load Schedule Results
┌─────────────────────────────────────────────────────────────┐
│ Load ID │ Load Name     │ Power │ Voltage │ Confidence │Action │
├─────────┼───────────────┼───────┼─────────┼────────────┼───────┤
│ L001    │ Main Pump     │ 5.5kW │ 400V    │    96%     │ ✅    │
│ L002    │ HVAC Unit 1   │ 15kW  │ 400V    │    94%     │ ✅    │
│ L003    │ Lighting Bank │ 2.2kW │ 230V    │    89%     │ ✅    │
│ L004    │ Motor-01      │ 7.5kW │ 400V    │    67%     │ ⚠️    │
└─────────────────────────────────────────────────────────────┘

[Edit Selected] [Validate All] [Export Loads]
```

#### 🔌 Cables Tab
```
Cable Schedule Results
┌─────────────────────────────────────────────────────────────┐
│ Cable ID │ From      │ To      │ Size    │ Length │Conf.  │
├──────────┼───────────┼─────────┼─────────┼────────┼───────┤
│ C001     │ DB-01     │ L001    │ 2.5mm²  │ 25m    │ 92%   │
│ C002     │ DB-01     │ L002    │ 4.0mm²  │ 30m    │ 95%   │
│ C003     │ DB-01     │ L003    │ 1.5mm²  │ 15m    │ 88%   │
│ C004     │ B001      │ L004    │ 2.5mm²  │ 40m    │ 73%   │
└─────────────────────────────────────────────────────────────┘

[Edit Selected] [Validate All] [Export Cables]
```

#### ⚡ Buses Tab
```
Bus/Board Results
┌─────────────────────────────────────────────────────────────┐
│ Bus ID │ Bus Name              │ Voltage │ Current │Conf.  │
├────────┼───────────────────────┼─────────┼─────────┼───────┤
│ B001   │ Main Distribution     │ 400V    │ 630A    │ 98%   │
│ B002   │ Panel A               │ 230V    │ 125A    │ 85%   │
└─────────────────────────────────────────────────────────────┘

[Edit Selected] [Validate All] [Export Buses]
```

#### 📋 Validation Tab
```
Electrical Engineering Validation
┌─────────────────────────────────────────────────────────────┐
│ ✅ PASSED: Power Balance Check                              │
│ ✅ PASSED: Cable Current Ratings                            │
│ ⚠️  WARNING: High voltage drop on Cable C004 (4.2%)         │
│ ✅ PASSED: Breaker Coordination                             │
│ ✅ PASSED: Standards Compliance (IEC)                       │
│                                                             │
│ Quality Score: 96% (Very High)                             │
│                                                             │
│ [View Details] [Export Validation Report]                   │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Manual Correction Interface

### Low Confidence Items

When the AI identifies uncertain extractions, it presents them for manual review:

```
🔧 Manual Corrections Required
┌─────────────────────────────────────────────────────────────┐
│ 3 items require your attention                               │
│                                                             │
│ Item 1: Load Mapping Uncertainty                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Load ID: L004                                           │ │
│ │ Description: "Motor-01"                                 │ │
│ │ Confidence: 67% (Below threshold)                       │ │
│ │ AI Suggestions:                                         │ │
│ │ • Map to: "3-Phase Motor"                               │ │
│ │ • Power Factor: 0.85 (typical for motors)              │ │
│ │ • Efficiency: 0.88 (estimated)                         │ │
│ │                                                         │ │
│ │ Your Correction: [Motor-01 ________________]           │ │
│ │ Load Type: [Motor ▼] Power: [7.5___] kW               │ │
│ │                                                         │ │
│ │ Confidence Rating: ████████░░ 80%                      │ │
│ │                                             [✅Approve] │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Item 2: Voltage Unit Correction                             │
│ [Similar interface for voltage unit issues]                 │
│                                                             │
│ Item 3: Missing Cable Specification                         │
│ [Similar interface for cable issues]                        │
│                                                             │
│ [✅ Approve All] [🔄 Re-run Extraction] [❌ Cancel]         │
└─────────────────────────────────────────────────────────────┘
```

### Interactive Correction Tools

#### Field Editors
- **Text Input**: For component names and descriptions
- **Dropdown Lists**: For standardized values (load types, standards)
- **Numeric Inputs**: For power, voltage, current values
- **Unit Selectors**: For different measurement units

#### Validation Assistance
- **Real-time Validation**: Immediate feedback on corrections
- **Suggested Values**: AI-recommended values based on patterns
- **Range Checking**: Ensure values are within reasonable bounds
- **Standards Checking**: Verify compliance with selected standards

#### Batch Operations
- **Select Multiple**: Choose multiple items for batch correction
- **Apply Pattern**: Apply same correction to similar items
- **Undo/Redo**: Reverse or repeat correction actions
- **Auto-complete**: Fill similar fields automatically

## 🎯 User Interaction Features

### Interactive Elements

#### Hover Information
- **Tooltips**: Detailed information on hover
- **Field Help**: Context-sensitive help for each field
- **Validation Hints**: Tips for entering valid data
- **Examples**: Sample values for guidance

#### Click Actions
- **Expand Details**: Click to see more information
- **Edit Mode**: Click to modify extracted values
- **Validation Results**: Click to view detailed checks
- **Related Items**: Click to see connected components

#### Keyboard Shortcuts
- **Ctrl+Enter**: Approve current corrections
- **Ctrl+Z**: Undo last action
- **Ctrl+Y**: Redo last action
- **F1**: Show help for current section
- **Esc**: Cancel current operation

### Responsive Design

#### Desktop Interface
- **Wide Layout**: Optimized for large screens
- **Multi-column**: Efficient use of screen space
- **Full Features**: Complete functionality available
- **Keyboard Support**: Full keyboard navigation

#### Tablet Interface
- **Adaptive Layout**: Adjusts to screen size
- **Touch Friendly**: Large touch targets
- **Simplified View**: Focus on essential features
- **Swipe Navigation**: Tab switching via swipe

#### Mobile Interface
- **Stacked Layout**: Single column design
- **Simplified Controls**: Essential features only
- **Tap Targets**: Large buttons for touch
- **Progressive Disclosure**: Show details on demand

## 📱 Navigation and Workflow

### Main Workflow Steps

#### 1. Upload Phase
1. **Access AI Import**: From main menu
2. **Upload File**: Drag & drop or browse
3. **Validate**: Automatic format checking
4. **Configure**: Set processing options

#### 2. Processing Phase
1. **Monitor Progress**: Real-time status updates
2. **Manage Processing**: Pause, resume, or cancel if needed
3. **View Logs**: Detailed processing information
4. **Handle Errors**: Address any processing issues

#### 3. Review Phase
1. **Summary Review**: Overall results assessment
2. **Detailed Check**: Tab-by-tab component review
3. **Manual Corrections**: Fix low-confidence items
4. **Validation Review**: Check engineering validation

#### 4. Export Phase
1. **Choose Format**: Select export options
2. **Configure Output**: Customize export settings
3. **Download**: Save results to local system
4. **Integration**: Import into main application

### Navigation Breadcrumbs

```
Home > AI Excel Import > Upload > Processing > Results > Export
```

#### Quick Navigation
- **Back Buttons**: Return to previous steps
- **Jump Navigation**: Direct access to any step
- **Progress Tracking**: Visual progress indication
- **Step Validation**: Ensure completion before advancing

### Context Switching

#### Multi-Project Management
- **Project Tabs**: Work on multiple projects simultaneously
- **Quick Switch**: Easy switching between projects
- **Session Management**: Restore previous sessions
- **Auto-save**: Preserve work automatically

#### Integration Navigation
- **Main Application**: Seamless transition to full design tools
- **Calculation Engine**: Direct access to electrical calculations
- **Standards Manager**: Quick access to compliance tools
- **Export Manager**: Multiple export format options

## 🎨 Customization and Preferences

### User Preferences

#### Interface Customization
- **Theme Selection**: Light/dark mode options
- **Layout Preferences**: Customize dashboard layout
- **Font Size**: Adjust text size for readability
- **Color Schemes**: High contrast or color-blind friendly

#### Processing Preferences
- **Default Confidence**: Set personal threshold
- **Auto-corrections**: Preferred correction settings
- **Standards**: Default electrical standards
- **Export Format**: Preferred export formats

#### Performance Settings
- **Processing Speed**: Balance speed vs. accuracy
- **Memory Usage**: Optimize for system resources
- **Background Processing**: Enable/disable background tasks
- **Notifications**: Control alert preferences

### Session Management

#### Save/Load Sessions
- **Session Save**: Preserve current work state
- **Session Load**: Resume previous work
- **Auto-recovery**: Restore after system interruption
- **Session History**: Access to recent sessions

#### Sharing and Collaboration
- **Export Sessions**: Share configuration settings
- **Import Settings**: Apply team standards
- **Version Control**: Track configuration changes
- **Team Profiles**: Organization-wide settings

---

**Next**: Learn how to review and correct extracted data in the Data Review Process guide.