# AI Excel Extraction - Quick Start Guide

## Getting Started in 5 Minutes

Welcome to the AI-Powered Excel Extraction system! This guide will help you extract electrical distribution data from Excel files in just a few minutes.

## 🚀 What You'll Accomplish

- **Upload Excel files** containing electrical project data
- **Extract components** automatically (loads, cables, buses)
- **Review and correct** any low-confidence items
- **Generate complete projects** with calculations and validation
- **Export results** in multiple formats

## 📋 Prerequisites

Before you begin, ensure you have:
- ✅ Excel files with electrical distribution data
- ✅ Modern web browser (Chrome, Firefox, Safari, Edge)
- ✅ Files in supported formats (.xlsx, .xls)
- ✅ Data in standard electrical project layouts

## 🎯 Step-by-Step Process

### Step 1: Access AI Excel Import
1. Open the **Electrical Design Automation System**
2. From the main menu, select **"🤖 AI Excel Import"**
3. You'll see the AI extraction dashboard

### Step 2: Upload Your Excel File
1. **Drag and drop** your Excel file into the upload area, or
2. **Click "Browse"** to select files from your computer
3. **Supported formats**: .xlsx (Excel 2007+), .xls (Excel 97-2003)
4. **File size**: Up to 50MB recommended

### Step 3: Configure Extraction Settings
Click "⚙️ Advanced Options" to configure:

| Setting | Description | Recommended |
|---------|-------------|-------------|
| **Confidence Threshold** | Minimum confidence for auto-processing | 80% |
| **Auto-Corrections** | Allow AI to fix common issues | ✅ Enabled |
| **Electrical Standard** | Standards for validation (IEC, IS, NEC) | IEC |

### Step 4: Monitor Processing
Watch real-time progress as the AI processes your file:

```
🔄 Analyzing file structure...    [■■■■■□□□□□] 20%
🔄 Identifying electrical patterns... [■■■■■■■■□□] 60%  
🔄 Extracting components...       [■■■■■■■■■□] 80%
🔄 Validating electrical consistency... [■■■■■■■■■■] 100%
✅ Processing completed!
```

### Step 5: Review Results
The system displays a comprehensive results dashboard:

#### Summary Metrics
- **Overall Confidence**: How reliable the extraction is
- **Components Extracted**: Number of loads, cables, buses found
- **Processing Time**: Time taken to process the file
- **Data Quality Score**: Overall data completeness and accuracy

#### Detailed Results Tabs
- **🔌 Loads**: All electrical loads with specifications
- **🔌 Cables**: Cable schedules and connections  
- **⚡ Buses**: Distribution panels and connections
- **📋 Validation**: Electrical engineering validation results

### Step 6: Handle Low-Confidence Items
If the AI found uncertain items, you'll see a **"🔧 Manual Corrections"** section:

1. **Review each item** flagged with confidence below threshold
2. **Accept AI suggestions** or **provide corrections**
3. **Click "✅ Approve"** to confirm each change
4. **Rate your confidence** in the correction (0-100%)

### Step 7: Export Your Project
Once satisfied with results:

1. **Click "Create Project"** to generate a complete project
2. **Choose export format**:
   - **Excel**: Load lists, cable schedules
   - **JSON**: Complete project data
   - **PDF**: Professional reports
3. **Download files** or **continue editing** in the main application

## 📊 Understanding Confidence Scores

The AI provides confidence scores to help you trust the results:

| Confidence | Interpretation | Action Required |
|------------|----------------|-----------------|
| **90-100%** | Very High | ✅ Accept results |
| **80-89%** | High | ✅ Generally reliable |
| **70-79%** | Medium | 👀 Review key items |
| **60-69%** | Low | 🔧 Manual review needed |
| **<60%** | Very Low | 🔧 Significant corrections required |

## 🎯 Supported Excel Structures

The AI works best with these common electrical project formats:

### Load Schedule Format
| Column | Example Headers | AI Recognition |
|--------|-----------------|----------------|
| **Load ID** | "Load ID", "Equipment ID", "Tag" | ✅ Auto-detected |
| **Load Name** | "Load Name", "Description", "Equipment Name" | ✅ Auto-detected |
| **Power (kW)** | "Power (kW)", "Rating kW", "Capacity" | ✅ Auto-detected |
| **Voltage (V)** | "Voltage", "System Voltage", "V" | ✅ Auto-detected |
| **Type** | "Load Type", "Category", "Equipment Type" | ✅ Auto-detected |
| **Source Bus** | "Source Bus", "Panel", "Distribution Board" | ✅ Auto-detected |

### Cable Schedule Format  
| Column | Example Headers | AI Recognition |
|--------|-----------------|----------------|
| **Cable ID** | "Cable ID", "Cable Ref", "Tag" | ✅ Auto-detected |
| **From Equipment** | "From", "Source", "Origin" | ✅ Auto-detected |
| **To Equipment** | "To", "Destination", "Load" | ✅ Auto-detected |
| **Size (mm²)** | "Size", "Cross Section", "mm²" | ✅ Auto-detected |
| **Cores** | "Cores", "Core Count", "No of Cores" | ✅ Auto-detected |
| **Length (m)** | "Length", "Run Length", "Distance" | ✅ Auto-detected |

## 💡 Pro Tips for Best Results

### Excel File Preparation
- ✅ **Use clear headers** in the first row
- ✅ **Include electrical terms** in column names
- ✅ **Use consistent formatting** across sheets
- ✅ **Organize data in tables** with clear structure
- ❌ **Avoid merged cells** in data areas
- ❌ **Don't mix units** in the same column
- ❌ **Avoid special characters** in headers

### Getting Higher Confidence Scores
1. **Use standard electrical terminology**:
   - "Load ID" instead of "ID_LD"
   - "Power (kW)" instead of "P_kW"
   - "Source Bus" instead of "Bus_Src"

2. **Include all necessary columns**:
   - Essential: Load ID, Name, Power, Voltage
   - Helpful: Type, Source Bus, Cable Length
   - Optional: Power Factor, Efficiency, Priority

3. **Maintain data consistency**:
   - Use consistent power units (all kW)
   - Use standard voltage values (230V, 400V, 415V)
   - Use proper electrical component names

## 🆘 Quick Troubleshooting

### File Upload Issues
**Problem**: "File format not supported"
- **Solution**: Convert .csv to .xlsx using Excel or LibreOffice

**Problem**: "File too large"
- **Solution**: Split large files into smaller sections (<50MB each)

**Problem**: "No readable data found"
- **Solution**: Ensure data starts in row 1 with headers

### Processing Issues  
**Problem**: "Low confidence scores"
- **Solution**: 
  1. Check column headers match expected formats
  2. Review and correct flagged items manually
  3. Consider retraining with better file structure

**Problem**: "Missing components"
- **Solution**:
  1. Verify data is in standard table format
  2. Check for merged cells or formatting issues
  3. Ensure electrical terms are used in headers

### Results Issues
**Problem**: "Broken relationships"
- **Solution**: The AI automatically fixes these, but you can:
  1. Review the corrections made
  2. Manually assign loads to correct buses
  3. Verify cable connections are logical

## 🎓 Next Steps

Once you're comfortable with the basics:

1. **Read the Excel Format Guide** for advanced structuring
2. **Explore the Data Review Process** for quality assurance  
3. **Learn about Export Options** for different use cases
4. **Check Integration Examples** for workflow automation

## 📞 Getting Help

- **Built-in Help**: Click the "❓" icons throughout the interface
- **Demo Files**: Use sample files to practice extraction
- **Validation Reports**: Review electrical engineering validation results
- **Support Documentation**: Check troubleshooting guides for detailed solutions

---

**Ready to get started?** Upload your first Excel file and experience the power of AI-driven electrical data extraction!