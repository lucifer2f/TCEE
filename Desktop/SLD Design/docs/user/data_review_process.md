# Data Review Process - Quality Assurance and Corrections

## Overview

This guide explains how to review, validate, and correct extracted data to ensure high-quality electrical project data. The AI provides confidence scores and flags uncertain items for manual review.

## 📊 Understanding Extraction Results

### Confidence Scoring System

The AI assigns confidence scores to help you evaluate extraction reliability:

#### Confidence Levels
| Score Range | Reliability | Action Required | Color Code |
|-------------|-------------|-----------------|------------|
| **90-100%** | Very High | ✅ Accept results | Green |
| **80-89%** | High | ✅ Generally reliable | Light Green |
| **70-79%** | Medium | 👀 Review key items | Yellow |
| **60-69%** | Low | 🔧 Manual review needed | Orange |
| **<60%** | Very Low | 🔧 Significant corrections required | Red |

#### Confidence Factors
The AI calculates confidence based on:

- **Data Quality**: Completeness and consistency of source data
- **Pattern Match**: How well data matches electrical engineering patterns
- **Relationship Integrity**: Consistency of connections between components
- **Standards Compliance**: Adherence to electrical engineering standards
- **Logical Consistency**: Internal consistency of electrical parameters

### Quality Metrics Dashboard

After extraction, review the quality metrics:

#### Overall Quality Indicators
```
📊 Quality Dashboard
┌─────────────────────────────────────────────────────────────┐
│ Extraction Quality: 94% (Excellent)                         │
│                                                             │
│ Data Completeness: ████████████████████ 96%                │
│ Relationship Integrity: ████████████████░░ 88%             │
│ Standards Compliance: ████████████████████ 98%             │
│ Electrical Consistency: ███████████████░░░ 85%             │
│                                                             │
│ Issues Found: 3 (Low Priority)                              │
│ Warnings: 2 (Informational)                                 │
│ Corrections Made: 15 (Automatic)                            │
└─────────────────────────────────────────────────────────────┘
```

#### Component-Specific Metrics
- **Loads**: Data completeness, electrical parameter validity
- **Cables**: Connection integrity, sizing adequacy
- **Buses**: Rating consistency, load assignments
- **System**: Overall electrical balance, standards compliance

## 🔍 Detailed Review Process

### Step 1: Summary Assessment

#### Overall Results Review
1. **Check Overall Confidence**: Should be >80% for acceptable quality
2. **Review Component Count**: Verify expected number of components extracted
3. **Assess Processing Time**: Unusually long times may indicate issues
4. **Examine Corrections**: Review what the AI automatically fixed

#### Critical Issues Scan
Look for these red flags:
- ❌ **Overall confidence <70%**
- ❌ **Missing critical components** (loads, cables, buses)
- ❌ **Broken relationships** (loads not connected to cables)
- ❌ **Electrical violations** (overloaded cables, undersized breakers)

### Step 2: Component-by-Component Review

#### Load Schedule Review
```
🔌 Load Review Checklist
┌─────────────────────────────────────────────────────────────┐
│ ID: L001 | Name: Main Pump Motor | Power: 5.5kW | 400V    │
│                                                             │
│ ✅ Power rating reasonable for motor type                    │
│ ✅ Voltage matches expected system voltage                   │
│ ✅ Load type correctly identified (Motor)                    │
│ ✅ Source bus assignment logical                             │
│ ⚠️  Power factor 0.75 (low for motor - typical is 0.85)     │
│                                                             │
│ [✅ Accept] [🔧 Edit] [🚫 Reject]                           │
└─────────────────────────────────────────────────────────────┘
```

**Review Each Load for**:
- ✅ **ID Format**: Consistent with project naming convention
- ✅ **Name Clarity**: Descriptive and meaningful
- ✅ **Power Rating**: Reasonable for equipment type
- ✅ **Voltage**: Standard system voltage
- ✅ **Load Type**: Correctly classified
- ✅ **Source Bus**: Logically connected to correct bus
- ✅ **Electrical Parameters**: Power factor, efficiency within range

#### Cable Schedule Review
```
🔌 Cable Review Checklist
┌─────────────────────────────────────────────────────────────┐
│ ID: C001 | From: DB-01 | To: L001 | Size: 2.5mm² | 25m    │
│                                                             │
│ ✅ Connection to existing load (L001)                       │
│ ✅ Cable size adequate for load current                     │
│ ✅ Length reasonable for installation                       │
│ ✅ Installation method appropriate                          │
│ ✅ Armoring suitable for application                        │
│                                                             │
│ [✅ Accept] [🔧 Edit] [🚫 Reject]                           │
└─────────────────────────────────────────────────────────────┘
```

**Review Each Cable for**:
- ✅ **ID Format**: Consistent naming convention
- ✅ **End Points**: Both from/to equipment exist
- ✅ **Connection Logic**: Makes electrical sense
- ✅ **Sizing**: Adequate for connected load
- ✅ **Installation**: Appropriate method selected
- ✅ **Length**: Reasonable for installation

#### Bus/Board Review
```
⚡ Bus Review Checklist
┌─────────────────────────────────────────────────────────────┐
│ ID: B001 | Name: Main Distribution | 400V | 630A           │
│                                                             │
│ ✅ Voltage matches system voltage                           │
│ ✅ Current rating appropriate for connected loads           │
│ ✅ Short circuit rating adequate                            │
│ ✅ Load assignments logical                                 │
│                                                             │
│ [✅ Accept] [🔧 Edit] [🚫 Reject]                           │
└─────────────────────────────────────────────────────────────┘
```

**Review Each Bus for**:
- ✅ **ID Format**: Consistent naming convention
- ✅ **Voltage Level**: Matches system voltage
- ✅ **Current Rating**: Adequate for connected loads
- ✅ **Short Circuit Rating**: Adequate for system fault level
- ✅ **Load Balance**: Connected loads appropriate

### Step 3: Relationship Validation

#### Load-to-Cable Connections
Verify each load is properly connected:

```
Connection Matrix
Load L001 (Main Pump 5.5kW)
├─ ✅ Connected to C001 (DB-01 → L001)
├─ ✅ Cable size 2.5mm² adequate for 8.9A load current
└─ ✅ Length 25m reasonable

Load L002 (HVAC Unit 15kW)
├─ ✅ Connected to C002 (DB-01 → L002)
├─ ✅ Cable size 4.0mm² adequate for 24.2A load current
└─ ✅ Length 30m reasonable
```

#### Bus Load Assignments
Check load distribution across buses:

```
Bus Assignment Review
B001 - Main Distribution (400V, 630A)
├─ ✅ Total connected load: 45.2kW
├─ ✅ Load current: 73.1A (within 80% of bus rating)
└─ ✅ Breaker coordination maintained

B002 - Panel A (230V, 125A)
├─ ✅ Total connected load: 8.5kW
├─ ✅ Load current: 37.0A (within capacity)
└─ ✅ Appropriate for lighting and small loads
```

### Step 4: Electrical Engineering Validation

#### Power Balance Check
```
⚡ Power Balance Analysis
System Total Load: 53.7kW
Bus Loading:
├─ B001 (400V): 45.2kW (84% of capacity)
├─ B002 (230V): 8.5kW (68% of capacity)
└─ Total: 53.7kW (matches sum of loads ✅)
```

#### Cable Rating Validation
```
Cable Current Check
C001: 2.5mm² Cu cable
├─ Current carrying capacity: 24A (IEC standard)
├─ Load current: 8.9A
├─ Utilization: 37% (adequate ✅)
└─ Voltage drop: 1.8% (within 5% limit ✅)
```

#### Standards Compliance
```
Standards Compliance Check
IEC 60364 Compliance:
├─ ✅ Load categorization appropriate
├─ ✅ Cable sizing meets standard requirements
├─ ✅ Protection coordination maintained
└─ ✅ Installation methods compliant

Electrical Safety:
├─ ✅ No overloaded cables detected
├─ ✅ Adequate short circuit ratings
└─ ✅ Proper earthing connections assumed
```

## 🔧 Manual Correction Process

### Types of Corrections

#### 1. Data Value Corrections
**When**: Incorrect values detected during review
**How**: Edit field values directly

```
Example: Correcting Load Power
Before: Power = 5.5kW (extracted from "5.5")
After:  Power = 7.5kW (correct motor rating)

AI Suggestion: "5.5kW seems low for a pump motor.
                Typical pump motors are 7.5-15kW.
                Please verify actual rating."
```

#### 2. Component Type Corrections
**When**: Load or cable type incorrectly identified
**How**: Change component type from dropdown

```
Example: Load Type Correction
Before: Load Type = "General"
AI Suggestion: Load appears to be a motor based on name "Main Pump"
After:  Load Type = "Motor" (user confirmed)
```

#### 3. Relationship Corrections
**When**: Incorrect connections between components
**How**: Reassign connections manually

```
Example: Cable Connection Correction
Before: Cable C004 connects B001 → L099 (non-existent load)
Issue:  L099 doesn't exist
After:  Cable C004 connects B001 → L005 (existing motor)
```

#### 4. Parameter Validation
**When**: Electrical parameters outside normal ranges
**How**: Enter correct values with AI guidance

```
Example: Power Factor Validation
Before: Power Factor = 0.95 (unusually high for motor)
AI Suggestion: "Typical motor power factor is 0.80-0.90.
                High values may indicate measurement error."
After:  Power Factor = 0.85 (corrected value)
```

### Correction Workflow

#### 1. Identify Issues
```
🔍 Issue Detection
Issues Found: 5
├─ ⚠️  L004: Low confidence load type (67%)
├─ ⚠️  C003: Missing installation method
├─ ⚠️  B001: High bus utilization (94%)
├─ ❌ L008: Power rating seems incorrect
└─ ❌ C007: Cable length unrealistic (250m)
```

#### 2. Prioritize Corrections
**Priority 1 (Critical)**: Electrical safety issues
- Undersized cables
- Overloaded buses
- Missing protection

**Priority 2 (Important)**: Data accuracy issues
- Incorrect equipment ratings
- Wrong component types
- Broken relationships

**Priority 3 (Minor)**: Consistency improvements
- Naming conventions
- Non-critical parameters
- Formatting issues

#### 3. Apply Corrections
```
Correction Interface
┌─────────────────────────────────────────────────────────────┐
│ Item: Load L008 - Compressor Motor                          │
│                                                             │
│ Issue: Power rating 25kW seems high for listed application │
│                                                             │
│ Current Values:                                              │
│ Power: [25.0________] kW                                     │
│ Voltage: [400_______] V                                      │
│ Load Type: [Motor ▼]                                        │
│                                                             │
│ Suggested Values:                                            │
│ • Typical compressor motor: 15-22kW                          │
│ • Verify actual motor nameplate rating                       │
│                                                             │
│ Your Correction:                                             │
│ Power: [18.5________] kW (user input)                       │
│                                                             │
│ Confidence: ██████████░░ 90%                                 │
│                                   [✅ Apply] [❌ Cancel]     │
└─────────────────────────────────────────────────────────────┘
```

#### 4. Validate Corrections
After making corrections, the system re-validates:

```
Validation Results
✅ Power balance maintained
✅ Cable ratings still adequate
✅ Bus loading within limits
✅ Standards compliance verified

Overall Quality: 97% (Improved from 94%)
```

### Batch Correction Tools

#### Multi-Select Corrections
For multiple similar issues:

```
Batch Operations
Select multiple loads with similar issues:
├─ ☐ L004: Power factor correction needed
├─ ☐ L007: Power factor correction needed  
├─ ☐ L012: Power factor correction needed
└─ ☐ L015: Power factor correction needed

Action: [Apply Pattern Correction]
Pattern: "Set motor power factor to 0.85"
Apply to: 4 selected loads
Confirm: [✅ Yes] [❌ No]
```

#### Find and Replace
For systematic corrections:

```
Find & Replace
Find: "Power Factor" < 0.75
Replace: "Set to 0.85 (typical motor value)"
Scope: All motor loads
Preview: 6 changes to be made
Execute: [Apply Changes]
```

## 📋 Quality Assurance Checklist

### Pre-Approval Checklist

Before accepting the extraction results:

#### Data Completeness
- [ ] **All expected loads extracted**
- [ ] **All expected cables identified**
- [ ] **Buses/boards properly configured**
- [ ] **No missing critical parameters**

#### Electrical Consistency
- [ ] **Power balance maintained**
- [ ] **Cable ratings adequate for loads**
- [ ] **Bus loading within capacity**
- [ ] **Voltage levels consistent**

#### Relationship Integrity
- [ ] **Every load has a connecting cable**
- [ ] **Every cable connects existing endpoints**
- [ ] **Bus assignments are logical**
- [ ] **No orphaned components**

#### Standards Compliance
- [ ] **IEC/IS/NEC requirements met**
- [ ] **Safety margins maintained**
- [ ] **Installation methods appropriate**
- [ ] **Protection coordination adequate**

#### Quality Metrics
- [ ] **Overall confidence >80%**
- [ ] **Data quality score >90%**
- [ ] **No critical validation errors**
- [ ] **Acceptable number of corrections made**

### Final Approval Process

#### Summary Review
```
Final Review Summary
┌─────────────────────────────────────────────────────────────┐
│ Total Components: 31 (Loads: 18, Cables: 10, Buses: 3)      │
│ Overall Confidence: 94% (Excellent)                         │
│ Data Quality Score: 96% (Very High)                         │
│ Critical Issues: 0                                           │
│ Warnings: 2 (Informational)                                 │
│ Corrections Applied: 8 (All reviewed and approved)          │
│                                                             │
│ ✅ Ready for project integration                             │
│                                                             │
│ [✅ Approve and Create Project] [🔧 Review Again]            │
└─────────────────────────────────────────────────────────────┘
```

#### Approval Actions
- **Approve and Create Project**: Accept results and generate project
- **Review Again**: Return to detailed review
- **Re-run Extraction**: Start over with different settings
- **Export for Manual Review**: Save current state for external review

## 🎯 Best Practices for Data Review

### Efficient Review Process

#### 1. Start with Summary
- Check overall confidence and quality scores
- Review the list of corrections made
- Identify any critical issues flagged

#### 2. Focus on High-Impact Items
- Review low-confidence extractions first
- Check electrical safety-related parameters
- Validate critical relationships

#### 3. Use Batch Operations
- Apply similar corrections to multiple items
- Use find-and-replace for systematic issues
- Leverage AI suggestions for common problems

#### 4. Maintain Documentation
- Record rationale for major corrections
- Note patterns that need attention in source files
- Document lessons learned for future extractions

### Common Correction Patterns

#### Power Factor Corrections
**Issue**: Motor loads with non-standard power factors
**Correction**: Set to 0.85 (typical motor value)
**Pattern**: Most motor loads should be 0.80-0.90

#### Cable Sizing Issues
**Issue**: Cables sized for full load current
**Correction**: Apply diversity factors where appropriate
**Pattern**: Consider load diversity in cable sizing

#### Naming Convention Standardization
**Issue**: Inconsistent component naming
**Correction**: Apply standardized naming format
**Pattern**: Use consistent prefix (L001, C001, B001)

#### Voltage Level Validation
**Issue**: Unusual voltage levels (380V, 420V)
**Correction**: Map to standard voltages (400V, 415V)
**Pattern**: Use standard electrical system voltages

### Quality Improvement Tips

#### Improve Source Data Quality
- Use consistent column headers
- Include all necessary parameters
- Avoid merged cells in data areas
- Use standard electrical terminology

#### Optimize AI Settings
- Adjust confidence threshold based on data quality
- Enable auto-corrections for common issues
- Select appropriate electrical standards
- Use batch processing for multiple files

#### Validate Results Thoroughly
- Check electrical relationships
- Verify parameter reasonableness
- Confirm standards compliance
- Document any deviations

---

**Next**: Learn about export options and how to use extracted data in Export Options guide.