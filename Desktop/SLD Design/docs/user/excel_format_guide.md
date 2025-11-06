# Excel File Format Guide for AI Extraction

## Overview

This guide provides detailed specifications for structuring Excel files to achieve optimal AI extraction results. Following these best practices will maximize extraction accuracy and confidence scores.

## 📊 Supported Excel Structures

The AI Excel extraction system supports multiple Excel file formats and structures commonly used in electrical distribution projects.

### File Format Compatibility

| Format | Extension | Support Level | Notes |
|--------|-----------|---------------|-------|
| **Excel 2007+** | .xlsx | ✅ Full Support | Recommended format |
| **Excel 97-2003** | .xls | ✅ Full Support | Legacy format |
| **Excel with Macros** | .xlsm | ⚠️ Partial | Macros ignored |
| **Password Protected** | .xlsx | ❌ Not Supported | Remove password first |
| **Read-Only Files** | .xlsx | ⚠️ Limited | May have access restrictions |

### Sheet Structure Requirements

#### ✅ Supported Sheet Types
- **Load Schedules**: Electrical equipment and power requirements
- **Cable Schedules**: Wiring specifications and connections
- **Bus/Board Schedules**: Distribution panels and switchboards
- **Transformer Schedules**: Power transformer specifications
- **Project Information**: General project metadata

#### ✅ Sheet Organization
- **Multiple sheets** in one file supported
- **Sheet names** can be descriptive (e.g., "Load Schedule", "Cable List")
- **Sheet order** doesn't matter for AI processing
- **Empty sheets** are automatically ignored

## 🏗️ Load Schedule Format Specifications

### Essential Column Structure

The AI can extract load data from various column arrangements. Here are the optimal formats:

#### Standard Load Schedule

| Column | Required | AI Recognition Patterns | Example Values |
|--------|----------|------------------------|----------------|
| **Load ID** | ✅ Yes | `load id`, `load_id`, `equipment id`, `tag` | L001, MTR-01, PUMP-001 |
| **Load Name** | ✅ Yes | `load name`, `description`, `equipment name`, `load` | "Main Pump Motor", "HVAC Unit 1" |
| **Power (kW)** | ✅ Yes | `power (kw)`, `power`, `rating kw`, `capacity` | 5.5, 15.0, 45 |
| **Voltage (V)** | ✅ Yes | `voltage (v)`, `voltage`, `system voltage`, `v` | 230, 400, 415 |
| **Phases** | 🔄 Optional | `phases`, `phase`, `no of phases` | 1, 3, "3-Phase" |
| **Load Type** | 🔄 Optional | `load type`, `type`, `category`, `equipment type` | Motor, HVAC, Lighting |
| **Power Factor** | 🔄 Optional | `power factor`, `pf`, `cos phi` | 0.85, 0.9, 0.8 |
| **Efficiency** | 🔄 Optional | `efficiency`, `eff`, `motor efficiency` | 0.88, 0.9, 0.85 |
| **Source Bus** | 🔄 Optional | `source bus`, `bus`, `panel`, `distribution board` | DB-01, B001, Main Panel |
| **Priority** | 🔄 Optional | `priority`, `importance`, `criticality` | Essential, Normal, Critical |
| **Cable Length** | 🔄 Optional | `cable length`, `length`, `distance` | 25, 50, 100 |
| **Installation Method** | 🔄 Optional | `installation`, `method`, `cable installation` | Tray, Conduit, Buried |

#### Extended Load Schedule

For more comprehensive projects, include these additional columns:

| Column | Purpose | AI Recognition | Example Values |
|--------|---------|----------------|----------------|
| **Starting Method** | Motor starting type | `starting method`, `start type`, `motor start` | DOL, Star-Delta, VFD |
| **Duty Cycle** | Operating pattern | `duty cycle`, `operating mode`, `usage` | Continuous, Intermittent |
| **Diversity Factor** | Load diversity | `diversity factor`, `demand factor` | 1.0, 0.8, 0.7 |
| **Ambient Temperature** | Environmental condition | `ambient temp`, `temperature`, `amb` | 35, 40, 45 |
| **Protection Rating** | IP rating | `protection`, `ip rating`, `enclosure` | IP54, IP65, IP44 |
| **Manufacturer** | Equipment make | `manufacturer`, `make`, `brand` | ABB, Siemens, Schneider |

### Data Quality Requirements

#### ✅ High-Quality Data Characteristics
- **Consistent units**: All power in kW, all voltages in V
- **Standard values**: Common voltage levels (230V, 400V, 415V)
- **Proper formatting**: No merged cells in data areas
- **Clear headers**: Descriptive column names in row 1
- **Complete rows**: Minimal empty cells in critical columns

#### ❌ Problematic Data Patterns
- **Mixed units**: Power in kW and HP in same column
- **Non-standard voltages**: Unusual values like 380V
- **Merged cells**: Headers or data spanning multiple cells
- **Special characters**: Unusual symbols in column names
- **Inconsistent naming**: Random abbreviations or codes

### Load Type Classification

The AI automatically classifies loads based on name patterns:

#### Motor Loads
**Detection Patterns**: motor, drive, pump, compressor, fan, conveyor
**Expected Values**:
- Power: 0.75kW - 500kW
- Voltage: 230V, 400V, 415V
- Phases: Usually 3-phase
- Power Factor: 0.8 - 0.9
- Efficiency: 0.85 - 0.95

#### HVAC Loads  
**Detection Patterns**: hvac, air conditioning, ventilation, chiller, ahu
**Expected Values**:
- Power: 5kW - 100kW
- Voltage: 230V, 400V, 415V
- Phases: 1-phase or 3-phase
- Power Factor: 0.85 - 0.9

#### Lighting Loads
**Detection Patterns**: lighting, light, led, lamp, luminaire
**Expected Values**:
- Power: 0.1kW - 10kW
- Voltage: 230V, 400V
- Phases: Usually 1-phase
- Power Factor: 0.9 - 1.0

#### Heating Loads
**Detection Patterns**: heater, heating, resistance, furnace
**Expected Values**:
- Power: 1kW - 50kW
- Voltage: 230V, 400V
- Power Factor: 1.0 (resistive)

## 🔌 Cable Schedule Format Specifications

### Standard Cable Schedule

| Column | Required | AI Recognition Patterns | Example Values |
|--------|----------|------------------------|----------------|
| **Cable ID** | ✅ Yes | `cable id`, `cable_id`, `cable ref`, `tag` | C001, CB-01, FEEDER-01 |
| **From Equipment** | ✅ Yes | `from`, `source`, `origin`, `from equipment` | DB-01, Main Panel, B001 |
| **To Equipment** | ✅ Yes | `to`, `destination`, `load`, `to equipment` | L001, PUMP-01, Motor-1 |
| **Cores** | ✅ Yes | `cores`, `core`, `no of cores`, `conductor count` | 2, 3, 4 |
| **Size (mm²)** | ✅ Yes | `size (mm²)`, `size`, `cross section`, `mm²` | 1.5, 2.5, 4.0, 6.0 |
| **Cable Type** | 🔄 Optional | `cable type`, `type`, `specification` | XLPE/PVC, PVC/PVC |
| **Insulation** | 🔄 Optional | `insulation`, `insulation type` | PVC, XLPE, EPR |
| **Length (m)** | 🔄 Optional | `length (m)`, `length`, `run length` | 25, 50, 100, 250 |
| **Installation Method** | 🔄 Optional | `installation`, `method`, `installation method` | Tray, Conduit, Buried |
| **Armored** | 🔄 Optional | `armored`, `armoured`, `swa` | Yes/No, True/False, Y/N |
| **Voltage Drop (%)** | 🔄 Optional | `voltage drop`, `voltage drop %`, `vd %` | 1.2, 2.5, 3.8 |

### Cable Type Specifications

#### Core Configuration Standards
- **2-Core**: Single-phase loads (phase + neutral)
- **3-Core**: Three-phase loads (3 phases, no neutral)
- **4-Core**: Three-phase with neutral (3 phases + neutral)
- **Special**: 3+1, 2+1 configurations

#### Size Standards (mm²)
**Standard Sizes**: 1.5, 2.5, 4.0, 6.0, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240, 300, 400

#### Installation Methods
- **Tray**: Cable tray systems
- **Conduit**: Conduit installations
- **Buried**: Direct buried cables
- **Air**: Free air installation
- **Duct**: Underground duct systems

## 🏭 Bus/Board Schedule Format

### Distribution Board Schedule

| Column | Required | AI Recognition | Example Values |
|--------|----------|----------------|----------------|
| **Bus ID** | ✅ Yes | `bus id`, `bus_id`, `board id`, `panel id` | B001, DB-01, Panel-A |
| **Bus Name** | ✅ Yes | `bus name`, `name`, `description`, `board name` | "Main Distribution Board", "Panel A" |
| **Voltage (V)** | ✅ Yes | `voltage (v)`, `voltage`, `rated voltage` | 400, 230, 415 |
| **Phases** | ✅ Yes | `phases`, `phase`, `no of phases` | 3, 1, "3-Phase" |
| **Rated Current (A)** | ✅ Yes | `rated current (a)`, `current`, `ampere rating` | 630, 400, 250 |
| **Short Circuit Rating (kA)** | 🔄 Optional | `short circuit rating`, `sc rating`, `fault level` | 50, 35, 25 |
| **Connected Loads** | 🔄 Optional | `connected loads`, `loads`, `total load` | 15, 22, 35 |

## 🏗️ Multi-Sheet Project Files

For comprehensive projects, organize data across multiple sheets:

### Recommended Sheet Organization

#### Sheet 1: "Load Schedule"
- Primary electrical loads
- Motor specifications
- HVAC equipment
- Lighting systems

#### Sheet 2: "Cable Schedule"  
- Cable specifications
- Connection details
- Installation methods

#### Sheet 3: "Project Info"
- Project details
- Standards used
- Environmental conditions

#### Sheet 4: "Buses/Boards"
- Distribution panels
- Switchboards
- Bus ratings

### Cross-Sheet Relationships

The AI automatically establishes relationships between sheets:

```
Load Sheet → Cable Sheet
    ↓           ↓
Source Bus → Destination Equipment
    ↓           ↓
Bus/Board ← Cable Connections
```

## 📋 Data Validation Patterns

### Unit Recognition

#### Power Units
| Unit | Conversion | AI Recognition |
|------|------------|----------------|
| **kW** | Direct | `kw`, `kilowatt`, `power kw` |
| **W** | ÷1000 to kW | `w`, `watt`, `power w` |
| **HP** | ×0.746 to kW | `hp`, `horsepower`, `motor power` |

#### Voltage Units
| Unit | Conversion | AI Recognition |
|------|------------|----------------|
| **V** | Direct | `v`, `volt`, `voltage` |
| **kV** | ×1000 to V | `kv`, `kilovolt`, `voltage kv` |

#### Current Units
| Unit | Conversion | AI Recognition |
|------|------------|----------------|
| **A** | Direct | `a`, `amp`, `current`, `amperes` |
| **mA** | ÷1000 to A | `ma`, `milliamp`, `current ma` |

### Smart Value Mapping

The AI automatically maps non-standard values to standards:

#### Voltage Mapping
- **220V** → **230V** (standard single-phase)
- **380V** → **400V** (standard three-phase)
- **420V** → **415V** (standard industrial)

#### Power Factor Mapping
- **0.8** → **0.85** (typical motor PF)
- **1** → **0.9** (typical lighting PF)

#### Core Count Mapping
- **3+1** → **4** (3 phases + neutral)
- **2+1** → **3** (2 phases + neutral)

## 🎯 Optimization Strategies

### For Maximum Extraction Accuracy

#### 1. Header Naming Best Practices
✅ **Good Headers**:
- `Load ID`, `Power (kW)`, `Voltage (V)`, `Source Bus`
- `Cable ID`, `From Equipment`, `Size (mm²)`
- `Bus ID`, `Rated Current (A)`, `Short Circuit Rating (kA)`

❌ **Poor Headers**:
- `ID`, `Pwr`, `Volts`, `Bus`
- `CB`, `From`, `Size`
- `B_ID`, `I_rating`, `SC_rating`

#### 2. Data Consistency Guidelines
- **Use consistent casing**: "Load" not "load" and "LOAD"
- **Standardize units**: Choose either "Power (kW)" or "Power kW" throughout
- **Consistent formatting**: Same date format, number format across sheets
- **No special characters**: Avoid symbols like @, #, $ in headers

#### 3. File Structure Recommendations
- **Headers in row 1**: Always place column headers in the first row
- **No merged cells**: Keep data areas free of merged cells
- **Clear sheet names**: Use descriptive names like "Load Schedule" not "Sheet1"
- **Logical organization**: Group related data together

### Advanced Configuration

#### Custom Column Mapping
For non-standard formats, the AI can be configured to recognize custom headers:

```python
# Example custom mappings
custom_load_headers = {
    'load_identifier': 'Load ID',
    'equipment_description': 'Load Name', 
    'power_capacity_kilowatts': 'Power (kW)',
    'operating_voltage_volts': 'Voltage (V)'
}
```

#### Pattern Customization
Advanced users can customize recognition patterns:

```python
# Custom load type patterns
load_type_patterns = {
    'pump': ['pump', 'centrifugal', 'submersible'],
    'compressor': ['compressor', 'air compressor', 'gas compressor'],
    'crane': ['crane', 'hoist', 'crane motor']
}
```

## 🔧 Troubleshooting Format Issues

### Common Problems and Solutions

#### Low Extraction Confidence
**Problem**: Confidence scores below 70%
**Solutions**:
1. Check column headers match expected patterns
2. Ensure data starts in row 1 with headers
3. Remove merged cells and formatting issues
4. Verify electrical terminology is used

#### Missing Components
**Problem**: Fewer components extracted than expected
**Solutions**:
1. Check for merged cells hiding data
2. Verify critical columns are not empty
3. Look for unusual characters in data
4. Check sheet naming and organization

#### Broken Relationships
**Problem**: Loads not connected to cables or buses
**Solutions**:
1. Ensure "Source Bus" columns are populated
2. Check cable "To Equipment" matches load IDs
3. Verify equipment naming consistency
4. Review AI auto-corrections in results

### Format Validation Checklist

Before uploading, verify your Excel file:

- [ ] **Headers in Row 1**: Column names are in the first row
- [ ] **No Merged Cells**: Data areas don't have merged cells
- [ ] **Consistent Units**: All power in kW, all voltages in V
- [ ] **Standard Values**: Common voltage levels and component types
- [ ] **Clear Names**: Descriptive column headers using electrical terms
- [ ] **Complete Data**: Minimal empty cells in critical columns
- [ ] **Sheet Organization**: Logical sheet names and structure

## 📊 Performance Expectations

### Extraction Accuracy by Format Quality

| Format Quality | Expected Confidence | Typical Components | Processing Time |
|----------------|-------------------|-------------------|-----------------|
| **Excellent** | 95-100% | 20-50 per file | < 1 second |
| **Good** | 85-94% | 15-30 per file | 1-3 seconds |
| **Fair** | 70-84% | 10-20 per file | 2-5 seconds |
| **Poor** | 50-69% | 5-15 per file | 3-10 seconds |

### File Size Guidelines

| File Size | Recommended Use | Expected Performance |
|-----------|----------------|---------------------|
| **< 1MB** | Standard projects | Fast processing (<1s) |
| **1-5MB** | Complex projects | Good performance (1-5s) |
| **5-10MB** | Large projects | Acceptable (5-15s) |
| **> 10MB** | Very large projects | Consider splitting |

---

**Next Steps**: Once your Excel files follow these guidelines, proceed to the User Interface Guide to learn how to navigate the AI extraction interface.