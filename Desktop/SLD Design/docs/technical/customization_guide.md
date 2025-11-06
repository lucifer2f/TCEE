# Customization Guide - AI Excel Extraction System

## Overview

This guide provides comprehensive instructions for customizing and extending the AI Excel Extraction System to meet specific organizational requirements and workflows.

## Table of Contents

1. [Configuration System](#configuration-system)
2. [Pattern Customization](#pattern-customization)
3. [Model Extensions](#model-extensions)
4. [UI Customization](#ui-customization)
5. [Validation Rules](#validation-rules)
6. [Export Templates](#export-templates)
7. [Plugin Development](#plugin-development)
8. [Performance Tuning](#performance-tuning)

---

## Configuration System

### Configuration Management
```python
# Centralized configuration system
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import os

@dataclass
class ExtractionConfig:
    """Comprehensive configuration for AI extraction"""
    
    # AI Processing Settings
    confidence_threshold: float = 0.8
    auto_corrections: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Standards Configuration
    default_standard: str = "IEC"
    supported_standards: List[str] = None
    
    # Pattern Recognition
    pattern_sensitivity: float = 0.7
    custom_patterns: Dict[str, List[str]] = None
    
    # Data Quality
    min_data_completeness: float = 0.6
    enable_data_enhancement: bool = True
    
    # File Handling
    max_file_size_mb: int = 50
    allowed_file_types: List[str] = None
    temp_dir: str = "/tmp"
    
    # Export Settings
    default_export_format: str = "excel"
    include_calculations: bool = True
    include_validation: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    audit_enabled: bool = True
    
    def __post_init__(self):
        if self.supported_standards is None:
            self.supported_standards = ["IEC", "IS", "NEC"]
        
        if self.allowed_file_types is None:
            self.allowed_file_types = [".xlsx", ".xls"]
        
        if self.custom_patterns is None:
            self.custom_patterns = {}

class ConfigurableExtractor:
    """Extractor with comprehensive configuration support"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components with configuration"""
        # Initialize AI components with custom settings
        self.sheet_classifier = CustomSheetClassifier(
            sensitivity=self.config.pattern_sensitivity,
            custom_patterns=self.config.custom_patterns
        )
        
        self.column_mapper = CustomColumnMapper(
            confidence_threshold=self.config.confidence_threshold
        )
        
        self.data_enhancer = ConfigurableDataEnhancer(
            enable_enhancement=self.config.enable_data_enhancement
        )
        
        self.validation_engine = ConfigurableValidationEngine(
            min_completeness=self.config.min_data_completeness
        )
    
    @classmethod
    def from_file(cls, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = ExtractionConfig(**config_data)
        return cls(config)
    
    def save_config(self, config_path: str):
        """Save configuration to JSON file"""
        config_data = {
            field.name: getattr(self.config, field.name)
            for field in self.config.__dataclass_fields__.values()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
```

### Configuration Files
```json
{
    "confidence_threshold": 0.85,
    "auto_corrections": true,
    "enable_parallel_processing": true,
    "max_workers": 8,
    "default_standard": "IEC",
    "supported_standards": ["IEC", "IS", "NEC"],
    "pattern_sensitivity": 0.8,
    "custom_patterns": {
        "load_schedule": [
            "custom_equipment_id",
            "custom_power_field"
        ],
        "cable_schedule": [
            "custom_cable_tag",
            "custom_routing_info"
        ]
    },
    "min_data_completeness": 0.7,
    "enable_data_enhancement": true,
    "max_file_size_mb": 100,
    "allowed_file_types": [".xlsx", ".xls", ".xlsm"],
    "temp_dir": "/var/tmp/ai_extraction",
    "default_export_format": "excel",
    "include_calculations": true,
    "include_validation": true,
    "log_level": "DEBUG",
    "log_file": "/var/log/ai_extraction.log",
    "audit_enabled": true
}
```

---

## Pattern Customization

### Custom Sheet Patterns
```python
# Extended pattern recognition for custom sheet types
class CustomSheetClassifier(SheetClassifier):
    """Extensible sheet classifier with custom patterns"""
    
    def __init__(self, sensitivity: float = 0.7, custom_patterns: Dict[str, List[str]] = None):
        super().__init__()
        self.sensitivity = sensitivity
        self.custom_patterns = custom_patterns or {}
        
        # Load organization-specific patterns
        self._load_organization_patterns()
    
    def _load_organization_patterns(self):
        """Load organization-specific patterns"""
        # Add custom load schedule patterns
        if 'custom_load_patterns' in self.custom_patterns:
            self.load_patterns['custom'] = self.custom_patterns['custom_load_patterns']
        
        # Add custom cable schedule patterns
        if 'custom_cable_patterns' in self.custom_patterns:
            self.cable_patterns['custom'] = self.custom_patterns['custom_cable_patterns']
        
        # Add custom equipment patterns
        self.equipment_patterns = {
            'transformer': [
                r'transformer\s*id', r'rating\s*\(\s*kva\s*\)',
                r'primary\s*voltage', r'secondary\s*voltage',
                r'connection\s*type', r'vector\s*group'
            ],
            'panel': [
                r'panel\s*id', r'panel\s*name', r'board\s*type',
                r'number\s*of\s*ways', r'rating\s*\(\s*a\s*\)'
            ],
            'protection': [
                r'protection\s*device', r'breaker\s*rating',
                r'fuse\s*rating', r'relay\s*type', r'setting'
            ]
        }
        
        # Add custom sheet type patterns
        self.custom_sheet_types = {
            'equipment_schedule': {
                'patterns': [
                    r'equipment\s*id', r'equipment\s*name',
                    r'equipment\s*type', r'specification'
                ],
                'model_mapping': 'Equipment'
            },
            'protection_schedule': {
                'patterns': [
                    r'protection\s*device', r'breaker\s*rating',
                    r'setting', r'coordination'
                ],
                'model_mapping': 'Protection'
            },
            'earthing_schedule': {
                'patterns': [
                    r'earthing\s*electrode', r'earth\s*resistance',
                    r'conductor\s*size', r'testing'
                ],
                'model_mapping': 'Earthing'
            }
        }
    
    def add_custom_pattern(self, sheet_type: str, pattern: str, weight: float = 1.0):
        """Add a custom pattern for sheet type recognition"""
        if sheet_type not in self.custom_sheet_types:
            self.custom_sheet_types[sheet_type] = {
                'patterns': [],
                'model_mapping': None
            }
        
        self.custom_sheet_types[sheet_type]['patterns'].append(pattern)
    
    def classify_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Enhanced sheet classification with custom patterns"""
        # Start with base classification
        result = super().classify_sheet(df, sheet_name)
        
        # Check custom sheet types
        headers_text = ' '.join(str(h).lower() for h in df.columns.tolist())
        
        for sheet_type, type_info in self.custom_sheet_types.items():
            score = 0.0
            matches = []
            
            for pattern in type_info['patterns']:
                if re.search(pattern, headers_text, re.IGNORECASE):
                    score += self.sensitivity
                    matches.append(pattern)
            
            # Update result if custom type scores higher
            if score > result.get('confidence', 0.0):
                result.update({
                    'sheet_type': sheet_type,
                    'confidence': min(score, 1.0),
                    'evidence': matches,
                    'recommended_model_mapping': type_info['model_mapping']
                })
        
        return result

# Example custom patterns for different industries
def get_industry_patterns(industry: str) -> Dict[str, List[str]]:
    """Get industry-specific patterns"""
    patterns = {
        'manufacturing': {
            'load_schedule': [
                r'machine\s*id', r'cnc\s*machine', r'production\s*line',
                r'workstation', r'assembly\s*line', r'conveyor'
            ],
            'equipment_schedule': [
                r'machine\s*type', r'production\s*capacity',
                r'operational\s*hours', r'maintenance\s*schedule'
            ]
        },
        'hospital': {
            'load_schedule': [
                r'medical\s*equipment', r'patient\s*area',
                r'operating\s*room', r'icu', r'radiology'
            ],
            'critical_loads': [
                r'life\s*support', r'emergency\s*power',
                r'ups\s*load', r'generator\s*backup'
            ]
        },
        'data_center': {
            'load_schedule': [
                r'server\s*rack', r'it\s*equipment', r'cooling\s*unit',
                r'ups\s*system', r'rack\s*id'
            ],
            'power_distribution': [
                r'pdu\s*rating', r'rack\s*power', r'redundancy\s*level',
                r'power\s*chain'
            ]
        }
    }
    
    return patterns.get(industry, {})

# Usage
industry_patterns = get_industry_patterns('manufacturing')
custom_classifier = CustomSheetClassifier(
    sensitivity=0.8,
    custom_patterns=industry_patterns
)
```

### Custom Column Mappings
```python
# Extended column mapping with organization-specific terminology
class CustomColumnMapper(ColumnMapper):
    """Extensible column mapper with custom field definitions"""
    
    def __init__(self, confidence_threshold: float = 0.6, custom_fields: Dict = None):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.custom_fields = custom_fields or {}
        
        # Load custom field mappings
        self._load_custom_mappings()
    
    def _load_custom_mappings(self):
        """Load custom field mappings from configuration"""
        # Add custom load fields
        if 'load_fields' in self.custom_fields:
            for field_name, patterns in self.custom_fields['load_fields'].items():
                if 'Load' not in self.field_mappings:
                    self.field_mappings['Load'] = {}
                self.field_mappings['Load'][field_name] = patterns
        
        # Add custom cable fields
        if 'cable_fields' in self.custom_fields:
            for field_name, patterns in self.custom_fields['cable_fields'].items():
                if 'Cable' not in self.field_mappings:
                    self.field_mappings['Cable'] = {}
                self.field_mappings['Cable'][field_name] = patterns
        
        # Add custom bus fields
        if 'bus_fields' in self.custom_fields:
            for field_name, patterns in self.custom_fields['bus_fields'].items():
                if 'Bus' not in self.field_mappings:
                    self.field_mappings['Bus'] = {}
                self.field_mappings['Bus'][field_name] = patterns
    
    def add_custom_field(self, model_type: str, field_name: str, patterns: List[str]):
        """Add a custom field mapping"""
        if model_type not in self.field_mappings:
            self.field_mappings[model_type] = {}
        
        self.field_mappings[model_type][field_name] = patterns
    
    def add_smart_mapping(self, model_type: str, column: str, field_name: str):
        """Add smart mapping based on user preference"""
        # Learn from user corrections
        if not hasattr(self, 'learned_mappings'):
            self.learned_mappings = {}
        
        if model_type not in self.learned_mappings:
            self.learned_mappings[model_type] = {}
        
        self.learned_mappings[model_type][column] = field_name
    
    def map_columns(self, columns: List[str], model_type: str, sheet_context: str = "") -> Dict[str, Any]:
        """Enhanced column mapping with learning capabilities"""
        # Start with base mapping
        result = super().map_columns(columns, model_type, sheet_context)
        
        # Apply learned mappings
        if hasattr(self, 'learned_mappings') and model_type in self.learned_mappings:
            learned = self.learned_mappings[model_type]
            
            for column in columns:
                if column in learned:
                    field_name = learned[column]
                    
                    if field_name in result['field_mappings']:
                        # Update existing mapping with learned preference
                        result['field_mappings'][field_name]['mapped_columns'].append(column)
                        result['field_mappings'][field_name]['confidence'] = 1.0
                    else:
                        # Add new learned mapping
                        result['field_mappings'][field_name] = {
                            'mapped_columns': [column],
                            'confidence': 1.0,
                            'data_type': 'str',
                            'pattern_match': 'learned'
                        }
        
        return result

# Example custom field definitions
custom_column_fields = {
    'load_fields': {
        'equipment_category': [
            'equipment category', 'equipment type', 'machine category',
            'production category', 'functional area'
        ],
        'operational_hours': [
            'operational hours', 'working hours', 'duty hours',
            'running hours per day', 'shift pattern'
        ],
        'maintenance_frequency': [
            'maintenance frequency', 'service interval',
            'maintenance schedule', 'pm schedule'
        ],
        'criticality_level': [
            'criticality level', 'importance rating',
            'business impact', 'downtime impact'
        ]
    },
    'cable_fields': {
        'cable_function': [
            'cable function', 'cable purpose', 'signal type',
            'communication cable', 'power cable'
        ],
        'installation_risk': [
            'installation risk', 'environmental hazard',
            'special handling', 'installation complexity'
        ],
        'maintenance_access': [
            'maintenance access', 'accessibility',
            'maintenance requirements', 'testing frequency'
        ]
    },
    'bus_fields': {
        'bus_function': [
            'bus function', 'distribution function',
            'switchboard type', 'panel function'
        ],
        'redundancy_level': [
            'redundancy level', 'n+1', 'double bus',
            'backup configuration', 'availability'
        ]
    }
}

# Usage
custom_mapper = CustomColumnMapper(
    confidence_threshold=0.7,
    custom_fields=custom_column_fields
)
```

---

## Model Extensions

### Custom Data Models
```python
# Extended data models for specific requirements
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

class EquipmentCategory(Enum):
    PRODUCTION = "production"
    UTILITY = "utility"
    SAFETY = "safety"
    HVAC = "hvac"
    LIGHTING = "lighting"
    OFFICE = "office"

class CriticalityLevel(Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    STANDARD = "standard"
    NON_ESSENTIAL = "non_essential"

@dataclass
class ExtendedLoad:
    """Extended Load model with additional fields"""
    
    # Standard fields
    load_id: str
    load_name: str
    power_kw: float
    voltage: float
    phases: int
    
    # Extended fields
    equipment_category: EquipmentCategory = EquipmentCategory.STANDARD
    operational_hours: float = 24.0
    maintenance_frequency_days: int = 90
    criticality_level: CriticalityLevel = CriticalityLevel.STANDARD
    startup_power_factor: float = 0.6
    running_power_factor: float = 0.85
    harmonic_content: float = 0.0
    motor_efficiency_class: str = "IE2"
    installation_environment: str = "indoor"
    special_requirements: List[str] = None
    
    def __post_init__(self):
        if self.special_requirements is None:
            self.special_requirements = []

@dataclass
class ExtendedCable:
    """Extended Cable model with additional fields"""
    
    # Standard fields
    cable_id: str
    from_equipment: str
    to_equipment: str
    size_sqmm: float
    cores: int
    
    # Extended fields
    cable_function: str = "power"
    insulation_class: str = "PVC"
    fire_resistance: bool = False
    halogen_free: bool = False
    uv_resistant: bool = False
    chemical_resistance: str = "standard"
    installation_risk_level: str = "low"
    maintenance_access: str = "good"
    testing_frequency: int = 12  # months
    replacement_life_years: int = 25
    special_markings: List[str] = None
    
    def __post_init__(self):
        if self.special_markings is None:
            self.special_markings = []

@dataclass
class ExtendedBus:
    """Extended Bus model with additional fields"""
    
    # Standard fields
    bus_id: str
    bus_name: str
    voltage: float
    phases: int
    rated_current_a: float
    
    # Extended fields
    bus_function: str = "distribution"
    redundancy_level: int = 1
    switchgear_type: str = "metal_clad"
    protection_class: str = "IP54"
    busbar_material: str = "copper"
    cooling_method: str = "natural"
    monitoring_system: bool = True
    remote_control: bool = False
    fault_indication: bool = True
    maintenance_access: str = "front"

class ExtendedDataExtractor(DataExtractor):
    """Extended data extractor for custom models"""
    
    def extract_extended_loads(self, df: pd.DataFrame, field_mapping: Dict) -> Tuple[List[ExtendedLoad], ExtractionResult]:
        """Extract extended load objects"""
        extracted_loads = []
        issues = []
        
        for index, row in df.iterrows():
            try:
                load = self._create_extended_load_from_row(row, field_mapping)
                if load:
                    extracted_loads.append(load)
            except Exception as e:
                issues.append(f"Row {index + 1}: Failed to create extended load - {str(e)}")
        
        confidence = self._calculate_extraction_confidence(extracted_loads, len(df))
        quality_score = self._assess_extended_load_data_quality(extracted_loads)
        
        result = ExtractionResult(
            success=True,
            confidence=confidence,
            sheet_type='extended_load_schedule',
            components_extracted=len(extracted_loads),
            data_quality_score=quality_score,
            issues=issues,
            extracted_data={'extended_loads': [self._extended_load_to_dict(load) for load in extracted_loads]}
        )
        
        return extracted_loads, result
    
    def _create_extended_load_from_row(self, row: pd.Series, field_mapping: Dict) -> Optional[ExtendedLoad]:
        """Create extended load from data row"""
        try:
            load_data = {}
            for field_name, mapping_info in field_mapping.get('field_mappings', {}).items():
                columns = mapping_info.get('mapped_columns', [])
                if columns:
                    column_name = columns[0]
                    if column_name in row.index:
                        load_data[field_name] = row[column_name]
            
            # Create extended load with all fields
            extended_load = ExtendedLoad(
                load_id=self._extract_load_id(load_data),
                load_name=self._extract_load_name(load_data),
                power_kw=self._extract_power_kw(load_data),
                voltage=self._extract_voltage(load_data),
                phases=self._extract_phases(load_data),
                equipment_category=self._extract_equipment_category(load_data),
                operational_hours=self._extract_operational_hours(load_data),
                maintenance_frequency_days=self._extract_maintenance_frequency(load_data),
                criticality_level=self._extract_criticality_level(load_data),
                startup_power_factor=self._extract_startup_power_factor(load_data),
                running_power_factor=self._extract_running_power_factor(load_data),
                harmonic_content=self._extract_harmonic_content(load_data),
                motor_efficiency_class=self._extract_efficiency_class(load_data),
                installation_environment=self._extract_environment(load_data),
                special_requirements=self._extract_special_requirements(load_data)
            )
            
            return extended_load
            
        except Exception as e:
            logger.error(f"Error creating extended load: {e}")
            return None
    
    def _extract_equipment_category(self, data: Dict) -> EquipmentCategory:
        """Extract equipment category"""
        category_str = str(data.get('equipment_category', 'standard')).lower()
        
        mapping = {
            'production': EquipmentCategory.PRODUCTION,
            'utility': EquipmentCategory.UTILITY,
            'safety': EquipmentCategory.SAFETY,
            'hvac': EquipmentCategory.HVAC,
            'lighting': EquipmentCategory.LIGHTING,
            'office': EquipmentCategory.OFFICE
        }
        
        for key, value in mapping.items():
            if key in category_str:
                return value
        
        return EquipmentCategory.STANDARD
    
    def _extract_criticality_level(self, data: Dict) -> CriticalityLevel:
        """Extract criticality level"""
        criticality_str = str(data.get('criticality_level', 'standard')).lower()
        
        mapping = {
            'critical': CriticalityLevel.CRITICAL,
            'important': CriticalityLevel.IMPORTANT,
            'standard': CriticalityLevel.STANDARD,
            'non-essential': CriticalityLevel.NON_ESSENTIAL
        }
        
        for key, value in mapping.items():
            if key in criticality_str:
                return value
        
        return CriticalityLevel.STANDARD

# Usage
extended_extractor = ExtendedDataExtractor()
extended_loads, result = extended_extractor.extract_extended_loads(df, field_mapping)
```

### Custom Validation Rules
```python
# Organization-specific validation rules
class CustomValidationEngine(ValidationEngine):
    """Extended validation engine with custom rules"""
    
    def __init__(self, standard: str = "IEC", custom_rules: Dict = None):
        super().__init__(standard)
        self.custom_rules = custom_rules or {}
        self.organization_standards = self._load_organization_standards()
    
    def _load_organization_standards(self) -> Dict:
        """Load organization-specific standards"""
        return {
            'max_motor_power_kw': 500,
            'max_panel_current_a': 630,
            'required_power_factor_range': (0.8, 0.95),
            'mandatory_load_categories': ['safety', 'emergency'],
            'prohibited_cable_types': ['aluminum'],  # For copper-only policy
            'min_cable_size_sqmm': 1.5,
            'max_cable_length_m': 500,
            'required_redundancy_level': 1
        }
    
    def validate_with_custom_rules(self, project: Project) -> Dict[str, Any]:
        """Validate project with organization-specific rules"""
        # Start with standard validation
        base_results = self.validate_project(project)
        
        # Apply custom validation rules
        custom_results = self._apply_custom_rules(project)
        
        # Combine results
        combined_results = {
            **base_results,
            **custom_results,
            'is_valid': base_results['is_valid'] and custom_results.get('is_valid', True),
            'organization_compliance': self._check_organization_compliance(project)
        }
        
        return combined_results
    
    def _apply_custom_rules(self, project: Project) -> Dict[str, Any]:
        """Apply organization-specific validation rules"""
        errors = []
        warnings = []
        recommendations = []
        
        # Check motor power limits
        for load in project.loads:
            if load.load_type == LoadType.MOTOR:
                if load.power_kw > self.organization_standards['max_motor_power_kw']:
                    errors.append(f"Motor {load.load_id}: Power {load.power_kw}kW exceeds organizational limit of {self.organization_standards['max_motor_power_kw']}kW")
        
        # Check panel current limits
        for bus in project.buses:
            if bus.rated_current_a > self.organization_standards['max_panel_current_a']:
                warnings.append(f"Panel {bus.bus_id}: Current rating {bus.rated_current_a}A exceeds organizational limit of {self.organization_standards['max_panel_current_a']}A")
        
        # Check power factor requirements
        for load in project.loads:
            pf = load.power_factor
            min_pf, max_pf = self.organization_standards['required_power_factor_range']
            if not (min_pf <= pf <= max_pf):
                warnings.append(f"Load {load.load_id}: Power factor {pf} outside organizational range {min_pf}-{max_pf}")
        
        # Check mandatory load categories
        load_categories = set(load.load_type.value for load in project.loads)
        mandatory = set(self.organization_standards['mandatory_load_categories'])
        missing_categories = mandatory - load_categories
        
        if missing_categories:
            recommendations.append(f"Missing mandatory load categories: {', '.join(missing_categories)}")
        
        # Check cable restrictions
        for cable in project.cables:
            if 'aluminum' in cable.cable_type.lower():
                errors.append(f"Cable {cable.cable_id}: Aluminum cables not permitted by organizational policy")
            
            if cable.size_sqmm < self.organization_standards['min_cable_size_sqmm']:
                errors.append(f"Cable {cable.cable_id}: Size {cable.size_sqmm}mm² below minimum organizational standard")
            
            if cable.length_m > self.organization_standards['max_cable_length_m']:
                warnings.append(f"Cable {cable.cable_id}: Length {cable.length_m}m exceeds organizational maximum")
        
        # Check redundancy requirements
        for bus in project.buses:
            if hasattr(bus, 'redundancy_level') and bus.redundancy_level < self.organization_standards['required_redundancy_level']:
                recommendations.append(f"Bus {bus.bus_id}: Redundancy level {bus.redundancy_level} below organizational requirement")
        
        return {
            'errors': base_results.get('errors', []) + errors,
            'warnings': base_results.get('warnings', []) + warnings,
            'recommendations': base_results.get('recommendations', []) + recommendations,
            'is_valid': len(errors) == 0
        }
    
    def _check_organization_compliance(self, project: Project) -> Dict[str, Any]:
        """Check overall organization compliance"""
        compliance_score = 100.0
        
        # Calculate compliance score based on various factors
        
        # Load category compliance
        required_categories = set(self.organization_standards['mandatory_load_categories'])
        actual_categories = set(load.load_type.value for load in project.loads)
        missing_categories = required_categories - actual_categories
        
        if missing_categories:
            compliance_score -= len(missing_categories) * 10
        
        # Safety compliance
        safety_loads = [load for load in project.loads if load.priority == Priority.CRITICAL]
        if len(safety_loads) == 0:
            compliance_score -= 20
        
        # Cable standards compliance
        non_compliant_cables = [
            cable for cable in project.cables
            if 'aluminum' in cable.cable_type.lower() or 
               cable.size_sqmm < self.organization_standards['min_cable_size_sqmm']
        ]
        
        compliance_score -= len(non_compliant_cables) * 5
        
        return {
            'compliance_score': max(compliance_score, 0.0),
            'compliance_level': (
                'Excellent' if compliance_score >= 90 else
                'Good' if compliance_score >= 80 else
                'Fair' if compliance_score >= 70 else
                'Poor'
            ),
            'missing_requirements': missing_categories,
            'non_compliant_items': len(non_compliant_cables)
        }
```

---

## UI Customization

### Streamlit Customization
```python
# Custom Streamlit interface
import streamlit as st
from typing import Dict, Any, List

class CustomAIExtractionApp:
    """Customizable Streamlit application"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.setup_page_config()
        self.setup_custom_css()
    
    def setup_page_config(self):
        """Custom page configuration"""
        st.set_page_config(
            page_title="Custom AI Excel Extraction",
            page_icon="🏢",  # Custom icon
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://example.com/help',
                'Report a bug': 'https://example.com/bug',
                'About': 'Custom AI Excel Extraction System v2.0'
            }
        )
    
    def setup_custom_css(self):
        """Custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #17becf);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        
        .confidence-high { color: #28a745; }
        .confidence-medium { color: #ffc107; }
        .confidence-low { color: #dc3545; }
        
        .custom-sidebar {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        .extraction-progress {
            background-color: #e9ecef;
            border-radius: 0.5rem;
            padding: 0.5rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_custom_header(self):
        """Render custom application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏢 Custom AI Excel Extraction System</h1>
            <p>Intelligent Electrical Project Data Processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_custom_sidebar(self):
        """Render customizable sidebar"""
        with st.sidebar:
            st.markdown('<div class="custom-sidebar">', unsafe_allow_html=True)
            
            st.header("⚙️ Configuration")
            
            # Organization settings
            organization_name = st.text_input("Organization Name", value="My Company")
            project_default_path = st.text_input("Default Project Path", value="/projects")
            
            # Custom standards
            st.subheader("📋 Standards")
            custom_standards = st.multiselect(
                "Active Standards",
                self.config.supported_standards,
                default=[self.config.default_standard]
            )
            
            # Processing settings
            st.subheader("⚡ Processing")
            custom_confidence = st.slider(
                "Custom Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=self.config.confidence_threshold,
                step=0.05
            )
            
            auto_enhance = st.checkbox(
                "Auto-Enhancement",
                value=self.config.auto_corrections
            )
            
            # Export preferences
            st.subheader("📤 Export")
            export_format = st.selectbox(
                "Default Export Format",
                ["Excel", "JSON", "CSV", "PDF"],
                index=["Excel", "JSON", "CSV", "PDF"].index(self.config.default_export_format.title())
            )
            
            include_calculations = st.checkbox(
                "Include Calculations",
                value=self.config.include_calculations
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Update config
            self.config.default_standard = custom_standards[0] if custom_standards else "IEC"
            self.config.confidence_threshold = custom_confidence
            self.config.auto_corrections = auto_enhance
            self.config.default_export_format = export_format.lower()
            self.config.include_calculations = include_calculations
            
            return {
                'organization_name': organization_name,
                'project_path': project_default_path,
                'custom_standards': custom_standards
            }
    
    def render_extraction_progress(self, progress_data: Dict[str, Any]):
        """Render custom extraction progress"""
        st.markdown("### 🔄 Processing Progress")
        
        # Overall progress
        if 'overall_progress' in progress_data:
            st.progress(progress_data['overall_progress'])
        
        # Step-by-step progress
        steps = [
            "📁 Analyzing file structure",
            "🔍 Identifying patterns",
            "📊 Mapping columns",
            "⚡ Extracting data",
            "🔧 Enhancing quality",
            "✅ Validating results"
        ]
        
        for i, step in enumerate(steps):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if i < progress_data.get('current_step', 0):
                    st.success("✅")
                elif i == progress_data.get('current_step', 0):
                    st.info("🔄")
                else:
                    st.text("⏳")
            
            with col2:
                status_text = "Completed" if i < progress_data.get('current_step', 0) else \
                             "In Progress" if i == progress_data.get('current_step', 0) else "Pending"
                st.text(f"{step} - {status_text}")
        
        # Real-time metrics
        if 'metrics' in progress_data:
            metrics = progress_data['metrics']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Processing Time", f"{metrics.get('elapsed_time', 0):.1f}s")
            with col2:
                st.metric("Sheets Processed", f"{metrics.get('sheets_processed', 0)}")
            with col3:
                st.metric("Components Found", f"{metrics.get('components_found', 0)}")
    
    def render_confidence_visualization(self, report: ProcessingReport):
        """Render custom confidence visualization"""
        if not report.sheet_results:
            return
        
        st.subheader("📈 Confidence Analysis")
        
        # Confidence by sheet
        confidence_data = []
        for sheet_name, result in report.sheet_results.items():
            confidence_data.append({
                'Sheet': sheet_name,
                'Confidence': result.confidence * 100,
                'Components': result.components_extracted,
                'Quality Score': result.data_quality_score * 100
            })
        
        if confidence_data:
            df_confidence = pd.DataFrame(confidence_data)
            
            # Confidence chart
            fig = px.bar(
                df_confidence,
                x='Sheet',
                y='Confidence',
                color='Confidence',
                color_continuous_scale='RdYlGn',
                title="Sheet Processing Confidence"
            )
            fig.add_hline(y=self.config.confidence_threshold * 100, 
                         line_dash="dash", line_color="red",
                         annotation_text=f"Threshold ({self.config.confidence_threshold * 100:.0f}%)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            with st.expander("📋 Detailed Confidence Metrics"):
                st.dataframe(df_confidence, use_container_width=True)
    
    def render_custom_dashboard(self, report: ProcessingReport):
        """Render custom organizational dashboard"""
        st.subheader("🏢 Organization Dashboard")
        
        # Custom KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Processing Efficiency",
                f"{(report.overall_confidence * 100):.1f}%",
                delta="High" if report.overall_confidence > 0.8 else "Medium"
            )
        
        with col2:
            st.metric(
                "Data Quality Score",
                f"{(sum(r.data_quality_score for r in report.sheet_results.values()) / len(report.sheet_results) * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Auto-Corrections",
                len(report.corrections_made),
                delta=f"+{len(report.corrections_made)} improvements"
            )
        
        with col4:
            processing_rate = report.total_components / max(report.processing_time_seconds, 1)
            st.metric(
                "Processing Rate",
                f"{processing_rate:.1f} comp/s",
                delta="Fast" if processing_rate > 10 else "Standard"
            )
        
        # Organization-specific insights
        if report.project_data:
            st.subheader("📊 Project Insights")
            
            # Load distribution by category
            if hasattr(report.project_data, 'loads') and report.project_data.loads:
                load_categories = {}
                for load in report.project_data.loads:
                    if hasattr(load, 'equipment_category'):
                        category = load.equipment_category.value
                        load_categories[category] = load_categories.get(category, 0) + 1
                
                if load_categories:
                    fig = px.pie(
                        values=list(load_categories.values()),
                        names=list(load_categories.keys()),
                        title="Load Distribution by Equipment Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Criticality analysis
            if hasattr(report.project_data, 'loads'):
                critical_loads = [load for load in report.project_data.loads 
                                if hasattr(load, 'criticality_level') and 
                                load.criticality_level.value == 'critical']
                
                if critical_loads:
                    st.warning(f"⚠️  {len(critical_loads)} critical loads require special attention")
                    for load in critical_loads:
                        st.text(f"- {load.load_id}: {load.load_name} ({load.power_kw}kW)")

# Custom application initialization
def create_custom_app():
    """Create and run custom AI extraction application"""
    
    # Load configuration
    config = ExtractionConfig.from_file("custom_config.json")
    
    # Create and run app
    app = CustomAIExtractionApp(config)
    
    # Main application logic
    app.render_custom_header()
    
    # Sidebar configuration
    sidebar_config = app.render_custom_sidebar()
    
    # Main content
    st.markdown("---")
    
    # File upload section
    st.header("📁 File Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload electrical project Excel file for AI processing"
        )
    
    with col2:
        if uploaded_file:
            st.info(f"File: {uploaded_file.name}")
            st.info(f"Size: {uploaded_file.size / 1024:.1f} KB")
    
    # Processing section
    if uploaded_file:
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Start AI Processing", type="primary"):
            # Save uploaded file
            with open("temp_upload.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize extractor with custom config
            extractor = AIExcelExtractor(standard=config.default_standard)
            
            # Show progress
            progress_placeholder = st.empty()
            
            with st.spinner("🤖 Processing with AI..."):
                # Simulate progress updates
                for i in range(6):
                    progress_data = {
                        'current_step': i,
                        'overall_progress': (i + 1) / 6,
                        'metrics': {
                            'elapsed_time': i * 2,
                            'sheets_processed': min(i, 3),
                            'components_found': i * 5
                        }
                    }
                    
                    with progress_placeholder.container():
                        app.render_extraction_progress(progress_data)
                    
                    time.sleep(1)
            
            # Process file
            report = extractor.process_excel_file("temp_upload.xlsx")
            
            # Display results
            st.markdown("---")
            app.render_custom_dashboard(report)
            app.render_confidence_visualization(report)
            
            # Cleanup
            os.unlink("temp_upload.xlsx")

# Run the application
if __name__ == "__main__":
    create_custom_app()
```

This customization guide provides comprehensive examples for extending and adapting the AI Excel Extraction System to meet specific organizational requirements, industry standards, and workflow preferences.