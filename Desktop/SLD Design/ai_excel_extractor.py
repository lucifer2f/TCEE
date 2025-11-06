"""
AI-Powered Excel Extraction System for Electrical Distribution Projects

This module implements intelligent Excel data extraction for electrical engineering projects,
featuring domain-specific AI components for pattern recognition, data mapping, quality enhancement,
and validation.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import math
import json
from datetime import datetime

# Import existing models and utilities
from models import Load, Cable, Breaker, Bus, Transformer, Project, LoadType, InstallationMethod, DutyCycle, Priority
from calculations import ElectricalCalculationEngine
from standards import StandardsFactory


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Container for extraction results with confidence scoring"""
    success: bool
    confidence: float
    sheet_type: str
    components_extracted: int
    data_quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extracted_data: Optional[Dict] = None


@dataclass
class ProcessingReport:
    """Comprehensive processing report for the entire Excel file"""
    overall_confidence: float
    total_components: int
    processing_time_seconds: float
    sheet_results: Dict[str, ExtractionResult]
    project_data: Optional[Project] = None
    corrections_made: List[Dict] = field(default_factory=list)
    validation_issues: List[str] = field(default_factory=list)


class SheetClassifier:
    """
    Identifies sheet types (Load, Cable, Bus, etc.) using pattern matching
    based on electrical engineering domain knowledge
    """

    def __init__(self):
        # Define patterns for different sheet types
        self.load_patterns = {
            'primary': [
                r'load\s*id', r'power\s*\(\s*kw\s*\)', r'voltage\s*\(\s*v\s*\)',
                r'load\s*name', r'load\s*type', r'phases', r'current\s*\(\s*a\s*\)'
            ],
            'secondary': [
                r'power\s*factor', r'efficiency', r'design\s*current',
                r'cable\s*size', r'breaker\s*rating', r'voltage\s*drop',
                r'source\s*bus', r'priority', r'starting\s*method'
            ]
        }

        self.cable_patterns = {
            'primary': [
                r'cable\s*id', r'from\s*equipment', r'to\s*equipment',
                r'specification', r'cores', r'size\s*\(\s*mm²\s*\)'
            ],
            'secondary': [
                r'length\s*\(\s*m\s*\)', r'installation', r'current\s*rating',
                r'voltage\s*drop\s*\(\s*v\s*\)', r'voltage\s*drop\s*%'
            ]
        }

        self.bus_patterns = {
            'primary': [
                r'bus\s*id', r'bus\s*name', r'voltage\s*\(\s*v\s*\)',
                r'rated\s*current', r'short\s*circuit\s*rating'
            ],
            'secondary': [
                r'phases', r'frequency', r'parent\s*bus', r'connected\s*loads'
            ]
        }

        self.transformer_patterns = {
            'primary': [
                r'transformer\s*id', r'rating\s*\(\s*kva\s*\)',
                r'primary\s*voltage', r'secondary\s*voltage'
            ],
            'secondary': [
                r'impedance\s*%', r'vector\s*group', r'cooling', r'windings'
            ]
        }

        self.sheet_type_weights = {
            'load_schedule': {'primary': 3, 'secondary': 1},
            'cable_schedule': {'primary': 3, 'secondary': 1},
            'bus_schedule': {'primary': 3, 'secondary': 1},
            'transformer_schedule': {'primary': 3, 'secondary': 1},
            'project_info': {'primary': 2, 'secondary': 1},
            'unknown': {'primary': 0, 'secondary': 0}
        }

    def classify_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        Classify sheet based on content patterns
        
        Args:
            df: DataFrame with sheet data
            sheet_name: Name of the sheet
            
        Returns:
            Dictionary with classification results
        """
        if df.empty:
            return {
                'sheet_type': 'unknown',
                'confidence': 0.0,
                'evidence': [],
                'recommended_model_mapping': None
            }

        # Get all column headers as string
        headers = df.columns.tolist()
        headers_text = ' '.join(str(h).lower() for h in headers)

        # Calculate pattern match scores
        scores = {}
        evidence = {}

        # Check load schedule patterns
        load_score = self._calculate_pattern_score(headers_text, self.load_patterns, 'load_schedule')
        scores['load_schedule'] = load_score['score']
        evidence['load_schedule'] = load_score['matches']

        # Check cable schedule patterns
        cable_score = self._calculate_pattern_score(headers_text, self.cable_patterns, 'cable_schedule')
        scores['cable_schedule'] = cable_score['score']
        evidence['cable_schedule'] = cable_score['matches']

        # Check bus schedule patterns
        bus_score = self._calculate_pattern_score(headers_text, self.bus_patterns, 'bus_schedule')
        scores['bus_schedule'] = bus_score['score']
        evidence['bus_schedule'] = bus_score['matches']

        # Check transformer schedule patterns
        transformer_score = self._calculate_pattern_score(headers_text, self.transformer_patterns, 'transformer_schedule')
        scores['transformer_schedule'] = transformer_score['score']
        evidence['transformer_schedule'] = transformer_score['matches']

        # Check for project info patterns
        project_score = self._calculate_project_info_score(headers_text, sheet_name)
        scores['project_info'] = project_score['score']
        evidence['project_info'] = project_score['matches']

        # Determine best match
        if max(scores.values()) == 0:
            best_type = 'unknown'
            confidence = 0.0
        else:
            best_type = max(scores, key=scores.get)
            max_score = scores[best_type]
            
            # Normalize confidence based on max possible score
            weights = self.sheet_type_weights.get(best_type, {'primary': 3, 'secondary': 1})
            max_possible = weights['primary'] * 3 + weights['secondary'] * 2
            confidence = min(max_score / max_possible, 1.0) if max_possible > 0 else 0.0

        # Map to model type
        model_mapping = self._get_model_mapping(best_type)

        return {
            'sheet_type': best_type,
            'confidence': confidence,
            'evidence': evidence.get(best_type, []),
            'recommended_model_mapping': model_mapping,
            'all_scores': scores
        }

    def _calculate_pattern_score(self, headers_text: str, patterns: Dict, sheet_type: str) -> Dict:
        """Calculate pattern matching score for a sheet type"""
        matches = []
        score = 0.0

        for pattern_category, pattern_list in patterns.items():
            category_score = 0.0
            category_matches = []

            for pattern in pattern_list:
                if re.search(pattern, headers_text, re.IGNORECASE):
                    weight = self.sheet_type_weights.get(sheet_type, {}).get(pattern_category, 1)
                    category_score += weight
                    category_matches.append(pattern)

            score += category_score
            matches.extend([(m, pattern_category) for m in category_matches])

        return {
            'score': score,
            'matches': matches
        }

    def _calculate_project_info_score(self, headers_text: str, sheet_name: str) -> Dict:
        """Calculate score for project information sheets"""
        project_patterns = [
            r'project\s*name', r'standard', r'voltage\s*system',
            r'ambient\s*temperature', r'project\s*id', r'created'
        ]

        name_bonus = 0.0
        if re.search(r'project|info|summary', sheet_name.lower()):
            name_bonus = 1.0

        matches = []
        score = name_bonus

        for pattern in project_patterns:
            if re.search(pattern, headers_text, re.IGNORECASE):
                score += 1.0
                matches.append(pattern)

        return {
            'score': score,
            'matches': matches
        }

    def _get_model_mapping(self, sheet_type: str) -> Optional[str]:
        """Map sheet type to data model"""
        mapping = {
            'load_schedule': 'Load',
            'cable_schedule': 'Cable',
            'bus_schedule': 'Bus',
            'transformer_schedule': 'Transformer',
            'project_info': 'Project'
        }
        return mapping.get(sheet_type)


class ColumnMapper:
    """
    Intelligent column header mapping with fuzzy string matching
    """

    def __init__(self):
        # Define target fields for each model type
        self.field_mappings = {
            'Load': {
                'load_id': [
                    'load id', 'load_id', 'id', 'load', 'equipment id',
                    'asset id', 'tag', 'load number', 'ld id', 'load ref'
                ],
                'load_name': [
                    'load name', 'load_name', 'name', 'description',
                    'equipment name', 'load description', 'name of load',
                    'equipment description', 'load desc', 'load'
                ],
                'power_kw': [
                    'power (kw)', 'power', 'kw', 'power kw', 'rating kw',
                    'rated power', 'load power', 'capacity kw', 'power rating',
                    'motor power', 'load capacity'
                ],
                'voltage': [
                    'voltage (v)', 'voltage', 'v', 'volts', 'operating voltage',
                    'system voltage', 'rated voltage', 'supply voltage'
                ],
                'phases': [
                    'phases', 'phase', 'no of phases', 'number of phases',
                    'phase connection', '3 phase', '1 phase', 'phase count'
                ],
                'load_type': [
                    'load type', 'type', 'equipment type', 'category',
                    'load category', 'equipment category', 'load nature'
                ],
                'power_factor': [
                    'power factor', 'pf', 'cos phi', 'cosφ', 'power factor (pf)',
                    'power fact', 'power fact.'
                ],
                'efficiency': [
                    'efficiency', 'eff', 'efficiency %', 'motor efficiency',
                    'power efficiency', 'η', 'efficiency ratio'
                ],
                'source_bus': [
                    'source bus', 'bus', 'supply point', 'source',
                    'distribution board', 'panel', 'bus bar', 'source panel'
                ],
                'priority': [
                    'priority', 'importance', 'criticality', 'load priority',
                    'essential', 'critical', 'non-essential', 'priority level'
                ],
                'cable_length': [
                    'cable length', 'length', 'distance', 'run length',
                    'cable run', 'length (m)', 'distance (m)'
                ],
                'installation_method': [
                    'installation', 'method', 'cable installation',
                    'installation method', 'routing', 'cable method'
                ]
            },
            'Cable': {
                'cable_id': [
                    'cable id', 'cable_id', 'cable', 'id', 'cable ref',
                    'cable number', 'asset id', 'tag'
                ],
                'from_equipment': [
                    'from', 'from equipment', 'source', 'origin',
                    'from equipment', 'cable from', 'source equipment'
                ],
                'to_equipment': [
                    'to', 'to equipment', 'destination', 'load',
                    'to equipment', 'cable to', 'target equipment'
                ],
                'cores': [
                    'cores', 'core', 'number of cores', 'no of cores',
                    'core count', 'cable cores', 'conductor count'
                ],
                'size_sqmm': [
                    'size (mm²)', 'size', 'mm2', 'cross section',
                    'cable size', 'conductor size', 'area', 'mm²'
                ],
                'cable_type': [
                    'cable type', 'type', 'specification', 'cable spec',
                    'cable category', 'insulation type', 'cable construction'
                ],
                'insulation': [
                    'insulation', 'insulation type', 'insul',
                    'insulating material', 'insulation material'
                ],
                'length_m': [
                    'length (m)', 'length', 'm', 'cable length',
                    'run length', 'distance', 'cable distance'
                ],
                'installation_method': [
                    'installation', 'method', 'installation method',
                    'routing', 'cable routing', 'installation type'
                ],
                'armored': [
                    'armored', 'armour', 'armoured', 'swa', 'armour type',
                    'armor', 'armoured cable', 'steel wire armour'
                ]
            },
            'Bus': {
                'bus_id': [
                    'bus id', 'bus_id', 'bus', 'id', 'bus ref',
                    'bus number', 'panel id', 'distribution board id'
                ],
                'bus_name': [
                    'bus name', 'bus_name', 'name', 'description',
                    'panel name', 'board name', 'bus description'
                ],
                'voltage': [
                    'voltage (v)', 'voltage', 'v', 'rated voltage',
                    'system voltage', 'bus voltage'
                ],
                'phases': [
                    'phases', 'phase', 'no of phases', 'number of phases'
                ],
                'rated_current_a': [
                    'rated current (a)', 'current', 'rated current',
                    'ampere rating', 'current rating'
                ],
                'short_circuit_rating_ka': [
                    'short circuit rating', 'sc rating', 'fault level',
                    'short circuit (ka)', 'fault rating'
                ]
            }
        }

        # Data type inference patterns
        self.data_type_patterns = {
            'int': [r'^\d+$', r'phase', r'cores', r'poles'],
            'float': [r'power', r'voltage', r'current', r'length', r'size', r'efficiency'],
            'str': [r'id', r'name', r'type', r'description', r'priority'],
            'bool': [r'armored', r'armoured', r'shielded', r'redundancy']
        }

    def map_columns(self, columns: List[str], model_type: str, sheet_context: str = "") -> Dict[str, Any]:
        """
        Map Excel columns to model fields with confidence scores
        
        Args:
            columns: List of Excel column headers
            model_type: Target model type ('Load', 'Cable', 'Bus', etc.)
            sheet_context: Additional context about the sheet
            
        Returns:
            Dictionary mapping target fields to column mappings
        """
        if model_type not in self.field_mappings:
            return {}

        target_fields = self.field_mappings[model_type]
        mapping_result = {}
        unmapped_columns = list(columns)

        # Fuzzy match each target field to columns
        for field_name, field_patterns in target_fields.items():
            best_match, confidence = self._find_best_column_match(
                field_patterns, unmapped_columns
            )
            
            if best_match and confidence > 0.6:  # Minimum confidence threshold
                mapping_result[field_name] = {
                    'mapped_columns': [best_match],
                    'confidence': confidence,
                    'data_type': self._infer_data_type(field_name, columns),
                    'pattern_match': self._get_match_pattern(field_patterns, best_match)
                }
                
                # Remove matched column from unmapped list
                unmapped_columns = [col for col in unmapped_columns if col != best_match]

        # Map remaining columns with lower confidence
        for column in unmapped_columns:
            field_name = self._find_best_field_match(column, target_fields)
            if field_name and self._calculate_match_confidence(column, field_name) > 0.3:
                if field_name not in mapping_result:
                    mapping_result[field_name] = {
                        'mapped_columns': [],
                        'confidence': 0.0,
                        'data_type': self._infer_data_type(field_name, columns),
                        'pattern_match': None
                    }
                
                mapping_result[field_name]['mapped_columns'].append(column)
                current_conf = mapping_result[field_name]['confidence']
                new_conf = self._calculate_match_confidence(column, field_name)
                mapping_result[field_name]['confidence'] = max(current_conf, new_conf)

        # Calculate overall mapping confidence
        total_confidence = sum(m['confidence'] for m in mapping_result.values())
        field_count = len(target_fields)
        overall_confidence = total_confidence / field_count if field_count > 0 else 0.0

        return {
            'field_mappings': mapping_result,
            'overall_confidence': overall_confidence,
            'unmapped_columns': unmapped_columns,
            'mapping_quality': self._assess_mapping_quality(mapping_result, target_fields)
        }

    def _find_best_column_match(self, field_patterns: List[str], columns: List[str]) -> Tuple[Optional[str], float]:
        """Find best column match for a field using fuzzy matching"""
        best_match = None
        best_confidence = 0.0

        for column in columns:
            for pattern in field_patterns:
                # Use fuzzy string matching
                confidence = fuzz.partial_ratio(pattern.lower(), column.lower()) / 100.0
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = column

        return best_match, best_confidence

    def _find_best_field_match(self, column: str, target_fields: Dict[str, List[str]]) -> Optional[str]:
        """Find best target field for an unmapped column"""
        best_field = None
        best_confidence = 0.0

        for field_name, field_patterns in target_fields.items():
            confidence = self._calculate_match_confidence(column, field_name)
            if confidence > best_confidence:
                best_confidence = confidence
                best_field = field_name

        return best_field

    def _calculate_match_confidence(self, column: str, field_name: str) -> float:
        """Calculate confidence for column-field match"""
        if field_name not in self.field_mappings:
            return 0.0

        # Direct name similarity
        direct_confidence = fuzz.ratio(column.lower(), field_name.lower()) / 100.0

        # Pattern matching
        pattern_confidence = 0.0
        patterns = self.field_mappings[field_name]
        for pattern in patterns:
            pattern_confidence = max(pattern_confidence, 
                                   fuzz.partial_ratio(pattern.lower(), column.lower()) / 100.0)

        # Combine both approaches
        return max(direct_confidence, pattern_confidence)

    def _infer_data_type(self, field_name: str, all_columns: List[str]) -> str:
        """Infer appropriate data type for field based on name and patterns"""
        # Use data type patterns to infer type
        for data_type, patterns in self.data_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, field_name, re.IGNORECASE):
                    return data_type

        # Default to string
        return 'str'

    def _get_match_pattern(self, patterns: List[str], matched_column: str) -> str:
        """Get the pattern that matched the column"""
        best_pattern = ""
        best_confidence = 0.0

        for pattern in patterns:
            confidence = fuzz.partial_ratio(pattern.lower(), matched_column.lower()) / 100.0
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = pattern

        return best_pattern

    def _assess_mapping_quality(self, mapping_result: Dict, target_fields: Dict) -> str:
        """Assess overall mapping quality"""
        mapped_fields = len(mapping_result)
        total_fields = len(target_fields)
        coverage_ratio = mapped_fields / total_fields if total_fields > 0 else 0.0

        if coverage_ratio >= 0.8:
            return 'excellent'
        elif coverage_ratio >= 0.6:
            return 'good'
        elif coverage_ratio >= 0.4:
            return 'fair'
        else:
            return 'poor'


class DataExtractor:
    """
    Extract and validate data using existing Load, Cable, etc. models
    """

    def __init__(self):
        self.calculation_engine = ElectricalCalculationEngine()

        # Load type mapping patterns
        self.load_type_mappings = {
            'motor': ['motor', 'drive', 'pump', 'compressor', 'fan', 'conveyor'],
            'hvac': ['hvac', 'air conditioning', 'ac', 'ventilation', 'chiller', 'ahu'],
            'lighting': ['lighting', 'light', 'led', 'lamp', 'luminaire', 'illumination'],
            'heater': ['heater', 'heating', 'heater', 'resistance', 'furnace'],
            'ups': ['ups', 'uninterruptible', 'battery', 'backup power'],
            'general': ['general', 'misc', 'other', 'auxiliary', 'miscellaneous']
        }

        # Installation method mappings
        self.installation_mappings = {
            'conduit': ['conduit', 'pipe', 'tubing', 'raceway'],
            'tray': ['tray', 'cable tray', 'ladder', 'channel'],
            'buried': ['buried', 'direct buried', 'underground', 'duct bank'],
            'air': ['air', 'free air', 'exposed'],
            'duct': ['duct', 'underground duct', 'subway']
        }

        # Priority mappings
        self.priority_mappings = {
            'critical': ['critical', 'essential', 'safety', 'emergency'],
            'essential': ['essential', 'important', 'priority'],
            'non-essential': ['non-essential', 'normal', 'general', 'standard']
        }

    def extract_loads(self, df: pd.DataFrame, field_mapping: Dict) -> Tuple[List[Load], ExtractionResult]:
        """Extract Load objects from DataFrame"""
        extracted_loads = []
        issues = []
        warnings = []

        try:
            for index, row in df.iterrows():
                try:
                    load = self._create_load_from_row(row, field_mapping)
                    if load:
                        extracted_loads.append(load)
                except Exception as e:
                    issues.append(f"Row {index + 1}: Failed to create load - {str(e)}")
                    logger.warning(f"Failed to create load from row {index + 1}: {e}")

            # Calculate confidence and quality score
            confidence = self._calculate_extraction_confidence(extracted_loads, len(df))
            quality_score = self._assess_load_data_quality(extracted_loads)

            result = ExtractionResult(
                success=True,
                confidence=confidence,
                sheet_type='load_schedule',
                components_extracted=len(extracted_loads),
                data_quality_score=quality_score,
                issues=issues,
                warnings=warnings,
                extracted_data={'loads': [self._load_to_dict(load) for load in extracted_loads]}
            )

        except Exception as e:
            result = ExtractionResult(
                success=False,
                confidence=0.0,
                sheet_type='load_schedule',
                components_extracted=0,
                data_quality_score=0.0,
                issues=[f"Critical extraction failure: {str(e)}"]
            )

        return extracted_loads, result

    def extract_cables(self, df: pd.DataFrame, field_mapping: Dict) -> Tuple[List[Cable], ExtractionResult]:
        """Extract Cable objects from DataFrame"""
        extracted_cables = []
        issues = []
        warnings = []

        try:
            for index, row in df.iterrows():
                try:
                    cable = self._create_cable_from_row(row, field_mapping)
                    if cable:
                        extracted_cables.append(cable)
                except Exception as e:
                    issues.append(f"Row {index + 1}: Failed to create cable - {str(e)}")
                    logger.warning(f"Failed to create cable from row {index + 1}: {e}")

            confidence = self._calculate_extraction_confidence(extracted_cables, len(df))
            quality_score = self._assess_cable_data_quality(extracted_cables)

            result = ExtractionResult(
                success=True,
                confidence=confidence,
                sheet_type='cable_schedule',
                components_extracted=len(extracted_cables),
                data_quality_score=quality_score,
                issues=issues,
                warnings=warnings,
                extracted_data={'cables': [self._cable_to_dict(cable) for cable in extracted_cables]}
            )

        except Exception as e:
            result = ExtractionResult(
                success=False,
                confidence=0.0,
                sheet_type='cable_schedule',
                components_extracted=0,
                data_quality_score=0.0,
                issues=[f"Critical extraction failure: {str(e)}"]
            )

        return extracted_cables, result

    def _create_load_from_row(self, row: pd.Series, field_mapping: Dict) -> Optional[Load]:
        """Create a Load object from a data row"""
        try:
            # Map fields using the provided mapping
            load_data = {}
            for field_name, mapping_info in field_mapping.get('field_mappings', {}).items():
                columns = mapping_info.get('mapped_columns', [])
                if columns:
                    # Take the first mapped column
                    column_name = columns[0]
                    if column_name in row.index:
                        load_data[field_name] = row[column_name]

            # Extract and validate required fields
            load_id = self._extract_load_id(load_data)
            load_name = self._extract_load_name(load_data)
            power_kw = self._extract_power_kw(load_data)
            voltage = self._extract_voltage(load_data)

            if not all([load_id, load_name, power_kw, voltage]):
                return None

            # Create Load object with extracted and defaulted values
            load = Load(
                load_id=load_id,
                load_name=load_name,
                power_kw=power_kw,
                voltage=voltage,
                phases=self._extract_phases(load_data),
                load_type=self._extract_load_type(load_data),
                power_factor=self._extract_power_factor(load_data),
                efficiency=self._extract_efficiency(load_data),
                duty_cycle=self._extract_duty_cycle(load_data),
                cable_length=self._extract_cable_length(load_data),
                installation_method=self._extract_installation_method(load_data),
                source_bus=self._extract_source_bus(load_data),
                priority=self._extract_priority(load_data)
            )

            # Calculate electrical parameters
            load = self.calculation_engine.calculate_load(load)

            return load

        except Exception as e:
            logger.error(f"Error creating load from row data: {e}")
            return None

    def _create_cable_from_row(self, row: pd.Series, field_mapping: Dict) -> Optional[Cable]:
        """Create a Cable object from a data row"""
        try:
            cable_data = {}
            for field_name, mapping_info in field_mapping.get('field_mappings', {}).items():
                columns = mapping_info.get('mapped_columns', [])
                if columns:
                    column_name = columns[0]
                    if column_name in row.index:
                        cable_data[field_name] = row[column_name]

            # Extract and validate required fields
            cable_id = self._extract_cable_id(cable_data)
            from_equipment = self._extract_from_equipment(cable_data)
            to_equipment = self._extract_to_equipment(cable_data)

            if not all([cable_id, from_equipment, to_equipment]):
                return None

            # Create Cable object
            cable = Cable(
                cable_id=cable_id,
                from_equipment=from_equipment,
                to_equipment=to_equipment,
                cores=self._extract_cores(cable_data),
                size_sqmm=self._extract_size_sqmm(cable_data),
                cable_type=self._extract_cable_type(cable_data),
                insulation=self._extract_insulation(cable_data),
                length_m=self._extract_length_m(cable_data),
                installation_method=self._extract_installation_method(cable_data),
                armored=self._extract_armored(cable_data)
            )

            return cable

        except Exception as e:
            logger.error(f"Error creating cable from row data: {e}")
            return None

    def _extract_load_id(self, data: Dict) -> Optional[str]:
        """Extract load ID with smart generation if missing"""
        load_id = data.get('load_id')
        if load_id:
            return str(load_id).strip()
        
        # Generate ID based on other data if possible
        load_name = data.get('load_name', '')
        if load_name:
            # Extract number or create from name
            numbers = re.findall(r'\d+', str(load_name))
            if numbers:
                return f"L{numbers[0].zfill(3)}"
            else:
                # Create from first letters
                words = str(load_name).split()[:2]
                return ''.join(w[0].upper() for w in words) + "001"
        
        return None

    def _extract_load_name(self, data: Dict) -> str:
        """Extract load name"""
        return str(data.get('load_name', 'Unknown Load')).strip()

    def _extract_power_kw(self, data: Dict) -> float:
        """Extract power in kW"""
        power_str = str(data.get('power_kw', '0')).replace(',', '').strip()
        try:
            return float(power_str)
        except ValueError:
            return 0.0

    def _extract_voltage(self, data: Dict) -> float:
        """Extract voltage with standard value mapping"""
        voltage_str = str(data.get('voltage', '400')).replace(',', '').strip()
        try:
            voltage = float(voltage_str)
            # Map to standard voltages
            if voltage < 300:
                return 230
            elif voltage < 500:
                return 400
            elif voltage < 1000:
                return 415
            else:
                return voltage
        except ValueError:
            return 400  # Default to 400V

    def _extract_phases(self, data: Dict) -> int:
        """Extract number of phases"""
        phases_str = str(data.get('phases', '3')).strip().lower()
        if '1' in phases_str or 'single' in phases_str:
            return 1
        else:
            return 3  # Default to 3-phase

    def _extract_load_type(self, data: Dict) -> LoadType:
        """Extract load type using pattern matching"""
        type_str = str(data.get('load_type', 'general')).lower()
        
        for load_type, patterns in self.load_type_mappings.items():
            for pattern in patterns:
                if pattern in type_str:
                    return LoadType(load_type)
        
        return LoadType.GENERAL

    def _extract_power_factor(self, data: Dict) -> float:
        """Extract power factor"""
        pf_str = str(data.get('power_factor', '0.85')).replace(',', '.').strip()
        try:
            pf = float(pf_str)
            return max(0.1, min(1.0, pf))  # Clamp between 0.1 and 1.0
        except ValueError:
            return 0.85

    def _extract_efficiency(self, data: Dict) -> float:
        """Extract efficiency"""
        eff_str = str(data.get('efficiency', '0.9')).replace(',', '.').strip()
        try:
            eff = float(eff_str)
            return max(0.1, min(1.0, eff))  # Clamp between 0.1 and 1.0
        except ValueError:
            return 0.9

    def _extract_duty_cycle(self, data: Dict) -> DutyCycle:
        """Extract duty cycle"""
        duty_str = str(data.get('duty_cycle', 'continuous')).lower()
        if 'intermittent' in duty_str:
            return DutyCycle.INTERMITTENT
        elif 'short' in duty_str:
            return DutyCycle.SHORT_TIME
        else:
            return DutyCycle.CONTINUOUS

    def _extract_cable_length(self, data: Dict) -> float:
        """Extract cable length in meters"""
        length_str = str(data.get('cable_length', '25')).replace(',', '').strip()
        try:
            return float(length_str)
        except ValueError:
            return 25.0

    def _extract_installation_method(self, data: Dict) -> InstallationMethod:
        """Extract installation method"""
        install_str = str(data.get('installation_method', 'tray')).lower()
        
        for method, patterns in self.installation_mappings.items():
            for pattern in patterns:
                if pattern in install_str:
                    return InstallationMethod(method)
        
        return InstallationMethod.TRAY

    def _extract_source_bus(self, data: Dict) -> Optional[str]:
        """Extract source bus"""
        return str(data.get('source_bus', '')).strip() or None

    def _extract_priority(self, data: Dict) -> Priority:
        """Extract priority"""
        priority_str = str(data.get('priority', 'non-essential')).lower()
        
        for priority, patterns in self.priority_mappings.items():
            for pattern in patterns:
                if pattern in priority_str:
                    return Priority(priority)
        
        return Priority.NON_ESSENTIAL

    def _extract_cable_id(self, data: Dict) -> Optional[str]:
        """Extract cable ID"""
        cable_id = data.get('cable_id')
        if cable_id:
            return str(cable_id).strip()
        return None

    def _extract_from_equipment(self, data: Dict) -> str:
        """Extract from equipment"""
        return str(data.get('from_equipment', 'Unknown Source')).strip()

    def _extract_to_equipment(self, data: Dict) -> str:
        """Extract to equipment"""
        return str(data.get('to_equipment', 'Unknown Destination')).strip()

    def _extract_cores(self, data: Dict) -> int:
        """Extract number of cores"""
        cores_str = str(data.get('cores', '4')).strip()
        try:
            return int(cores_str)
        except ValueError:
            return 4

    def _extract_size_sqmm(self, data: Dict) -> float:
        """Extract cable size in mm²"""
        size_str = str(data.get('size_sqmm', '2.5')).replace(',', '.').strip()
        try:
            return float(size_str)
        except ValueError:
            return 2.5

    def _extract_cable_type(self, data: Dict) -> str:
        """Extract cable type"""
        cable_type = data.get('cable_type')
        if cable_type:
            return str(cable_type).strip()
        
        # Generate based on other properties
        cores = self._extract_cores(data)
        if cores == 3:
            return "XLPE/PVC"  # 3-core
        else:
            return "XLPE/PVC"  # 4-core (3+neutral)

    def _extract_insulation(self, data: Dict) -> str:
        """Extract insulation type"""
        return str(data.get('insulation', 'PVC')).strip()

    def _extract_length_m(self, data: Dict) -> float:
        """Extract cable length in meters"""
        length_str = str(data.get('length_m', '25')).replace(',', '').strip()
        try:
            return float(length_str)
        except ValueError:
            return 25.0

    def _extract_armored(self, data: Dict) -> bool:
        """Extract armored flag"""
        armored_str = str(data.get('armored', 'false')).lower()
        return any(word in armored_str for word in ['true', 'yes', 'y', 'swa', 'armored', 'armoured'])

    def _calculate_extraction_confidence(self, extracted_items: List, total_rows: int) -> float:
        """Calculate extraction confidence based on success rate and data quality"""
        if total_rows == 0:
            return 0.0

        success_rate = len(extracted_items) / total_rows
        
        # Weight the success rate heavily
        base_confidence = success_rate * 0.8
        
        # Add bonus for complete extractions
        if success_rate > 0.9:
            base_confidence += 0.1
        elif success_rate > 0.8:
            base_confidence += 0.05

        return min(base_confidence, 1.0)

    def _assess_load_data_quality(self, loads: List[Load]) -> float:
        """Assess data quality for extracted loads"""
        if not loads:
            return 0.0

        quality_scores = []
        for load in loads:
            score = 1.0
            
            # Check for missing critical data
            if not load.load_id:
                score -= 0.3
            if not load.load_name:
                score -= 0.2
            if load.power_kw <= 0:
                score -= 0.3
            if load.voltage not in [230, 400, 415]:
                score -= 0.1
            
            # Check for reasonable values
            if not 0.1 <= load.power_factor <= 1.0:
                score -= 0.1
            if not 0.1 <= load.efficiency <= 1.0:
                score -= 0.1
            if load.cable_length < 0.1 or load.cable_length > 1000:
                score -= 0.1
                
            quality_scores.append(max(score, 0.0))

        return sum(quality_scores) / len(quality_scores)

    def _assess_cable_data_quality(self, cables: List[Cable]) -> float:
        """Assess data quality for extracted cables"""
        if not cables:
            return 0.0

        quality_scores = []
        for cable in cables:
            score = 1.0
            
            # Check for missing critical data
            if not cable.cable_id:
                score -= 0.3
            if not cable.from_equipment:
                score -= 0.2
            if not cable.to_equipment:
                score -= 0.2
            if cable.cores <= 0:
                score -= 0.2
            if cable.size_sqmm <= 0:
                score -= 0.2
            
            quality_scores.append(max(score, 0.0))

        return sum(quality_scores) / len(quality_scores)

    def _load_to_dict(self, load: Load) -> Dict:
        """Convert Load object to dictionary"""
        return {
            'load_id': load.load_id,
            'load_name': load.load_name,
            'power_kw': load.power_kw,
            'voltage': load.voltage,
            'phases': load.phases,
            'load_type': load.load_type.value,
            'power_factor': load.power_factor,
            'efficiency': load.efficiency,
            'source_bus': load.source_bus,
            'priority': load.priority.value,
            'cable_length': load.cable_length,
            'installation_method': load.installation_method.value,
            'current_a': load.current_a,
            'design_current_a': load.design_current_a,
            'apparent_power_kva': load.apparent_power_kva
        }

    def _cable_to_dict(self, cable: Cable) -> Dict:
        """Convert Cable object to dictionary"""
        return {
            'cable_id': cable.cable_id,
            'from_equipment': cable.from_equipment,
            'to_equipment': cable.to_equipment,
            'cores': cable.cores,
            'size_sqmm': cable.size_sqmm,
            'cable_type': cable.cable_type,
            'insulation': cable.insulation,
            'length_m': cable.length_m,
            'installation_method': cable.installation_method.value,
            'armored': cable.armored
        }


class DataEnhancer:
    """
    Auto-correct common issues (broken IDs, missing relationships)
    """

    def __init__(self):
        self.id_patterns = {
            'load': r'^[Ll]?\d{3}$',
            'cable': r'^[Cc]?\d{3}$',
            'bus': r'^[Bb]?\d{3}$'
        }

    def enhance_project_data(self, project: Project, extraction_results: List[ExtractionResult]) -> Dict[str, Any]:
        """
        Enhance extracted project data by fixing common issues
        
        Args:
            project: Project object to enhance
            extraction_results: List of extraction results
            
        Returns:
            Dictionary with enhancement report
        """
        corrections_made = []
        
        # Fix broken IDs
        id_corrections = self._fix_broken_ids(project)
        corrections_made.extend(id_corrections)
        
        # Establish missing relationships
        relationship_corrections = self._establish_missing_relationships(project)
        corrections_made.extend(relationship_corrections)
        
        # Standardize naming conventions
        naming_corrections = self._standardize_naming_conventions(project)
        corrections_made.extend(naming_corrections)
        
        # Fill missing calculated values
        calculated_corrections = self._fill_missing_calculated_values(project)
        corrections_made.extend(calculated_corrections)
        
        return {
            'corrections_made': corrections_made,
            'correction_count': len(corrections_made),
            'enhancement_success': True,
            'final_project': project
        }

    def _fix_broken_ids(self, project: Project) -> List[Dict]:
        """Fix broken or missing IDs"""
        corrections = []
        counter = {'load': 1, 'cable': 1, 'bus': 1}
        
        # Fix load IDs
        for load in project.loads:
            if not load.load_id or not re.match(self.id_patterns['load'], load.load_id):
                new_id = f"L{counter['load']:03d}"
                corrections.append({
                    'type': 'load_id_fixed',
                    'original': load.load_id,
                    'corrected': new_id,
                    'reason': 'invalid_or_missing_load_id'
                })
                load.load_id = new_id
                counter['load'] += 1
                
                # Update any cable connections that reference the old ID
                for cable in project.cables:
                    if cable.to_equipment == load.load_id:
                        cable.to_equipment = new_id
                        corrections.append({
                            'type': 'cable_connection_updated',
                            'cable_id': cable.cable_id,
                            'old_destination': load.load_id,
                            'new_destination': new_id,
                            'reason': 'load_id_change'
                        })
        
        # Fix cable IDs
        for cable in project.cables:
            if not cable.cable_id or not re.match(self.id_patterns['cable'], cable.cable_id):
                new_id = f"C{counter['cable']:03d}"
                corrections.append({
                    'type': 'cable_id_fixed',
                    'original': cable.cable_id,
                    'corrected': new_id,
                    'reason': 'invalid_or_missing_cable_id'
                })
                cable.cable_id = new_id
                counter['cable'] += 1
        
        # Fix bus IDs
        for bus in project.buses:
            if not bus.bus_id or not re.match(self.id_patterns['bus'], bus.bus_id):
                new_id = f"B{counter['bus']:03d}"
                corrections.append({
                    'type': 'bus_id_fixed',
                    'original': bus.bus_id,
                    'corrected': new_id,
                    'reason': 'invalid_or_missing_bus_id'
                })
                bus.bus_id = new_id
                counter['bus'] += 1
        
        return corrections

    def _establish_missing_relationships(self, project: Project) -> List[Dict]:
        """Establish missing relationships between components"""
        corrections = []
        
        # Create default buses if none exist
        if not project.buses:
            # Create main bus
            main_bus = Bus(
                bus_id="B001",
                bus_name="Main Distribution Bus",
                voltage=400,
                phases=3,
                rated_current_a=630,
                short_circuit_rating_ka=50
            )
            project.buses.append(main_bus)
            corrections.append({
                'type': 'bus_created',
                'bus_id': 'B001',
                'reason': 'missing_bus_system'
            })
        
        # Assign loads to buses if not assigned
        main_bus = project.buses[0] if project.buses else None
        if main_bus:
            for load in project.loads:
                if not load.source_bus:
                    load.source_bus = main_bus.bus_id
                    main_bus.add_load(load.load_id)
                    corrections.append({
                        'type': 'load_bus_assignment',
                        'load_id': load.load_id,
                        'bus_id': main_bus.bus_id,
                        'reason': 'missing_bus_assignment'
                    })
        
        # Create cables for loads that don't have them
        load_ids_with_cables = {cable.to_equipment for cable in project.cables}
        for load in project.loads:
            if load.load_id not in load_ids_with_cables and load.cable_length > 0:
                # Create a basic cable for this load
                cable = Cable(
                    cable_id=f"C{len(project.cables) + 1:03d}",
                    from_equipment=load.source_bus or "B001",
                    to_equipment=load.load_id,
                    cores=4 if load.phases == 3 else 2,
                    size_sqmm=max(2.5, load.cable_size_sqmm or 2.5),
                    cable_type="XLPE/PVC",
                    insulation="PVC",
                    length_m=load.cable_length,
                    installation_method=load.installation_method,
                    armored=load.installation_method in [InstallationMethod.BURIED, InstallationMethod.DUCT]
                )
                project.cables.append(cable)
                corrections.append({
                    'type': 'cable_created',
                    'cable_id': cable.cable_id,
                    'load_id': load.load_id,
                    'reason': 'missing_cable_for_load'
                })
        
        return corrections

    def _standardize_naming_conventions(self, project: Project) -> List[Dict]:
        """Standardize naming conventions across components"""
        corrections = []
        
        # Standardize load names
        for load in project.loads:
            original_name = load.load_name
            standardized_name = self._standardize_load_name(original_name)
            if standardized_name != original_name:
                load.load_name = standardized_name
                corrections.append({
                    'type': 'load_name_standardized',
                    'load_id': load.load_id,
                    'original': original_name,
                    'standardized': standardized_name,
                    'reason': 'naming_convention_standardization'
                })
        
        return corrections

    def _standardize_load_name(self, name: str) -> str:
        """Standardize individual load names"""
        if not name:
            return "Unknown Load"
        
        # Clean up common issues
        name = str(name).strip()
        
        # Title case
        name = ' '.join(word.capitalize() for word in name.split())
        
        # Common replacements
        replacements = {
            'Hvac': 'HVAC',
            'Ups': 'UPS',
            'Led': 'LED',
            'Dc': 'DC',
            'Ac': 'AC',
            'Motor': 'Motor',
            'Pump': 'Pump',
            'Fan': 'Fan'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        return name

    def _fill_missing_calculated_values(self, project: Project) -> List[Dict]:
        """Fill in missing calculated values using electrical engineering rules"""
        corrections = []
        engine = ElectricalCalculationEngine()
        
        # Recalculate loads
        for load in project.loads:
            try:
                original_current = load.current_a
                engine.calculate_load(load)
                
                if load.current_a != original_current:
                    corrections.append({
                        'type': 'load_current_recalculated',
                        'load_id': load.load_id,
                        'original_current': original_current,
                        'corrected_current': load.current_a,
                        'reason': 'electrical_calculation_update'
                    })
            except Exception as e:
                logger.warning(f"Failed to recalculate load {load.load_id}: {e}")
        
        return corrections


class ValidationEngine:
    """
    Cross-check with electrical engineering rules
    """

    def __init__(self, standard: str = "IEC"):
        self.standard = StandardsFactory.get_standard(standard)
        self.calculation_engine = ElectricalCalculationEngine(standard)

    def validate_project(self, project: Project) -> Dict[str, Any]:
        """
        Validate entire project against electrical engineering rules
        
        Args:
            project: Project object to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'electrical_violations': [],
            'quality_score': 0.0
        }
        
        error_count = 0
        warning_count = 0
        
        # Validate individual components
        for load in project.loads:
            load_validation = self._validate_load(load)
            validation_results['errors'].extend(load_validation['errors'])
            validation_results['warnings'].extend(load_validation['warnings'])
            validation_results['electrical_violations'].extend(load_validation['violations'])
            
            error_count += len(load_validation['errors'])
            warning_count += len(load_validation['warnings'])
        
        for cable in project.cables:
            cable_validation = self._validate_cable(cable)
            validation_results['errors'].extend(cable_validation['errors'])
            validation_results['warnings'].extend(cable_validation['warnings'])
            validation_results['electrical_violations'].extend(cable_validation['violations'])
            
            error_count += len(cable_validation['errors'])
            warning_count += len(cable_validation['warnings'])
        
        # Validate system-level consistency
        system_validation = self._validate_system_consistency(project)
        validation_results['errors'].extend(system_validation['errors'])
        validation_results['warnings'].extend(system_validation['warnings'])
        validation_results['recommendations'].extend(system_validation['recommendations'])
        
        error_count += len(system_validation['errors'])
        warning_count += len(system_validation['warnings'])
        
        # Determine overall validity
        validation_results['is_valid'] = error_count == 0
        
        # Calculate quality score
        total_items = len(project.loads) + len(project.cables) + len(project.buses)
        if total_items > 0:
            quality_score = (total_items - error_count - warning_count * 0.5) / total_items
            validation_results['quality_score'] = max(quality_score, 0.0)
        
        return validation_results

    def _validate_load(self, load: Load) -> Dict[str, List[str]]:
        """Validate individual load"""
        result = {
            'errors': [],
            'warnings': [],
            'violations': []
        }
        
        # Basic validation
        if not load.load_id:
            result['errors'].append(f"Load {load.load_name}: Missing load ID")
        
        if load.power_kw <= 0:
            result['errors'].append(f"Load {load.load_id}: Power must be positive")
        
        if load.voltage not in [230, 400, 415, 440, 690, 3300, 6600, 11000, 33000]:
            result['warnings'].append(f"Load {load.load_id}: Non-standard voltage {load.voltage}V")
        
        if not (0.1 <= load.power_factor <= 1.0):
            result['errors'].append(f"Load {load.load_id}: Power factor out of range (0.1-1.0)")
        
        if not (0.1 <= load.efficiency <= 1.0):
            result['errors'].append(f"Load {load.load_id}: Efficiency out of range (0.1-1.0)")
        
        # Electrical validation
        if load.current_a and load.design_current_a:
            if load.breaker_rating_a and load.breaker_rating_a < load.design_current_a:
                result['errors'].append(f"Load {load.load_id}: Breaker rating {load.breaker_rating_a}A too low for design current {load.design_current_a}A")
            
            if load.voltage_drop_percent and load.voltage_drop_percent > 5.0:
                result['warnings'].append(f"Load {load.load_id}: High voltage drop {load.voltage_drop_percent:.1f}%")
        
        # Load type specific validation
        if load.load_type == LoadType.MOTOR:
            if load.power_kw > 500:
                result['warnings'].append(f"Load {load.load_id}: Large motor {load.power_kw}kW - consider special protection")
        
        return result

    def _validate_cable(self, cable: Cable) -> Dict[str, List[str]]:
        """Validate individual cable"""
        result = {
            'errors': [],
            'warnings': [],
            'violations': []
        }
        
        # Basic validation
        if not cable.cable_id:
            result['errors'].append(f"Cable from {cable.from_equipment}: Missing cable ID")
        
        if not cable.from_equipment:
            result['errors'].append(f"Cable {cable.cable_id}: Missing source equipment")
        
        if not cable.to_equipment:
            result['errors'].append(f"Cable {cable.cable_id}: Missing destination equipment")
        
        if cable.cores not in [2, 3, 4]:
            result['warnings'].append(f"Cable {cable.cable_id}: Unusual core count {cable.cores}")
        
        if cable.size_sqmm < 1.5:
            result['warnings'].append(f"Cable {cable.cable_id}: Very small cable size {cable.size_sqmm}mm²")
        
        if cable.size_sqmm > 500:
            result['warnings'].append(f"Cable {cable.cable_id}: Very large cable size {cable.size_sqmm}mm²")
        
        if cable.length_m < 1:
            result['warnings'].append(f"Cable {cable.cable_id}: Very short cable length {cable.length_m}m")
        
        if cable.length_m > 1000:
            result['warnings'].append(f"Cable {cable.cable_id}: Very long cable length {cable.length_m}m")
        
        return result

    def _validate_system_consistency(self, project: Project) -> Dict[str, List[str]]:
        """Validate system-level consistency"""
        result = {
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for duplicate IDs
        load_ids = [l.load_id for l in project.loads]
        if len(load_ids) != len(set(load_ids)):
            result['errors'].append("Duplicate load IDs found")
        
        cable_ids = [c.cable_id for c in project.cables]
        if len(cable_ids) != len(set(cable_ids)):
            result['errors'].append("Duplicate cable IDs found")
        
        # Check bus assignments
        bus_ids = {b.bus_id for b in project.buses}
        for load in project.loads:
            if load.source_bus and load.source_bus not in bus_ids:
                result['warnings'].append(f"Load {load.load_id}: References unknown bus {load.source_bus}")
        
        # Check load balance
        total_power = sum(load.power_kw for load in project.loads)
        if total_power > 1000:  # 1 MW threshold
            result['recommendations'].append(f"Large total load {total_power}kW - consider multiple feeders")
        
        # Check voltage consistency
        voltages = set(load.voltage for load in project.loads)
        if len(voltages) > 2:
            result['recommendations'].append(f"Multiple voltage levels detected: {sorted(voltages)}V")
        
        return result


class AIExcelExtractor:
    """
    Main orchestrator for AI-powered Excel extraction
    """

    def __init__(self, standard: str = "IEC"):
        self.standard = standard
        self.sheet_classifier = SheetClassifier()
        self.column_mapper = ColumnMapper()
        self.data_extractor = DataExtractor()
        self.data_enhancer = DataEnhancer()
        self.validation_engine = ValidationEngine(standard)

    def process_excel_file(self, file_path: str) -> ProcessingReport:
        """
        Process Excel file and extract all electrical components
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            ProcessingReport with comprehensive results
        """
        start_time = datetime.now()
        logger.info(f"Starting Excel file processing: {file_path}")
        
        try:
            # Read Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            logger.info(f"Read {len(excel_data)} sheets from file")
            
            # Process each sheet
            sheet_results = {}
            all_loads = []
            all_cables = []
            all_buses = []
            
            for sheet_name, df in excel_data.items():
                logger.info(f"Processing sheet: {sheet_name}")
                
                # Classify sheet
                classification = self.sheet_classifier.classify_sheet(df, sheet_name)
                logger.info(f"Sheet '{sheet_name}' classified as: {classification['sheet_type']} (confidence: {classification['confidence']:.2f})")
                
                # Map columns if we have a supported type
                if classification['recommended_model_mapping'] in ['Load', 'Cable', 'Bus']:
                    field_mapping = self.column_mapper.map_columns(
                        df.columns.tolist(),
                        classification['recommended_model_mapping'],
                        sheet_name
                    )
                    
                    # Extract data based on sheet type
                    if classification['sheet_type'] == 'load_schedule':
                        loads, result = self.data_extractor.extract_loads(df, field_mapping)
                        all_loads.extend(loads)
                    elif classification['sheet_type'] == 'cable_schedule':
                        cables, result = self.data_extractor.extract_cables(df, field_mapping)
                        all_cables.extend(cables)
                    else:
                        result = ExtractionResult(
                            success=True,
                            confidence=classification['confidence'],
                            sheet_type=classification['sheet_type'],
                            components_extracted=0,
                            data_quality_score=classification['confidence'],
                            extracted_data={}
                        )
                else:
                    result = ExtractionResult(
                        success=True,
                        confidence=classification['confidence'],
                        sheet_type=classification['sheet_type'],
                        components_extracted=0,
                        data_quality_score=classification['confidence'],
                        extracted_data={}
                    )
                
                sheet_results[sheet_name] = result
            
            # Create project from extracted data
            project = self._create_project_from_extracted_data(
                all_loads, all_cables, all_buses, sheet_results
            )
            
            # Enhance project data
            enhancement_results = self.data_enhancer.enhance_project_data(project, list(sheet_results.values()))
            
            # Validate project
            validation_results = self.validation_engine.validate_project(project)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(sheet_results)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create final report
            report = ProcessingReport(
                overall_confidence=overall_confidence,
                total_components=len(all_loads) + len(all_cables) + len(all_buses),
                processing_time_seconds=processing_time,
                sheet_results=sheet_results,
                project_data=project,
                corrections_made=enhancement_results['corrections_made'],
                validation_issues=validation_results['errors'] + validation_results['warnings']
            )
            
            logger.info(f"Processing completed: {report.total_components} components extracted, {overall_confidence:.2f} confidence")
            return report
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return ProcessingReport(
                overall_confidence=0.0,
                total_components=0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                sheet_results={},
                validation_issues=[f"Processing failed: {str(e)}"]
            )

    def _create_project_from_extracted_data(self, loads: List[Load], cables: List[Cable], 
                                           buses: List[Bus], sheet_results: Dict) -> Project:
        """Create Project object from extracted data"""
        project_name = "AI Extracted Project"
        
        # Try to extract project name from sheet results
        for sheet_name, result in sheet_results.items():
            if result.sheet_type == 'project_info' and result.extracted_data:
                project_info = result.extracted_data.get('project_info', {})
                if project_info.get('name'):
                    project_name = project_info['name']
                    break
        
        project = Project(
            project_name=project_name,
            standard=self.standard,
            voltage_system="LV"
        )
        
        # Add components
        project.loads = loads
        project.cables = cables
        project.buses = buses
        
        return project

    def _calculate_overall_confidence(self, sheet_results: Dict[str, ExtractionResult]) -> float:
        """Calculate overall confidence from all sheet results"""
        if not sheet_results:
            return 0.0
        
        confidences = [result.confidence for result in sheet_results.values()]
        return sum(confidences) / len(confidences)


# Example usage and testing functions
def demo_extraction():
    """Demo function showing how to use the AIExcelExtractor"""
    extractor = AIExcelExtractor()
    
    # Process a sample Excel file
    try:
        report = extractor.process_excel_file("sample_electrical_project.xlsx")
        
        print(f"Processing Results:")
        print(f"Overall Confidence: {report.overall_confidence:.2%}")
        print(f"Total Components: {report.total_components}")
        print(f"Processing Time: {report.processing_time_seconds:.2f}s")
        
        print(f"\nSheet Results:")
        for sheet_name, result in report.sheet_results.items():
            print(f"  {sheet_name}: {result.sheet_type} ({result.confidence:.2%} confidence, {result.components_extracted} components)")
        
        if report.project_data:
            print(f"\nProject Summary:")
            print(f"  Loads: {len(report.project_data.loads)}")
            print(f"  Cables: {len(report.project_data.cables)}")
            print(f"  Buses: {len(report.project_data.buses)}")
        
        print(f"\nCorrections Made: {len(report.corrections_made)}")
        for correction in report.corrections_made:
            print(f"  - {correction['type']}: {correction.get('reason', 'N/A')}")
        
        print(f"\nValidation Issues: {len(report.validation_issues)}")
        for issue in report.validation_issues:
            print(f"  - {issue}")
            
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    # Run demo if this file is executed directly
    demo_extraction()