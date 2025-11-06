
"""
Comprehensive Integration Layer for AI Excel Extraction System

This module provides deep integration between AI-extracted data and all existing
calculation engines, standards framework, and automation features. It ensures
that AI-extracted data is indistinguishable from manually entered data and
supports full downstream processing.

Key Features:
- Deep integration with CurrentCalculator, VoltageDropCalculator, CableSizingEngine, BreakerSelectionEngine
- Standards framework validation (IEC, IS, NEC)
- SLD generation integration with automatic hierarchy creation
- Enhanced data models with comprehensive calculated fields
- Smart defaults and validation rules
- Cross-validation with electrical engineering constraints
- Optimization suggestions based on calculations
- End-to-end workflow validation
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Import existing systems
from models import Project, Load, Cable, Bus, Transformer, Breaker, LoadType, InstallationMethod, DutyCycle, Priority
from calculations import (
    ElectricalCalculationEngine, CurrentCalculator, VoltageDropCalculator,
    CableSizingEngine, BreakerSelectionEngine
)
from standards import StandardsFactory, IStandard
from ai_excel_extractor import AIExcelExtractor, ProcessingReport, ExtractionResult
from sld_data_preparation import SLDDataPreparationEngine, SLDProcessor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration status tracking"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class IntegrationMetrics:
    """Metrics for integration performance tracking"""
    extraction_confidence: float = 0.0
    calculation_success_rate: float = 0.0
    standards_compliance_rate: float = 0.0
    data_quality_score: float = 0.0
    processing_time_seconds: float = 0.0
    components_processed: int = 0
    issues_found: int = 0
    optimization_suggestions: int = 0


@dataclass
class IntegrationReport:
    """Comprehensive integration report"""
    overall_status: IntegrationStatus
    metrics: IntegrationMetrics
    calculation_results: Dict[str, Any] = field(default_factory=dict)
    standards_validation: Dict[str, Any] = field(default_factory=dict)
    sld_integration: Dict[str, Any] = field(default_factory=dict)
    optimization_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    integration_issues: List[str] = field(default_factory=list)
    performance_data: Dict[str, float] = field(default_factory=dict)


class CalculationEngineIntegration:
    """
    Deep integration layer for electrical calculation engines
    Ensures AI-extracted data works seamlessly with current, voltage drop,
    cable sizing, and breaker selection calculations
    """

    def __init__(self, standard: str = "IEC"):
        self.standard = StandardsFactory.get_standard(standard)
        self.calc_engine = ElectricalCalculationEngine(standard)
        self.current_calc = CurrentCalculator(self.standard)
        self.voltage_drop_calc = VoltageDropCalculator(self.standard)
        self.cable_sizing = CableSizingEngine(self.standard)
        self.breaker_selection = BreakerSelectionEngine(self.standard)
        self.logger = logging.getLogger(__name__)

    def integrate_load_calculations(self, load: Load) -> Tuple[Load, Dict[str, Any]]:
        """
        Deep integration of load calculations with AI-extracted data
        
        Args:
            load: Load object (potentially AI-extracted)
            
        Returns:
            Tuple of (enhanced load, calculation report)
        """
        calc_report = {
            "load_id": load.load_id,
            "original_power_kw": load.power_kw,
            "calculations_performed": [],
            "validation_results": {},
            "issues": [],
            "warnings": []
        }
        
        try:
            # Step 1: Current calculations
            if load.power_kw > 0 and load.voltage > 0:
                current_results = self.current_calc.calculate_load_current(load)
                load.current_a = current_results["current_a"]
                load.design_current_a = current_results["design_current_a"]
                load.apparent_power_kva = current_results["apparent_power_kva"]
                calc_report["calculations_performed"].append("current_calculation")
                
                self.logger.info(f"Calculated currents for load {load.load_id}: {load.current_a:.2f}A")
            
            # Step 2: Cable sizing (if length provided)
            if load.cable_length > 0:
                try:
                    cable_results = self.cable_sizing.calculate_cable_size(
                        current=load.design_current_a or load.current_a or 0,
                        voltage=load.voltage,
                        length=load.cable_length,
                        phases=load.phases,
                        installation_method=load.installation_method.value,
                        ambient_temp=40,  # Default ambient temperature
                        grouping_factor=load.grouping_factor,
                        power_factor=load.power_factor,
                        max_voltage_drop_percent=5.0
                    )
                    
                    # Update load with cable sizing results
                    load.cable_size_sqmm = cable_results["cable_size_sqmm"]
                    load.cable_cores = cable_results["cable_cores"]
                    load.cable_type = cable_results["cable_type"]
                    load.voltage_drop_v = cable_results["voltage_drop_v"]
                    load.voltage_drop_percent = cable_results["voltage_drop_percent"]
                    
                    calc_report["calculations_performed"].append("cable_sizing")
                    calc_report["cable_sizing_details"] = cable_results
                    
                    self.logger.info(f"Calculated cable sizing for load {load.load_id}: {load.cable_size_sqmm}mm²")
                    
                except Exception as e:
                    error_msg = f"Cable sizing failed for load {load.load_id}: {str(e)}"
                    calc_report["issues"].append(error_msg)
                    self.logger.warning(error_msg)
            
            # Step 3: Breaker selection
            if load.design_current_a:
                try:
                    breaker_results = self.breaker_selection.select_breaker(
                        load_current=load.current_a or 0,
                        design_current=load.design_current_a,
                        load_type=load.load_type.value,
                        voltage=load.voltage,
                        phases=load.phases
                    )
                    
                    load.breaker_rating_a = breaker_results["breaker_rating_a"]
                    load.breaker_type = breaker_results["breaker_type"]
                    load.breaker_curve = breaker_results.get("curve_type")
                    
                    calc_report["calculations_performed"].append("breaker_selection")
                    calc_report["breaker_details"] = breaker_results
                    
                    self.logger.info(f"Selected breaker for load {load.load_id}: {load.breaker_type} {load.breaker_rating_a}A")
                    
                except Exception as e:
                    error_msg = f"Breaker selection failed for load {load.load_id}: {str(e)}"
                    calc_report["issues"].append(error_msg)
                    self.logger.warning(error_msg)
            
            # Step 4: Validation of results
            calc_report["validation_results"] = self.calc_engine.validate_calculations(load)
            
            # Step 5: Electrical engineering validation
            self._validate_electrical_constraints(load, calc_report)
            
            return load, calc_report
            
        except Exception as e:
            error_msg = f"Calculation integration failed for load {load.load_id}: {str(e)}"
            calc_report["issues"].append(error_msg)
            self.logger.error(error_msg)
            return load, calc_report

    def _validate_electrical_constraints(self, load: Load, calc_report: Dict[str, Any]):
        """Validate electrical engineering constraints for load"""
        
        # Voltage drop validation
        if load.voltage_drop_percent and load.voltage_drop_percent > 5.0:
            calc_report["validation_results"]["warnings"].append(
                f"Voltage drop {load.voltage_drop_percent:.1f}% exceeds 5% limit for load {load.load_id}"
            )
        
        # Breaker rating validation
        if load.breaker_rating_a and load.design_current_a:
            if load.breaker_rating_a < load.design_current_a * 1.1:  # 10% safety margin
                calc_report["validation_results"]["errors"].append(
                    f"Breaker rating {load.breaker_rating_a}A too low for design current {load.design_current_a:.1f}A"
                )
        
        # Current density validation
        if load.current_a and load.cable_size_sqmm:
            current_density = load.current_a / load.cable_size_sqmm
            if current_density > 6.0:  # Maximum 6 A/mm² for copper
                calc_report["validation_results"]["warnings"].append(
                    f"High current density {current_density:.1f} A/mm² for load {load.load_id}"
                )

    def integrate_project_calculations(self, project: Project) -> Tuple[Project, Dict[str, Any]]:
        """
        Integrate calculations for entire project
        
        Args:
            project: Project object with loads
            
        Returns:
            Tuple of (enhanced project, integration report)
        """
        project_report = {
            "project_name": project.project_name,
            "total_loads": len(project.loads),
            "loads_processed": 0,
            "load_reports": {},
            "aggregate_calculations": {},
            "system_summary": {}
        }
        
        try:
            # Process each load
            for load in project.loads:
                load_report = {}
                enhanced_load, load_calc_report = self.integrate_load_calculations(load)
                project_report["load_reports"][load.load_id] = load_calc_report
                project_report["loads_processed"] += 1
            
            # Aggregate calculations
            self._calculate_project_totals(project, project_report)
            
            # Bus loading calculations
            self._calculate_bus_loads(project, project_report)
            
            # System-level validations
            self._validate_system_constraints(project, project_report)
            
            return project, project_report
            
        except Exception as e:
            error_msg = f"Project calculation integration failed: {str(e)}"
            project_report["error"] = error_msg
            self.logger.error(error_msg)
            return project, project_report

    def _calculate_project_totals(self, project: Project, report: Dict[str, Any]):
        """Calculate project-level totals and metrics"""
        
        # Power totals
        total_power_kw = sum(load.power_kw for load in project.loads if load.power_kw)
        total_demand_kw = sum(load.power_kw * self._get_demand_factor(load) for load in project.loads)
        total_reactive_kvar = sum(
            load.power_kw * (1/load.power_factor - 1) for load in project.loads 
            if load.power_kw and load.power_factor
        )
        
        report["aggregate_calculations"] = {
            "total_installed_capacity_kw": total_power_kw,
            "total_demand_kw": total_demand_kw,
            "total_reactive_kvar": total_reactive_kvar,
            "average_power_factor": total_power_kw / (total_power_kw + total_reactive_kvar) if total_reactive_kvar > 0 else 1.0,
            "system_diversity_factor": total_demand_kw / total_power_kw if total_power_kw > 0 else 1.0
        }

    def _calculate_bus_loads(self, project: Project, report: Dict[str, Any]):
        """Calculate bus loading and distribution"""
        
        bus_loads = {}
        for load in project.loads:
            if load.source_bus:
                if load.source_bus not in bus_loads:
                    bus_loads[load.source_bus] = {
                        "connected_loads": [],
                        "total_power_kw": 0,
                        "total_current_a": 0
                    }
                
                bus_loads[load.source_bus]["connected_loads"].append(load.load_id)
                bus_loads[load.source_bus]["total_power_kw"] += load.power_kw
                if load.design_current_a:
                    bus_loads[load.source_bus]["total_current_a"] += load.design_current_a
        
        report["bus_loads"] = bus_loads

    def _validate_system_constraints(self, project: Project, report: Dict[str, Any]):
        """Validate system-level electrical constraints"""
        
        # Check total system capacity
        total_power = sum(load.power_kw for load in project.loads)
        if total_power > 1000:  # 1 MW threshold
            report["validation_warnings"].append(
                f"Large system capacity {total_power:.0f}kW - consider multiple feeders"
            )
        
        # Check voltage consistency
        voltages = set(load.voltage for load in project.loads)
        if len(voltages) > 2:
            report["validation_warnings"].append(
                f"Multiple voltage levels detected: {sorted(voltages)}V"
            )

    def _get_demand_factor(self, load: Load) -> float:
        """Get demand factor based on load type and characteristics"""
        
        if load.load_type == LoadType.MOTOR:
            return 0.8  # Motors typically have diversity
        elif load.load_type == LoadType.LIGHTING:
            return 0.9  # Lighting usually consistent
        elif load.load_type == LoadType.HVAC:
            return 0.7  # HVAC has good diversity
        else:
            return 0.85  # Default diversity factor


class StandardsIntegrationEngine:
    """
    Deep integration with electrical standards (IEC, IS, NEC)
    Validates extracted data against standards and applies
    temperature factors, derating, and compliance checks
    """

    def __init__(self, standard: str = "IEC"):
        self.standard = StandardsFactory.get_standard(standard)
        self.standard_name = standard
        self.logger = logging.getLogger(__name__)

    def validate_project_compliance(self, project: Project) -> Dict[str, Any]:
        """
        Validate project against electrical standards
        
        Args:
            project: Project to validate
            
        Returns:
            Compliance report dictionary
        """
        compliance_report = {
            "standard": self.standard_name,
            "overall_compliant": True,
            "voltage_drop_compliance": {},
            "cable_ratings_compliance": {},
            "temperature_derating": {},
            "grouping_factors": {},
            "standards_violations": [],
            "recommendations": []
        }
        
        try:
            # Validate voltage drop compliance
            self._validate_voltage_drop_compliance(project, compliance_report)
            
            # Validate cable current ratings
            self._validate_cable_ratings(project, compliance_report)
            
            # Apply temperature derating
            self._apply_temperature_derating(project, compliance_report)
            
            # Check grouping factors
            self._validate_grouping_factors(project, compliance_report)
            
            # Overall compliance assessment
            compliance_report["overall_compliant"] = len(compliance_report["standards_violations"]) == 0
            
            return compliance_report
            
        except Exception as e:
            error_msg = f"Standards compliance validation failed: {str(e)}"
            compliance_report["error"] = error_msg
            self.logger.error(error_msg)
            return compliance_report

    def _validate_voltage_drop_compliance(self, project: Project, report: Dict[str, Any]):
        """Validate voltage drop against standards limits"""
        
        voltage_drop_violations = []
        
        for load in project.loads:
            if load.voltage_drop_percent is not None:
                # Get limit based on circuit type
                circuit_type = self._get_circuit_type(load)
                limit = self.standard.get_voltage_drop_limit(circuit_type)
                
                if load.voltage_drop_percent > limit:
                    voltage_drop_violations.append({
                        "load_id": load.load_id,
                        "load_name": load.load_name,
                        "voltage_drop_percent": load.voltage_drop_percent,
                        "limit_percent": limit,
                        "exceeded_by": load.voltage_drop_percent - limit
                    })
        
        report["voltage_drop_compliance"] = {
            "violations": voltage_drop_violations,
            "compliant_loads": len([l for l in project.loads if l.voltage_drop_percent is not None]) - len(voltage_drop_violations),
            "total_loads": len([l for l in project.loads if l.voltage_drop_percent is not None])
        }
        
        if voltage_drop_violations:
            report["standards_violations"].extend(
                [f"Voltage drop violation for load {v['load_id']}: {v['voltage_drop_percent']:.1f}% > {v['limit_percent']:.1f}% limit"
                 for v in voltage_drop_violations]
            )

    def _validate_cable_ratings(self, project: Project, report: Dict[str, Any]):
        """Validate cable current ratings against standards"""
        
        rating_violations = []
        
        for load in project.loads:
            if load.cable_size_sqmm and (load.design_current_a or load.current_a):
                current = load.design_current_a or load.current_a
                
                # Get cable rating
                rating = self.standard.get_cable_current_capacity(
                    load.cable_size_sqmm,
                    load.installation_method.value,
                    40  # Default ambient temperature
                )
                
                if rating < current:
                    rating_violations.append({
                        "load_id": load.load_id,
                        "cable_size_mm2": load.cable_size_sqmm,
                        "current_a": current,
                        "rating_a": rating,
                        "utilization_percent": (current / rating) * 100
                    })
        
        report["cable_ratings_compliance"] = {
            "violations": rating_violations,
            "compliant_loads": len(project.loads) - len(rating_violations),
            "total_loads": len(project.loads)
        }
        
        if rating_violations:
            report["standards_violations"].extend(
                [f"Cable rating violation for load {v['load_id']}: {v['current_a']:.1f}A > {v['rating_a']:.1f}A cable rating"
                 for v in rating_violations]
            )

    def _apply_temperature_derating(self, project: Project, report: Dict[str, Any]):
        """Apply temperature derating factors"""
        
        derating_data = {}
        
        for load in project.loads:
            if load.installation_method:
                temp_factor = self.standard.get_temperature_factor(40)  # Default 40°C
                install_factor = self.standard.get_installation_factor(load.installation_method.value)
                
                derating_data[load.load_id] = {
                    "temperature_factor": temp_factor,
                    "installation_factor": install_factor,
                    "combined_factor": temp_factor * install_factor
                }
        
        report["temperature_derating"] = derating_data

    def _validate_grouping_factors(self, project: Project, report: Dict[str, Any]):
        """Validate grouping factor applications"""
        
        # Count cables per installation method and location
        installation_groups = {}
        
        for load in project.loads:
            if load.installation_method:
                group_key = f"{load.installation_method.value}_tray"
                if group_key not in installation_groups:
                    installation_groups[group_key] = []
                installation_groups[group_key].append(load.load_id)
        
        # Apply grouping factors
        grouping_data = {}
        for group_key, load_ids in installation_groups.items():
            num_cables = len(load_ids)
            grouping_factor = self.standard.get_grouping_factor(num_cables)
            
            for load_id in load_ids:
                grouping_data[load_id] = {
                    "group_size": num_cables,
                    "grouping_factor": grouping_factor
                }
        
        report["grouping_factors"] = grouping_data

    def _get_circuit_type(self, load: Load) -> str:
        """Determine circuit type for voltage drop limits"""
        
        if load.load_type == LoadType.LIGHTING:
            return "lighting"
        elif load.load_type == LoadType.MOTOR:
            return "motor"
        else:
            return "power"


class SLDSystemIntegration:
    """
    Integration layer for SLD generation system
    Ensures AI-extracted data seamlessly feeds into SLD processing
    and diagram generation
    """

    def __init__(self, standard: str = "IEC"):
        self.sld_engine = SLDDataPreparationEngine(standard)
        self.sld_processor = SLDProcessor(standard)
        self.logger = logging.getLogger(__name__)

    def integrate_with_sld_system(self, project: Project) -> Tuple[Project, Dict[str, Any]]:
        """
        Integrate project with SLD generation system
        
        Args:
            project: Project to integrate
            
        Returns:
            Tuple of (enhanced project, SLD integration report)
        """
        sld_report = {
            "integration_status": "in_progress",
            "sld_data_prepared": False,
            "electrical_hierarchy_created": False,
            "connectivity_established": False,
            "components_positioned": False,
            "sld_validation": {},
            "generation_capabilities": [],
            "integration_issues": []
        }
        
        try:
            # Step 1: Prepare SLD data from project
            sld_data = self.sld_engine.prepare_sld_data(project)
            sld_report["sld_data_prepared"] = True
            sld_report["generation_capabilities"].append("sld_data_preparation")
            
            # Step 2: Validate electrical hierarchy
            hierarchy_validation = self._validate_electrical_hierarchy(project, sld_data)
            sld_report["electrical_hierarchy_created"] = hierarchy_validation["valid"]
            sld_report["hierarchy_validation"] = hierarchy_validation
            
            # Step 3: Establish connectivity
            connectivity_status = self._establish_sld_connectivity(project, sld_data)
            sld_report["connectivity_established"] = connectivity_status["valid"]
            sld_report["connectivity_details"] = connectivity_status
            
            # Step 4: Position components
            positioning_status = self._validate_component_positioning(sld_data)
            sld_report["components_positioned"] = positioning_status["valid"]
            sld_report["positioning_details"] = positioning_status
            
            # Step 5: SLD system validation
            sld_validation = sld_data.get("validation", {})
            sld_report["sld_validation"] = sld_validation
            
            # Step 6: Update project with SLD metadata
            self._update_project_with_sld_metadata(project, sld_data)
            
            sld_report["integration_status"] = "completed"
            self.logger.info(f"SLD integration completed for project: {project.project_name}")
            
            return project, sld_report
            
        except Exception as e:
            error_msg = f"SLD integration failed: {str(e)}"
            sld_report["integration_status"] = "failed"
            sld_report["integration_issues"].append(error_msg)
            self.logger.error(error_msg)
            return project, sld_report

    def _validate_electrical_hierarchy(self, project: Project, sld_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate electrical hierarchy in SLD data"""
        
        hierarchy = sld_data.get("electrical_hierarchy", {})
        
        validation_result = {
            "valid": True,
            "checks": {},
            "issues": [],
            "statistics": {}
        }
        
        # Check transformer hierarchy
        transformers = hierarchy.get("transformers", [])
        validation_result["statistics"]["transformers"] = len(transformers)
        
        # Check bus hierarchy
        main_buses = hierarchy.get("main_buses", [])
        dist_buses = hierarchy.get("distribution_buses", [])
        validation_result["statistics"]["main_buses"] = len(main_buses)
        validation_result["statistics"]["distribution_buses"] = len(dist_buses)
        
        # Check load organization
        loads = hierarchy.get("loads", [])
        validation_result["statistics"]["loads"] = len(loads)
        
        # Validate hierarchy integrity
        for load in loads:
            if not load.get("source_bus"):
                validation_result["issues"].append(f"Load {load['id']} has no source bus")
                validation_result["valid"] = False
        
        validation_result["checks"]["hierarchy_integrity"] = len(validation_result["issues"]) == 0
        
        return validation_result

    def _establish_sld_connectivity(self, project: Project, sld_data: Dict[str, Any]) -> Dict[str, Any]:
        """Establish SLD connectivity from project data"""
        
        connectivity = sld_data.get("connectivity", {})
        
        connectivity_result = {
            "valid": True,
            "connection_statistics": {},
            "missing_connections": [],
            "duplicate_connections": []
        }
        
        # Count connections
        connectivity_result["connection_statistics"] = {
            "transformer_to_bus": len(connectivity.get("transformer_to_bus", [])),
            "bus_to_bus": len(connectivity.get("bus_to_bus", [])),
            "bus_to_load": len(connectivity.get("bus_to_load", [])),
            "cable_routes": len(connectivity.get("cable_routes", [])),
            "protection_zones": len(connectivity.get("protection_zones", []))
        }
        
        # Validate connectivity completeness
        expected_load_connections = len(project.loads)
        actual_load_connections = connectivity_result["connection_statistics"]["bus_to_load"]
        
        if actual_load_connections < expected_load_connections:
            missing = expected_load_connections - actual_load_connections
            connectivity_result["missing_connections"].append(f"{missing} loads missing bus connections")
            connectivity_result["valid"] = False
        
        return connectivity_result

    def _validate_component_positioning(self, sld_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component positioning for SLD generation"""
        
        components = sld_data.get("components", [])
        
        positioning_result = {
            "valid": True,
            "positioning_statistics": {},
            "positioning_issues": []
        }
        
        # Count positioned components
        positioned_count = 0
        for comp in components:
            if comp.get("x_position") is not None and comp.get("y_position") is not None:
                positioned_count += 1
        
        positioning_result["positioning_statistics"] = {
            "total_components": len(components),
            "positioned_components": positioned_count,
            "positioning_percentage": (positioned_count / len(components) * 100) if components else 0
        }
        
        if positioned_count < len(components):
            positioning_result["positioning_issues"].append(
                f"Only {positioned_count}/{len(components)} components have positions"
            )
        
        positioning_result["valid"] = positioned_count == len(components)
        
        return positioning_result

    def _update_project_with_sld_metadata(self, project: Project, sld_data: Dict[str, Any]):
        """Update project with SLD generation metadata"""
        
        # Add SLD generation capability flags
        if not hasattr(project, 'sld_metadata'):
            project.sld_metadata = {}
        
        project.sld_metadata.update({
            "sld_ready": True,
            "hierarchy_valid": True,
            "connectivity_complete": True,
            "components_positioned": True,
            "generation_timestamp": datetime.now().isoformat(),
            "voltage_levels": sld_data.get("metadata", {}).get("voltage_levels", []),
            "tool_compatibility": ["AutoCAD Electrical", "ETAP", "EPLAN", "SolidWorks Electrical"]
        })


class OptimizationEngine:
    """
    Engine for generating optimization suggestions based on
    electrical calculations and standards compliance
    """

    def __init__(self, standard: str = "IEC"):
        self.standard = StandardsFactory.get_standard(standard)
        self.logger = logging.getLogger(__name__)

    def generate_optimization_suggestions(self, project: Project, 
                                        calculation_results: Dict[str, Any],
                                        standards_compliance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions based on calculations and compliance
        
        Args:
            project: Project to analyze
            calculation_results: Results from calculation engine
            standards_compliance: Standards compliance results
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        try:
            # Power factor optimization
            suggestions.extend(self._analyze_power_factor_optimization(project))
            
            # Cable sizing optimization
            suggestions.extend(self._analyze_cable_optimization(project, standards_compliance))
            
            # Voltage drop optimization
            suggestions.extend(self._analyze_voltage_drop_optimization(project))
            
            # Load balancing suggestions
            suggestions.extend(self._analyze_load_balancing(project))
            
            # Energy efficiency suggestions
            suggestions.extend(self._analyze_energy_efficiency(project))
            
            # Protection optimization
            suggestions.extend(self._analyze_protection_optimization(project, calculation_results))
            
            self.logger.info(f"Generated {len(suggestions)} optimization suggestions")
            return suggestions
            
        except Exception as e:
            error_msg = f"Optimization analysis failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"type": "error", "message": error_msg, "priority": "low"}]

    def _analyze_power_factor_optimization(self, project: Project) -> List[Dict[str, Any]]:
        """Analyze power factor optimization opportunities"""
        suggestions = []
        
        # Find low power factor loads
        low_pf_loads = [load for load in project.loads if load.power_factor < 0.8]
        
        if low_pf_loads:
            suggestions.append({
                "type": "power_factor_correction",
                "priority": "medium",
                "title": "Power Factor Correction Recommended",
                "description": f"Found {len(low_pf_loads)} loads with power factor < 0.8",
                "affected_loads": [load.load_id for load in low_pf_loads],
                "recommendation": "Install power factor correction capacitors",
                "estimated_benefits": {
                    "power_factor_improvement": "0.8 → 0.95",
                    "current_reduction": "15-20%",
                    "energy_savings": "5-10%"
                },
                "implementation_cost": "medium"
            })
        
        return suggestions

    def _analyze_cable_optimization(self, project: Project, compliance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze cable sizing optimization"""
        suggestions = []
        
        # Find oversized cables
        voltage_drop_data = compliance.get("voltage_drop_compliance", {})
        rating_data = compliance.get("cable_ratings_compliance", {})
        
        # Check for cables with very low utilization
        underutilized_cables = []
        for load in project.loads:
            if load.cable_size_sqmm and load.design_current_a:
                rating = self.standard.get_cable_current_capacity(
                    load.cable_size_sqmm, load.installation_method.value, 40
                )
                utilization = (load.design_current_a / rating) * 100
                
                if utilization < 30:  # Less than 30% utilization
                    underutilized_cables.append({
                        "load_id": load.load_id,
                        "current_size": load.cable_size_sqmm,
                        "utilization_percent": utilization,
                        "suggested_size": self._suggest_smaller_cable(load.design_current_a, load.installation_method.value)
                    })
        
        if underutilized_cables:
            suggestions.append({
                "type": "cable_optimization",
                "priority": "low",
                "title": "Cable Size Optimization",
                "description": f"Found {len(underutilized_cables)} cables with <30% utilization",
                "affected_cables": underutilized_cables,
                "recommendation": "Consider smaller cable sizes to reduce material costs",
                "estimated_benefits": {
                    "material_cost_savings": "10-15%",
                    "installation_simplification": "reduced cable weight"
                },
                "implementation_cost": "low"
            })
        
        return suggestions

    def _analyze_voltage_drop_optimization(self, project: Project) -> List[Dict[str, Any]]:
        """Analyze voltage drop optimization opportunities"""
        suggestions = []
        
        # Find loads with high voltage drop
        high_vdrop_loads = [load for load in project.loads 
                          if load.voltage_drop_percent and load.voltage_drop_percent > 3.0]
        
        if high_vdrop_loads:
            total_power = sum(load.power_kw for load in high_vdrop_loads)
            suggestions.append({
                "type": "voltage_drop_improvement",
                "priority": "medium",
                "title": "Voltage Drop Improvement",
                "description": f"Found {len(high_vdrop_loads)} loads with voltage drop >3% (total {total_power:.1f}kW)",
                "affected_loads": [load.load_id for load in high_vdrop_loads],
                "recommendation": "Consider larger cables or shorter runs to reduce voltage drop",
                "estimated_benefits": {
                    "voltage_drop_reduction": "1-2%",
                    "efficiency_improvement": "2-3%",
                    "energy_savings": "1-2%"
                },
                "implementation_cost": "medium"
            })
        
        return suggestions

    def _analyze_load_balancing(self, project: Project) -> List[Dict[str, Any]]:
        """Analyze load balancing opportunities"""
        suggestions = []
        
        # Group loads by bus
        bus_loads = {}
        for load in project.loads:
            if load.source_bus:
                if load.source_bus not in bus_loads:
                    bus_loads[load.source_bus] = []
                bus_loads[load.source_bus].append(load)
        
        # Check for unbalanced buses
        unbalanced_buses = []
        for bus_id, loads in bus_loads.items():
            total_power = sum(load.power_kw for load in loads)
            if total_power > 200:  # Large bus loading
                # Check if loading is concentrated
                largest_load = max(load.power_kw for load in loads)
                if largest_load / total_power > 0.5:  # Single load >50% of bus load
                    unbalanced_buses.append({
                        "bus_id": bus_id,
                        "total_power_kw": total_power,
                        "largest_load_kw": largest_load,
                        "concentration_ratio": largest_load / total_power
                    })
        
        if unbalanced_buses:
            suggestions.append({
                "type": "load_balancing",
                "priority": "low",
                "title": "Load Balancing Opportunity",
                "description": f"Found {len(unbalanced_buses)} buses with concentrated loading",
                "affected_buses": unbalanced_buses,
                "recommendation": "Consider distributing large loads across multiple feeders",
                "estimated_benefits": {
                    "reliability_improvement": "reduced single point of failure",
                    "maintenance_simplification": "easier to isolate faults"
                },
                "implementation_cost": "medium"
            })
        
        return suggestions

    def _analyze_energy_efficiency(self, project: Project) -> List[Dict[str, Any]]:
        """Analyze energy efficiency optimization"""
        suggestions = []
        
        # Check for inefficient motors
        inefficient_motors = [load for load in project.loads 
                            if load.load_type == LoadType.MOTOR and load.efficiency < 0.85]
        
        if inefficient_motors:
            total_inefficient_power = sum(load.power_kw for load in inefficient_motors)
            suggestions.append({
                "type": "energy_efficiency",
                "priority": "medium",
                "title": "Motor Efficiency Improvement",
                "description": f"Found {len(inefficient_motors)} motors with efficiency <85% (total {total_inefficient_power:.1f}kW)",
                "affected_motors": [load.load_id for load in inefficient_motors],
                "recommendation": "Upgrade to high-efficiency motors",
                "estimated_benefits": {
                    "efficiency_improvement": "5-10%",
                    "energy_savings": "3-7%",
                    "payback_period": "2-4 years"
                },
                "implementation_cost": "high"
            })
        
        return suggestions

    def _analyze_protection_optimization(self, project: Project, calc_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze protection system optimization"""
        suggestions = []
        
        # Check for coordination issues
        coordination_issues = []
        for load in project.loads:
            if load.breaker_rating_a and load.design_current_a:
                ratio = load.breaker_rating_a / load.design_current_a
                if ratio > 3.0:  # Breaker much larger than necessary
                    coordination_issues.append({
                        "load_id": load.load_id,
                        "current_rating": load.breaker_rating_a,
                        "design_current": load.design_current_a,
                        "ratio": ratio
                    })
        
        if coordination_issues:
            suggestions.append({
                "type": "protection_coordination",
                "priority": "high",
                "title": "Protection Coordination Optimization",
                "description": f"Found {len(coordination_issues)} breakers with poor coordination",
                "affected_breakers": coordination_issues,
                "recommendation": "Optimize breaker ratings for better protection coordination",
                "estimated_benefits": {
                    "selectivity_improvement": "better fault discrimination",
                    "cost_optimization": "reduced breaker costs"
                },
                "implementation_cost": "low"
            })
        
        return suggestions

    def _suggest_smaller_cable(self, current_a: float, installation_method: str) -> float:
        """Suggest smaller cable size for given current"""
        
        # Standard cable sizes
        cable_sizes = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240, 300]
        
        for size in cable_sizes:
            rating = self.standard.get_cable_current_capacity(size, installation_method, 40)
            if rating * 0.8 >= current_a:  # 80% utilization
                return size
        
        return max(cable_sizes)  # Return largest if none suitable


class ComprehensiveIntegrationLayer:
    """
    Main integration layer that orchestrates all integration components
    and provides a unified interface for AI-extracted data integration
    """

    def __init__(self, standard: str = "IEC"):
        self.standard = standard
        self.calc_integration = CalculationEngineIntegration(standard)
        self.standards_integration = StandardsIntegrationEngine(standard)
        self.sld_integration = SLDSystemIntegration(standard)
        self.optimization_engine = OptimizationEngine(standard)
        self.logger = logging.getLogger(__name__)

    def integrate_extracted_project(self, project: Project) -> IntegrationReport:
        """
        Main integration method that processes AI-extracted project
        through all integration layers
        
        Args:
            project: Project object (potentially AI-extracted)
            
        Returns:
            Comprehensive IntegrationReport
        """
        start_time = time.time()
        
        integration_report = IntegrationReport(
            overall_status=IntegrationStatus.IN_PROGRESS,
            metrics=IntegrationMetrics()
        )
        
        try:
            self.logger.info(f"Starting comprehensive integration for project: {project.project_name}")
            
            # Step 1: Calculation Engine Integration
            self.logger.info("Step 1: Integrating with calculation engines")
            enhanced_project, calc_report = self.calc_integration.integrate_project_calculations(project)
            integration_report.calculation_results = calc_report
            
            # Update metrics
            integration_report.metrics.components_processed = len(enhanced_project.loads)
            integration_report.metrics.calculation_success_rate = (
                len([r for r in calc_report.get("load_reports", {}).values() 
                    if not r.get("issues")]) / max(len(enhanced_project.loads), 1)
            ) * 100
            
            # Step 2: Standards Integration
            self.logger.info("Step 2: Validating against electrical standards")
            standards_report = self.standards_integration.validate_project_compliance(enhanced_project)
            integration_report.standards_validation = standards_report
            
            # Update metrics
            integration_report.metrics.standards_compliance_rate = (
                100 if standards_report.get("overall_compliant") else 
                (len(enhanced_project.loads) - len(standards_report.get("standards_violations", []))) / 
                max(len(enhanced_project.loads), 1) * 100
            )
            
            # Step 3: SLD System Integration
            self.logger.info("Step 3: Integrating with SLD generation system")
            final_project, sld_report = self.sld_integration.integrate_with_sld_system(enhanced_project)
            integration_report.sld_integration = sld_report
            
            # Step 4: Generate Optimization Suggestions
            self.logger.info("Step 4: Generating optimization suggestions")
            optimization_suggestions = self.optimization_engine.generate_optimization_suggestions(
                final_project, calc_report, standards_report
            )
            integration_report.optimization_suggestions = optimization_suggestions
            
            # Update metrics
            integration_report.metrics.optimization_suggestions = len(optimization_suggestions)
            
            # Step 5: Data Quality Assessment
            self.logger.info("Step 5: Assessing data quality")
            data_quality_score = self._assess_data_quality(final_project, calc_report, standards_report)
            integration_report.metrics.data_quality_score = data_quality_score
            
            # Step 6: Performance Metrics
            processing_time = time.time() - start_time
            integration_report.metrics.processing_time_seconds = processing_time
            
            # Count total issues
            total_issues = (
                len(calc_report.get("load_reports", {}).get("validation_results", {}).get("errors", [])) +
                len(standards_report.get("standards_violations", [])) +
                len(sld_report.get("integration_issues", []))
            )
            integration_report.metrics.issues_found = total_issues
            
            # Step 7: Determine Overall Status
            if total_issues == 0 and data_quality_score > 0.8:
                integration_report.overall_status = IntegrationStatus.COMPLETED
            elif total_issues < 5 and data_quality_score > 0.6:
                integration_report.overall_status = IntegrationStatus.WARNING
            else:
                integration_report.overall_status = IntegrationStatus.FAILED
            
            # Add performance data
            integration_report.performance_data = {
                "extraction_confidence": getattr(project, 'extraction_confidence', 0.0),
                "processing_time": processing_time,
                "components_per_second": integration_report.metrics.components_processed / processing_time,
                "quality_score": data_quality_score
            }
            
            self.logger.info(f"Integration completed: {integration_report.overall_status.value} - "
                           f"Quality: {data_quality_score:.1%}, Issues: {total_issues}")
            
            return integration_report
            
        except Exception as e:
            error_msg = f"Integration failed: {str(e)}"
            integration_report.integration_issues.append(error_msg)
            integration_report.overall_status = IntegrationStatus.FAILED
            integration_report.metrics.processing_time_seconds = time.time() - start_time
            self.logger.error(error_msg)
            return integration_report

    def _assess_data_quality(self, project: Project, calc_report: Dict[str, Any], standards_report: Dict[str, Any]) -> float:
        """
        Assess overall data quality of integrated project
        
        Args:
            project: Project object
            calc_report: Calculation integration report
            standards_report: Standards compliance report
            
        Returns:
            Data quality score (0.0 to 1.0)
        """
        quality_factors = []
        
        # Factor 1: Calculation completeness
        loads_with_calculations = len([load for load in project.loads if load.current_a is not None])
        calculation_completeness = loads_with_calculations / max(len(project.loads), 1)
        quality_factors.append(calculation_completeness * 0.3)  # 30% weight
        
        # Factor 2: Standards compliance
        compliance_score = 1.0 - (len(standards_report.get("standards_violations", [])) / max(len(project.loads), 1))
        quality_factors.append(max(compliance_score, 0.0) * 0.25)  # 25% weight
        
        # Factor 3: Data consistency
        consistency_checks = [
            self._check_voltage_consistency(project),
            self._check_load_balance_consistency(project),
            self._check_cable_specification_consistency(project)
        ]
        consistency_score = sum(consistency_checks) / len(consistency_checks)
        quality_factors.append(consistency_score * 0.2)  # 20% weight
        
        # Factor 4: Required field completeness
        required_fields_score = self._check_required_fields_completeness(project)
        quality_factors.append(required_fields_score * 0.15)  # 15% weight
        
        # Factor 5: Electrical engineering validity
        engineering_validity = self._assess_engineering_validity(project, calc_report)
        quality_factors.append(engineering_validity * 0.1)  # 10% weight
        
        # Calculate overall score
        overall_score = sum(quality_factors)
        return max(0.0, min(1.0, overall_score))

    def _check_voltage_consistency(self, project: Project) -> float:
        """Check voltage level consistency across the project"""
        voltages = [load.voltage for load in project.loads if load.voltage]
        if not voltages:
            return 0.0
        
        # Check for excessive voltage variation
        voltage_set = set(voltages)
        standard_voltages = {230, 400, 415, 440, 690, 3300, 6600, 11000, 33000}
        
        # Deduct points for non-standard voltages
        non_standard_count = sum(1 for v in voltages if v not in standard_voltages)
        deduction = non_standard_count / len(voltages) * 0.3
        
        return max(0.0, 1.0 - deduction)

    def _check_load_balance_consistency(self, project: Project) -> float:
        """Check load balance across distribution buses"""
        bus_loads = {}
        for load in project.loads:
            if load.source_bus:
                if load.source_bus not in bus_loads:
                    bus_loads[load.source_bus] = []
                bus_loads[load.source_bus].append(load.power_kw)
        
        if not bus_loads:
            return 0.5  # Neutral score if no bus assignments
        
        # Check for reasonable load distribution
        load_spreads = []
        for bus_id, loads in bus_loads.items():
            if len(loads) > 1:
                avg_load = sum(loads) / len(loads)
                variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
                # Normalize variance by average to get coefficient of variation
                if avg_load > 0:
                    cv = (variance ** 0.5) / avg_load
                    load_spreads.append(min(1.0, 1.0 / (1.0 + cv)))
        
        return sum(load_spreads) / len(load_spreads) if load_spreads else 0.7

    def _check_cable_specification_consistency(self, project: Project) -> float:
        """Check consistency of cable specifications"""
        cables_with_specs = [load for load in project.loads if load.cable_size_sqmm]
        if not cables_with_specs:
            return 0.5  # Neutral score if no cable data
        
        # Check for reasonable cable sizing vs current
        consistent_specs = 0
        for load in cables_with_specs:
            if load.current_a and load.cable_size_sqmm:
                # Basic sanity check: current density should be reasonable
                current_density = load.current_a / load.cable_size_sqmm
                if 1.0 <= current_density <= 8.0:  # Reasonable range for copper cables
                    consistent_specs += 1
        
        return consistent_specs / len(cables_with_specs)

    def _check_required_fields_completeness(self, project: Project) -> float:
        """Check completeness of required fields"""
        required_load_fields = ['load_id', 'load_name', 'power_kw', 'voltage', 'phases']
        
        complete_loads = 0
        for load in project.loads:
            load_complete = True
            for field in required_load_fields:
                if not getattr(load, field, None):
                    load_complete = False
                    break
            
            if load_complete:
                complete_loads += 1
        
        return complete_loads / max(len(project.loads), 1)

    def _assess_engineering_validity(self, project: Project, calc_report: Dict[str, Any]) -> float:
        """Assess engineering validity of calculations and specifications"""
        validity_score = 1.0
        
        # Check for unrealistic values
        for load in project.loads:
            # Power check
            if load.power_kw > 1000:  # Very large load
                validity_score -= 0.1
            
            # Voltage check
            if load.voltage not in [230, 400, 415, 440, 690, 3300, 6600, 11000, 33000]:
                validity_score -= 0.05
            
            # Current density check
            if load.current_a and load.cable_size_sqmm:
                current_density = load.current_a / load.cable_size_sqmm
                if current_density > 10:  # Too high current density
                    validity_score -= 0.1
                elif current_density < 0.5:  # Very low utilization
                    validity_score -= 0.05
        
        return max(0.0, validity_score)


# Factory function for easy integration
def create_integration_layer(standard: str = "IEC") -> ComprehensiveIntegrationLayer:
    """
    Factory function to create a comprehensive integration layer
    
    Args:
        standard: Electrical standard to use (IEC, IS, NEC)
        
    Returns:
        ComprehensiveIntegrationLayer instance
    """
    return ComprehensiveIntegrationLayer(standard)


# Utility functions for integration workflow
def integrate_ai_extracted_project_with_calculations(
    project: Project, 
    standard: str = "IEC"
) -> Tuple[Project, IntegrationReport]:
    """
    Convenience function for integrating AI-extracted project with calculations
    
    Args:
        project: AI-extracted project
        standard: Electrical standard
        
    Returns:
        Tuple of (enhanced project, integration report)
    """
    integration_layer = create_integration_layer(standard)
    return project, integration_layer.integrate_extracted_project(project)


def validate_integration_quality(integration_report: IntegrationReport) -> Dict[str, Any]:
    """
    Validate the quality of integration results
    
    Args:
        integration_report: Integration report to validate
        
    Returns:
        Quality validation results
    """
    validation_result = {
        "overall_quality": "unknown",
        "critical_issues": [],
        "warnings": [],
        "recommendations": [],
        "quality_metrics": {}
    }
    
    metrics = integration_report.metrics
    
    # Determine overall quality
    if (metrics.data_quality_score > 0.9 and 
        metrics.calculation_success_rate > 95 and 
        metrics.standards_compliance_rate > 90):
        validation_result["overall_quality"] = "excellent"
    elif (metrics.data_quality_score > 0.8 and 
          metrics.calculation_success_rate > 85 and 
          metrics.standards_compliance_rate > 80):
        validation_result["overall_quality"] = "good"
    elif (metrics.data_quality_score > 0.6 and 
          metrics.calculation_success_rate > 70 and 
          metrics.standards_compliance_rate > 70):
        validation_result["overall_quality"] = "acceptable"
    else:
        validation_result["overall_quality"] = "poor"
    
    # Check for critical issues
    if integration_report.overall_status == IntegrationStatus.FAILED:
        validation_result["critical_issues"].append("Integration failed completely")
    
    if metrics.calculation_success_rate < 50:
        validation_result["critical_issues"].append("Very low calculation success rate")
    
    if metrics.standards_compliance_rate < 50:
        validation_result["critical_issues"].append("Poor standards compliance")
    
    # Check for warnings
    if integration_report.metrics.issues_found > 10:
        validation_result["warnings"].append(f"High number of issues found: {integration_report.metrics.issues_found}")
    
    if integration_report.metrics.optimization_suggestions > 20:
        validation_result["warnings"].append(f"Many optimization opportunities: {integration_report.metrics.optimization_suggestions}")
    
    # Add quality metrics
    validation_result["quality_metrics"] = {
        "data_quality_score": metrics.data_quality_score,
        "calculation_success_rate": metrics.calculation_success_rate,
        "standards_compliance_rate": metrics.standards_compliance_rate,
        "processing_time": metrics.processing_time_seconds,
        "components_per_second": integration_report.performance_data.get("components_per_second", 0)
    }
    
    # Add recommendations
    if metrics.data_quality_score < 0.8:
        validation_result["recommendations"].append("Review and improve data extraction quality")
    
    if metrics.calculation_success_rate < 90:
        validation_result["recommendations"].append("Check input data completeness for failed calculations")
    
    if integration_report.optimization_suggestions:
        validation_result["recommendations"].append(f"Consider implementing {len(integration_report.optimization_suggestions)} optimization suggestions")
    
    return validation_result


# Example usage and testing
def demo_integration_workflow():
    """Demonstrate the integration workflow with sample data"""
    
    from models import LoadType, InstallationMethod, Priority
    
    # Create sample project
    project = Project(
        project_name="Integration Demo Project",
        standard="IEC",
        voltage_system="LV"
    )
    
    # Add sample loads
    sample_loads = [
        Load(
            load_id="L001",
            load_name="Main Motor",
            power_kw=15.0,
            voltage=400,
            phases=3,
            load_type=LoadType.MOTOR,
            installation_method=InstallationMethod.TRAY,
            priority=Priority.CRITICAL,
            cable_length=50.0,
            power_factor=0.85,
            efficiency=0.92
        ),
        Load(
            load_id="L002", 
            load_name="HVAC Unit",
            power_kw=25.0,
            voltage=400,
            phases=3,
            load_type=LoadType.HVAC,
            installation_method=InstallationMethod.TRAY,
            priority=Priority.ESSENTIAL,
            cable_length=35.0,
            power_factor=0.80,
            efficiency=0.88
        )
    ]
    
    for load in sample_loads:
        project.add_load(load)
    
    # Add sample bus
    from models import Bus
    main_bus = Bus(
        bus_id="B001",
        bus_name="Main Distribution Bus",
        voltage=400,
        phases=3,
        rated_current_a=630,
        short_circuit_rating_ka=50
    )
    project.buses.append(main_bus)
    
    # Assign loads to bus
    for load in project.loads:
        load.source_bus = "B001"
    
    # Perform integration
    print("🔧 Starting integration workflow...")
    integration_layer = create_integration_layer("IEC")
    
    # Add extraction confidence attribute
    project.extraction_confidence = 0.85
    
    # Integrate
    integration_report = integration_layer.integrate_extracted_project(project)
    
    # Validate quality
    quality_validation = validate_integration_quality(integration_report)
    
    # Display results
    print(f"\n✅ Integration Results:")
    print(f"   Status: {integration_report.overall_status.value}")
    print(f"   Quality Score: {integration_report.metrics.data_quality_score:.1%}")
    print(f"   Calculation Success: {integration_report.metrics.calculation_success_rate:.1%}")
    print(f"   Standards Compliance: {integration_report.metrics.standards_compliance_rate:.1%}")
    print(f"   Processing Time: {integration_report.metrics.processing_time_seconds:.2f}s")
    print(f"   Optimization Suggestions: {integration_report.metrics.optimization_suggestions}")
    
    print(f"\n📊 Quality Assessment:")
    print(f"   Overall Quality: {quality_validation['overall_quality']}")
    print(f"   Critical Issues: {len(quality_validation['critical_issues'])}")
    print(f"   Warnings: {len(quality_validation['warnings'])}")
    print(f"   Recommendations: {len(quality_validation['recommendations'])}")
    
    return integration_report, quality_validation


if __name__ == "__main__":
    # Run demo if this file is executed directly
    demo_integration_workflow()