"""
End-to-End Integration Testing Framework for AI Excel Extraction System

This module provides comprehensive testing capabilities for validating the complete
integration workflow from AI Excel extraction through calculations, standards compliance,
SLD generation, and optimization suggestions.

Key Features:
- Complete workflow testing from Excel to SLD generation
- Data consistency validation across all integration layers
- Performance benchmarking and monitoring
- Automated regression testing
- Quality assurance reporting
- Integration metrics tracking
"""

import unittest
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

# Import all integration components
from models import Project, Load, Cable, Bus, Transformer, LoadType, InstallationMethod, Priority
from calculations import ElectricalCalculationEngine
from standards import StandardsFactory
from ai_excel_extractor import AIExcelExtractor, ProcessingReport
from integration_layer import (
    ComprehensiveIntegrationLayer, IntegrationStatus, IntegrationMetrics,
    IntegrationReport, create_integration_layer, validate_integration_quality
)
from sld_data_preparation import SLDProcessor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    description: str
    excel_files: List[str]
    expected_results: Dict[str, Any]
    performance_targets: Dict[str, float]
    quality_thresholds: Dict[str, float]


@dataclass
class TestResult:
    """Test execution result"""
    scenario_name: str
    success: bool
    execution_time_seconds: float
    integration_report: IntegrationReport
    quality_validation: Dict[str, Any]
    performance_metrics: Dict[str, float]
    data_consistency_checks: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DataConsistencyValidator:
    """
    Validates data consistency across all integration layers
    Ensures AI-extracted data maintains integrity through the entire pipeline
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_end_to_end_consistency(self, project: Project, integration_report: IntegrationReport) -> Dict[str, Any]:
        """
        Validate data consistency from extraction through final integration
        
        Args:
            project: Final integrated project
            integration_report: Integration report with all validation results
            
        Returns:
            Consistency validation report
        """
        consistency_report = {
            "overall_consistent": True,
            "checks_performed": [],
            "consistency_issues": [],
            "data_flow_validation": {},
            "calculation_consistency": {},
            "standards_consistency": {}
        }
        
        try:
            # Check 1: Load data consistency
            load_consistency = self._validate_load_data_consistency(project)
            consistency_report["checks_performed"].append("load_data_consistency")
            consistency_report["load_consistency"] = load_consistency
            
            if not load_consistency["consistent"]:
                consistency_report["consistency_issues"].extend(load_consistency["issues"])
                consistency_report["overall_consistent"] = False
            
            # Check 2: Calculation consistency
            calc_consistency = self._validate_calculation_consistency(project, integration_report)
            consistency_report["checks_performed"].append("calculation_consistency")
            consistency_report["calculation_consistency"] = calc_consistency
            
            if not calc_consistency["consistent"]:
                consistency_report["consistency_issues"].extend(calc_consistency["issues"])
                consistency_report["overall_consistent"] = False
            
            # Check 3: Standards compliance consistency
            standards_consistency = self._validate_standards_consistency(project, integration_report)
            consistency_report["checks_performed"].append("standards_consistency")
            consistency_report["standards_consistency"] = standards_consistency
            
            if not standards_consistency["consistent"]:
                consistency_report["consistency_issues"].extend(standards_consistency["issues"])
                consistency_report["overall_consistent"] = False
            
            # Check 4: SLD integration consistency
            sld_consistency = self._validate_sld_consistency(project, integration_report)
            consistency_report["checks_performed"].append("sld_consistency")
            consistency_report["sld_consistency"] = sld_consistency
            
            if not sld_consistency["consistent"]:
                consistency_report["consistency_issues"].extend(sld_consistency["issues"])
                consistency_report["overall_consistent"] = False
            
            # Check 5: Cross-reference validation
            cross_ref_validation = self._validate_cross_references(project, integration_report)
            consistency_report["checks_performed"].append("cross_reference_validation")
            consistency_report["cross_reference_validation"] = cross_ref_validation
            
            if not cross_ref_validation["consistent"]:
                consistency_report["consistency_issues"].extend(cross_ref_validation["issues"])
                consistency_report["overall_consistent"] = False
            
            return consistency_report
            
        except Exception as e:
            error_msg = f"Consistency validation failed: {str(e)}"
            consistency_report["consistency_issues"].append(error_msg)
            consistency_report["overall_consistent"] = False
            self.logger.error(error_msg)
            return consistency_report

    def _validate_load_data_consistency(self, project: Project) -> Dict[str, Any]:
        """Validate load data consistency"""
        result = {
            "consistent": True,
            "issues": [],
            "checks": {}
        }
        
        # Check required fields
        for load in project.loads:
            if not all([load.load_id, load.load_name, load.power_kw, load.voltage, load.phases]):
                result["issues"].append(f"Load {load.load_id}: Missing required fields")
                result["consistent"] = False
        
        # Check electrical parameter consistency
        for load in project.loads:
            # Voltage consistency
            if load.voltage not in [230, 400, 415, 440, 690, 3300, 6600, 11000, 33000]:
                result["issues"].append(f"Load {load.load_id}: Non-standard voltage {load.voltage}V")
                result["consistent"] = False
            
            # Power factor reasonableness
            if load.power_factor and not (0.1 <= load.power_factor <= 1.0):
                result["issues"].append(f"Load {load.load_id}: Unrealistic power factor {load.power_factor}")
                result["consistent"] = False
            
            # Efficiency reasonableness
            if load.efficiency and not (0.1 <= load.efficiency <= 1.0):
                result["issues"].append(f"Load {load.load_id}: Unrealistic efficiency {load.efficiency}")
                result["consistent"] = False
        
        result["checks"]["required_fields"] = len([l for l in project.loads if all([l.load_id, l.load_name, l.power_kw, l.voltage, l.phases])])
        result["checks"]["total_loads"] = len(project.loads)
        
        return result

    def _validate_calculation_consistency(self, project: Project, integration_report: IntegrationReport) -> Dict[str, Any]:
        """Validate calculation consistency"""
        result = {
            "consistent": True,
            "issues": [],
            "checks": {}
        }
        
        calc_results = integration_report.calculation_results
        
        # Check if calculations were performed
        loads_with_calculations = sum(1 for load in project.loads if load.current_a is not None)
        result["checks"]["loads_with_calculations"] = loads_with_calculations
        result["checks"]["total_loads"] = len(project.loads)
        
        if loads_with_calculations < len(project.loads):
            missing_calc_loads = [load.load_id for load in project.loads if load.current_a is None]
            result["issues"].append(f"Missing calculations for loads: {missing_calc_loads}")
            result["consistent"] = False
        
        # Validate calculation logic
        for load in project.loads:
            if load.current_a and load.design_current_a:
                # Design current should be >= actual current
                if load.design_current_a < load.current_a:
                    result["issues"].append(f"Load {load.load_id}: Design current {load.design_current_a}A < actual current {load.current_a}A")
                    result["consistent"] = False
            
            # Check cable sizing if available
            if load.current_a and load.cable_size_sqmm:
                current_density = load.current_a / load.cable_size_sqmm
                if current_density > 10:  # Extremely high current density
                    result["issues"].append(f"Load {load.load_id}: Very high current density {current_density:.1f} A/mm²")
                    result["consistent"] = False
        
        return result

    def _validate_standards_consistency(self, project: Project, integration_report: IntegrationReport) -> Dict[str, Any]:
        """Validate standards compliance consistency"""
        result = {
            "consistent": True,
            "issues": [],
            "checks": {}
        }
        
        standards_validation = integration_report.standards_validation
        
        # Check voltage drop compliance
        voltage_drop_data = standards_validation.get("voltage_drop_compliance", {})
        violations = voltage_drop_data.get("violations", [])
        
        if violations:
            for violation in violations:
                result["issues"].append(
                    f"Voltage drop violation for {violation['load_id']}: {violation['voltage_drop_percent']:.1f}% > {violation['limit_percent']:.1f}%"
                )
            result["consistent"] = False
        
        # Check cable rating compliance
        rating_data = standards_validation.get("cable_ratings_compliance", {})
        rating_violations = rating_data.get("violations", [])
        
        if rating_violations:
            for violation in rating_violations:
                result["issues"].append(
                    f"Cable rating violation for {violation['load_id']}: {violation['current_a']:.1f}A > {violation['rating_a']:.1f}A"
                )
            result["consistent"] = False
        
        result["checks"]["voltage_drop_violations"] = len(violations)
        result["checks"]["rating_violations"] = len(rating_violations)
        result["checks"]["total_loads"] = len(project.loads)
        
        return result

    def _validate_sld_consistency(self, project: Project, integration_report: IntegrationReport) -> Dict[str, Any]:
        """Validate SLD integration consistency"""
        result = {
            "consistent": True,
            "issues": [],
            "checks": {}
        }
        
        sld_integration = integration_report.sld_integration
        
        # Check SLD integration status
        if sld_integration.get("integration_status") == "failed":
            result["issues"].append("SLD integration failed")
            result["consistent"] = False
        
        # Check hierarchy validation
        hierarchy_validation = sld_integration.get("hierarchy_validation", {})
        if not hierarchy_validation.get("valid", True):
            hierarchy_issues = hierarchy_validation.get("issues", [])
            result["issues"].extend([f"SLD hierarchy issue: {issue}" for issue in hierarchy_issues])
            result["consistent"] = False
        
        # Check connectivity validation
        connectivity_details = sld_integration.get("connectivity_details", {})
        if not connectivity_details.get("valid", True):
            connectivity_issues = connectivity_details.get("missing_connections", [])
            result["issues"].extend([f"SLD connectivity issue: {issue}" for issue in connectivity_issues])
            result["consistent"] = False
        
        result["checks"]["hierarchy_valid"] = hierarchy_validation.get("valid", False)
        result["checks"]["connectivity_valid"] = connectivity_details.get("valid", False)
        
        return result

    def _validate_cross_references(self, project: Project, integration_report: IntegrationReport) -> Dict[str, Any]:
        """Validate cross-references between components"""
        result = {
            "consistent": True,
            "issues": [],
            "checks": {}
        }
        
        # Check load-to-bus assignments
        assigned_loads = set()
        for load in project.loads:
            if load.source_bus:
                assigned_loads.add(load.load_id)
        
        orphaned_loads = set(load.load_id for load in project.loads) - assigned_loads
        if orphaned_loads:
            result["issues"].append(f"Loads without bus assignment: {orphaned_loads}")
            result["consistent"] = False
        
        # Check bus existence
        existing_bus_ids = set(bus.bus_id for bus in project.buses)
        for load in project.loads:
            if load.source_bus and load.source_bus not in existing_bus_ids:
                result["issues"].append(f"Load {load.load_id} references non-existent bus {load.source_bus}")
                result["consistent"] = False
        
        # Check calculation vs specification consistency
        for load in project.loads:
            if load.current_a and load.cable_size_sqmm and load.voltage_drop_percent:
                # Basic sanity check: higher power should generally mean higher current
                if load.power_kw > 0 and load.current_a < (load.power_kw * 1000) / (load.voltage * 1.732 * 0.85):
                    result["issues"].append(f"Load {load.load_id}: Current seems low for power rating")
                    result["consistent"] = False
        
        result["checks"]["assigned_loads"] = len(assigned_loads)
        result["checks"]["total_loads"] = len(project.loads)
        result["checks"]["existing_buses"] = len(existing_bus_ids)
        
        return result


class PerformanceBenchmark:
    """
    Performance benchmarking and monitoring for the integration system
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.benchmark_results = []

    def benchmark_integration_workflow(self, project: Project, integration_layer: ComprehensiveIntegrationLayer) -> Dict[str, Any]:
        """
        Benchmark the complete integration workflow
        
        Args:
            project: Project to benchmark
            integration_layer: Integration layer to test
            
        Returns:
            Performance benchmark results
        """
        benchmark_result = {
            "timestamp": datetime.now().isoformat(),
            "project_name": project.project_name,
            "total_components": len(project.loads) + len(project.cables) + len(project.buses),
            "workflow_timings": {},
            "throughput_metrics": {},
            "resource_usage": {},
            "bottlenecks": []
        }
        
        try:
            # Benchmark individual components
            # 1. Calculation Engine Integration
            start_time = time.time()
            enhanced_project, calc_report = integration_layer.calc_integration.integrate_project_calculations(project)
            calc_time = time.time() - start_time
            benchmark_result["workflow_timings"]["calculations"] = calc_time
            
            # 2. Standards Integration
            start_time = time.time()
            standards_report = integration_layer.standards_integration.validate_project_compliance(enhanced_project)
            standards_time = time.time() - start_time
            benchmark_result["workflow_timings"]["standards_validation"] = standards_time
            
            # 3. SLD Integration
            start_time = time.time()
            sld_project, sld_report = integration_layer.sld_integration.integrate_with_sld_system(enhanced_project)
            sld_time = time.time() - start_time
            benchmark_result["workflow_timings"]["sld_integration"] = sld_time
            
            # 4. Optimization Engine
            start_time = time.time()
            optimization_suggestions = integration_layer.optimization_engine.generate_optimization_suggestions(
                sld_project, calc_report, standards_report
            )
            optimization_time = time.time() - start_time
            benchmark_result["workflow_timings"]["optimization"] = optimization_time
            
            # 5. Overall Integration
            start_time = time.time()
            integration_report = integration_layer.integrate_extracted_project(project)
            total_time = time.time() - start_time
            benchmark_result["workflow_timings"]["total_integration"] = total_time
            
            # Calculate throughput metrics
            total_components = len(project.loads) + len(project.cables) + len(project.buses)
            benchmark_result["throughput_metrics"] = {
                "components_per_second": total_components / total_time,
                "calculations_per_second": len(project.loads) / calc_time,
                "standards_checks_per_second": len(project.loads) / standards_time,
                "sld_components_per_second": total_components / sld_time
            }
            
            # Performance targets comparison
            performance_targets = {
                "total_integration_time": 10.0,  # 10 seconds max
                "calculations_per_second": 10.0,  # 10 calculations per second min
                "throughput_components_per_second": 1.0  # 1 component per second min
            }
            
            for metric, target in performance_targets.items():
                if metric == "total_integration_time":
                    actual = total_time
                    benchmark_result["bottlenecks"].append(f"Slow integration: {actual:.2f}s > {target}s") if actual > target else None
                elif metric == "throughput_components_per_second":
                    actual = benchmark_result["throughput_metrics"]["components_per_second"]
                    benchmark_result["bottlenecks"].append(f"Low throughput: {actual:.2f} < {target}") if actual < target else None
            
            self.benchmark_results.append(benchmark_result)
            self.logger.info(f"Benchmark completed: {total_time:.2f}s for {total_components} components")
            
            return benchmark_result
            
        except Exception as e:
            error_msg = f"Benchmark failed: {str(e)}"
            benchmark_result["error"] = error_msg
            self.logger.error(error_msg)
            return benchmark_result


class IntegrationTestSuite:
    """
    Comprehensive test suite for validating the complete integration workflow
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.consistency_validator = DataConsistencyValidator()
        self.performance_benchmark = PerformanceBenchmark()
        self.test_results = []

    def run_complete_integration_test(self, test_scenarios: List[TestScenario]) -> Dict[str, Any]:
        """
        Run complete integration tests for all scenarios
        
        Args:
            test_scenarios: List of test scenarios to execute
            
        Returns:
            Complete test suite results
        """
        suite_results = {
            "test_suite_name": "AI Excel Extraction Integration Test Suite",
            "execution_timestamp": datetime.now().isoformat(),
            "total_scenarios": len(test_scenarios),
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "scenario_results": [],
            "overall_performance": {},
            "quality_summary": {},
            "recommendations": []
        }
        
        try:
            self.logger.info(f"Starting integration test suite with {len(test_scenarios)} scenarios")
            
            for scenario in test_scenarios:
                scenario_result = self._run_single_scenario_test(scenario)
                suite_results["scenario_results"].append(scenario_result)
                
                if scenario_result.success:
                    suite_results["passed_scenarios"] += 1
                else:
                    suite_results["failed_scenarios"] += 1
            
            # Calculate overall performance
            total_execution_time = sum(r.execution_time_seconds for r in self.test_results)
            avg_quality_score = sum(
                r.integration_report.metrics.data_quality_score 
                for r in self.test_results
            ) / len(self.test_results)
            
            suite_results["overall_performance"] = {
                "total_execution_time_seconds": total_execution_time,
                "average_quality_score": avg_quality_score,
                "average_calculation_success_rate": sum(
                    r.integration_report.metrics.calculation_success_rate 
                    for r in self.test_results
                ) / len(self.test_results),
                "average_standards_compliance_rate": sum(
                    r.integration_report.metrics.standards_compliance_rate 
                    for r in self.test_results
                ) / len(self.test_results)
            }
            
            # Quality summary
            total_issues = sum(len(r.errors) + len(r.warnings) for r in self.test_results)
            total_optimization_suggestions = sum(
                len(r.integration_report.optimization_suggestions) 
                for r in self.test_results
            )
            
            suite_results["quality_summary"] = {
                "total_issues_found": total_issues,
                "total_optimization_suggestions": total_optimization_suggestions,
                "quality_distribution": self._calculate_quality_distribution(),
                "consistency_rate": self._calculate_consistency_rate()
            }
            
            # Generate recommendations
            suite_results["recommendations"] = self._generate_test_recommendations(suite_results)
            
            self.logger.info(f"Test suite completed: {suite_results['passed_scenarios']}/{suite_results['total_scenarios']} passed")
            
            return suite_results
            
        except Exception as e:
            error_msg = f"Test suite execution failed: {str(e)}"
            suite_results["error"] = error_msg
            self.logger.error(error_msg)
            return suite_results

    def _run_single_scenario_test(self, scenario: TestScenario) -> TestResult:
        """Run a single test scenario"""
        self.logger.info(f"Running test scenario: {scenario.name}")
        
        result = TestResult(
            scenario_name=scenario.name,
            success=False,
            execution_time_seconds=0.0,
            integration_report=IntegrationReport(
                overall_status=IntegrationStatus.FAILED,
                metrics=IntegrationMetrics()
            ),
            quality_validation={},
            performance_metrics={},
            data_consistency_checks={}
        )
        
        try:
            start_time = time.time()
            
            # Create integration layer
            integration_layer = create_integration_layer("IEC")
            
            # Create test project (simplified for demo)
            test_project = self._create_test_project(scenario)
            
            # Add extraction confidence
            test_project.extraction_confidence = 0.85
            
            # Run integration
            integration_report = integration_layer.integrate_extracted_project(test_project)
            result.integration_report = integration_report
            
            # Validate integration quality
            quality_validation = validate_integration_quality(integration_report)
            result.quality_validation = quality_validation
            
            # Run data consistency validation
            consistency_report = self.consistency_validator.validate_end_to_end_consistency(
                test_project, integration_report
            )
            result.data_consistency_checks = consistency_report
            
            # Run performance benchmark
            performance_metrics = self.performance_benchmark.benchmark_integration_workflow(
                test_project, integration_layer
            )
            result.performance_metrics = performance_metrics
            
            # Evaluate success criteria
            result.success = self._evaluate_success_criteria(
                scenario, integration_report, quality_validation, consistency_report, performance_metrics
            )
            
            # Collect issues and warnings
            result.errors = integration_report.integration_issues
            if not consistency_report["overall_consistent"]:
                result.warnings.extend(consistency_report["consistency_issues"])
            
            result.execution_time_seconds = time.time() - start_time
            self.test_results.append(result)
            
            self.logger.info(f"Scenario {scenario.name}: {'PASSED' if result.success else 'FAILED'}")
            
            return result
            
        except Exception as e:
            error_msg = f"Scenario {scenario.name} failed: {str(e)}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
            return result

    def _create_test_project(self, scenario: TestScenario) -> Project:
        """Create a test project based on scenario"""
        project = Project(
            project_name=f"Test Project - {scenario.name}",
            standard="IEC",
            voltage_system="LV"
        )
        
        # Create sample loads for testing
        sample_loads = [
            Load(
                load_id="L001",
                load_name="Test Motor 1",
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
            ),
            Load(
                load_id="L003",
                load_name="Lighting Panel",
                power_kw=8.2,
                voltage=230,
                phases=1,
                load_type=LoadType.LIGHTING,
                installation_method=InstallationMethod.CONDUIT,
                priority=Priority.NON_ESSENTIAL,
                cable_length=25.0,
                power_factor=0.90,
                efficiency=0.95
            )
        ]
        
        for load in sample_loads:
            project.add_load(load)
        
        # Add sample bus
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
        
        return project

    def _evaluate_success_criteria(self, scenario: TestScenario, integration_report: IntegrationReport, 
                                 quality_validation: Dict, consistency_report: Dict, 
                                 performance_metrics: Dict) -> bool:
        """Evaluate if test scenario meets success criteria"""
        
        # Check quality thresholds
        quality_score = integration_report.metrics.data_quality_score
        calculation_success = integration_report.metrics.calculation_success_rate
        standards_compliance = integration_report.metrics.standards_compliance_rate
        
        min_quality = scenario.quality_thresholds.get("data_quality_score", 0.7)
        min_calculation_success = scenario.quality_thresholds.get("calculation_success_rate", 80.0)
        min_standards_compliance = scenario.quality_thresholds.get("standards_compliance_rate", 80.0)
        
        quality_passed = (quality_score >= min_quality and 
                         calculation_success >= min_calculation_success and 
                         standards_compliance >= min_standards_compliance)
        
        # Check consistency
        consistency_passed = consistency_report["overall_consistent"]
        
        # Check performance targets
        performance_passed = True
        total_time = performance_metrics.get("workflow_timings", {}).get("total_integration", 0)
        max_time = scenario.performance_targets.get("max_integration_time", 10.0)
        
        if total_time > max_time:
            performance_passed = False
        
        return quality_passed and consistency_passed and performance_passed

    def _calculate_quality_distribution(self) -> Dict[str, int]:
        """Calculate distribution of quality scores across tests"""
        distribution = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        
        for result in self.test_results:
            quality_score = result.integration_report.metrics.data_quality_score
            
            if quality_score >= 0.9:
                distribution["excellent"] += 1
            elif quality_score >= 0.8:
                distribution["good"] += 1
            elif quality_score >= 0.6:
                distribution["acceptable"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution

    def _calculate_consistency_rate(self) -> float:
        """Calculate overall data consistency rate"""
        if not self.test_results:
            return 0.0
        
        consistent_count = sum(
            1 for result in self.test_results 
            if result.data_consistency_checks.get("overall_consistent", False)
        )
        
        return consistent_count / len(self.test_results)

    def _generate_test_recommendations(self, suite_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check overall success rate
        success_rate = suite_results["passed_scenarios"] / suite_results["total_scenarios"]
        if success_rate < 0.8:
            recommendations.append("Overall success rate is low. Review integration logic and test scenarios.")
        
        # Check quality distribution
        quality_dist = suite_results["quality_summary"]["quality_distribution"]
        if quality_dist["poor"] > 0:
            recommendations.append("Some tests show poor quality. Investigate data extraction and validation issues.")
        
        # Check consistency rate
        consistency_rate = suite_results["quality_summary"]["consistency_rate"]
        if consistency_rate < 0.9:
            recommendations.append("Data consistency issues detected. Review cross-layer validation logic.")
        
        # Check performance bottlenecks
        performance = suite_results["overall_performance"]
        if performance.get("average_quality_score", 0) < 0.8:
            recommendations.append("Average quality score is below target. Improve data extraction accuracy.")
        
        return recommendations


# Test scenarios for demonstration
def create_demo_test_scenarios() -> List[TestScenario]:
    """Create demonstration test scenarios"""
    
    scenarios = [
        TestScenario(
            name="Basic Manufacturing Plant",
            description="Test integration with standard manufacturing plant electrical layout",
            excel_files=["Advanced_Manufacturing_Plant_-_Electrical_Distribution_LoadList.xlsx"],
            expected_results={
                "min_loads": 10,
                "expected_quality_score": 0.85
            },
            performance_targets={
                "max_integration_time": 5.0,
                "min_throughput": 2.0
            },
            quality_thresholds={
                "data_quality_score": 0.8,
                "calculation_success_rate": 85.0,
                "standards_compliance_rate": 80.0
            }
        ),
        TestScenario(
            name="Complex Distribution System",
            description="Test integration with complex multi-voltage distribution system",
            excel_files=["New_Electrical_Project_LoadList.xlsx"],
            expected_results={
                "min_loads": 5,
                "expected_quality_score": 0.75
            },
            performance_targets={
                "max_integration_time": 8.0,
                "min_throughput": 1.0
            },
            quality_thresholds={
                "data_quality_score": 0.7,
                "calculation_success_rate": 75.0,
                "standards_compliance_rate": 70.0
            }
        )
    ]
    
    return scenarios


# Main execution function
def run_integration_test_suite():
    """Run the complete integration test suite"""
    
    print("🚀 Starting AI Excel Extraction Integration Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = IntegrationTestSuite()
    
    # Create test scenarios
    scenarios = create_demo_test_scenarios()
    
    # Run complete test suite
    suite_results = test_suite.run_complete_integration_test(scenarios)
    
    # Display results
    print(f"\n📊 Test Suite Results:")
    print(f"   Total Scenarios: {suite_results['total_scenarios']}")
    print(f"   Passed: {suite_results['passed_scenarios']}")
    print(f"   Failed: {suite_results['failed_scenarios']}")
    print(f"   Success Rate: {(suite_results['passed_scenarios']/suite_results['total_scenarios']*100):.1f}%")
    
    # Overall performance
    perf = suite_results["overall_performance"]
    print(f"\n⚡ Performance Summary:")
    print(f"   Total Execution Time: {perf['total_execution_time_seconds']:.2f}s")
    print(f"   Average Quality Score: {perf['average_quality_score']:.1%}")
    print(f"   Average Calculation Success: {perf['average_calculation_success_rate']:.1f}%")
    print(f"   Average Standards Compliance: {perf['average_standards_compliance_rate']:.1f}%")
    
    # Quality distribution
    quality_dist = suite_results["quality_summary"]["quality_distribution"]
    print(f"\n🎯 Quality Distribution:")
    for quality, count in quality_dist.items():
        print(f"   {quality.title()}: {count}")
    
    # Consistency rate
    consistency_rate = suite_results["quality_summary"]["consistency_rate"]
    print(f"\n🔗 Data Consistency Rate: {consistency_rate:.1%}")
    
    # Recommendations
    if suite_results["recommendations"]:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(suite_results["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    # Save detailed results
    results_file = f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(suite_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return suite_results


if __name__ == "__main__":
    # Run integration test suite
    run_integration_test_suite()