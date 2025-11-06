"""
Unified Data Processing Pipeline for Electrical Design Automation

This module provides a comprehensive pipeline that orchestrates AI-powered Excel extraction
with existing manual data entry workflows, validation, and integration with the calculation engine.
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import io
from dataclasses import asdict

# Import existing systems
from models import Project, Load, Cable, Bus, Transformer, LoadType, InstallationMethod, Priority
from ai_excel_extractor import AIExcelExtractor, ProcessingReport, ExtractionResult
from calculations import ElectricalCalculationEngine
from standards import StandardsFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus:
    """Status tracking for the unified processing pipeline"""
    
    def __init__(self):
        self.current_step = ""
        self.progress_percent = 0
        self.status_message = ""
        self.is_complete = False
        self.extraction_report = None
        self.validation_results = {}
        self.corrections_needed = []
        self.final_project = None
        
    def update_progress(self, step: str, progress: int, message: str = ""):
        """Update processing progress"""
        self.current_step = step
        self.progress_percent = progress
        self.status_message = message
        logger.info(f"Pipeline Progress: {step} - {progress}% - {message}")
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for session state storage"""
        return {
            'current_step': self.current_step,
            'progress_percent': self.progress_percent,
            'status_message': self.status_message,
            'is_complete': self.is_complete,
            'has_extraction_report': self.extraction_report is not None,
            'validation_results': self.validation_results,
            'corrections_needed': self.corrections_needed,
            'has_final_project': self.final_project is not None
        }


class UnifiedDataProcessor:
    """
    Main orchestrator for unified data processing pipeline
    
    Coordinates AI Excel extraction with manual workflows, validation, and integration
    """
    
    def __init__(self, standard: str = "IEC"):
        self.standard = standard
        self.ai_extractor = AIExcelExtractor(standard)
        self.calc_engine = ElectricalCalculationEngine(standard)
        self.standards = StandardsFactory.get_standard(standard)
        
        # Initialize processing status
        self.status = ProcessingStatus()
        
    def process_excel_upload(self, uploaded_file, project_name: str = "AI Extracted Project") -> Tuple[bool, str, Optional[Project]]:
        """
        Process Excel file upload through AI extraction pipeline with comprehensive error handling
        
        Args:
            uploaded_file: Streamlit uploaded file object
            project_name: Name for the extracted project
            
        Returns:
            Tuple of (success, message, project)
        """
        try:
            # Initialize status
            self.status.update_progress("Initializing", 5, "Setting up AI extraction pipeline")
            
            # Validate uploaded file
            if not self._validate_uploaded_file(uploaded_file):
                return False, "Invalid file format. Please upload a valid Excel file (.xlsx or .xls)", None
            
            # Save uploaded file temporarily
            try:
                with st.spinner("Preparing Excel file for processing..."):
                    file_path = self._save_uploaded_file(uploaded_file)
                
                self.status.update_progress("File Preparation", 10, "Excel file saved and validated")
            except Exception as e:
                error_msg = f"Failed to process uploaded file: {str(e)}"
                logger.error(error_msg)
                self.status.update_progress("File Error", 0, error_msg)
                return False, error_msg, None
            
            # Start AI extraction process
            try:
                with st.spinner("🤖 AI extraction in progress..."):
                    self.status.update_progress("AI Extraction", 20, "Analyzing Excel structure and content")
                    
                    # Perform AI extraction
                    extraction_report = self.ai_extractor.process_excel_file(file_path)
                    
                    # Check if extraction was successful
                    if not extraction_report or extraction_report.total_components == 0:
                        return False, "No electrical components could be extracted from the Excel file. Please check the file format and content.", None
                    
                    self.status.extraction_report = extraction_report
                    self.status.update_progress("AI Extraction", 40, f"Extracted {extraction_report.total_components} components")
                    
            except Exception as e:
                error_msg = f"AI extraction failed: {str(e)}"
                logger.error(error_msg)
                self.status.update_progress("Extraction Error", 0, error_msg)
                # Continue with fallback processing if possible
                return False, f"AI extraction failed: {error_msg}. Please check your Excel file format.", None
            
            # Validate extracted data
            try:
                with st.spinner("Validating extracted data..."):
                    self.status.update_progress("Data Validation", 60, "Validating against electrical engineering rules")
                    
                    validation_results = self._validate_extracted_data(extraction_report)
                    self.status.validation_results = validation_results
                    
            except Exception as e:
                warning_msg = f"Validation failed: {str(e)}. Continuing with basic validation."
                logger.warning(warning_msg)
                self.status.validation_results = {'errors': [], 'warnings': [warning_msg], 'recommendations': []}
            
            # Enhance and correct data
            try:
                with st.spinner("Enhancing data quality..."):
                    self.status.update_progress("Data Enhancement", 70, "Auto-correcting issues and standardizing")
                    
                    enhanced_project = self._enhance_extracted_project(extraction_report, validation_results)
                    
            except Exception as e:
                warning_msg = f"Data enhancement failed: {str(e)}. Using basic extracted data."
                logger.warning(warning_msg)
                enhanced_project = extraction_report.project_data
                if not enhanced_project:
                    return False, "Failed to create project from extracted data", None
            
            # Calculate electrical parameters
            try:
                with st.spinner("Performing electrical calculations..."):
                    self.status.update_progress("Electrical Calculations", 80, "Calculating currents, cable sizes, and protection")
                    
                    final_project = self._calculate_project_parameters(enhanced_project)
                    
            except Exception as e:
                warning_msg = f"Electrical calculations failed: {str(e)}. Basic project created without calculations."
                logger.warning(warning_msg)
                final_project = enhanced_project
                # Add warning but continue
            
            # Finalize
            try:
                self.status.update_progress("Finalization", 90, "Creating project and generating reports")
                
                # Set project name
                final_project.project_name = project_name
                final_project.created_date = datetime.now().isoformat()
                
                self.status.final_project = final_project
                self.status.update_progress("Complete", 100, "AI extraction completed successfully")
                self.status.is_complete = True
                
                # Store in session state
                st.session_state.unified_processing_status = self.status.to_dict()
                st.session_state.project = final_project
                
            except Exception as e:
                error_msg = f"Finalization failed: {str(e)}"
                logger.error(error_msg)
                return False, f"Project creation failed: {error_msg}", None
            
            # Success summary
            summary_msg = f"Successfully extracted {extraction_report.total_components} components"
            if extraction_report.total_components > 0:
                summary_msg += f" with {extraction_report.overall_confidence:.1%} confidence"
            
            return True, summary_msg, final_project
            
        except Exception as e:
            error_msg = f"Unexpected error during AI extraction: {str(e)}"
            logger.error(error_msg)
            self.status.update_progress("System Error", 0, error_msg)
            return False, f"System error: {error_msg}. Please try again or contact support.", None
    
    def _validate_uploaded_file(self, uploaded_file) -> bool:
        """Validate uploaded file before processing"""
        if not uploaded_file:
            return False
        
        # Check file extension
        allowed_extensions = ['.xlsx', '.xls']
        file_name = uploaded_file.name.lower()
        
        if not any(file_name.endswith(ext) for ext in allowed_extensions):
            return False
        
        # Check file size (max 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
            return False
        
        # Check if file is readable
        try:
            # Try to read a small sample
            sample_data = uploaded_file.read(1024)  # Read first 1KB
            uploaded_file.seek(0)  # Reset file pointer
            return True
        except Exception:
            return False
    
    def process_manual_workflow(self, existing_project: Project) -> Tuple[bool, str, Project]:
        """
        Process existing project through manual workflow enhancement
        
        Args:
            existing_project: Project from manual data entry
            
        Returns:
            Tuple of (success, message, enhanced project)
        """
        try:
            self.status.update_progress("Manual Processing", 20, "Processing manually entered project data")
            
            # Validate existing project
            validation_results = self._validate_existing_project(existing_project)
            self.status.validation_results = validation_results
            
            self.status.update_progress("Manual Processing", 40, "Validating project consistency")
            
            # Enhance existing project
            enhanced_project = self._enhance_existing_project(existing_project)
            
            self.status.update_progress("Manual Processing", 60, "Auto-enhancing project data")
            
            # Calculate electrical parameters
            final_project = self._calculate_project_parameters(enhanced_project)
            
            self.status.update_progress("Manual Processing", 80, "Performing electrical calculations")
            
            # Finalize
            self.status.final_project = final_project
            self.status.update_progress("Manual Processing", 100, "Manual workflow completed")
            self.status.is_complete = True
            
            # Store in session state
            st.session_state.unified_processing_status = self.status.to_dict()
            st.session_state.project = final_project
            
            return True, "Manual workflow enhanced successfully", final_project
            
        except Exception as e:
            error_msg = f"Manual processing failed: {str(e)}"
            logger.error(error_msg)
            self.status.update_progress("Error", 0, error_msg)
            return False, error_msg, None
    
    def get_correction_interface_data(self) -> Dict[str, Any]:
        """
        Get data for manual correction interface
        
        Returns:
            Dictionary with correction interface data
        """
        if not self.status.extraction_report:
            return {}
        
        correction_data = {
            'low_confidence_items': [],
            'validation_issues': self.status.validation_results.get('errors', []) + self.status.validation_results.get('warnings', []),
            'extraction_confidence': self.status.extraction_report.overall_confidence,
            'sheet_results': {}
        }
        
        # Get low confidence items from extraction
        for sheet_name, result in self.status.extraction_report.sheet_results.items():
            if result.confidence < 0.8 or result.data_quality_score < 0.7:
                correction_data['low_confidence_items'].append({
                    'sheet_name': sheet_name,
                    'confidence': result.confidence,
                    'quality_score': result.data_quality_score,
                    'issues': result.issues
                })
            
            correction_data['sheet_results'][sheet_name] = {
                'confidence': result.confidence,
                'components_extracted': result.components_extracted,
                'data_quality_score': result.data_quality_score
            }
        
        # Get validation issues
        correction_data['validation_issues'] = self.status.validation_results.get('errors', []) + self.status.validation_results.get('warnings', [])
        
        return correction_data
    
    def apply_manual_corrections(self, corrections: Dict[str, Any]) -> Tuple[bool, str, Project]:
        """
        Apply manual corrections to the extracted project
        
        Args:
            corrections: Dictionary with manual corrections
            
        Returns:
            Tuple of (success, message, corrected project)
        """
        try:
            if not self.status.final_project:
                return False, "No project available for correction", None
            
            # Apply corrections to project
            corrected_project = self._apply_corrections_to_project(self.status.final_project, corrections)
            
            # Recalculate with corrections
            final_project = self._calculate_project_parameters(corrected_project)
            
            # Update status
            self.status.final_project = final_project
            self.status.is_complete = True
            
            # Store in session state
            st.session_state.unified_processing_status = self.status.to_dict()
            st.session_state.project = final_project
            
            return True, "Manual corrections applied successfully", final_project
            
        except Exception as e:
            error_msg = f"Failed to apply corrections: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def get_processing_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data for results dashboard
        
        Returns:
            Dictionary with dashboard data
        """
        if not self.status.final_project:
            return {}
        
        dashboard_data = {
            'project_summary': self._get_project_summary(self.status.final_project),
            'extraction_metrics': self._get_extraction_metrics(),
            'validation_status': self.status.validation_results,
            'confidence_breakdown': self._get_confidence_breakdown(),
            'component_statistics': self._get_component_statistics(),
            'processing_history': self._get_processing_history()
        }
        
        return dashboard_data
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file temporarily for processing"""
        file_bytes = uploaded_file.getvalue()
        file_path = f"temp_{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        
        return file_path
    
    def _validate_extracted_data(self, extraction_report: ProcessingReport) -> Dict[str, Any]:
        """Validate extracted data against electrical engineering rules"""
        if not extraction_report.project_data:
            return {'errors': ['No project data extracted'], 'warnings': [], 'recommendations': []}
        
        # Use the AI extractor's validation engine
        return self.ai_extractor.validation_engine.validate_project(extraction_report.project_data)
    
    def _enhance_extracted_project(self, extraction_report: ProcessingReport, validation_results: Dict) -> Project:
        """Enhance extracted project data"""
        if not extraction_report.project_data:
            raise ValueError("No project data available for enhancement")
        
        project = extraction_report.project_data
        
        # Use AI extractor's enhancement capabilities
        enhancement_results = self.ai_extractor.data_enhancer.enhance_project_data(project, list(extraction_report.sheet_results.values()))
        
        # Apply any corrections made during enhancement
        if enhancement_results['corrections_made']:
            logger.info(f"Applied {len(enhancement_results['corrections_made'])} auto-corrections")
        
        return enhancement_results['final_project']
    
    def _calculate_project_parameters(self, project: Project) -> Project:
        """Calculate electrical parameters for the project"""
        # Calculate loads
        for load in project.loads:
            try:
                calculated_load = self.calc_engine.calculate_load(load)
                # Update the load in the project
                idx = project.loads.index(load)
                project.loads[idx] = calculated_load
            except Exception as e:
                logger.warning(f"Error calculating load {load.load_id}: {e}")
        
        # Calculate bus loads
        for bus in project.buses:
            total_load = bus.calculate_total_load(project.loads)
            demand_load = total_load * bus.diversity_factor
            bus.demand_kw = demand_load
            bus.demand_kva = demand_load / 0.85  # Assuming average PF
        
        # Calculate project totals
        project.total_installed_capacity_kw = sum(load.power_kw for load in project.loads)
        project.total_demand_kw = sum(bus.demand_kw for bus in project.buses if bus.demand_kw)
        project.system_diversity_factor = project.total_demand_kw / project.total_installed_capacity_kw if project.total_installed_capacity_kw > 0 else 1.0
        
        return project
    
    def _validate_existing_project(self, project: Project) -> Dict[str, Any]:
        """Validate existing project data"""
        # Use the AI extractor's validation engine
        return self.ai_extractor.validation_engine.validate_project(project)
    
    def _enhance_existing_project(self, project: Project) -> Project:
        """Enhance existing project data"""
        # Apply enhancement without AI extraction
        enhancement_results = self.ai_extractor.data_enhancer.enhance_project_data(project, [])
        return enhancement_results['final_project']
    
    def _apply_corrections_to_project(self, project: Project, corrections: Dict[str, Any]) -> Project:
        """Apply manual corrections to project"""
        corrected_project = project
        
        # Apply load corrections
        if 'load_corrections' in corrections:
            for load_correction in corrections['load_corrections']:
                load_id = load_correction['load_id']
                load = next((l for l in corrected_project.loads if l.load_id == load_id), None)
                if load:
                    # Apply corrections
                    for field, value in load_correction['corrections'].items():
                        if hasattr(load, field):
                            setattr(load, field, value)
        
        # Apply cable corrections
        if 'cable_corrections' in corrections:
            for cable_correction in corrections['cable_corrections']:
                cable_id = cable_correction['cable_id']
                cable = next((c for c in corrected_project.cables if c.cable_id == cable_id), None)
                if cable:
                    # Apply corrections
                    for field, value in cable_correction['corrections'].items():
                        if hasattr(cable, field):
                            setattr(cable, field, value)
        
        return corrected_project
    
    def _get_project_summary(self, project: Project) -> Dict[str, Any]:
        """Get project summary data"""
        return {
            'project_name': project.project_name,
            'standard': project.standard,
            'total_loads': len(project.loads),
            'total_cables': len(project.cables),
            'total_buses': len(project.buses),
            'total_transformers': len(project.transformers),
            'total_power_kw': project.total_installed_capacity_kw or 0,
            'total_demand_kw': project.total_demand_kw or 0,
            'system_diversity_factor': project.system_diversity_factor or 0,
            'main_transformer_kva': project.main_transformer_rating_kva or 0
        }
    
    def _get_extraction_metrics(self) -> Dict[str, Any]:
        """Get extraction-specific metrics"""
        if not self.status.extraction_report:
            return {}
        
        return {
            'overall_confidence': self.status.extraction_report.overall_confidence,
            'total_components': self.status.extraction_report.total_components,
            'processing_time_seconds': self.status.extraction_report.processing_time_seconds,
            'corrections_made': len(self.status.extraction_report.corrections_made),
            'sheet_results': {
                name: {
                    'confidence': result.confidence,
                    'components_extracted': result.components_extracted,
                    'data_quality_score': result.data_quality_score
                }
                for name, result in self.status.extraction_report.sheet_results.items()
            }
        }
    
    def _get_confidence_breakdown(self) -> Dict[str, float]:
        """Get confidence breakdown by component type"""
        if not self.status.extraction_report:
            return {}
        
        breakdown = {}
        for sheet_name, result in self.status.extraction_report.sheet_results.items():
            breakdown[sheet_name] = result.confidence
        
        return breakdown
    
    def _get_component_statistics(self) -> Dict[str, Any]:
        """Get component statistics"""
        if not self.status.final_project:
            return {}
        
        project = self.status.final_project
        
        # Load statistics
        load_stats = {
            'total_power_kw': sum(load.power_kw for load in project.loads),
            'average_power_kw': sum(load.power_kw for load in project.loads) / len(project.loads) if project.loads else 0,
            'load_types': {}
        }
        
        for load in project.loads:
            load_type = load.load_type.value
            if load_type not in load_stats['load_types']:
                load_stats['load_types'][load_type] = 0
            load_stats['load_types'][load_type] += 1
        
        # Cable statistics
        cable_stats = {
            'total_length_m': sum(cable.length_m for cable in project.cables),
            'average_length_m': sum(cable.length_m for cable in project.cables) / len(project.cables) if project.cables else 0,
            'size_distribution': {}
        }
        
        for cable in project.cables:
            size = cable.size_sqmm
            if size not in cable_stats['size_distribution']:
                cable_stats['size_distribution'][size] = 0
            cable_stats['size_distribution'][size] += 1
        
        return {
            'loads': load_stats,
            'cables': cable_stats,
            'buses': len(project.buses),
            'transformers': len(project.transformers)
        }
    
    def _get_processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history log"""
        history = []
        
        # Add extraction details if available
        if self.status.extraction_report:
            history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'AI Extraction Completed',
                'details': f"Extracted {self.status.extraction_report.total_components} components with {self.status.extraction_report.overall_confidence:.1%} confidence"
            })
        
        # Add validation details
        if self.status.validation_results:
            error_count = len(self.status.validation_results.get('errors', []))
            warning_count = len(self.status.validation_results.get('warnings', []))
            
            if error_count > 0 or warning_count > 0:
                history.append({
                    'timestamp': datetime.now().isoformat(),
                    'event': 'Data Validation',
                    'details': f"Found {error_count} errors and {warning_count} warnings"
                })
        
        # Add enhancement details
        if self.status.extraction_report and self.status.extraction_report.corrections_made:
            history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'Auto-Enhancement',
                'details': f"Applied {len(self.status.extraction_report.corrections_made)} automatic corrections"
            })
        
        return history


class ProcessingInterface:
    """
    User interface components for the unified processing pipeline
    """
    
    @staticmethod
    def render_processing_status(status: ProcessingStatus):
        """Render processing status display"""
        if status.progress_percent > 0:
            st.progress(status.progress_percent / 100)
            
            # Status information
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Current Step:** {status.current_step}")
                if status.status_message:
                    st.write(f"**Status:** {status.status_message}")
            
            with col2:
                if status.is_complete:
                    st.success("✅ Complete")
                elif status.progress_percent > 0:
                    st.info(f"{status.progress_percent}% Complete")
                else:
                    st.warning("⏳ Pending")
        
        # Show extraction report if available
        if status.extraction_report and status.is_complete:
            with st.expander("📊 Extraction Report", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Confidence", f"{status.extraction_report.overall_confidence:.1%}")
                
                with col2:
                    st.metric("Components Extracted", status.extraction_report.total_components)
                
                with col3:
                    st.metric("Processing Time", f"{status.extraction_report.processing_time_seconds:.1f}s")
                
                # Sheet results
                st.subheader("Sheet Results")
                for sheet_name, result in status.extraction_report.sheet_results.items():
                    st.write(f"**{sheet_name}**: {result.sheet_type} - {result.components_extracted} components ({result.confidence:.1%} confidence)")
    
    @staticmethod
    def render_confidence_visualization(extraction_report: ProcessingReport):
        """Render confidence score visualization"""
        if not extraction_report:
            return
        
        st.markdown("### 🎯 Confidence Analysis")
        
        # Overall confidence gauge
        col1, col2 = st.columns([1, 2])
        
        with col1:
            confidence_percent = extraction_report.overall_confidence * 100
            
            if confidence_percent >= 80:
                confidence_color = "🟢"
                confidence_status = "High"
            elif confidence_percent >= 60:
                confidence_color = "🟡"
                confidence_status = "Medium"
            else:
                confidence_color = "🔴"
                confidence_status = "Low"
            
            st.metric(
                label="Overall Confidence",
                value=f"{confidence_percent:.1f}%",
                delta=confidence_status
            )
        
        with col2:
            # Confidence breakdown by sheet
            sheet_data = []
            for sheet_name, result in extraction_report.sheet_results.items():
                sheet_data.append({
                    'Sheet': sheet_name,
                    'Confidence': result.confidence * 100,
                    'Components': result.components_extracted,
                    'Quality Score': result.data_quality_score * 100
                })
            
            if sheet_data:
                df_confidence = pd.DataFrame(sheet_data)
                st.dataframe(df_confidence, use_container_width=True)
        
        # Confidence interpretation
        st.markdown("#### Confidence Interpretation:")
        if confidence_percent >= 80:
            st.success("✅ **High Confidence**: Data extraction is reliable. Minimal manual review required.")
        elif confidence_percent >= 60:
            st.warning("⚠️ **Medium Confidence**: Data extraction is good. Some manual review recommended.")
        else:
            st.error("❌ **Low Confidence**: Significant manual review and correction required.")
    
    @staticmethod
    def render_validation_issues(validation_results: Dict[str, List[str]]):
        """Render validation issues for user review"""
        if not validation_results:
            return
        
        errors = validation_results.get('errors', [])
        warnings = validation_results.get('warnings', [])
        recommendations = validation_results.get('recommendations', [])
        
        if errors:
            st.error("🚫 **Validation Errors**")
            for error in errors:
                st.error(f"• {error}")
        
        if warnings:
            st.warning("⚠️ **Validation Warnings**")
            for warning in warnings:
                st.warning(f"• {warning}")
        
        if recommendations:
            st.info("💡 **Recommendations**")
            for rec in recommendations:
                st.info(f"• {rec}")
        
        if not errors and not warnings:
            st.success("✅ All validation checks passed!")
    
    @staticmethod
    def render_correction_interface(correction_data: Dict[str, Any]):
        """Render manual correction interface"""
        if not correction_data:
            return
        
        st.markdown("### ✏️ Manual Correction Interface")
        
        # Low confidence items
        low_confidence_items = correction_data.get('low_confidence_items', [])
        if low_confidence_items:
            st.warning("⚠️ **Items Requiring Manual Review**")
            
            for item in low_confidence_items:
                with st.expander(f"Sheet: {item['sheet_name']} (Confidence: {item['confidence']:.1%})"):
                    st.write(f"**Quality Score:** {item['quality_score']:.1%}")
                    if item['issues']:
                        st.write("**Issues:**")
                        for issue in item['issues']:
                            st.write(f"• {issue}")
                    
                    # Add correction form for this sheet
                    st.text_area(
                        f"Corrections for {item['sheet_name']}",
                        placeholder="Enter corrections or notes for this sheet...",
                        key=f"correction_{item['sheet_name']}"
                    )
        
        # General corrections
        st.markdown("#### General Corrections")
        
        correction_form = st.form("manual_corrections")
        
        with correction_form:
            st.text_area(
                "Load Corrections",
                placeholder="Enter load corrections in JSON format, e.g., {\"L001\": {\"power_kw\": 15.5, \"voltage\": 415}}",
                help="Provide corrections for individual loads in JSON format"
            )
            
            st.text_area(
                "Cable Corrections", 
                placeholder="Enter cable corrections in JSON format...",
                help="Provide corrections for individual cables in JSON format"
            )
            
            st.text_area(
                "General Notes",
                placeholder="Enter any general notes or corrections..."
            )
            
            submitted = st.form_submit_button("Apply Corrections")
        
        if submitted:
            # Process corrections will be handled by the main processor
            st.success("Corrections submitted for processing!")
            st.rerun()


# Factory function for easy integration
def create_unified_processor(standard: str = "IEC") -> UnifiedDataProcessor:
    """
    Factory function to create a unified data processor
    
    Args:
        standard: Electrical standard to use (IEC, IS, NEC)
        
    Returns:
        UnifiedDataProcessor instance
    """
    return UnifiedDataProcessor(standard)


# Convenience functions for Streamlit integration
def initialize_processing_status():
    """Initialize processing status in session state"""
    if 'unified_processing_status' not in st.session_state:
        st.session_state.unified_processing_status = ProcessingStatus().to_dict()

def get_processing_status() -> Optional[ProcessingStatus]:
    """Get current processing status from session state"""
    if 'unified_processing_status' in st.session_state:
        status_dict = st.session_state.unified_processing_status
        status = ProcessingStatus()
        status.current_step = status_dict.get('current_step', '')
        status.progress_percent = status_dict.get('progress_percent', 0)
        status.status_message = status_dict.get('status_message', '')
        status.is_complete = status_dict.get('is_complete', False)
        status.validation_results = status_dict.get('validation_results', {})
        status.corrections_needed = status_dict.get('corrections_needed', [])
        return status
    return None