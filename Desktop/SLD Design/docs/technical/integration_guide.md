# Integration Guide - AI Excel Extraction System

## Overview

This guide provides comprehensive instructions for integrating the AI Excel Extraction System into existing applications, workflows, and enterprise environments.

## Table of Contents

1. [Integration Patterns](#integration-patterns)
2. [Web Application Integration](#web-application-integration)
3. [REST API Integration](#rest-api-integration)
4. [Database Integration](#database-integration)
5. [Workflow Automation](#workflow-automation)
6. [Enterprise Integration](#enterprise-integration)
7. [Security Considerations](#security-considerations)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)

---

## Integration Patterns

### Direct API Integration
```python
# Direct integration with Python applications
from ai_excel_extractor import AIExcelExtractor
import pandas as pd

class IntegratedElectricalSystem:
    def __init__(self):
        self.ai_extractor = AIExcelExtractor(standard="IEC")
        self.calculation_engine = ElectricalCalculationEngine()
        self.project_manager = ProjectManager()
    
    def process_electrical_project(self, excel_file_path: str):
        """Complete electrical project processing"""
        
        # Step 1: AI extraction
        extraction_report = self.ai_extractor.process_excel_file(excel_file_path)
        
        if extraction_report.overall_confidence > 0.7:
            project = extraction_report.project_data
            
            # Step 2: Enhanced calculations
            for load in project.loads:
                self.calculation_engine.calculate_load(load)
                self.calculation_engine.calculate_cable_for_load(load)
            
            # Step 3: System validation
            validation_results = self.calculation_engine.validate_project_data(project)
            
            # Step 4: Project management
            saved_project = self.project_manager.save_project(project)
            
            return {
                'success': True,
                'project_id': saved_project.id,
                'confidence': extraction_report.overall_confidence,
                'validation': validation_results
            }
        else:
            return {
                'success': False,
                'reason': 'Low extraction confidence',
                'confidence': extraction_report.overall_confidence
            }

# Usage
electrical_system = IntegratedElectricalSystem()
result = electrical_system.process_electrical_project("project.xlsx")
```

### Plugin Architecture
```python
# Plugin-based integration for existing systems
class AIExtractionPlugin:
    """Base plugin for AI extraction integration"""
    
    def __init__(self, config: dict):
        self.config = config
        self.extractor = AIExcelExtractor(**config)
    
    def get_name(self) -> str:
        return "AI Excel Extraction Plugin"
    
    def process_file(self, file_path: str) -> dict:
        """Main processing method - override in subclasses"""
        raise NotImplementedError
    
    def validate_config(self) -> bool:
        """Validate plugin configuration"""
        return True

class StreamlitPlugin(AIExtractionPlugin):
    """Plugin for Streamlit applications"""
    
    def process_file(self, file_path: str) -> dict:
        report = self.extractor.process_excel_file(file_path)
        
        return {
            'confidence': report.overall_confidence,
            'components': report.total_components,
            'project': report.project_data,
            'corrections': report.corrections_made
        }

class FastAPIPlugin(AIExtractionPlugin):
    """Plugin for FastAPI applications"""
    
    def process_file(self, file_path: str) -> dict:
        try:
            report = self.extractor.process_excel_file(file_path)
            return {
                'status': 'success',
                'data': {
                    'confidence': report.overall_confidence,
                    'project_name': report.project_data.project_name if report.project_data else None,
                    'loads_count': len(report.project_data.loads) if report.project_data else 0,
                    'cables_count': len(report.project_data.cables) if report.project_data else 0
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

# Plugin registration
class PluginRegistry:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name: str, plugin: AIExtractionPlugin):
        self.plugins[name] = plugin
    
    def get_plugin(self, name: str) -> AIExtractionPlugin:
        return self.plugins.get(name)

registry = PluginRegistry()
registry.register_plugin("streamlit", StreamlitPlugin(config))
registry.register_plugin("fastapi", FastAPIPlugin(config))
```

---

## Web Application Integration

### Streamlit Integration
```python
# Complete Streamlit application integration
import streamlit as st
from ai_excel_extractor import AIExcelExtractor
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class AIExtractionStreamlitApp:
    def __init__(self):
        self.extractor = AIExcelExtractor()
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="AI Excel Extraction",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_sidebar(self):
        """Render configuration sidebar"""
        st.sidebar.header("🤖 AI Configuration")
        
        standard = st.sidebar.selectbox(
            "Electrical Standard",
            ["IEC", "IS", "NEC"],
            index=0
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
        
        auto_corrections = st.sidebar.checkbox(
            "Enable Auto-Corrections",
            value=True
        )
        
        return {
            'standard': standard,
            'confidence_threshold': confidence_threshold,
            'auto_corrections': auto_corrections
        }
    
    def render_main_page(self, config):
        """Render main application page"""
        st.title("🤖 AI-Powered Excel Extraction")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload Excel file containing electrical distribution data"
        )
        
        if uploaded_file:
            # Save uploaded file
            with open("temp_upload.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process with AI
            with st.spinner("Processing with AI..."):
                report = self.extractor.process_excel_file("temp_upload.xlsx")
            
            # Display results
            self.render_results_dashboard(report)
            
            # Render detailed views
            self.render_detailed_views(report)
    
    def render_results_dashboard(self, report: ProcessingReport):
        """Render results dashboard"""
        st.subheader("📊 Extraction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Confidence",
                f"{report.overall_confidence:.1%}",
                delta=f"{report.overall_confidence-0.5:+.1%}" if report.overall_confidence > 0.5 else None
            )
        
        with col2:
            st.metric(
                "Components Extracted",
                report.total_components,
                delta=report.total_components if report.total_components > 0 else None
            )
        
        with col3:
            st.metric(
                "Processing Time",
                f"{report.processing_time_seconds:.2f}s"
            )
        
        with col4:
            st.metric(
                "Corrections Made",
                len(report.corrections_made),
                delta=len(report.corrections_made) if report.corrections_made else None
            )
        
        # Confidence visualization
        if report.sheet_results:
            confidence_data = {
                'Sheet': list(report.sheet_results.keys()),
                'Confidence': [result.confidence for result in report.sheet_results.values()]
            }
            
            fig = px.bar(
                confidence_data,
                x='Sheet',
                y='Confidence',
                title="Sheet Processing Confidence",
                labels={'Confidence': 'Confidence Score'},
                color='Confidence',
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                         annotation_text="Confidence Threshold")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_views(self, report: ProcessingReport):
        """Render detailed component views"""
        st.markdown("---")
        
        if report.project_data:
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔌 Loads", "🔌 Cables", "⚡ Buses", "📋 Validation"
            ])
            
            with tab1:
                self.render_loads_view(report.project_data.loads)
            
            with tab2:
                self.render_cables_view(report.project_data.cables)
            
            with tab3:
                self.render_buses_view(report.project_data.buses)
            
            with tab4:
                self.render_validation_view(report)
    
    def render_loads_view(self, loads):
        """Render loads data table"""
        if loads:
            load_data = []
            for load in loads:
                load_data.append({
                    'Load ID': load.load_id,
                    'Name': load.load_name,
                    'Power (kW)': load.power_kw,
                    'Voltage (V)': load.voltage,
                    'Type': load.load_type.value,
                    'Current (A)': f"{load.current_a:.2f}" if load.current_a else "N/A",
                    'Source Bus': load.source_bus or "Not assigned"
                })
            
            df = pd.DataFrame(load_data)
            st.dataframe(df, use_container_width=True)
            
            # Power distribution chart
            if len(loads) > 1:
                fig = px.pie(
                    values=[load.power_kw for load in loads],
                    names=[load.load_name for load in loads],
                    title="Power Distribution by Load"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No loads extracted from the file.")
    
    def run(self):
        """Run the Streamlit application"""
        config = self.render_sidebar()
        self.render_main_page(config)

# Initialize and run app
if __name__ == "__main__":
    app = AIExtractionStreamlitApp()
    app.run()
```

### React/Web Integration
```javascript
// Frontend integration with React
import React, { useState } from 'react';
import axios from 'axios';

const AIExtractionInterface = () => {
    const [file, setFile] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [results, setResults] = useState(null);

    const handleFileUpload = async () => {
        if (!file) return;

        setProcessing(true);
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await axios.post('/api/extract', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    // Update progress bar
                }
            });
            
            setResults(response.data);
        } catch (error) {
            console.error('Extraction failed:', error);
        } finally {
            setProcessing(false);
        }
    };

    return (
        <div className="ai-extraction-interface">
            <h1>🤖 AI Excel Extraction</h1>
            
            <div className="upload-section">
                <input
                    type="file"
                    accept=".xlsx,.xls"
                    onChange={(e) => setFile(e.target.files[0])}
                />
                <button 
                    onClick={handleFileUpload}
                    disabled={!file || processing}
                >
                    {processing ? 'Processing...' : 'Extract Data'}
                </button>
            </div>
            
            {results && (
                <div className="results-section">
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <h3>Confidence</h3>
                            <p>{results.confidence}%</p>
                        </div>
                        <div className="metric-card">
                            <h3>Components</h3>
                            <p>{results.components}</p>
                        </div>
                        <div className="metric-card">
                            <h3>Processing Time</h3>
                            <p>{results.processing_time}s</p>
                        </div>
                    </div>
                    
                    <div className="components-section">
                        <h2>Extracted Components</h2>
                        <div className="tabs">
                            <button className="tab-button">Loads ({results.loads_count})</button>
                            <button className="tab-button">Cables ({results.cables_count})</button>
                            <button className="tab-button">Buses ({results.buses_count})</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIExtractionInterface;
```

---

## REST API Integration

### FastAPI Implementation
```python
# Complete FastAPI implementation
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import tempfile
import os
from datetime import datetime

# Pydantic models for API
class ExtractionResponse(BaseModel):
    success: bool
    confidence: float
    components_count: int
    processing_time: float
    project_name: Optional[str] = None
    loads_count: Optional[int] = None
    cables_count: Optional[int] = None
    buses_count: Optional[int] = None
    corrections_made: List[Dict[str, Any]] = []
    validation_issues: List[str] = []
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]

# Initialize FastAPI app
app = FastAPI(
    title="AI Excel Extraction API",
    description="REST API for AI-powered electrical project data extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global extractor instance
extractor = AIExcelExtractor()

@app.get("/", response_model=dict)
async def root():
    """API root endpoint"""
    return {
        "message": "AI Excel Extraction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test extractor components
        classifier_health = "ok" if hasattr(extractor, 'sheet_classifier') else "error"
        mapper_health = "ok" if hasattr(extractor, 'column_mapper') else "error"
        extractor_health = "ok" if hasattr(extractor, 'data_extractor') else "error"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            components={
                "sheet_classifier": classifier_health,
                "column_mapper": mapper_health,
                "data_extractor": extractor_health
            }
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/extract", response_model=ExtractionResponse)
async def extract_excel(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Excel file to process"),
    standard: str = "IEC",
    confidence_threshold: float = 0.8
):
    """Extract electrical data from Excel file"""
    
    # Validate file
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .xlsx and .xls files are supported."
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Create extractor with specified standard
        extractor_instance = AIExcelExtractor(standard=standard)
        
        # Process file
        start_time = datetime.now()
        report = extractor_instance.process_excel_file(tmp_path)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = ExtractionResponse(
            success=True,
            confidence=report.overall_confidence,
            components_count=report.total_components,
            processing_time=processing_time,
            project_name=report.project_data.project_name if report.project_data else None,
            loads_count=len(report.project_data.loads) if report.project_data else 0,
            cables_count=len(report.project_data.cables) if report.project_data else 0,
            buses_count=len(report.project_data.buses) if report.project_data else 0,
            corrections_made=report.corrections_made,
            validation_issues=report.validation_issues
        )
        
        # Background task: clean up temporary file
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        return response
        
    except Exception as e:
        # Clean up on error
        cleanup_temp_file(tmp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )

@app.post("/extract/batch", response_model=List[ExtractionResponse])
async def extract_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple Excel files to process")
):
    """Process multiple Excel files"""
    
    async def process_single_file(file: UploadFile):
        try:
            # Validate file
            if not file.filename.endswith(('.xlsx', '.xls')):
                return ExtractionResponse(
                    success=False,
                    confidence=0.0,
                    components_count=0,
                    processing_time=0.0,
                    error=f"Invalid file format: {file.filename}"
                )
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Process file
            extractor_instance = AIExcelExtractor()
            start_time = datetime.now()
            report = extractor_instance.process_excel_file(tmp_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = ExtractionResponse(
                success=True,
                confidence=report.overall_confidence,
                components_count=report.total_components,
                processing_time=processing_time,
                project_name=report.project_data.project_name if report.project_data else None,
                loads_count=len(report.project_data.loads) if report.project_data else 0,
                cables_count=len(report.project_data.cables) if report.project_data else 0,
                buses_count=len(report.project_data.buses) if report.project_data else 0,
                corrections_made=report.corrections_made,
                validation_issues=report.validation_issues
            )
            
            # Background cleanup
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            return response
            
        except Exception as e:
            cleanup_temp_file(tmp_path)
            return ExtractionResponse(
                success=False,
                confidence=0.0,
                components_count=0,
                processing_time=0.0,
                error=str(e)
            )
    
    # Process all files concurrently
    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    return results

@app.get("/standards")
async def get_supported_standards():
    """Get list of supported electrical standards"""
    return {
        "supported_standards": ["IEC", "IS", "NEC"],
        "descriptions": {
            "IEC": "International Electrotechnical Commission",
            "IS": "Indian Standards",
            "NEC": "National Electrical Code"
        }
    }

@app.post("/validate/config")
async def validate_extraction_config(config: Dict[str, Any]):
    """Validate extraction configuration"""
    try:
        # Validate standard
        if config.get('standard') not in ['IEC', 'IS', 'NEC']:
            return {"valid": False, "error": "Invalid standard specified"}
        
        # Validate confidence threshold
        confidence = config.get('confidence_threshold', 0.8)
        if not 0.0 <= confidence <= 1.0:
            return {"valid": False, "error": "Confidence threshold must be between 0.0 and 1.0"}
        
        return {"valid": True, "message": "Configuration is valid"}
        
    except Exception as e:
        return {"valid": False, "error": f"Configuration validation failed: {str(e)}"}

# Utility functions
def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Failed to cleanup temp file {file_path}: {e}")

# Run with: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### API Client Examples

#### Python Client
```python
# Python client for the API
import requests
import json
from typing import List, Dict, Optional

class AIExtractionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def extract_file(self, file_path: str, standard: str = "IEC", 
                    confidence_threshold: float = 0.8) -> Dict:
        """Extract data from Excel file"""
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'standard': standard,
                'confidence_threshold': confidence_threshold
            }
            
            response = self.session.post(
                f"{self.base_url}/extract",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Extraction failed: {response.text}")
    
    def extract_batch(self, file_paths: List[str], 
                     standard: str = "IEC") -> List[Dict]:
        """Extract data from multiple files"""
        
        files = [('files', open(path, 'rb')) for path in file_paths]
        data = {'standard': standard}
        
        try:
            response = self.session.post(
                f"{self.base_url}/extract/batch",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Batch extraction failed: {response.text}")
        finally:
            # Close all file handles
            for _, file_handle in files:
                file_handle.close()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_standards(self) -> Dict:
        """Get supported standards"""
        response = self.session.get(f"{self.base_url}/standards")
        return response.json()
    
    def validate_config(self, config: Dict) -> Dict:
        """Validate extraction configuration"""
        response = self.session.post(
            f"{self.base_url}/validate/config",
            json=config
        )
        return response.json()

# Usage example
client = AIExtractionClient("http://localhost:8000")

# Health check
health = client.health_check()
print(f"API Health: {health['status']}")

# Extract single file
try:
    result = client.extract_file("project.xlsx", standard="IEC")
    print(f"Extraction successful: {result['confidence']:.1%} confidence")
    print(f"Components extracted: {result['components_count']}")
except Exception as e:
    print(f"Extraction failed: {e}")

# Extract batch
file_paths = ["project1.xlsx", "project2.xlsx", "project3.xlsx"]
results = client.extract_batch(file_paths)

for i, result in enumerate(results):
    if result['success']:
        print(f"File {i+1}: {result['confidence']:.1%} confidence")
    else:
        print(f"File {i+1}: Failed - {result['error']}")
```

#### JavaScript/Node.js Client
```javascript
// Node.js client for the API
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class AIExtractionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }

    async extractFile(filePath, options = {}) {
        const { standard = 'IEC', confidenceThreshold = 0.8 } = options;
        
        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));
        formData.append('standard', standard);
        formData.append('confidence_threshold', confidenceThreshold);

        try {
            const response = await axios.post(`${this.baseUrl}/extract`, formData, {
                headers: formData.getHeaders(),
                maxContentLength: Infinity,
                maxBodyLength: Infinity
            });

            return response.data;
        } catch (error) {
            if (error.response) {
                throw new Error(`Extraction failed: ${error.response.data.detail}`);
            } else {
                throw new Error(`Network error: ${error.message}`);
            }
        }
    }

    async extractBatch(filePaths, options = {}) {
        const { standard = 'IEC' } = options;
        
        const formData = new FormData();
        filePaths.forEach(path => {
            formData.append('files', fs.createReadStream(path));
        });
        formData.append('standard', standard);

        try {
            const response = await axios.post(`${this.baseUrl}/extract/batch`, formData, {
                headers: formData.getHeaders(),
                maxContentLength: Infinity,
                maxBodyLength: Infinity
            });

            return response.data;
        } catch (error) {
            if (error.response) {
                throw new Error(`Batch extraction failed: ${error.response.data.detail}`);
            } else {
                throw new Error(`Network error: ${error.message}`);
            }
        }
    }

    async healthCheck() {
        const response = await axios.get(`${this.baseUrl}/health`);
        return response.data;
    }

    async getStandards() {
        const response = await axios.get(`${this.baseUrl}/standards`);
        return response.data;
    }

    async validateConfig(config) {
        const response = await axios.post(`${this.baseUrl}/validate/config`, config);
        return response.data;
    }
}

// Usage example
async function exampleUsage() {
    const client = new AIExtractionClient('http://localhost:8000');
    
    try {
        // Health check
        const health = await client.healthCheck();
        console.log(`API Health: ${health.status}`);
        
        // Extract single file
        const result = await client.extractFile('project.xlsx', {
            standard: 'IEC',
            confidenceThreshold: 0.8
        });
        
        console.log(`Extraction successful: ${(result.confidence * 100).toFixed(1)}% confidence`);
        console.log(`Components extracted: ${result.components_count}`);
        
        // Extract batch
        const batchResults = await client.extractBatch([
            'project1.xlsx',
            'project2.xlsx',
            'project3.xlsx'
        ], { standard: 'IEC' });
        
        batchResults.forEach((result, index) => {
            if (result.success) {
                console.log(`File ${index + 1}: ${(result.confidence * 100).toFixed(1)}% confidence`);
            } else {
                console.log(`File ${index + 1}: Failed - ${result.error}`);
            }
        });
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

module.exports = AIExtractionClient;
```

---

## Database Integration

### SQL Database Integration
```python
# SQL database integration
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class DatabaseExtractor:
    """Database integration for AI extraction results"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.create_tables()
    
    def create_tables(self):
        """Create database tables"""
        Base.metadata.create_all(self.engine)
    
    def save_extraction_result(self, report: ProcessingReport, source_file: str) -> int:
        """Save extraction result to database"""
        
        # Save project
        project_id = self._save_project(report.project_data, source_file)
        
        # Save loads
        if report.project_data:
            for load in report.project_data.loads:
                self._save_load(load, project_id)
            
            # Save cables
            for cable in report.project_data.cables:
                self._save_cable(cable, project_id)
            
            # Save buses
            for bus in report.project_data.buses:
                self._save_bus(bus, project_id)
        
        # Save extraction metadata
        extraction_id = self._save_extraction_metadata(report, project_id)
        
        return extraction_id
    
    def _save_project(self, project: Project, source_file: str) -> int:
        """Save project to database"""
        project_record = ProjectRecord(
            name=project.project_name if project else "AI Extracted Project",
            standard=project.standard if project else "IEC",
            voltage_system=project.voltage_system if project else "LV",
            source_file=source_file,
            created_at=datetime.now()
        )
        
        self.session.add(project_record)
        self.session.commit()
        return project_record.id
    
    def _save_load(self, load: Load, project_id: int):
        """Save load to database"""
        load_record = LoadRecord(
            project_id=project_id,
            load_id=load.load_id,
            load_name=load.load_name,
            power_kw=load.power_kw,
            voltage=load.voltage,
            phases=load.phases,
            load_type=load.load_type.value,
            power_factor=load.power_factor,
            efficiency=load.efficiency,
            source_bus=load.source_bus,
            priority=load.priority.value,
            cable_length=load.cable_length,
            installation_method=load.installation_method.value,
            current_a=load.current_a,
            design_current_a=load.design_current_a,
            apparent_power_kva=load.apparent_power_kva
        )
        
        self.session.add(load_record)
    
    def _save_cable(self, cable: Cable, project_id: int):
        """Save cable to database"""
        cable_record = CableRecord(
            project_id=project_id,
            cable_id=cable.cable_id,
            from_equipment=cable.from_equipment,
            to_equipment=cable.to_equipment,
            cores=cable.cores,
            size_sqmm=cable.size_sqmm,
            cable_type=cable.cable_type,
            insulation=cable.insulation,
            length_m=cable.length_m,
            installation_method=cable.installation_method.value,
            armored=cable.armored
        )
        
        self.session.add(cable_record)
    
    def _save_bus(self, bus: Bus, project_id: int):
        """Save bus to database"""
        bus_record = BusRecord(
            project_id=project_id,
            bus_id=bus.bus_id,
            bus_name=bus.bus_name,
            voltage=bus.voltage,
            phases=bus.phases,
            rated_current_a=bus.rated_current_a,
            short_circuit_rating_ka=bus.short_circuit_rating_ka
        )
        
        self.session.add(bus_record)
    
    def _save_extraction_metadata(self, report: ProcessingReport, project_id: int) -> int:
        """Save extraction metadata"""
        metadata_record = ExtractionMetadataRecord(
            project_id=project_id,
            overall_confidence=report.overall_confidence,
            total_components=report.total_components,
            processing_time_seconds=report.processing_time_seconds,
            extraction_date=datetime.now(),
            corrections_count=len(report.corrections_made),
            validation_issues=json.dumps(report.validation_issues)
        )
        
        self.session.add(metadata_record)
        self.session.commit()
        return metadata_record.id
    
    def get_projects(self) -> pd.DataFrame:
        """Get all projects as DataFrame"""
        query = """
        SELECT 
            p.id,
            p.name,
            p.standard,
            p.created_at,
            COUNT(DISTINCT l.id) as load_count,
            COUNT(DISTINCT c.id) as cable_count,
            COUNT(DISTINCT b.id) as bus_count,
            em.overall_confidence,
            em.processing_time_seconds
        FROM projects p
        LEFT JOIN loads l ON p.id = l.project_id
        LEFT JOIN cables c ON p.id = c.project_id
        LEFT JOIN buses b ON p.id = b.project_id
        LEFT JOIN extraction_metadata em ON p.id = em.project_id
        GROUP BY p.id
        ORDER BY p.created_at DESC
        """
        
        return pd.read_sql(query, self.engine)
    
    def get_project_details(self, project_id: int) -> Dict:
        """Get detailed project information"""
        # Get project
        project = self.session.query(ProjectRecord).filter_by(id=project_id).first()
        
        # Get loads
        loads = self.session.query(LoadRecord).filter_by(project_id=project_id).all()
        
        # Get cables
        cables = self.session.query(CableRecord).filter_by(project_id=project_id).all()
        
        # Get buses
        buses = self.session.query(BusRecord).filter_by(project_id=project_id).all()
        
        # Get extraction metadata
        metadata = self.session.query(ExtractionMetadataRecord).filter_by(
            project_id=project_id
        ).first()
        
        return {
            'project': project,
            'loads': loads,
            'cables': cables,
            'buses': buses,
            'metadata': metadata
        }

# SQLAlchemy models
class ProjectRecord(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    standard = Column(String(10), nullable=False)
    voltage_system = Column(String(10))
    source_file = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)

class LoadRecord(Base):
    __tablename__ = 'loads'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, nullable=False)
    load_id = Column(String(50), nullable=False)
    load_name = Column(String(255), nullable=False)
    power_kw = Column(Float, nullable=False)
    voltage = Column(Float, nullable=False)
    phases = Column(Integer, nullable=False)
    load_type = Column(String(50), nullable=False)
    power_factor = Column(Float)
    efficiency = Column(Float)
    source_bus = Column(String(50))
    priority = Column(String(50))
    cable_length = Column(Float)
    installation_method = Column(String(50))
    current_a = Column(Float)
    design_current_a = Column(Float)
    apparent_power_kva = Column(Float)

class CableRecord(Base):
    __tablename__ = 'cables'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, nullable=False)
    cable_id = Column(String(50), nullable=False)
    from_equipment = Column(String(255), nullable=False)
    to_equipment = Column(String(255), nullable=False)
    cores = Column(Integer, nullable=False)
    size_sqmm = Column(Float, nullable=False)
    cable_type = Column(String(100))
    insulation = Column(String(50))
    length_m = Column(Float, nullable=False)
    installation_method = Column(String(50))
    armored = Column(Boolean, default=False)

class BusRecord(Base):
    __tablename__ = 'buses'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, nullable=False)
    bus_id = Column(String(50), nullable=False)
    bus_name = Column(String(255), nullable=False)
    voltage = Column(Float, nullable=False)
    phases = Column(Integer, nullable=False)
    rated_current_a = Column(Float, nullable=False)
    short_circuit_rating_ka = Column(Float)

class ExtractionMetadataRecord(Base):
    __tablename__ = 'extraction_metadata'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, nullable=False)
    overall_confidence = Column(Float, nullable=False)
    total_components = Column(Integer, nullable=False)
    processing_time_seconds = Column(Float, nullable=False)
    extraction_date = Column(DateTime, default=datetime.now)
    corrections_count = Column(Integer, default=0)
    validation_issues = Column(Text)
```

### MongoDB Integration
```python
# MongoDB integration
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import json

class MongoDBExtractor:
    """MongoDB integration for AI extraction results"""
    
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Create indexes
        self.db.projects.create_index([("created_at", -1)])
        self.db.loads.create_index([("project_id", 1)])
        self.db.cables.create_index([("project_id", 1)])
        self.db.buses.create_index([("project_id", 1)])
    
    def save_extraction_result(self, report: ProcessingReport, source_file: str) -> str:
        """Save extraction result to MongoDB"""
        
        # Save project
        project_doc = {
            "name": report.project_data.project_name if report.project_data else "AI Extracted Project",
            "standard": report.project_data.standard if report.project_data else "IEC",
            "voltage_system": report.project_data.voltage_system if report.project_data else "LV",
            "source_file": source_file,
            "created_at": datetime.now(),
            "extraction_metadata": {
                "overall_confidence": report.overall_confidence,
                "total_components": report.total_components,
                "processing_time_seconds": report.processing_time_seconds,
                "corrections_count": len(report.corrections_made),
                "validation_issues": report.validation_issues
            }
        }
        
        project_id = self.db.projects.insert_one(project_doc).inserted_id
        
        # Save components if project data exists
        if report.project_data:
            self._save_loads_to_mongo(report.project_data.loads, project_id)
            self._save_cables_to_mongo(report.project_data.cables, project_id)
            self._save_buses_to_mongo(report.project_data.buses, project_id)
        
        return str(project_id)
    
    def _save_loads_to_mongo(self, loads, project_id: str):
        """Save loads to MongoDB"""
        load_docs = []
        for load in loads:
            load_doc = {
                "project_id": ObjectId(project_id),
                "load_id": load.load_id,
                "load_name": load.load_name,
                "power_kw": load.power_kw,
                "voltage": load.voltage,
                "phases": load.phases,
                "load_type": load.load_type.value,
                "power_factor": load.power_factor,
                "efficiency": load.efficiency,
                "source_bus": load.source_bus,
                "priority": load.priority.value,
                "cable_length": load.cable_length,
                "installation_method": load.installation_method.value,
                "current_a": load.current_a,
                "design_current_a": load.design_current_a,
                "apparent_power_kva": load.apparent_power_kva
            }
            load_docs.append(load_doc)
        
        if load_docs:
            self.db.loads.insert_many(load_docs)
    
    def _save_cables_to_mongo(self, cables, project_id: str):
        """Save cables to MongoDB"""
        cable_docs = []
        for cable in cables:
            cable_doc = {
                "project_id": ObjectId(project_id),
                "cable_id": cable.cable_id,
                "from_equipment": cable.from_equipment,
                "to_equipment": cable.to_equipment,
                "cores": cable.cores,
                "size_sqmm": cable.size_sqmm,
                "cable_type": cable.cable_type,
                "insulation": cable.insulation,
                "length_m": cable.length_m,
                "installation_method": cable.installation_method.value,
                "armored": cable.armored
            }
            cable_docs.append(cable_doc)
        
        if cable_docs:
            self.db.cables.insert_many(cable_docs)
    
    def _save_buses_to_mongo(self, buses, project_id: str):
        """Save buses to MongoDB"""
        bus_docs = []
        for bus in buses:
            bus_doc = {
                "project_id": ObjectId(project_id),
                "bus_id": bus.bus_id,
                "bus_name": bus.bus_name,
                "voltage": bus.voltage,
                "phases": bus.phases,
                "rated_current_a": bus.rated_current_a,
                "short_circuit_rating_ka": bus.short_circuit_rating_ka
            }
            bus_docs.append(bus_doc)
        
        if bus_docs:
            self.db.buses.insert_many(bus_docs)
    
    def get_projects_summary(self) -> list:
        """Get projects summary"""
        pipeline = [
            {
                "$lookup": {
                    "from": "loads",
                    "localField": "_id",
                    "foreignField": "project_id",
                    "as": "loads"
                }
            },
            {
                "$lookup": {
                    "from": "cables",
                    "localField": "_id",
                    "foreignField": "project_id",
                    "as": "cables"
                }
            },
            {
                "$lookup": {
                    "from": "buses",
                    "localField": "_id",
                    "foreignField": "project_id",
                    "as": "buses"
                }
            },
            {
                "$project": {
                    "name": 1,
                    "standard": 1,
                    "created_at": 1,
                    "load_count": {"$size": "$loads"},
                    "cable_count": {"$size": "$cables"},
                    "bus_count": {"$size": "$buses"},
                    "extraction_confidence": "$extraction_metadata.overall_confidence"
                }
            },
            {
                "$sort": {"created_at": -1}
            }
        ]
        
        return list(self.db.projects.aggregate(pipeline))
```

---

This integration guide provides comprehensive examples for integrating the AI Excel Extraction System into various environments and applications. The examples cover web applications, REST APIs, databases, and enterprise systems, enabling seamless adoption of the AI-powered extraction capabilities.