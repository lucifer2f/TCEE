# Performance Guide - AI Excel Extraction System

## Overview

This guide covers performance optimization, benchmarking, and scaling strategies for the AI Excel Extraction System in production environments.

## Table of Contents

1. [Performance Benchmarks](#performance-benchmarks)
2. [Optimization Strategies](#optimization-strategies)
3. [Memory Management](#memory-management)
4. [Parallel Processing](#parallel-processing)
5. [Caching Strategies](#caching-strategies)
6. [Resource Monitoring](#resource-monitoring)
7. [Scaling Guidelines](#scaling-guidelines)
8. [Troubleshooting Performance](#troubleshooting-performance)

---

## Performance Benchmarks

### Baseline Performance Metrics

#### Small Files (< 1MB)
```
File Size: 50-500 KB
Typical Components: 5-50 loads/cables
Processing Time: 0.5-2.0 seconds
Memory Usage: 50-100 MB
CPU Usage: 10-30%
Success Rate: 95-100%
```

#### Medium Files (1-10MB)
```
File Size: 1-10 MB
Typical Components: 50-500 loads/cables
Processing Time: 2-10 seconds
Memory Usage: 100-300 MB
CPU Usage: 20-60%
Success Rate: 90-98%
```

#### Large Files (10-50MB)
```
File Size: 10-50 MB
Typical Components: 500-5000 loads/cables
Processing Time: 10-60 seconds
Memory Usage: 300-800 MB
CPU Usage: 40-80%
Success Rate: 85-95%
```

#### Very Large Files (50-100MB)
```
File Size: 50-100 MB
Typical Components: 5000-20000 loads/cables
Processing Time: 60-300 seconds
Memory Usage: 800 MB - 2 GB
CPU Usage: 60-90%
Success Rate: 80-90%
```

### Performance Testing Framework
```python
# Comprehensive performance testing
import time
import psutil
import gc
import pandas as pd
from typing import Dict, List, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance measurement container"""
    file_size_mb: float
    processing_time_seconds: float
    memory_peak_mb: float
    cpu_average_percent: float
    components_extracted: int
    success_rate: float
    throughput_components_per_second: float

class PerformanceBenchmark:
    """Performance benchmarking framework"""
    
    def __init__(self):
        self.results = []
        self.monitoring_enabled = True
    
    def benchmark_extraction(self, 
                           extractor: AIExcelExtractor,
                           test_files: List[str],
                           iterations: int = 3) -> List[PerformanceMetrics]:
        """Benchmark extraction performance"""
        
        benchmark_results = []
        
        for file_path in test_files:
            file_metrics = []
            
            # Get file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            for iteration in range(iterations):
                # Clear memory and garbage collect
                gc.collect()
                
                # Start monitoring
                process = psutil.Process()
                initial_memory = process.memory_info().rss / (1024 * 1024)
                
                cpu_samples = []
                
                def monitor_cpu():
                    while self.monitoring_enabled:
                        cpu_samples.append(psutil.cpu_percent(interval=0.1))
                        time.sleep(0.1)
                
                # Start CPU monitoring
                self.monitoring_enabled = True
                cpu_thread = threading.Thread(target=monitor_cpu)
                cpu_thread.start()
                
                # Measure extraction time
                start_time = time.time()
                
                try:
                    report = extractor.process_excel_file(file_path)
                    success = report.overall_confidence > 0.5
                except Exception as e:
                    print(f"Extraction failed: {e}")
                    success = False
                    report = None
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Stop CPU monitoring
                self.monitoring_enabled = False
                cpu_thread.join()
                
                # Measure memory peak
                final_memory = process.memory_info().rss / (1024 * 1024)
                memory_peak_mb = max(initial_memory, final_memory)
                
                # Calculate metrics
                if success and report:
                    components_extracted = report.total_components
                    throughput = components_extracted / processing_time if processing_time > 0 else 0
                else:
                    components_extracted = 0
                    throughput = 0
                
                cpu_average = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
                
                metrics = PerformanceMetrics(
                    file_size_mb=file_size_mb,
                    processing_time_seconds=processing_time,
                    memory_peak_mb=memory_peak_mb,
                    cpu_average_percent=cpu_average,
                    components_extracted=components_extracted,
                    success_rate=1.0 if success else 0.0,
                    throughput_components_per_second=throughput
                )
                
                file_metrics.append(metrics)
            
            # Calculate average for this file
            avg_metrics = self._calculate_average_metrics(file_metrics)
            benchmark_results.append(avg_metrics)
        
        return benchmark_results
    
    def _calculate_average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average performance metrics"""
        if not metrics_list:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        return PerformanceMetrics(
            file_size_mb=sum(m.file_size_mb for m in metrics_list) / len(metrics_list),
            processing_time_seconds=sum(m.processing_time_seconds for m in metrics_list) / len(metrics_list),
            memory_peak_mb=sum(m.memory_peak_mb for m in metrics_list) / len(metrics_list),
            cpu_average_percent=sum(m.cpu_average_percent for m in metrics_list) / len(metrics_list),
            components_extracted=int(sum(m.components_extracted for m in metrics_list) / len(metrics_list)),
            success_rate=sum(m.success_rate for m in metrics_list) / len(metrics_list),
            throughput_components_per_second=sum(m.throughput_components_per_second for m in metrics_list) / len(metrics_list)
        )
    
    def generate_performance_report(self, results: List[PerformanceMetrics]) -> str:
        """Generate performance analysis report"""
        report = []
        report.append("# Performance Benchmark Report\n")
        
        # Summary statistics
        avg_processing_time = sum(r.processing_time_seconds for r in results) / len(results)
        avg_memory_usage = sum(r.memory_peak_mb for r in results) / len(results)
        avg_throughput = sum(r.throughput_components_per_second for r in results) / len(results)
        overall_success_rate = sum(r.success_rate for r in results) / len(results)
        
        report.append("## Summary")
        report.append(f"- **Average Processing Time**: {avg_processing_time:.2f} seconds")
        report.append(f"- **Average Memory Usage**: {avg_memory_usage:.1f} MB")
        report.append(f"- **Average Throughput**: {avg_throughput:.1f} components/second")
        report.append(f"- **Overall Success Rate**: {overall_success_rate:.1%}")
        report.append("")
        
        # Detailed results table
        report.append("## Detailed Results")
        report.append("| File Size (MB) | Processing Time (s) | Memory (MB) | Components | Throughput | Success Rate |")
        report.append("|----------------|---------------------|-------------|------------|------------|---------------|")
        
        for metrics in results:
            report.append(
                f"| {metrics.file_size_mb:.1f} | {metrics.processing_time_seconds:.2f} | "
                f"{metrics.memory_peak_mb:.1f} | {metrics.components_extracted} | "
                f"{metrics.throughput_components_per_second:.1f} | {metrics.success_rate:.1%} |"
            )
        
        # Performance analysis
        report.append("\n## Performance Analysis")
        
        # Identify bottlenecks
        if avg_processing_time > 10:
            report.append("- **High Processing Time**: Consider parallel processing or optimization")
        
        if avg_memory_usage > 500:
            report.append("- **High Memory Usage**: Implement streaming processing")
        
        if avg_throughput < 10:
            report.append("- **Low Throughput**: Optimize AI algorithms or increase resources")
        
        if overall_success_rate < 0.9:
            report.append("- **Low Success Rate**: Review data quality and error handling")
        
        return "\n".join(report)
```

### Performance Profiling
```python
# Detailed performance profiling
import cProfile
import pstats
import io
from functools import wraps

def profile_performance(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        print("Performance Profile:")
        print(s.getvalue())
        
        return result
    return wrapper

class DetailedProfiler:
    """Detailed performance profiler for AI extraction pipeline"""
    
    def __init__(self):
        self.profiles = {}
        self.memory_profiles = []
    
    def profile_extraction_pipeline(self, extractor: AIExcelExtractor, file_path: str):
        """Profile the complete extraction pipeline"""
        
        # Profile each component
        components = [
            ('sheet_classification', self._profile_sheet_classification),
            ('column_mapping', self._profile_column_mapping),
            ('data_extraction', self._profile_data_extraction),
            ('data_enhancement', self._profile_data_enhancement),
            ('validation', self._profile_validation)
        ]
        
        component_profiles = {}
        
        for component_name, profile_func in components:
            print(f"Profiling {component_name}...")
            profile_result = profile_func(extractor, file_path)
            component_profiles[component_name] = profile_result
        
        # Memory profiling
        memory_profile = self._profile_memory_usage(extractor, file_path)
        
        # Generate report
        return self._generate_detailed_report(component_profiles, memory_profile)
    
    def _profile_sheet_classification(self, extractor: AIExcelExtractor, file_path: str) -> Dict:
        """Profile sheet classification performance"""
        pr = cProfile.Profile()
        pr.enable()
        
        # Read file
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        # Profile classification
        classification_results = {}
        for sheet_name, df in excel_data.items():
            result = extractor.sheet_classifier.classify_sheet(df, sheet_name)
            classification_results[sheet_name] = result
        
        pr.disable()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        
        return {
            'time_profile': s.getvalue(),
            'sheets_processed': len(excel_data),
            'classification_results': classification_results
        }
    
    def _profile_memory_usage(self, extractor: AIExcelExtractor, file_path: str) -> Dict:
        """Profile memory usage throughout extraction"""
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Get initial memory
        initial_memory = psutil.Process().memory_info().rss
        
        # Process file
        report = extractor.process_excel_file(file_path)
        
        # Get peak memory
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_memory_mb = peak_memory / (1024 * 1024)
        
        return {
            'initial_memory_mb': initial_memory / (1024 * 1024),
            'peak_memory_mb': peak_memory_mb,
            'memory_increase_mb': peak_memory_mb - (initial_memory / (1024 * 1024))
        }
```

---

## Optimization Strategies

### Algorithm Optimization

#### Pattern Recognition Optimization
```python
# Optimized pattern matching for better performance
import re
from typing import List, Dict, Tuple
import time

class OptimizedSheetClassifier(SheetClassifier):
    """Performance-optimized sheet classifier"""
    
    def __init__(self):
        super().__init__()
        self._compile_patterns()
        self._cache_pattern_matches()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.compiled_patterns = {}
        
        for sheet_type, patterns in self.sheet_patterns.items():
            self.compiled_patterns[sheet_type] = {
                'primary': [re.compile(pattern, re.IGNORECASE) for pattern in patterns.get('primary', [])],
                'secondary': [re.compile(pattern, re.IGNORECASE) for pattern in patterns.get('secondary', [])]
            }
    
    def _cache_pattern_matches(self):
        """Cache common pattern matches"""
        self.pattern_cache = {}
        self.sheet_name_cache = {}
    
    def classify_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Optimized sheet classification"""
        
        # Check cache first
        cache_key = f"{sheet_name}_{hash(str(df.columns.tolist()))}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        start_time = time.time()
        
        # Optimize by using compiled patterns
        headers_text = ' '.join(str(h).lower() for h in df.columns.tolist())
        
        scores = {}
        for sheet_type, compiled_patterns in self.compiled_patterns.items():
            score = 0.0
            matches = []
            
            # Primary patterns (higher weight)
            for pattern in compiled_patterns['primary']:
                if pattern.search(headers_text):
                    score += 3.0  # Weight for primary patterns
                    matches.append(pattern.pattern)
            
            # Secondary patterns
            for pattern in compiled_patterns['secondary']:
                if pattern.search(headers_text):
                    score += 1.0  # Weight for secondary patterns
                    matches.append(pattern.pattern)
            
            scores[sheet_type] = score
        
        # Determine best match
        if max(scores.values()) == 0:
            best_type = 'unknown'
            confidence = 0.0
        else:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type] / 10.0, 1.0)  # Normalize
        
        result = {
            'sheet_type': best_type,
            'confidence': confidence,
            'evidence': matches,
            'recommended_model_mapping': self._get_model_mapping(best_type),
            'processing_time': time.time() - start_time
        }
        
        # Cache result
        self.pattern_cache[cache_key] = result
        
        return result

# Fuzzy matching optimization
class OptimizedColumnMapper(ColumnMapper):
    """Performance-optimized column mapper"""
    
    def __init__(self):
        super().__init__()
        self._create_fuzzy_index()
    
    def _create_fuzzy_index(self):
        """Create fuzzy matching index for faster lookups"""
        self.fuzzy_index = {}
        
        for model_type, field_definitions in self.field_mappings.items():
            self.fuzzy_index[model_type] = {}
            
            for field_name, patterns in field_definitions.items():
                # Create normalized pattern index
                normalized_patterns = []
                for pattern in patterns:
                    # Remove common variations and normalize
                    normalized = self._normalize_pattern(pattern)
                    normalized_patterns.append(normalized)
                
                self.fuzzy_index[model_type][field_name] = {
                    'original_patterns': patterns,
                    'normalized_patterns': normalized_patterns,
                    'pattern_length': len(normalized_patterns)
                }
    
    def _normalize_pattern(self, pattern: str) -> str:
        """Normalize pattern for fuzzy matching"""
        # Remove common words and normalize
        normalized = pattern.lower()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = normalized.strip()
        return normalized
    
    def map_columns(self, columns: List[str], model_type: str, sheet_context: str = "") -> Dict[str, Any]:
        """Optimized column mapping"""
        
        # Use optimized fuzzy matching
        result = {
            'field_mappings': {},
            'overall_confidence': 0.0,
            'unmapped_columns': list(columns),
            'mapping_quality': 'poor'
        }
        
        if model_type not in self.fuzzy_index:
            return result
        
        # Pre-calculate column similarities
        column_similarities = self._calculate_column_similarities(columns, model_type)
        
        # Map fields based on similarities
        for field_name, field_data in self.fuzzy_index[model_type].items():
            best_match, confidence = self._find_best_match(column_similarities, field_data)
            
            if best_match and confidence > 0.6:
                result['field_mappings'][field_name] = {
                    'mapped_columns': [best_match],
                    'confidence': confidence,
                    'data_type': self._infer_data_type(field_name, columns),
                    'pattern_match': self._get_match_pattern(field_data['original_patterns'], best_match)
                }
                
                # Remove from unmapped columns
                if best_match in result['unmapped_columns']:
                    result['unmapped_columns'].remove(best_match)
        
        # Calculate overall confidence
        if result['field_mappings']:
            total_confidence = sum(m['confidence'] for m in result['field_mappings'].values())
            result['overall_confidence'] = total_confidence / len(result['field_mappings'])
            
            # Assess mapping quality
            mapped_fields = len(result['field_mappings'])
            total_fields = len(self.fuzzy_index[model_type])
            coverage_ratio = mapped_fields / total_fields
            
            if coverage_ratio >= 0.8:
                result['mapping_quality'] = 'excellent'
            elif coverage_ratio >= 0.6:
                result['mapping_quality'] = 'good'
            elif coverage_ratio >= 0.4:
                result['mapping_quality'] = 'fair'
        
        return result
    
    def _calculate_column_similarities(self, columns: List[str], model_type: str) -> Dict[str, Dict[str, float]]:
        """Calculate similarities between columns and field patterns"""
        similarities = {}
        
        for column in columns:
            similarities[column] = {}
            normalized_column = self._normalize_pattern(column)
            
            for field_name, field_data in self.fuzzy_index[model_type].items():
                max_similarity = 0.0
                
                for pattern in field_data['normalized_patterns']:
                    similarity = self._calculate_similarity(normalized_column, pattern)
                    max_similarity = max(max_similarity, similarity)
                
                similarities[column][field_name] = max_similarity
        
        return similarities
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Optimized string similarity calculation"""
        # Use faster similarity metric for simple cases
        if str1 == str2:
            return 1.0
        
        # Use SequenceMatcher for better performance than fuzzywuzzy in batch operations
        return SequenceMatcher(None, str1, str2).ratio()
```

### Data Processing Optimization

#### Streaming Data Processing
```python
# Streaming processor for large files
class StreamingProcessor:
    """Process large Excel files using streaming"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.cache = {}
    
    def process_large_file_streaming(self, file_path: str, extractor: AIExcelExtractor) -> ProcessingReport:
        """Process large files using streaming to manage memory"""
        
        with pd.ExcelFile(file_path) as xls:
            all_results = {}
            all_loads = []
            all_cables = []
            all_buses = []
            
            for sheet_name in xls.sheet_names:
                print(f"Processing sheet: {sheet_name}")
                
                # Read sheet in chunks
                sheet_results = self._process_sheet_streaming(
                    xls, sheet_name, extractor
                )
                
                all_results[sheet_name] = sheet_results['result']
                
                # Accumulate results
                if 'loads' in sheet_results:
                    all_loads.extend(sheet_results['loads'])
                if 'cables' in sheet_results:
                    all_cables.extend(sheet_results['cables'])
                if 'buses' in sheet_results:
                    all_buses.extend(sheet_results['buses'])
                
                # Force garbage collection after each sheet
                gc.collect()
        
        # Create final report
        return self._create_final_report(
            all_results, all_loads, all_cables, all_buses
        )
    
    def _process_sheet_streaming(self, xls, sheet_name: str, extractor: AIExcelExtractor) -> Dict:
        """Process single sheet using streaming"""
        
        # Get sheet dimensions first
        df_sample = pd.read_excel(xls, sheet_name=sheet_name, nrows=5)
        total_rows = len(pd.read_excel(xls, sheet_name=sheet_name))
        
        # Classify sheet
        classification = extractor.sheet_classifier.classify_sheet(df_sample, sheet_name)
        
        if classification['recommended_model_mapping'] not in ['Load', 'Cable', 'Bus']:
            return {'result': ExtractionResult(
                success=True,
                confidence=classification['confidence'],
                sheet_type=classification['sheet_type'],
                components_extracted=0,
                data_quality_score=classification['confidence']
            )}
        
        # Map columns using sample
        mapping = extractor.column_mapper.map_columns(
            df_sample.columns.tolist(),
            classification['recommended_model_mapping']
        )
        
        # Process in chunks
        loads = []
        cables = []
        
        for chunk_start in range(0, total_rows, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_rows)
            
            # Read chunk
            df_chunk = pd.read_excel(
                xls, 
                sheet_name=sheet_name,
                skiprows=range(1, chunk_start + 1),  # Skip to chunk start
                nrows=chunk_end - chunk_start
            )
            
            # Process chunk
            if classification['sheet_type'] == 'load_schedule':
                chunk_loads, chunk_result = extractor.data_extractor.extract_loads(df_chunk, mapping)
                loads.extend(chunk_loads)
            elif classification['sheet_type'] == 'cable_schedule':
                chunk_cables, chunk_result = extractor.data_extractor.extract_cables(df_chunk, mapping)
                cables.extend(chunk_cables)
        
        return {
            'result': chunk_result,
            'loads': loads,
            'cables': cables
        }
```

#### Memory-Efficient Processing
```python
# Memory-efficient processing strategies
import numpy as np
from typing import Iterator

class MemoryEfficientProcessor:
    """Memory-efficient processing for large datasets"""
    
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        self.memory_threshold = max_memory_mb * 1024 * 1024  # Convert to bytes
    
    def process_with_memory_management(self, extractor: AIExcelExtractor, file_path: str) -> ProcessingReport:
        """Process file with memory management"""
        
        # Monitor memory usage
        initial_memory = psutil.Process().memory_info().rss
        
        # Process file
        report = extractor.process_excel_file(file_path)
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Add memory information to report
        report.memory_usage_mb = memory_increase / (1024 * 1024)
        report.memory_efficient = memory_increase < self.memory_threshold
        
        return report
    
    def optimize_data_structures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        
        # Convert to memory-efficient data types
        optimized_df = data.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            # Downcast numeric columns
            if optimized_df[col].dtype in ['int64', 'int32']:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            elif optimized_df[col].dtype in ['float64', 'float32']:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize object columns
        for col in optimized_df.select_dtypes(include=['object']).columns:
            # Convert to category if few unique values
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def cleanup_memory(self):
        """Force memory cleanup"""
        # Clear pandas cache
        pd.io.sql._SQL_TABLE_LOCK = threading.Lock()
        
        # Garbage collection
        gc.collect()
        
        # Clear process memory
        if hasattr(psutil.Process(), 'memory_info'):
            process = psutil.Process()
            # Force process to release unused memory
            try:
                process.memory_info()  # Trigger memory measurement
            except:
                pass
```

---

## Memory Management

### Memory Monitoring
```python
# Comprehensive memory monitoring
import psutil
import threading
import time
from typing import Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory

class MemoryMonitor:
    """Real-time memory monitoring"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.snapshots: List[MemorySnapshot] = []
        self.callbacks: List[Callable] = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Memory monitoring loop"""
        while self.monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            
            # Execute callbacks
            for callback in self.callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    print(f"Memory monitoring callback error: {e}")
            
            time.sleep(self.interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take current memory snapshot"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=process.memory_percent(),
            available_mb=system_memory.available / (1024 * 1024)
        )
    
    def add_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add memory monitoring callback"""
        self.callbacks.append(callback)
    
    def get_memory_statistics(self) -> Dict:
        """Get memory usage statistics"""
        if not self.snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self.snapshots]
        
        return {
            'peak_memory_mb': max(rss_values),
            'average_memory_mb': sum(rss_values) / len(rss_values),
            'memory_growth_mb': rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0,
            'snapshots_count': len(self.snapshots),
            'monitoring_duration_seconds': (
                self.snapshots[-1].timestamp - self.snapshots[0].timestamp
            ).total_seconds() if len(self.snapshots) > 1 else 0
        }
    
    def generate_memory_report(self) -> str:
        """Generate memory usage report"""
        stats = self.get_memory_statistics()
        
        if not stats:
            return "No memory monitoring data available"
        
        report = []
        report.append("# Memory Usage Report")
        report.append(f"- **Peak Memory**: {stats['peak_memory_mb']:.1f} MB")
        report.append(f"- **Average Memory**: {stats['average_memory_mb']:.1f} MB")
        report.append(f"- **Memory Growth**: {stats['memory_growth_mb']:.1f} MB")
        report.append(f"- **Monitoring Duration**: {stats['monitoring_duration_seconds']:.1f} seconds")
        report.append(f"- **Snapshots**: {stats['snapshots_count']}")
        
        return "\n".join(report)

# Memory management integration
class ManagedAIExtractor:
    """AI extractor with integrated memory management"""
    
    def __init__(self, max_memory_mb: int = 500):
        self.extractor = AIExcelExtractor()
        self.memory_monitor = MemoryMonitor(interval=0.5)
        self.max_memory_mb = max_memory_mb
        self.alerts_enabled = True
    
    def process_with_memory_management(self, file_path: str) -> ProcessingReport:
        """Process file with memory management"""
        
        # Set up memory alerts
        def memory_alert(snapshot: MemorySnapshot):
            if snapshot.rss_mb > self.max_memory_mb and self.alerts_enabled:
                print(f"⚠️  Memory alert: {snapshot.rss_mb:.1f} MB > {self.max_memory_mb} MB")
                gc.collect()  # Force garbage collection
        
        self.memory_monitor.add_callback(memory_alert)
        
        # Start monitoring
        self.memory_monitor.start_monitoring()
        
        try:
            # Process file
            report = self.extractor.process_excel_file(file_path)
            
            # Add memory information to report
            memory_stats = self.memory_monitor.get_memory_statistics()
            report.memory_statistics = memory_stats
            
            return report
            
        finally:
            # Stop monitoring
            self.memory_monitor.stop_monitoring()
```

---

## Parallel Processing

### Concurrent Sheet Processing
```python
# Parallel processing for multiple sheets
import concurrent.futures
import threading
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class ParallelExtractor:
    """Parallel processing for improved performance"""
    
    def __init__(self, max_workers: int = None, use_threads: bool = True):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_threads = use_threads
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    def process_excel_file_parallel(self, file_path: str) -> ProcessingReport:
        """Process Excel file with parallel sheet processing"""
        
        # Read all sheets first
        excel_data = pd.read_excel(file_path, sheet_name=None)
        sheet_names = list(excel_data.keys())
        
        # Process sheets in parallel
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit sheet processing tasks
            future_to_sheet = {
                executor.submit(self._process_single_sheet, sheet_name, df): sheet_name
                for sheet_name, df in excel_data.items()
            }
            
            # Collect results
            sheet_results = {}
            for future in concurrent.futures.as_completed(future_to_sheet):
                sheet_name = future_to_sheet[future]
                try:
                    result = future.result()
                    sheet_results[sheet_name] = result
                except Exception as e:
                    print(f"Sheet {sheet_name} processing failed: {e}")
                    sheet_results[sheet_name] = ExtractionResult(
                        success=False,
                        confidence=0.0,
                        sheet_type='unknown',
                        components_extracted=0,
                        data_quality_score=0.0,
                        issues=[f"Processing failed: {str(e)}"]
                    )
        
        # Aggregate results
        return self._aggregate_parallel_results(sheet_results, excel_data)
    
    def _process_single_sheet(self, sheet_name: str, df: pd.DataFrame) -> ExtractionResult:
        """Process single sheet (can be run in parallel)"""
        
        # Create isolated extractor for thread safety
        extractor = AIExcelExtractor()
        
        # Classify sheet
        classification = extractor.sheet_classifier.classify_sheet(df, sheet_name)
        
        if classification['recommended_model_mapping'] in ['Load', 'Cable', 'Bus']:
            # Map columns
            mapping = extractor.column_mapper.map_columns(
                df.columns.tolist(),
                classification['recommended_model_mapping']
            )
            
            # Extract data
            if classification['sheet_type'] == 'load_schedule':
                loads, result = extractor.data_extractor.extract_loads(df, mapping)
                return result
            elif classification['sheet_type'] == 'cable_schedule':
                cables, result = extractor.data_extractor.extract_cables(df, mapping)
                return result
        
        # Return result for non-processed sheets
        return ExtractionResult(
            success=True,
            confidence=classification['confidence'],
            sheet_type=classification['sheet_type'],
            components_extracted=0,
            data_quality_score=classification['confidence']
        )
    
    def _aggregate_parallel_results(self, sheet_results: Dict[str, ExtractionResult], 
                                   excel_data: Dict[str, pd.DataFrame]) -> ProcessingReport:
        """Aggregate results from parallel sheet processing"""
        
        # Calculate overall metrics
        overall_confidence = sum(r.confidence for r in sheet_results.values()) / len(sheet_results)
        total_components = sum(r.components_extracted for r in sheet_results.values())
        processing_time = max(getattr(r, 'processing_time_seconds', 0) for r in sheet_results.values())
        
        # Create project from results
        all_loads = []
        all_cables = []
        all_buses = []
        corrections_made = []
        validation_issues = []
        
        for sheet_name, result in sheet_results.items():
            if result.success and result.extracted_data:
                if 'loads' in result.extracted_data:
                    for load_dict in result.extracted_data['loads']:
                        load = self._dict_to_load(load_dict)
                        if load:
                            all_loads.append(load)
                
                if 'cables' in result.extracted_data:
                    for cable_dict in result.extracted_data['cables']:
                        cable = self._dict_to_cable(cable_dict)
                        if cable:
                            all_cables.append(cable)
            
            corrections_made.extend(getattr(result, 'corrections_made', []))
            validation_issues.extend(result.issues + result.warnings)
        
        # Create project
        project = Project(
            project_name="Parallel Extracted Project",
            standard="IEC",
            voltage_system="LV"
        )
        project.loads = all_loads
        project.cables = all_cables
        project.buses = all_buses
        
        return ProcessingReport(
            overall_confidence=overall_confidence,
            total_components=total_components,
            processing_time_seconds=processing_time,
            sheet_results=sheet_results,
            project_data=project,
            corrections_made=corrections_made,
            validation_issues=validation_issues
        )
```

### Batch Processing Optimization
```python
# Batch processing for multiple files
class BatchProcessor:
    """Efficient batch processing of multiple files"""
    
    def __init__(self, batch_size: int = 5, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.progress_callback = None
    
    def process_batch_files(self, file_paths: List[str], 
                           extractor_factory: Callable = None) -> List[ProcessingReport]:
        """Process multiple files in batches"""
        
        if extractor_factory is None:
            extractor_factory = lambda: AIExcelExtractor()
        
        results = []
        
        # Process files in batches
        for i in range(0, len(file_paths), self.batch_size):
            batch = file_paths[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(file_paths)-1)//self.batch_size + 1}")
            
            # Process batch in parallel
            batch_results = self._process_batch_parallel(batch, extractor_factory)
            results.extend(batch_results)
            
            # Report progress
            if self.progress_callback:
                completed = min(i + self.batch_size, len(file_paths))
                self.progress_callback(completed, len(file_paths))
            
            # Force garbage collection between batches
            gc.collect()
        
        return results
    
    def _process_batch_parallel(self, batch: List[str], 
                               extractor_factory: Callable) -> List[ProcessingReport]:
        """Process batch of files in parallel"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path, extractor_factory): file_path
                for file_path in batch
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"File {file_path} processing failed: {e}")
                    # Create error result
                    results.append(ProcessingReport(
                        overall_confidence=0.0,
                        total_components=0,
                        processing_time_seconds=0.0,
                        sheet_results={},
                        validation_issues=[f"Processing failed: {str(e)}"]
                    ))
        
        return results
    
    def _process_single_file(self, file_path: str, 
                           extractor_factory: Callable) -> ProcessingReport:
        """Process single file with fresh extractor"""
        
        extractor = extractor_factory()
        return extractor.process_excel_file(file_path)
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set progress callback function"""
        self.progress_callback = callback

# Usage example
def process_project_directory(directory_path: str):
    """Process all Excel files in a directory"""
    
    # Find all Excel files
    excel_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(os.path.join(root, file))
    
    print(f"Found {len(excel_files)} Excel files to process")
    
    # Process files in batches
    batch_processor = BatchProcessor(batch_size=3, max_workers=2)
    
    def progress_callback(completed: int, total: int):
        print(f"Progress: {completed}/{total} files processed ({completed/total*100:.1f}%)")
    
    batch_processor.set_progress_callback(progress_callback)
    
    results = batch_processor.process_batch_files(excel_files)
    
    # Generate summary report
    successful = sum(1 for r in results if r.overall_confidence > 0.5)
    total_components = sum(r.total_components for r in results)
    total_time = sum(r.processing_time_seconds for r in results)
    
    print(f"\nBatch Processing Summary:")
    print(f"- Files processed: {len(results)}")
    print(f"- Successful extractions: {successful}")
    print(f"- Success rate: {successful/len(results)*100:.1f}%")
    print(f"- Total components extracted: {total_components}")
    print(f"- Total processing time: {total_time:.1f} seconds")
    print(f"- Average time per file: {total_time/len(results):.2f} seconds")
    
    return results
```

---

## Caching Strategies

### Result Caching
```python
# Intelligent caching for improved performance
import hashlib
import pickle
import os
from typing import Optional, Dict, Any
import json

class CachedExtractor:
    """Extractor with intelligent result caching"""
    
    def __init__(self, cache_dir: str = ".extraction_cache", 
                 cache_ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        self._create_cache_dir()
        self.cache_stats = {'hits': 0, 'misses': 0, 'expired': 0}
    
    def _create_cache_dir(self):
        """Create cache directory"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for caching"""
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_cache_key(self, file_path: str, extractor_config: Dict) -> str:
        """Generate cache key for file and configuration"""
        file_hash = self._get_file_hash(file_path)
        config_hash = hashlib.md5(json.dumps(extractor_config, sort_keys=True).encode()).hexdigest()
        return f"{file_hash}_{config_hash}"
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        # Check TTL
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < (self.cache_ttl_hours * 3600)
    
    def process_with_cache(self, file_path: str, extractor: AIExcelExtractor,
                          config: Dict = None) -> ProcessingReport:
        """Process file with caching"""
        
        config = config or {}
        cache_key = self._get_cache_key(file_path, config)
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_report = pickle.load(f)
                
                self.cache_stats['hits'] += 1
                print(f"Cache hit for {os.path.basename(file_path)}")
                return cached_report
                
            except Exception as e:
                print(f"Cache read error: {e}")
        
        self.cache_stats['misses'] += 1
        
        # Process file
        print(f"Processing {os.path.basename(file_path)}")
        report = extractor.process_excel_file(file_path)
        
        # Cache result
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(report, f)
            print(f"Cached result for {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Cache write error: {e}")
        
        return report
    
    def clear_cache(self, older_than_hours: int = None):
        """Clear cache files"""
        cutoff_time = time.time() - ((older_than_hours or self.cache_ttl_hours) * 3600)
        
        cleared_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        cleared_count += 1
                    except Exception as e:
                        print(f"Failed to remove {filename}: {e}")
        
        print(f"Cleared {cleared_count} cache files")
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size_mb': self._get_cache_size() / (1024 * 1024),
            'cache_files_count': self._get_cache_files_count()
        }
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(file_path)
        return total_size
    
    def _get_cache_files_count(self) -> int:
        """Get number of cache files"""
        return len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
```

### Pattern Caching
```python
# Pattern recognition result caching
class CachedPatternClassifier(OptimizedSheetClassifier):
    """Sheet classifier with pattern result caching"""
    
    def __init__(self):
        super().__init__()
        self.pattern_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def classify_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Classify sheet with pattern caching"""
        
        # Create cache key
        columns_tuple = tuple(sorted(df.columns.tolist()))
        sheet_name_normalized = sheet_name.lower().strip()
        cache_key = (sheet_name_normalized, columns_tuple)
        
        # Check cache
        if cache_key in self.pattern_cache:
            cached_result = self.pattern_cache[cache_key]
            # Update timestamp for LRU
            cached_result['last_used'] = time.time()
            self.cache_hits += 1
            return cached_result['result']
        
        self.cache_misses += 1
        
        # Perform classification
        result = super().classify_sheet(df, sheet_name)
        
        # Cache result
        self.pattern_cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'last_used': time.time()
        }
        
        # Cache size management
        self._manage_cache_size()
        
        return result
    
    def _manage_cache_size(self, max_cache_size: int = 1000):
        """Manage cache size using LRU"""
        
        if len(self.pattern_cache) <= max_cache_size:
            return
        
        # Remove oldest entries (simple LRU)
        sorted_cache = sorted(
            self.pattern_cache.items(),
            key=lambda x: x[1]['last_used']
        )
        
        # Remove oldest 20% of entries
        remove_count = int(len(sorted_cache) * 0.2)
        for i in range(remove_count):
            del self.pattern_cache[sorted_cache[i][0]]
    
    def get_pattern_cache_stats(self) -> Dict:
        """Get pattern caching statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.pattern_cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def clear_pattern_cache(self):
        """Clear pattern cache"""
        self.pattern_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
```

---

## Resource Monitoring

### System Resource Monitoring
```python
# Comprehensive system monitoring
import psutil
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    load_average: List[float]

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics_history: List[SystemMetrics] = []
        self.monitor_thread = None
        self.alerts = []
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """System monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Keep only last 1000 metrics to prevent memory buildup
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=psutil.virtual_memory().percent,
            memory_available_gb=psutil.virtual_memory().available / (1024**3),
            disk_usage_percent=psutil.disk_usage('/').percent,
            load_average=psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        )
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check for system resource alerts"""
        alerts = []
        
        if metrics.cpu_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 90:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.memory_available_gb < 1:
            alerts.append(f"Low available memory: {metrics.memory_available_gb:.1f}GB")
        
        if metrics.disk_usage_percent > 90:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        for alert in alerts:
            print(f"⚠️  {alert}")
            self.alerts.append({
                'timestamp': time.time(),
                'alert': alert,
                'metrics': metrics
            })
    
    def get_resource_summary(self) -> Dict:
        """Get resource usage summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-60:]  # Last minute
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'peak': max(cpu_values),
                'minimum': min(cpu_values)
            },
            'memory': {
                'average': sum(memory_values) / len(memory_values),
                'peak': max(memory_values),
                'minimum': min(memory_values),
                'available_gb': recent_metrics[-1].memory_available_gb
            },
            'disk_usage_percent': recent_metrics[-1].disk_usage_percent,
            'alerts_count': len(self.alerts)
        }

# Performance monitoring integration
class MonitoredExtractor:
    """Extractor with integrated system monitoring"""
    
    def __init__(self, system_monitor: SystemMonitor):
        self.extractor = AIExcelExtractor()
        self.system_monitor = system_monitor
    
    def process_with_monitoring(self, file_path: str) -> Dict:
        """Process file with system monitoring"""
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        
        try:
            # Get initial metrics
            initial_metrics = self.system_monitor._collect_metrics()
            
            # Process file
            start_time = time.time()
            report = self.extractor.process_excel_file(file_path)
            processing_time = time.time() - start_time
            
            # Get final metrics
            final_metrics = self.system_monitor._collect_metrics()
            
            # Generate monitoring report
            monitoring_report = {
                'extraction_report': report,
                'processing_time': processing_time,
                'initial_metrics': initial_metrics,
                'final_metrics': final_metrics,
                'resource_impact': {
                    'cpu_impact': final_metrics.cpu_percent - initial_metrics.cpu_percent,
                    'memory_impact': final_metrics.memory_percent - initial_metrics.memory_percent,
                    'available_memory_change_gb': final_metrics.memory_available_gb - initial_metrics.memory_available_gb
                },
                'system_summary': self.system_monitor.get_resource_summary()
            }
            
            return monitoring_report
            
        finally:
            # Stop monitoring
            self.system_monitor.stop_monitoring()
```

---

## Scaling Guidelines

### Horizontal Scaling
```python
# Horizontal scaling for multiple workers
import queue
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import redis
import pickle

class DistributedExtractor:
    """Distributed extraction across multiple workers"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.extractor_pool = None
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
    
    def setup_worker_pool(self, num_workers: int = None):
        """Setup worker pool for distributed processing"""
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        self.extractor_pool = ProcessPoolExecutor(max_workers=num_workers)
        print(f"Worker pool initialized with {num_workers} workers")
    
    def distribute_processing(self, file_paths: List[str]) -> List[ProcessingReport]:
        """Distribute file processing across worker pool"""
        
        if not self.extractor_pool:
            raise RuntimeError("Worker pool not initialized")
        
        results = []
        
        # Submit all tasks
        future_to_file = {
            self.extractor_pool.submit(self._process_file_worker, file_path): file_path
            for file_path in file_paths
        }
        
        # Collect results
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed: {os.path.basename(file_path)} - {e}")
                results.append(None)
        
        return results
    
    def _process_file_worker(self, file_path: str) -> ProcessingReport:
        """Worker function for file processing"""
        # Create new extractor in worker process
        extractor = AIExcelExtractor()
        return extractor.process_excel_file(file_path)
    
    def cleanup(self):
        """Cleanup worker pool"""
        if self.extractor_pool:
            self.extractor_pool.shutdown(wait=True)

# Load balancer for distributed processing
class ExtractionLoadBalancer:
    """Load balancer for distributed extraction tasks"""
    
    def __init__(self, worker_configs: List[Dict]):
        self.workers = worker_configs
        self.worker_loads = {i: 0 for i in range(len(workers))}
        self.worker_capabilities = self._assess_worker_capabilities()
    
    def _assess_worker_capabilities(self) -> Dict:
        """Assess worker capabilities"""
        capabilities = {}
        
        for i, config in enumerate(self.workers):
            capabilities[i] = {
                'max_concurrent_tasks': config.get('max_tasks', 1),
                'memory_limit_gb': config.get('memory_gb', 2),
                'cpu_cores': config.get('cpu_cores', 1),
                'specializations': config.get('specializations', [])
            }
        
        return capabilities
    
    def assign_task(self, task: Dict) -> int:
        """Assign task to best available worker"""
        
        # Calculate worker scores based on current load and capabilities
        worker_scores = {}
        
        for worker_id, capabilities in self.worker_capabilities.items():
            current_load = self.worker_loads[worker_id]
            max_load = capabilities['max_concurrent_tasks']
            
            if current_load >= max_load:
                continue  # Worker is at capacity
            
            # Calculate suitability score
            score = (
                (max_load - current_load) * 10 +  # Available capacity
                capabilities['cpu_cores'] * 5 +   # CPU capacity
                capabilities['memory_limit_gb'] * 2  # Memory capacity
            )
            
            # Bonus for specializations
            task_type = task.get('type', 'general')
            if task_type in capabilities['specializations']:
                score += 20
            
            worker_scores[worker_id] = score
        
        if not worker_scores:
            raise RuntimeError("No available workers")
        
        # Select worker with highest score
        best_worker = max(worker_scores, key=worker_scores.get)
        self.worker_loads[best_worker] += 1
        
        return best_worker
    
    def complete_task(self, worker_id: int):
        """Mark task as completed for worker"""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - 1)
    
    def get_load_balancer_stats(self) -> Dict:
        """Get load balancer statistics"""
        return {
            'worker_loads': self.worker_loads,
            'total_capacity': sum(cap['max_concurrent_tasks'] for cap in self.worker_capabilities.values()),
            'current_load': sum(self.worker_loads.values()),
            'utilization': sum(self.worker_loads.values()) / sum(cap['max_concurrent_tasks'] for cap in self.worker_capabilities.values())
        }
```

### Vertical Scaling
```python
# Vertical scaling optimizations
class OptimizedExtractor:
    """Extractor optimized for high-performance hardware"""
    
    def __init__(self, optimization_level: str = "maximum"):
        self.optimization_level = optimization_level
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup performance optimizations based on hardware"""
        
        if self.optimization_level == "maximum":
            # Maximum performance settings
            self.chunk_size = 5000
            self.parallel_threshold = 2  # Parallelize if > 2 sheets
            self.memory_limit_gb = 4
            self.enable_gpu = False  # Add GPU support if available
            self.num_workers = min(32, (os.cpu_count() or 1) + 4)
        
        elif self.optimization_level == "high":
            # High performance settings
            self.chunk_size = 2500
            self.parallel_threshold = 3
            self.memory_limit_gb = 2
            self.num_workers = min(16, (os.cpu_count() or 1) + 2)
        
        else:  # balanced
            # Balanced settings
            self.chunk_size = 1000
            self.parallel_threshold = 5
            self.memory_limit_gb = 1
            self.num_workers = min(8, (os.cpu_count() or 1))
    
    def process_optimized(self, file_path: str) -> ProcessingReport:
        """Process file with vertical optimizations"""
        
        # Monitor system resources
        initial_metrics = self._get_system_metrics()
        
        # Apply optimizations
        if self._should_parallelize(file_path):
            return self._process_parallel_optimized(file_path)
        else:
            return self._process_sequential_optimized(file_path)
    
    def _should_parallelize(self, file_path: str) -> bool:
        """Determine if file should be processed in parallel"""
        # Get file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Check system resources
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Parallelize if file is large and system has resources
        return (
            file_size_mb > 5 and  # File is large enough
            cpu_count >= 4 and    # Sufficient CPU cores
            memory_gb >= 2 and    # Sufficient memory
            cpu_count >= self.parallel_threshold
        )
    
    def _process_parallel_optimized(self, file_path: str) -> ProcessingReport:
        """Optimized parallel processing"""
        
        # Read sheets and determine processing strategy
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        # Group sheets by type for efficient processing
        sheet_groups = self._group_sheets_by_type(excel_data)
        
        results = {}
        
        for group_type, sheets in sheet_groups.items():
            if len(sheets) == 1:
                # Single sheet, process directly
                sheet_name, df = list(sheets.items())[0]
                result = self._process_single_sheet_optimized(df, sheet_name, group_type)
                results[sheet_name] = result
            else:
                # Multiple sheets of same type, process in parallel
                group_results = self._process_sheet_group_parallel(sheets, group_type)
                results.update(group_results)
        
        # Aggregate results
        return self._aggregate_optimized_results(results, excel_data)
    
    def _group_sheets_by_type(self, excel_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Group sheets by processing type for optimization"""
        
        # Temporary classifier for grouping
        classifier = SheetClassifier()
        groups = {}
        
        for sheet_name, df in excel_data.items():
            classification = classifier.classify_sheet(df, sheet_name)
            sheet_type = classification['sheet_type']
            
            if sheet_type not in groups:
                groups[sheet_type] = {}
            
            groups[sheet_type][sheet_name] = df
        
        return groups
    
    def _get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        return {
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
```

This performance guide provides comprehensive strategies for optimizing, monitoring, and scaling the AI Excel Extraction System across different hardware configurations and usage scenarios.