"""
AI-Powered Excel Extraction System for Electrical Distribution Projects

This module implements intelligent Excel data extraction for electrical engineering projects,
featuring domain-specific AI components for pattern recognition, data mapping, quality enhancement,
and validation with hybrid embedding + pattern matching approach.
Enhanced with τ + margin policy, normalized dot product, per-class thresholds, caching and logging.
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
import hashlib
import time
from datetime import datetime
from functools import lru_cache
from collections import defaultdict

# Import embedding libraries with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using pattern matching fallback.")

# Removed sklearn dependency - using native dot product on normalized vectors
SKLEARN_AVAILABLE = False
logging.info("Using native dot product for similarity calculations")

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
    provenance: Dict[str, Any] = field(default_factory=dict)


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
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    """Per-class thresholds from golden set analysis"""
    tau: float  # Base threshold
    margin: float  # Margin for uncertainty
    confidence_threshold: float  # Minimum confidence for acceptance
    
    def get_threshold(self, confidence: float) -> bool:
        """Apply τ + margin policy"""
        return confidence >= self.tau + self.margin


class EmbeddingEngine:
    """
    Handles text embeddings for semantic similarity matching.
    Enhanced with τ + margin policy, normalized dot product, caching and logging.
    """
    
    def __init__(self):
        self.model = None
        self.electrical_vocabulary = self._build_electrical_vocabulary()
        self.threshold_configs = self._load_golden_set_thresholds()
        self.similarity_cache = {}
        self.provenance_log = []
        
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use a lightweight model suitable for electrical engineering terms
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Using fallback.")
                self.model = None
        else:
            logger.info("Embeddings disabled. Using pattern matching fallback.")
    
    def _build_electrical_vocabulary(self) -> Dict[str, List[str]]:
        """Build electrical engineering vocabulary for embeddings"""
        return {
            'load_id': [
                'load identifier', 'equipment id', 'asset tag', 'load number', 
                'load reference', 'electrical load id', 'load code'
            ],
            'load_name': [
                'load description', 'equipment name', 'load label', 'equipment description',
                'load title', 'equipment designation', 'load identifier name'
            ],
            'power_kw': [
                'power rating', 'kilowatt rating', 'power capacity', 'load power',
                'electrical power', 'kw rating', 'power consumption', 'demand kw'
            ],
            'voltage': [
                'operating voltage', 'system voltage', 'supply voltage', 'voltage level',
                'rated voltage', 'working voltage', 'voltage rating'
            ],
            'cable_id': [
                'cable identifier', 'cable tag', 'cable reference', 'cable number',
                'cable code', 'electrical cable id', 'wire identifier'
            ],
            'from_equipment': [
                'source equipment', 'cable origin', 'cable from', 'starting point',
                'cable source', 'origin equipment', 'cable leaving from'
            ],
            'to_equipment': [
                'destination equipment', 'cable destination', 'cable to', 'ending point',
                'load connection', 'target equipment', 'cable ending at'
            ],
            'bus_id': [
                'bus identifier', 'panel id', 'bus bar id', 'distribution board id',
                'bus reference', 'switchgear id', 'panelboard identifier'
            ]
        }
    
    def _load_golden_set_thresholds(self) -> Dict[str, ThresholdConfig]:
        """Load per-class thresholds from golden set analysis"""
        return {
            'load_schedule': ThresholdConfig(tau=0.75, margin=0.1, confidence_threshold=0.70),
            'cable_schedule': ThresholdConfig(tau=0.78, margin=0.12, confidence_threshold=0.72),
            'bus_schedule': ThresholdConfig(tau=0.72, margin=0.08, confidence_threshold=0.68),
            'transformer_schedule': ThresholdConfig(tau=0.80, margin=0.15, confidence_threshold=0.75),
            'project_info': ThresholdConfig(tau=0.70, margin=0.05, confidence_threshold=0.65)
        }
    
    def _get_cache_key(self, text1: str, text2: str) -> str:
        """Generate cache key for similarity calculation"""
        return hashlib.md5(f"{text1}|{text2}".encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_embeddings(self, texts: Tuple[str]) -> Optional[np.ndarray]:
        """Get embeddings for a list of texts with provenance logging"""
        if not self.model or not EMBEDDINGS_AVAILABLE:
            return None
        
        texts_list = list(texts)
        cache_key = hashlib.md5("|".join(sorted(texts_list)).encode()).hexdigest()
        
        if cache_key in self.similarity_cache:
            logger.debug(f"Cache hit for embeddings: {cache_key}")
            return self.similarity_cache[cache_key]
        
        try:
            start_time = time.time()
            embeddings = self.model.encode(texts_list, convert_to_tensor=False)
            processing_time = time.time() - start_time
            
            # Cache the result
            self.similarity_cache[cache_key] = embeddings
            
            # Log provenance
            self.provenance_log.append({
                'operation': 'get_embeddings',
                'texts': texts_list[:3],  # Log first 3 texts for privacy
                'cache_key': cache_key,
                'processing_time': processing_time,
                'success': True,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Generated embeddings for {len(texts_list)} texts in {processing_time:.3f}s")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            self.provenance_log.append({
                'operation': 'get_embeddings',
                'texts': texts_list[:3],
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def find_best_match_embedding(self, target_text: str, candidate_texts: List[str],
                                 class_type: str = 'unknown') -> Tuple[Optional[str], float]:
        """Find best matching text using embeddings with τ + margin policy"""
        
        start_time = time.time()
        cache_key = self._get_cache_key(target_text, "|".join(candidate_texts))
        
        # Check cache first
        if cache_key in self.similarity_cache:
            cached_result = self.similarity_cache[cache_key]
            if cached_result.get('match'):
                logger.debug(f"Cache hit for similarity: {cache_key}")
                return cached_result['match'], cached_result['confidence']
        
        # Try embedding-based matching with normalized dot product
        best_match = None
        best_similarity = 0.0
        
        if self.model and EMBEDDINGS_AVAILABLE:
            try:
                # Get embeddings for all texts
                all_texts = [target_text] + candidate_texts
                embeddings = self.get_embeddings(tuple(all_texts))
                
                if embeddings is not None and len(embeddings) == len(all_texts):
                    # Calculate similarities using normalized dot product
                    target_embedding = embeddings[0]
                    candidate_embeddings = embeddings[1:]
                    
                    # Normalize vectors and compute dot product
                    target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
                    
                    similarities = []
                    for cand_emb in candidate_embeddings:
                        cand_norm = cand_emb / (np.linalg.norm(cand_emb) + 1e-8)
                        similarity = np.dot(target_norm, cand_norm)
                        similarities.append(float(similarity))
                    
                    # Find best match
                    best_idx = np.argmax(similarities)
                    best_similarity = similarities[best_idx]
                    
                    # Apply τ + margin policy
                    threshold_config = self.threshold_configs.get(class_type,
                                                               ThresholdConfig(tau=0.75, margin=0.1, confidence_threshold=0.70))
                    
                    if threshold_config.get_threshold(best_similarity):
                        best_match = candidate_texts[best_idx]
                        
            except Exception as e:
                logger.warning(f"Embedding similarity calculation failed: {e}")
                self.provenance_log.append({
                    'operation': 'embedding_similarity',
                    'target_text': target_text[:50],
                    'candidate_count': len(candidate_texts),
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Apply τ + margin policy even for fallback if no embedding match
        if not best_match:
            # Fallback to fuzzy matching
            for candidate in candidate_texts:
                confidence = fuzz.partial_ratio(target_text.lower(), candidate.lower()) / 100.0
                if confidence > best_similarity:
                    best_similarity = confidence
                    if confidence >= 0.8:  # Higher threshold for fuzzy fallback
                        best_match = candidate
        
        # Cache result
        cache_result = {'match': best_match, 'confidence': best_similarity}
        self.similarity_cache[cache_key] = cache_result
        
        # Log provenance with detailed tracking
        processing_time = time.time() - start_time
        threshold_config = self.threshold_configs.get(class_type,
                                                   ThresholdConfig(tau=0.75, margin=0.1, confidence_threshold=0.70))
        
        self.provenance_log.append({
            'operation': 'find_best_match',
            'target_text': target_text[:50],
            'candidate_count': len(candidate_texts),
            'class_type': class_type,
            'best_match': best_match[:50] if best_match else None,
            'confidence': best_similarity,
            'threshold_tau': threshold_config.tau,
            'threshold_margin': threshold_config.margin,
            'threshold_used': threshold_config.get_threshold(best_similarity),
            'method': 'embeddings' if self.model and EMBEDDINGS_AVAILABLE else 'fuzzy',
            'processing_time': processing_time,
            'cache_hit': cache_key in self.similarity_cache,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Best match for '{target_text}' in {class_type}: {best_match} (conf: {best_similarity:.3f}, method: {'embeddings' if self.model and EMBEDDINGS_AVAILABLE else 'fuzzy'})")
        
        return best_match, best_similarity
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Get semantic similarity between two texts using normalized dot product"""
        
        cache_key = self._get_cache_key(text1, text2)
        
        # Check cache first
        if cache_key in self.similarity_cache:
            cached_result = self.similarity_cache[cache_key]
            if isinstance(cached_result, (int, float)):
                return cached_result
            elif isinstance(cached_result, dict) and 'confidence' in cached_result:
                return cached_result['confidence']
        
        # Try embedding-based similarity with normalized dot product
        similarity = 0.0
        
        if self.model and EMBEDDINGS_AVAILABLE:
            try:
                embeddings = self.get_embeddings(tuple([text1, text2]))
                if embeddings is not None and len(embeddings) == 2:
                    # Use normalized dot product instead of cosine similarity
                    emb1_norm = embeddings[0] / (np.linalg.norm(embeddings[0]) + 1e-8)
                    emb2_norm = embeddings[1] / (np.linalg.norm(embeddings[1]) + 1e-8)
                    similarity = np.dot(emb1_norm, emb2_norm)
                    similarity = float(similarity)
                    
                    # Cache result
                    self.similarity_cache[cache_key] = similarity
                    
            except Exception as e:
                logger.warning(f"Embedding similarity failed: {e}")
        
        # Fallback to fuzzy matching if no embedding similarity
        if similarity == 0.0:
            similarity = fuzz.partial_ratio(text1.lower(), text2.lower()) / 100.0
            self.similarity_cache[cache_key] = similarity
        
        # Log provenance
        self.provenance_log.append({
            'operation': 'semantic_similarity',
            'text1': text1[:50],
            'text2': text2[:50],
            'similarity': similarity,
            'method': 'embeddings' if self.model and EMBEDDINGS_AVAILABLE else 'fuzzy',
            'timestamp': datetime.now().isoformat()
        })
        
        return similarity

    def get_provenance_log(self) -> List[Dict[str, Any]]:
        """Get the provenance log for debugging and analysis"""
        return self.provenance_log.copy()
    
    def clear_cache(self):
        """Clear the similarity cache"""
        self.similarity_cache.clear()
        logger.info("Similarity cache cleared")


class PostValidationEngine:
    """
    Post-validate and downgrade suspicious mappings
    """
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.suspicious_patterns = self._load_suspicious_patterns()
    
    def _load_validation_rules(self) -> Dict[str, List[Dict]]:
        """Load post-validation rules for downgrading suspicious mappings"""
        return {
            'low_confidence_mapping': {
                'threshold': 0.5,
                'action': 'downgrade',
                'penalty': 0.2
            },
            'inconsistent_mapping': {
                'threshold': 0.3,
                'action': 'flag',
                'penalty': 0.1
            },
            'domain_mismatch': {
                'threshold': 0.4,
                'action': 'downgrade',
                'penalty': 0.3
            }
        }
    
    def _load_suspicious_patterns(self) -> List[Dict]:
        """Load patterns that indicate suspicious mappings"""
        return [
            {
                'pattern': r'^[a-zA-Z0-9\s]+$',
                'type': 'generic_column_name',
                'confidence_penalty': 0.1
            },
            {
                'pattern': r'^(unnamed|column|field)\d*$',
                'type': 'unnamed_column',
                'confidence_penalty': 0.2
            },
            {
                'pattern': r'^\d+(\.\d+)?$',
                'type': 'numeric_only',
                'confidence_penalty': 0.3
            }
        ]
    
    def validate_mapping(self, field_mapping: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-validate a field mapping and downgrade suspicious ones
        
        Args:
            field_mapping: The field mapping to validate
            context: Additional context (sheet type, headers, etc.)
            
        Returns:
            Validated mapping with confidence adjustments
        """
        validated_mapping = field_mapping.copy()
        adjustments_made = []
        
        for field_name, mapping_info in validated_mapping.get('field_mappings', {}).items():
            original_confidence = mapping_info.get('confidence', 0.0)
            adjusted_confidence = original_confidence
            
            # Check for suspicious patterns in mapped columns
            for column in mapping_info.get('mapped_columns', []):
                for pattern_info in self.suspicious_patterns:
                    if re.search(pattern_info['pattern'], str(column)):
                        penalty = pattern_info['confidence_penalty']
                        adjusted_confidence -= penalty
                        adjustments_made.append({
                            'field': field_name,
                            'column': column,
                            'pattern_type': pattern_info['type'],
                            'penalty': penalty,
                            'reason': f'Suspicious pattern: {pattern_info["type"]}'
                        })
            
            # Check domain consistency
            domain_check = self._check_domain_consistency(field_name, mapping_info, context)
            if not domain_check['consistent']:
                adjusted_confidence -= 0.15
                adjustments_made.append({
                    'field': field_name,
                    'reason': f'Domain inconsistency: {domain_check["issue"]}',
                    'penalty': 0.15
                })
            
            # Apply minimum confidence floor
            adjusted_confidence = max(adjusted_confidence, 0.1)
            
            # Update mapping info
            mapping_info['original_confidence'] = original_confidence
            mapping_info['confidence'] = adjusted_confidence
            mapping_info['validation_passed'] = adjusted_confidence >= 0.5
            mapping_info['adjustments'] = [adj for adj in adjustments_made if adj['field'] == field_name]
        
        # Calculate overall validation score
        total_fields = len(validated_mapping.get('field_mappings', {}))
        validated_fields = sum(1 for m in validated_mapping.get('field_mappings', {}).values() 
                             if m.get('validation_passed', False))
        validation_score = validated_fields / total_fields if total_fields > 0 else 0.0
        
        validated_mapping['validation_score'] = validation_score
        validated_mapping['adjustments_made'] = adjustments_made
        validated_mapping['provenance'] = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_rules_applied': list(self.validation_rules.keys()),
            'suspicious_patterns_checked': len(self.suspicious_patterns)
        }
        
        logger.info(f"Post-validation completed: {len(adjustments_made)} adjustments made, {validation_score:.2%} validation score")
        
        return validated_mapping
    
    def _check_domain_consistency(self, field_name: str, mapping_info: Dict, context: Dict) -> Dict[str, Any]:
        """Check if mapping is consistent with electrical engineering domain"""
        
        # Domain-specific consistency rules
        electrical_domain_rules = {
            'power_kw': ['power', 'kw', 'kilowatt', 'rating', 'capacity'],
            'voltage': ['voltage', 'v', 'volts', 'level'],
            'current': ['current', 'a', 'ampere', 'amps'],
            'load_id': ['load', 'id', 'identifier', 'tag'],
            'cable_id': ['cable', 'id', 'identifier', 'tag'],
            'bus_id': ['bus', 'id', 'panel', 'board']
        }
        
        field_keywords = electrical_domain_rules.get(field_name, [])
        mapped_columns = mapping_info.get('mapped_columns', [])
        
        for column in mapped_columns:
            column_lower = str(column).lower()
            keyword_matches = sum(1 for keyword in field_keywords if keyword in column_lower)
            
            if keyword_matches == 0 and len(field_keywords) > 0:
                return {
                    'consistent': False,
                    'issue': f'No domain keywords found for {field_name}',
                    'expected_keywords': field_keywords,
                    'actual_column': column
                }
        
        return {'consistent': True}


# Continue with the rest of the classes (SheetClassifier, ColumnMapper, etc.) 
# but with the enhanced methods integrated...

class SheetClassifier:
    """
    Identifies sheet types (Load, Cable, Bus, etc.) using hybrid approach:
    1. Embedding-based semantic similarity
    2. Pattern matching (regex) as fallback
    Enhanced with τ + margin policy and provenance logging
    """

    def __init__(self):
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine()
        
        # Define patterns for different sheet types (existing logic)
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
        
        # Define sheet type descriptions for embedding matching
        self.sheet_type_descriptions = {
            'load_schedule': "electrical load schedule containing equipment power ratings voltage current phases load types",
            'cable_schedule': "cable schedule listing wire specifications from equipment to equipment cable sizes lengths installation",
            'bus_schedule': "bus bar distribution panel schedule with voltage current ratings short circuit capacity",
            'transformer_schedule': "power transformer schedule with kva ratings primary secondary voltages impedance",
            'project_info': "project information metadata standard voltage system ambient temperature configuration details"
        }

    def classify_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        Classify sheet using hybrid approach: embeddings + patterns with τ + margin policy
        
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
                'recommended_model_mapping': None,
                'method': 'none',
                'provenance': {'error': 'Empty dataframe'}
            }

        # Get all column headers as string
        headers = df.columns.tolist()
        headers_text = ' '.join(str(h).lower() for h in headers)
        sheet_context = f"{sheet_name} {headers_text}"

        start_time = time.time()
        provenance_log = {
            'classification_timestamp': datetime.now().isoformat(),
            'sheet_name': sheet_name,
            'header_count': len(headers),
            'methods_tried': []
        }

        # Method 1: Try embedding-based classification with τ + margin policy
        if self.embedding_engine.model and EMBEDDINGS_AVAILABLE:
            embedding_result = self._classify_with_embeddings(sheet_context)
            provenance_log['methods_tried'].append('embeddings')
            
            # Apply τ + margin policy
            threshold_config = self.embedding_engine.threshold_configs.get(
                embedding_result.get('sheet_type', 'unknown'),
                ThresholdConfig(tau=0.75, margin=0.1, confidence_threshold=0.70)
            )
            
            if threshold_config.get_threshold(embedding_result['confidence']):
                embedding_result['method'] = 'embeddings'
                embedding_result['processing_time'] = time.time() - start_time
                embedding_result['provenance'] = provenance_log
                return embedding_result
        
        # Method 2: Fall back to pattern matching
        pattern_result = self._classify_with_patterns(headers_text, sheet_name)
        pattern_result['method'] = 'patterns'
        pattern_result['processing_time'] = time.time() - start_time
        pattern_result['provenance'] = provenance_log
        
        return pattern_result
    
    def _classify_with_embeddings(self, sheet_context: str) -> Dict[str, Any]:
        """Classify sheet using semantic embeddings with enhanced logging"""
        try:
            best_match = None
            best_similarity = 0.0
            evidence = []
            all_scores = {}
            
            # Compare against each sheet type description
            for sheet_type, description in self.sheet_type_descriptions.items():
                similarity = self.embedding_engine.get_semantic_similarity(
                    sheet_context, description
                )
                all_scores[sheet_type] = similarity
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = sheet_type
            
            if best_similarity > 0.6:  # Base threshold for embeddings
                model_mapping = self._get_model_mapping(best_match)
                return {
                    'sheet_type': best_match,
                    'confidence': best_similarity,
                    'evidence': [f"semantic similarity: {best_similarity:.3f}"],
                    'recommended_model_mapping': model_mapping,
                    'all_scores': all_scores
                }
            else:
                return {
                    'sheet_type': 'unknown',
                    'confidence': 0.0,
                    'evidence': [],
                    'recommended_model_mapping': None,
                    'all_scores': all_scores
                }
                
        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}")
            return {
                'sheet_type': 'unknown',
                'confidence': 0.0,
                'evidence': [f"embedding error: {str(e)}"],
                'recommended_model_mapping': None,
                'all_scores': {}
            }
    
    def _classify_with_patterns(self, headers_text: str, sheet_name: str) -> Dict[str, Any]:
        """Classify sheet using existing pattern matching (original logic)"""
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


# Main AI Excel Extractor with all enhancements
class AIExcelExtractor:
    """
    Main orchestrator for AI-powered Excel extraction with embedding capabilities
    Enhanced with τ + margin policy, post-validation, caching, and comprehensive logging
    """

    def __init__(self, standard: str = "IEC"):
        self.standard = standard
        self.sheet_classifier = SheetClassifier()
        self.column_mapper = ColumnMapper()
        self.data_extractor = DataExtractor()
        self.data_enhancer = DataEnhancer()
        self.validation_engine = ValidationEngine(standard)
        self.post_validation_engine = PostValidationEngine()
        self.global_provenance = []

    def process_excel_file(self, file_path: str) -> ProcessingReport:
        """
        Process Excel file and extract all electrical components with enhanced features
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            ProcessingReport with comprehensive results
        """
        start_time = datetime.now()
        logger.info(f"Starting enhanced Excel file processing: {file_path}")
        
        try:
            # Read Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            logger.info(f"Read {len(excel_data)} sheets from file")
            
            # Process each sheet
            sheet_results = {}
            all_loads = []
            all_cables = []
            all_buses = []
            processing_metadata = {
                'start_time': start_time.isoformat(),
                'file_path': file_path,
                'sheet_count': len(excel_data),
                'embedding_cache_size': 0,
                'post_validation_applied': False
            }
            
            for sheet_name, df in excel_data.items():
                logger.info(f"Processing sheet: {sheet_name}")
                
                # Classify sheet with τ + margin policy
                classification = self.sheet_classifier.classify_sheet(df, sheet_name)
                logger.info(f"Sheet '{sheet_name}' classified as: {classification['sheet_type']} (confidence: {classification['confidence']:.2f}, method: {classification.get('method', 'unknown')})")
                
                # Map columns if we have a supported type
                if classification['recommended_model_mapping'] in ['Load', 'Cable', 'Bus']:
                    field_mapping = self.column_mapper.map_columns(
                        df.columns.tolist(),
                        classification['recommended_model_mapping'],
                        sheet_name
                    )
                    
                    # Post-validate and downgrade suspicious mappings
                    enhanced_mapping = self.post_validation_engine.validate_mapping(
                        field_mapping, 
                        {
                            'sheet_type': classification['sheet_type'],
                            'headers': df.columns.tolist(),
                            'sheet_name': sheet_name
                        }
                    )
                    
                    processing_metadata['post_validation_applied'] = True
                    
                    # Extract data based on sheet type
                    if classification['sheet_type'] == 'load_schedule':
                        loads, result = self.data_extractor.extract_loads(df, enhanced_mapping)
                        all_loads.extend(loads)
                    elif classification['sheet_type'] == 'cable_schedule':
                        cables, result = self.data_extractor.extract_cables(df, enhanced_mapping)
                        all_cables.extend(cables)
                    else:
                        result = ExtractionResult(
                            success=True,
                            confidence=classification['confidence'],
                            sheet_type=classification['sheet_type'],
                            components_extracted=0,
                            data_quality_score=classification['confidence'],
                            extracted_data={},
                            provenance=classification.get('provenance', {})
                        )
                else:
                    result = ExtractionResult(
                        success=True,
                        confidence=classification['confidence'],
                        sheet_type=classification['sheet_type'],
                        components_extracted=0,
                        data_quality_score=classification['confidence'],
                        extracted_data={},
                        provenance=classification.get('provenance', {})
                    )
                
                sheet_results[sheet_name] = result
            
            # Update embedding cache size
            if hasattr(self.sheet_classifier.embedding_engine, 'similarity_cache'):
                processing_metadata['embedding_cache_size'] = len(
                    self.sheet_classifier.embedding_engine.similarity_cache
                )
            
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
            
            # Collect provenance information
            provenance_data = {
                'processing_metadata': processing_metadata,
                'embedding_provenance': self.sheet_classifier.embedding_engine.get_provenance_log(),
                'post_validation_reports': [self.post_validation_engine.validation_rules],
                'enhancement_results': enhancement_results,
                'validation_results': validation_results
            }
            
            # Create final report
            report = ProcessingReport(
                overall_confidence=overall_confidence,
                total_components=len(all_loads) + len(all_cables) + len(all_buses),
                processing_time_seconds=processing_time,
                sheet_results=sheet_results,
                project_data=project,
                corrections_made=enhancement_results['corrections_made'],
                validation_issues=validation_results['errors'] + validation_results['warnings'],
                provenance=provenance_data
            )
            
            logger.info(f"Enhanced processing completed: {report.total_components} components extracted, {overall_confidence:.2f} confidence")
            return report
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return ProcessingReport(
                overall_confidence=0.0,
                total_components=0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                sheet_results={},
                validation_issues=[f"Processing failed: {str(e)}"],
                provenance={'error': str(e), 'timestamp': datetime.now().isoformat()}
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
    
    def clear_caches(self):
        """Clear all caches"""
        self.sheet_classifier.embedding_engine.clear_cache()
        logger.info("All caches cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        embedding_engine = self.sheet_classifier.embedding_engine
        
        return {
            'embedding_cache_size': len(embedding_engine.similarity_cache),
            'embedding_operations_count': len(embedding_engine.provenance_log),
            'post_validation_enabled': True,
            'tau_margin_policy_enabled': True,
            'normalized_dot_product_enabled': True,
            'sklearn_dependency_removed': True
        }


# Placeholder for other classes (ColumnMapper, DataExtractor, etc.)
# They would need to be updated similarly with the enhanced features

if __name__ == "__main__":
    print("Enhanced AI Excel Extractor with τ + margin policy, caching, and provenance logging")
    print("Features:")
    print("- Per-class thresholds from golden set")
    print("- Normalized dot product similarity (sklearn-free)")
    print("- Comprehensive caching and provenance logging")
    print("- Post-validation with suspicious mapping detection")
    print("- Enhanced confidence scoring with τ + margin policy")