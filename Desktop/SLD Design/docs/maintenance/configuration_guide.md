# Configuration Guide - AI Excel Extraction System

## Overview

This comprehensive configuration guide covers all aspects of configuring the AI-Powered Excel Extraction System for optimal performance, security, and reliability across different deployment scenarios.

## 📋 Configuration Architecture

### Configuration Hierarchy

The system uses a layered configuration approach:

1. **Default Configuration** - Built-in defaults
2. **Environment Configuration** - Environment-specific settings
3. **File Configuration** - Configuration files
4. **Runtime Configuration** - API/CLI overrides

### Configuration File Structure

```
~/.ai-excel-extractor/
├── config.yaml              # Main configuration
├── environment              # Environment variables
├── credentials.yaml         # API keys and secrets
├── models/                  # AI model configurations
├── templates/               # Processing templates
└── logs/                    # Log configuration
```

## 🔧 Core Configuration

### Primary Configuration File

#### config.yaml Structure
```yaml
# Main configuration file: ~/.ai-excel-extractor/config.yaml

# System configuration
system:
  name: "AI Excel Extractor"
  version: "2.1.0"
  environment: "production"  # development, staging, production
  debug: false
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  
# Application settings
application:
  name: "AI Excel Extraction Service"
  description: "Advanced AI-powered Excel data extraction and validation"
  author: "AI Excel Extractor Team"
  license: "Commercial"
  
# Processing configuration
processing:
  # File handling
  max_file_size: "100MB"
  allowed_extensions: [".xlsx", ".xls", ".csv"]
  temp_directory: "/tmp/ai-extractor"
  cleanup_interval: 3600  # seconds
  
  # AI processing
  concurrent_jobs: 4
  batch_size: 32
  timeout: 300
  retry_attempts: 3
  retry_delay: 5
  
  # Memory management
  max_memory_usage: "4GB"
  garbage_collection: "aggressive"
  cache_enabled: true
  cache_size: "1GB"
  
# AI model configuration
ai:
  # Model paths
  models_directory: "/opt/ai-excel-extractor/models"
  primary_model: "neural_network_v2.1"
  backup_model: "ensemble_model"
  
  # Processing parameters
  confidence_threshold: 0.85
  auto_correction: true
  validation_enabled: true
  learning_enabled: false  # Disable for production
  
  # Model-specific settings
  models:
    neural_network:
      type: "transformer"
      parameters:
        max_sequence_length: 2048
        temperature: 0.1
        top_p: 0.9
        top_k: 50
        repetition_penalty: 1.1
        
    ensemble:
      type: "ensemble"
      models: ["neural_network", "random_forest", "svm"]
      voting_strategy: "soft"
      
  # Validation settings
  validation:
    cross_validation: false  # Disable for performance
    validation_split: 0.2
    early_stopping: true
    epochs: 100
    
# API configuration
api:
  # Server settings
  host: "0.0.0.0"
  port: 8080
  workers: 4
  max_connections: 1000
  connection_timeout: 30
  
  # CORS settings
  enable_cors: true
  allowed_origins: ["*"]  # Configure for production
  allowed_methods: ["GET", "POST", "PUT", "DELETE"]
  allowed_headers: ["*"]
  
  # Security
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
    
  # SSL/TLS
  ssl:
    enabled: false  # Enable in production
    cert_file: null
    key_file: null
    verify_mode: "CERT_REQUIRED"
    
# Database configuration
database:
  # Primary database
  url: "sqlite:///var/data/ai_extractor.db"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  echo: false  # Set to true for SQL logging
  
  # Connection settings
  connect_args:
    timeout: 30
    check_same_thread: false
    
  # Migration settings
  migrate:
    enabled: true
    revision: "head"
    
# Cache configuration
cache:
  # Cache backend
  backend: "redis"  # redis, memory, disk
  ttl: 3600  # seconds
  
  # Redis settings (if using Redis)
  redis:
    host: "localhost"
    port: 6379
    db: 0
    password: null
    ssl: false
    connection_pool_size: 10
    
  # Memory cache settings
  memory:
    max_size: "100MB"
    ttl: 3600
    
  # Disk cache settings
  disk:
    directory: "/var/cache/ai-extractor"
    max_size: "1GB"
    compression: true
    
# Storage configuration
storage:
  # Input storage
  input:
    type: "local"  # local, s3, gcs, azure
    path: "/var/data/input"
    backup_enabled: true
    compression: "gzip"
    
  # Output storage
  output:
    type: "local"
    path: "/var/data/output"
    backup_enabled: true
    versioning: true
    
  # Temporary storage
  temp:
    type: "local"
    path: "/tmp/ai-extractor"
    auto_cleanup: true
    cleanup_interval: 3600
    
# Logging configuration
logging:
  # Log level
  level: "INFO"
  format: "json"  # json, text
  date_format: "%Y-%m-%d %H:%M:%S"
  
  # File logging
  file:
    enabled: true
    path: "/var/log/ai-excel-extractor/app.log"
    max_size: "100MB"
    backup_count: 5
    rotate_on_mention: true
    
  # Console logging
  console:
    enabled: true
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
  # Structured logging
  structured:
    enabled: true
    fields: ["timestamp", "level", "logger", "message", "module", "function", "line"]
    
# Monitoring configuration
monitoring:
  # Health checks
  health_check:
    enabled: true
    interval: 30  # seconds
    timeout: 5
    
  # Metrics
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
    format: "prometheus"
    
  # Performance monitoring
  performance:
    enabled: true
    sampling_rate: 0.1
    trace_sampling_rate: 0.01
    
  # Alerting
  alerting:
    enabled: true
    email:
      enabled: true
      smtp_server: "smtp.company.com"
      smtp_port: 587
      username: "alerts@company.com"
      password_file: "/etc/ai-excel-extractor/email_password"
      recipients: ["admin@company.com", "ops@company.com"]
      
    webhook:
      enabled: true
      url: "https://hooks.slack.com/your-webhook-url"
      method: "POST"
      timeout: 10
      
# Security configuration
security:
  # Authentication
  authentication:
    enabled: true
    method: "jwt"  # jwt, oauth2, ldap
    jwt:
      secret_key_file: "/etc/ai-excel-extractor/jwt_secret"
      algorithm: "HS256"
      expiration_time: 86400  # 24 hours in seconds
      issuer: "ai-excel-extractor"
      
  # Authorization
  authorization:
    enabled: true
    default_role: "viewer"
    roles:
      administrator:
        permissions: ["*"]
      engineer:
        permissions: ["read", "write", "process", "export"]
      viewer:
        permissions: ["read", "export"]
        
  # Encryption
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_file: "/etc/ai-excel-extractor/encryption_key"
    key_rotation_days: 90
    
  # Access control
  access_control:
    enabled: true
    rate_limiting:
      enabled: true
      requests_per_hour: 1000
      burst_size: 100
      
    ip_whitelist:
      enabled: false
      addresses: []
      
    ip_blacklist:
      enabled: false
      addresses: []
```

## 🔐 Security Configuration

### Authentication Setup

#### JWT Configuration
```yaml
# JWT authentication configuration
security:
  authentication:
    method: "jwt"
    jwt:
      secret_key_file: "/etc/ai-excel-extractor/jwt_secret"
      algorithm: "HS256"
      expiration_time: 86400  # 24 hours
      issuer: "ai-excel-extractor"
      audience: "ai-excel-users"
      
      # Refresh token settings
      refresh_token:
        enabled: true
        expiration_time: 604800  # 7 days
        
      # Blacklist settings
      blacklist:
        enabled: true
        storage: "redis"
        ttl: 86400
```

#### SSL/TLS Configuration
```yaml
# SSL/TLS configuration
api:
  ssl:
    enabled: true
    cert_file: "/etc/ssl/certs/ai-excel-extractor.crt"
    key_file: "/etc/ssl/private/ai-excel-extractor.key"
    ca_bundle: "/etc/ssl/certs/ca-bundle.crt"
    
    # Certificate validation
    verify_mode: "CERT_REQUIRED"  # CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED
    check_hostname: true
    
    # TLS versions
    min_tls_version: "1.2"  # 1.2, 1.3
    max_tls_version: "1.3"
    
    # HSTS
    hsts:
      enabled: true
      max_age: 31536000  # 1 year
      include_subdomains: true
      preload: true
      
    # Redirect HTTP to HTTPS
    redirect_http: true
    http_port: 8080
    https_port: 8443
```

## 📊 Performance Configuration

### Resource Management

#### Memory Configuration
```yaml
# Memory management settings
processing:
  memory:
    max_usage: "4GB"
    gc_strategy: "aggressive"  # conservative, aggressive, adaptive
    gc_threshold: 700  # MB
    gc_interval: 60  # seconds
    
  # Memory monitoring
  monitoring:
    enabled: true
    alert_threshold: "3.5GB"  # 87.5% of max
    critical_threshold: "3.8GB"  # 95% of max
```

### Caching Strategy

#### Multi-Level Caching
```yaml
# Multi-level caching configuration
cache:
  levels:
    # Level 1: In-memory cache (fastest)
    memory:
      enabled: true
      max_size: "512MB"
      ttl: 1800  # 30 minutes
      gc_interval: 300
      
    # Level 2: Redis cache (fast, shared)
    redis:
      enabled: true
      host: "localhost"
      port: 6379
      db: 0
      password: null
      ssl: false
      
      # Connection pool
      connection_pool:
        max_connections: 50
        retry_on_timeout: true
        
      # TTL settings
      ttl:
        processed_files: 3600  # 1 hour
        models: 86400  # 24 hours
        configurations: 3600  # 1 hour
        
    # Level 3: Disk cache (slow, large)
    disk:
      enabled: true
      path: "/var/cache/ai-excel-extractor"
      max_size: "50GB"
      compression: true
      encryption: true
```

## 🗄️ Database Configuration

### PostgreSQL Setup

#### Connection Configuration
```yaml
# PostgreSQL configuration
database:
  url: "postgresql://aiuser:password@localhost:5432/ai_extractor"
  
  # Connection pool
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  
  # Connection settings
  connect_args:
    connect_timeout: 10
    options: "-c timezone=UTC"
    application_name: "ai-excel-extractor"
    
  # SSL configuration
  ssl:
    enabled: true
    mode: "require"  # disable, allow, prefer, require, verify-ca, verify-full
    cert_file: "/etc/ssl/certs/client.crt"
    key_file: "/etc/ssl/private/client.key"
    ca_file: "/etc/ssl/certs/ca.crt"
```

## 📝 Logging Configuration

### Structured Logging

#### JSON Log Format
```yaml
# Structured logging configuration
logging:
  # Global settings
  level: "INFO"
  format: "json"
  date_format: "%Y-%m-%d %H:%M:%S"
  
  # File logging
  file:
    enabled: true
    path: "/var/log/ai-excel-extractor/app.log"
    max_size: "100MB"
    backup_count: 10
    rotate_on_mention: true
    
  # Console logging
  console:
    enabled: true
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
  # Structured logging fields
  structured:
    enabled: true
    fields:
      base:
        - "timestamp"
        - "level"
        - "logger"
        - "message"
        - "module"
        - "function"
        - "line"
        
      request:
        - "request_id"
        - "user_id"
        - "ip_address"
        - "user_agent"
        - "endpoint"
        - "method"
        - "status_code"
        - "response_time"
        
      processing:
        - "file_id"
        - "file_name"
        - "file_size"
        - "processing_time"
        - "confidence_score"
        - "error_count"
```

## 📈 Monitoring Configuration

### Metrics Collection

#### Prometheus Metrics
```yaml
# Metrics configuration
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
    format: "prometheus"
    
    # Custom metrics
    custom_metrics:
      enabled: true
      prefix: "ai_extractor_"
      
      # Business metrics
      metrics:
        files_processed_total:
          type: "counter"
          description: "Total number of files processed"
          
        processing_duration_seconds:
          type: "histogram"
          description: "Time spent processing files"
          buckets: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
          
        confidence_score:
          type: "histogram"
          description: "Distribution of confidence scores"
          buckets: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
```

### Health Checks

#### Comprehensive Health Monitoring
```yaml
# Health check configuration
monitoring:
  health_checks:
    enabled: true
    interval: 30
    timeout: 10
    
    # Component health checks
    components:
      database:
        enabled: true
        check_query: "SELECT 1"
        timeout: 5
        
      redis:
        enabled: true
        check_command: "PING"
        timeout: 3
        
      ai_models:
        enabled: true
        check_models: ["primary", "backup"]
        timeout: 10
        
      disk_space:
        enabled: true
        threshold: 85  # percentage
        paths: ["/var/data", "/tmp", "/var/log"]
        
      memory:
        enabled: true
        threshold: 90  # percentage
```

### Alerting Configuration

#### Multi-Channel Alerting
```yaml
# Alerting configuration
monitoring:
  alerting:
    enabled: true
    
    # Email alerts
    email:
      enabled: true
      smtp_server: "smtp.company.com"
      smtp_port: 587
      username: "alerts@company.com"
      password_file: "/etc/ai-excel-extractor/email_password"
      use_tls: true
      
      recipients:
        critical: ["admin@company.com", "ops@company.com"]
        warning: ["admin@company.com"]
        info: ["ops@company.com"]
        
    # Slack alerts
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/your-webhook"
      channel: "#ai-extractor-alerts"
      username: "AI Extractor Monitor"
```

## 🔄 Integration Configuration

### CAD Software Integration

#### AutoCAD Integration
```yaml
# AutoCAD integration
integrations:
  cad:
    enabled: false
    autocad:
      enabled: false
      version: "2023"
      api_key: null  # Set in credentials file
      
      # AutoCAD settings
      settings:
        working_directory: "/tmp/autocad"
        temporary_files: true
        cleanup_interval: 3600
        
      # DWG processing
      dwg:
        enabled: true
        supported_versions: ["R12", "R14", "R2000", "R2004", "R2007", "R2010", "R2013", "R2016", "R2018", "R2021"]
        output_format: "dxf"
```

### Cloud Storage Integration

#### AWS S3 Integration
```yaml
# AWS S3 integration
integrations:
  cloud_storage:
    aws_s3:
      enabled: false
      bucket: "ai-excel-extractor-data"
      region: "us-east-1"
      prefix: "ai-excel-extractor/"
      
      # Authentication
      access_key_id_file: "/etc/ai-excel-extractor/aws_access_key"
      secret_access_key_file: "/etc/ai-excel-extractor/aws_secret_key"
      
      # Connection settings
      connection_timeout: 30
      max_retries: 3
      use_ssl: true
      verify_ssl: true
      
      # Transfer settings
      multipart_threshold: "100MB"
      multipart_chunksize: "10MB"
      max_concurrency: 10
      
      # S3 features
      versioning: true
      encryption:
        enabled: true
        server_side_encryption: "AES256"  # AES256, aws:kms
        kms_key_id: null
```

## 🛠️ Configuration Management

### Configuration Validation

#### Schema Validation
```bash
# Validate configuration
ai-excel-extractor config validate --schema /opt/ai-extractor/config/schema.yaml

# Validate specific sections
ai-excel-extractor config validate --section ai --section database

# Generate sample configuration
ai-excel-extractor config generate --template production --output config.yaml
```

### Configuration Templates

#### Environment-Specific Templates
```yaml
# Development template
development_template:
  system:
    environment: "development"
    debug: true
    log_level: "DEBUG"
    
  processing:
    max_file_size: "50MB"
    concurrent_jobs: 2
    timeout: 300
    
  database:
    url: "sqlite:///dev_data/ai_extractor.db"
    echo: true  # Enable SQL logging
    
  security:
    authentication:
      enabled: false  # Disable auth for development
      
# Production template
production_template:
  system:
    environment: "production"
    debug: false
    log_level: "INFO"
    
  processing:
    max_file_size: "100MB"
    concurrent_jobs: 8
    timeout: 300
    
  database:
    url: "postgresql://user:pass@prod-db:5432/ai_extractor"
    pool_size: 20
    ssl:
      enabled: true
      
  security:
    authentication:
      enabled: true
      method: "jwt"
      
  monitoring:
    alerting:
      enabled: true
```

## 📋 Configuration Checklist

### Production Readiness Checklist

#### Security Configuration
- [ ] SSL/TLS certificates installed and configured
- [ ] Authentication enabled and configured
- [ ] Authorization policies implemented
- [ ] API keys and secrets secured
- [ ] Rate limiting configured
- [ ] CORS settings properly configured
- [ ] IP whitelisting/blacklisting configured

#### Performance Configuration
- [ ] Database connection pooling configured
- [ ] Cache layers configured and optimized
- [ ] Worker processes configured for workload
- [ ] Memory limits set appropriately
- [ ] CPU affinity configured if needed
- [ ] Load balancing configured for multiple instances

#### Monitoring Configuration
- [ ] Health checks configured for all components
- [ ] Metrics collection enabled
- [ ] Alerting configured with proper thresholds
- [ ] Log aggregation configured
- [ ] Performance monitoring enabled
- [ ] Backup procedures configured

#### Integration Configuration
- [ ] External database connections configured
- [ ] Cloud storage integrations configured
- [ ] CAD software integrations configured
- [ ] API integrations configured
- [ ] Webhook notifications configured

### Configuration Validation Commands

```bash
# Validate entire configuration
ai-excel-extractor config validate

# Validate specific sections
ai-excel-extractor config validate --section security
ai-excel-extractor config validate --section database
ai-excel-extractor config validate --section processing

# Test configuration in different environment
ai-excel-extractor config test --environment development
ai-excel-extractor config test --environment staging
ai-excel-extractor config test --environment production

# Generate configuration report
ai-excel-extractor config report --format json
ai-excel-extractor config report --format html
ai-excel-extractor config report --format pdf

# Compare configurations
ai-excel-extractor config diff --file1 config_dev.yaml --file2 config_prod.yaml

# Backup current configuration
ai-excel-extractor config backup --output config_backup_$(date +%Y%m%d_%H%M%S).yaml

# Restore configuration
ai-excel-extractor config restore --file config_backup_20251102_120000.yaml
```

## 🎯 Next Steps

After completing the configuration:

1. **Validate Configuration**: Run validation tests to ensure all settings are correct
2. **Test Functionality**: Perform end-to-end testing with sample data
3. **Performance Tuning**: Monitor performance and adjust settings as needed
4. **Security Audit**: Conduct security review of all configuration settings
5. **Backup Configuration**: Securely backup configuration files
6. **Documentation**: Document any customizations made to the base configuration

For troubleshooting configuration issues, refer to the [Troubleshooting Guide](troubleshooting_guide.md).