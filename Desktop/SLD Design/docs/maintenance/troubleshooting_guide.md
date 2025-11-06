# Troubleshooting Guide - AI Excel Extraction System

## Overview

This comprehensive troubleshooting guide provides solutions to common issues, diagnostic procedures, and resolution strategies for the AI-Powered Excel Extraction System. It covers installation, configuration, runtime, performance, and integration issues.

## 🔍 Diagnostic Framework

### Initial Diagnostics

#### System Health Check
```bash
# Run comprehensive health check
ai-excel-extractor health-check --verbose

# Check system status
ai-excel-extractor system-status

# Component health check
ai-excel-extractor component-health --all

# Performance check
ai-excel-extractor performance-check
```

#### Log Analysis
```bash
# View recent logs
tail -100 /var/log/ai-excel-extractor/app.log

# Search for errors
grep -i "error" /var/log/ai-excel-extractor/app.log

# Search for warnings
grep -i "warning" /var/log/ai-excel-extractor/app.log

# Analyze error patterns
ai-excel-extractor log-analyze --pattern "ERROR" --timeframe "1h"

# Monitor logs in real-time
tail -f /var/log/ai-excel-extractor/app.log | grep -E "(ERROR|WARNING|CRITICAL)"
```

### System Information Collection

#### Environment Information
```bash
# Collect system information
ai-excel-extractor system-info --output system_report.json

# Collect configuration
ai-excel-extractor config dump --output current_config.yaml

# Collect version information
ai-excel-extractor version --detailed

# Collect resource usage
ai-excel-extractor resource-usage --output resource_report.json
```

## 🚨 Installation Issues

### Common Installation Problems

#### Issue 1: Python Version Compatibility

**Symptoms:**
```
Error: Python 3.8+ required
Current version: 3.7.5
```

**Diagnosis:**
```bash
# Check Python version
python --version
python3 --version

# Check pip version
pip --version
```

**Resolution:**
```bash
# Ubuntu/Debian - Install Python 3.9+
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev

# CentOS/RHEL - Install Python 3.9+
sudo yum install python39 python39-devel python39-pip

# macOS - Install via Homebrew
brew install python@3.9

# Verify installation
python3.9 --version
```

**Prevention:**
```bash
# Set up Python version management
# Using pyenv
curl https://pyenv.run | bash
pyenv install 3.9.16
pyenv global 3.9.16

# Using conda
conda create -n ai-extractor python=3.9
conda activate ai-extractor
```

#### Issue 2: Permission Denied Errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/opt/ai-excel-extractor/logs'
OSError: [Errno 13] Permission denied: '/var/data/ai-excel-extractor'
```

**Diagnosis:**
```bash
# Check current user
whoami

# Check directory permissions
ls -la /opt/ai-excel-extractor/
ls -la /var/data/

# Check if directories exist
test -d /opt/ai-excel-extractor && echo "Directory exists" || echo "Directory does not exist"
```

**Resolution:**
```bash
# Create necessary directories
sudo mkdir -p /opt/ai-excel-extractor
sudo mkdir -p /var/data/ai-excel-extractor
sudo mkdir -p /var/log/ai-excel-extractor

# Set ownership
sudo chown -R $USER:$USER /opt/ai-excel-extractor
sudo chown -R $USER:$USER /var/data/ai-excel-extractor
sudo chown -R $USER:$USER /var/log/ai-excel-extractor

# Set permissions
sudo chmod -R 755 /opt/ai-excel-extractor
sudo chmod -R 755 /var/data/ai-excel-extractor
sudo chmod -R 755 /var/log/ai-excel-extractor

# Verify permissions
ls -la /var/log/ | grep ai-extractor
```

#### Issue 3: Database Connection Failures

**Symptoms:**
```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server
ConnectionError: Unable to connect to database
```

**Diagnosis:**
```bash
# Check database service status
sudo systemctl status postgresql
sudo systemctl status mysql

# Check if database is listening
netstat -tlnp | grep :5432  # PostgreSQL
netstat -tlnp | grep :3306  # MySQL

# Test database connection manually
psql -h localhost -U postgres -c "SELECT 1;"
mysql -h localhost -u root -e "SELECT 1;"

# Check configuration
ai-excel-extractor config show database
```

**Resolution:**
```bash
# PostgreSQL issues
# 1. Start the service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# 2. Check configuration
sudo -u postgres psql -c "SHOW config_file;"

# 3. Check connections
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"

# 4. Reset password if needed
sudo -u postgres psql
\password postgres

# MySQL issues
# 1. Start the service
sudo systemctl start mysql
sudo systemctl enable mysql

# 2. Check configuration
sudo mysqladmin variables | grep socket

# 3. Reset root password
sudo mysql_secure_installation

# Configuration update
ai-excel-extractor config set database.url "postgresql://aiuser:newpassword@localhost:5432/ai_extractor"
```

#### Issue 4: AI Model Loading Failures

**Symptoms:**
```
FileNotFoundError: Model file not found: /opt/ai-excel-extractor/models/neural_network_v2.1
RuntimeError: Failed to load AI model
```

**Diagnosis:**
```bash
# Check model directory
ls -la /opt/ai-excel-extractor/models/

# Check model files
find /opt/ai-excel-extractor/models/ -name "*.pt" -o -name "*.pkl" -o -name "*.h5"

# Verify model integrity
ai-excel-extractor models verify

# Check disk space
df -h /opt/ai-excel-extractor/

# Check configuration
ai-excel-extractor config show ai.models_directory
```

**Resolution:**
```bash
# 1. Download missing models
ai-excel-extractor models download --force

# 2. Verify model files
ai-excel-extractor models verify --verbose

# 3. Reinstall models if corrupted
ai-excel-extractor models reinstall --all

# 4. Fix permissions
sudo chown -R ai-excel:ai-excel /opt/ai-excel-extractor/models/
sudo chmod -R 644 /opt/ai-excel-extractor/models/*

# 5. Check disk space and clean if needed
df -h
sudo find /opt/ai-excel-extractor/ -type f -name "*.tmp" -delete

# 6. Update configuration if path is wrong
ai-excel-extractor config set ai.models_directory "/opt/ai-excel-extractor/models"
```

## ⚙️ Configuration Issues

### Configuration Problems

#### Issue 1: Configuration File Corruption

**Symptoms:**
```
yaml.scanner.ScannerError: mapping values are not allowed in this context
Configuration validation failed
```

**Diagnosis:**
```bash
# Validate configuration file
ai-excel-extractor config validate --verbose

# Check file syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check for special characters
file config.yaml
hexdump -C config.yaml | head -20
```

**Resolution:**
```bash
# 1. Backup current configuration
cp config.yaml config.yaml.backup

# 2. Generate fresh configuration
ai-excel-extractor config generate --template production --output config_new.yaml

# 3. Fix common YAML issues
# - Check indentation (use spaces, not tabs)
# - Check for special characters
# - Verify string quotes

# 4. Validate corrected file
ai-excel-extractor config validate --file config_new.yaml

# 5. Replace if valid
mv config_new.yaml config.yaml

# 6. Test configuration
ai-excel-extractor config test
```

#### Issue 2: Invalid Environment Variables

**Symptoms:**
```
ValueError: Invalid environment variable: AI_EXTRACTOR_DB_URL
EnvironmentError: Configuration value out of range
```

**Diagnosis:**
```bash
# Check environment variables
env | grep AI_EXTRACTOR

# Validate specific variable
ai-excel-extractor config validate-env AI_EXTRACTOR_DB_URL

# Check variable format
echo $AI_EXTRACTOR_DB_URL
```

**Resolution:**
```bash
# 1. Fix variable format
# For URLs: use proper protocol and format
export AI_EXTRACTOR_DB_URL="postgresql://user:password@host:port/database"

# For paths: use absolute paths
export AI_EXTRACTOR_LOG_PATH="/var/log/ai-extractor/app.log"

# For numbers: use proper numeric format
export AI_EXTRACTOR_WORKERS=8

# 2. Create proper environment file
cat > .env << EOF
AI_EXTRACTOR_ENV=production
AI_EXTRACTOR_DB_URL=postgresql://aiuser:password@localhost:5432/ai_extractor
AI_EXTRACTOR_LOG_LEVEL=INFO
AI_EXTRACTOR_API_PORT=8080
AI_EXTRACTOR_WORKERS=4
EOF

# 3. Validate environment
ai-excel-extractor config validate-env --file .env
```

## 🚀 Runtime Issues

### Processing Problems

#### Issue 1: File Processing Failures

**Symptoms:**
```
ProcessingError: Failed to process file sample.xlsx
ValueError: Unable to extract data from sheet
IndexError: Sheet index out of range
```

**Diagnosis:**
```bash
# Check file format
file sample.xlsx

# Validate file structure
python -c "
import pandas as pd
try:
    df = pd.read_excel('sample.xlsx')
    print('File structure:')
    print(f'Sheets: {pd.ExcelFile(\"sample.xlsx\").sheet_names}')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
except Exception as e:
    print(f'Error: {e}')
"

# Test with verbose logging
ai-excel-extractor process sample.xlsx --output test_output.json --verbose
```

**Resolution:**
```python
# 1. Fix Excel file issues
import pandas as pd

# Check and fix common Excel issues
def fix_excel_file(input_file, output_file):
    try:
        # Read Excel file with error handling
        df = pd.read_excel(input_file, sheet_name=0, engine='openpyxl')
        
        # Fix common issues
        df = df.dropna(how='all')  # Remove empty rows
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        
        # Ensure proper data types
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Save fixed file
        df.to_excel(output_file, index=False)
        print(f"Fixed file saved to: {output_file}")
        
    except Exception as e:
        print(f"Error fixing file: {e}")

# Use the fix function
fix_excel_file('sample.xlsx', 'sample_fixed.xlsx')
```

#### Issue 2: Memory Issues During Processing

**Symptoms:**
```
MemoryError: Unable to allocate array
SystemError: Cannot allocate memory
```

**Diagnosis:**
```bash
# Check memory usage
free -h
ps aux | grep ai-excel-extractor

# Check memory limits
ulimit -a | grep memory
cat /proc/meminfo

# Monitor memory during processing
ai-excel-extractor process large_file.xlsx --monitor-memory
```

**Resolution:**
```yaml
# 1. Adjust memory limits in configuration
processing:
  memory:
    max_usage: "2GB"  # Reduce from 4GB
    gc_strategy: "aggressive"
    chunk_size: 1000  # Process in smaller chunks
    
  # Enable streaming for large files
  streaming:
    enabled: true
    chunk_size: 500
    buffer_size: "64MB"
```

#### Issue 3: Timeout Errors

**Symptoms:**
```
TimeoutError: Processing timed out after 300 seconds
ConnectionTimeout: Request timeout
```

**Diagnosis:**
```bash
# Check timeout configuration
ai-excel-extractor config show processing.timeout

# Monitor processing time
time ai-excel-extractor process test_file.xlsx

# Check system load
uptime
iostat 1 5
```

**Resolution:**
```yaml
# 1. Increase timeout values
processing:
  timeout: 600  # Increase from 300 to 600 seconds
  retry_attempts: 5  # Increase retry attempts
  retry_delay: 10  # Increase delay between retries
  
# 2. Optimize processing
processing:
  # Enable parallel processing
  parallel:
    enabled: true
    max_workers: 8
    
  # Use faster algorithms
  optimization:
    use_fast_algorithms: true
    skip_optional_validations: true
```

### API Issues

#### Issue 1: API Connection Problems

**Symptoms:**
```
ConnectionError: Failed to connect to API server
HTTPError: 503 Service Unavailable
```

**Diagnosis:**
```bash
# Check API service status
sudo systemctl status ai-excel-extractor-api

# Test API endpoint
curl -X GET http://localhost:8080/health

# Check network connectivity
netstat -tlnp | grep 8080
lsof -i :8080
```

**Resolution:**
```bash
# 1. Restart API service
sudo systemctl restart ai-excel-extractor-api

# 2. Check logs for errors
sudo journalctl -u ai-excel-extractor-api -f

# 3. Verify configuration
ai-excel-extractor config validate --section api

# 4. Check port availability
sudo netstat -tlnp | grep :8080
sudo kill $(sudo lsof -t -i:8080)  # Kill process using port

# 5. Restart on different port
ai-excel-extractor config set api.port 8081
sudo systemctl restart ai-excel-extractor-api
```

#### Issue 2: Authentication Failures

**Symptoms:**
```
401 Unauthorized: Invalid credentials
403 Forbidden: Insufficient permissions
JWTError: Invalid token
```

**Diagnosis:**
```bash
# Test authentication
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'

# Check JWT configuration
ai-excel-extractor config show security.authentication

# Validate token
ai-excel-extractor auth validate-token <token>
```

**Resolution:**
```bash
# 1. Reset admin password
ai-excel-extractor auth reset-admin-password

# 2. Create new API key
ai-excel-extractor auth create-api-key --name "new_key" --permissions "read,write"

# 3. Fix JWT configuration
ai-excel-extractor config set security.authentication.jwt.secret_key "new_secret_key"

# 4. Disable auth for testing (development only)
ai-excel-extractor config set security.authentication.enabled false
```

## 🔧 Performance Issues

### Slow Processing

#### Issue 1: Slow File Processing

**Symptoms:**
```
Processing takes longer than expected
High CPU usage during processing
System becomes unresponsive
```

**Diagnosis:**
```bash
# Profile processing performance
ai-excel-extractor benchmark --files test_files/ --detailed

# Monitor system resources
top -p $(pgrep ai-excel-extractor)

# Check I/O performance
iostat -x 1

# Analyze processing steps
ai-excel-extractor process profile.xlsx --profile --output profile_report.json
```

**Resolution:**
```yaml
# 1. Optimize processing configuration
processing:
  # Use more workers
  workers: 8  # Increase from 4
  
  # Enable GPU acceleration
  gpu:
    enabled: true
    device_id: 0
    
  # Optimize batch processing
  batch:
    size: 64  # Increase batch size
    parallel: true
    
  # Use faster algorithms
  optimization:
    use_c_extensions: true
    parallel_processing: true
    skip_slow_validations: false  # Enable for speed

# 2. Hardware optimization
# - Use SSD storage
# - Increase RAM
# - Use multiple CPU cores
# - Enable GPU processing
```

#### Issue 2: Database Performance Issues

**Symptoms:**
```
Slow query responses
Database connection timeouts
High database CPU usage
```

**Diagnosis:**
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check database connections
SELECT count(*) FROM pg_stat_activity;

-- Check database locks
SELECT * FROM pg_locks WHERE NOT granted;

-- Check database size
SELECT pg_size_pretty(pg_database_size('ai_extractor'));
```

**Resolution:**
```yaml
# 1. Optimize database configuration
database:
  # Increase connection pool
  pool_size: 50
  max_overflow: 100
  
  # Optimize queries
  performance:
    query_cache:
      enabled: true
      size: "512MB"
      
    slow_query_log:
      enabled: true
      threshold: 1.0  # seconds
      
# 2. Add database indexes
-- Create indexes for common queries
CREATE INDEX idx_project_id ON extractions(project_id);
CREATE INDEX idx_created_at ON extractions(created_at);
CREATE INDEX idx_status ON extractions(status);

# 3. Optimize queries
-- Use prepared statements
-- Add proper indexes
-- Optimize JOIN operations
```

### Memory Issues

#### Issue 1: Memory Leaks

**Symptoms:**
```
Memory usage continuously increases
System becomes slow over time
Out of memory errors
```

**Diagnosis:**
```bash
# Monitor memory usage over time
while true; do
  ps aux | grep ai-excel-extractor | grep -v grep
  sleep 60
done

# Check memory patterns
ai-excel-extractor memory-profile --duration 3600

# Analyze garbage collection
python -c "
import gc
import psutil
import time

for i in range(60):
    gc.collect()
    print(f'Time: {time.ctime()}, Memory: {psutil.virtual_memory().percent}%')
    time.sleep(60)
"
```

**Resolution:**
```python
# 1. Implement proper memory management
import gc
import weakref

class MemoryManagedProcessor:
    def __init__(self):
        self.data_references = weakref.WeakSet()
        
    def process_file(self, file_path):
        try:
            # Process file with memory management
            df = pd.read_excel(file_path)
            self.data_references.add(df)
            
            # Process data
            result = self.process_data(df)
            
            # Clear references
            del df
            gc.collect()
            
            return result
            
        except MemoryError:
            # Force garbage collection
            gc.collect()
            # Process in chunks
            return self.process_file_chunks(file_path)
```

```yaml
# 2. Configure memory limits
processing:
  memory:
    max_usage: "2GB"
    gc_interval: 30  # seconds
    gc_strategy: "aggressive"
    
  # Enable memory monitoring
  monitoring:
    memory_alerts:
      enabled: true
      threshold: "1.8GB"
      action: "restart_process"
```

## 🌐 Integration Issues

### Database Integration Problems

#### Issue 1: PostgreSQL Connection Issues

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
sqlalchemy.exc.DisconnectionError: (psycopg2.DisconnectionError)
```

**Diagnosis:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection manually
psql -h localhost -U aiuser -d ai_extractor -c "SELECT 1;"

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# Check connection limits
sudo -u postgres psql -c "SHOW max_connections;"
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"
```

**Resolution:**
```bash
# 1. Restart PostgreSQL
sudo systemctl restart postgresql

# 2. Check and fix configuration
sudo -u postgres psql
-- Check current connections
SELECT * FROM pg_stat_activity;

-- Kill long-running connections
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction';

# 3. Update connection limits
-- Edit postgresql.conf
sudo nano /etc/postgresql/*/main/postgresql.conf
-- Set: max_connections = 200

# 4. Update application configuration
ai-excel-extractor config set database.pool_size 20
ai-excel-extractor config set database.max_overflow 40
```

#### Issue 2: Redis Connection Issues

**Symptoms:**
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379
redis.exceptions.TimeoutError: Timeout connecting to Redis
```

**Diagnosis:**
```bash
# Check Redis status
sudo systemctl status redis

# Test Redis connection
redis-cli ping

# Check Redis configuration
redis-cli config get "*"

# Monitor Redis performance
redis-cli monitor
```

**Resolution:**
```bash
# 1. Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# 2. Test and fix configuration
redis-cli config set maxmemory 512mb
redis-cli config set maxmemory-policy allkeys-lru

# 3. Update application configuration
ai-excel-extractor config set cache.redis.host localhost
ai-excel-extractor config set cache.redis.port 6379
ai-excel-extractor config set cache.redis.password null
```

### API Integration Issues

#### Issue 1: External API Integration Failures

**Symptoms:**
```
requests.exceptions.ConnectionError: HTTPSConnectionPool
APIError: Rate limit exceeded
TimeoutError: API request timeout
```

**Diagnosis:**
```bash
# Test external API
curl -X GET "https://api.example.com/health" -H "Authorization: Bearer token"

# Check rate limits
curl -X GET "https://api.example.com/status" -H "X-RateLimit-Remaining: 0"

# Monitor API calls
ai-excel-extractor api-monitor --external-apis
```

**Resolution:**
```yaml
# 1. Configure retry and timeout
integrations:
  external_apis:
    retry:
      max_attempts: 3
      backoff_factor: 2
      timeout: 30
      
    rate_limiting:
      enabled: true
      requests_per_minute: 60
      burst_size: 10
      
# 2. Implement circuit breaker
circuit_breaker:
  enabled: true
  failure_threshold: 5
  timeout: 60
  recovery_timeout: 30
```

## 🔒 Security Issues

### Authentication Problems

#### Issue 1: JWT Token Issues

**Symptoms:**
```
jwt.exceptions.InvalidTokenError: Invalid token
jwt.exceptions.ExpiredSignatureError: Token has expired
```

**Diagnosis:**
```python
# Test JWT token
import jwt
import datetime

def debug_jwt_token(token):
    try:
        # Decode without verification first
        decoded = jwt.decode(token, options={"verify_signature": False})
        print("Token payload:", decoded)
        
        # Check expiration
        if 'exp' in decoded:
            exp_timestamp = decoded['exp']
            exp_date = datetime.datetime.fromtimestamp(exp_timestamp)
            print(f"Expires: {exp_date}")
            print(f"Current: {datetime.datetime.now()}")
            print(f"Expired: {datetime.datetime.now() > exp_date}")
            
        return True
        
    except Exception as e:
        print(f"JWT Error: {e}")
        return False

# Use the function
debug_jwt_token("your_jwt_token_here")
```

**Resolution:**
```bash
# 1. Generate new JWT secret
ai-excel-extractor auth generate-secret --length 256

# 2. Update configuration
ai-excel-extractor config set security.authentication.jwt.secret_key "new_secret_here"

# 3. Clear existing tokens
ai-excel-extractor auth clear-tokens

# 4. Create new admin token
ai-excel-extractor auth create-token --user admin --role administrator --expiry 24h
```

### SSL/TLS Issues

#### Issue 1: Certificate Problems

**Symptoms:**
```
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

**Diagnosis:**
```bash
# Check certificate validity
openssl x509 -in /etc/ssl/certs/ai-excel-extractor.crt -text -noout

# Test SSL connection
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Check certificate chain
openssl verify -CAfile /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ai-excel-extractor.crt
```

**Resolution:**
```bash
# 1. Generate new self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout /etc/ssl/private/ai-excel-extractor.key \
  -out /etc/ssl/certs/ai-excel-extractor.crt -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=ai-excel-extractor"

# 2. Set proper permissions
sudo chmod 600 /etc/ssl/private/ai-excel-extractor.key
sudo chmod 644 /etc/ssl/certs/ai-excel-extractor.crt

# 3. Update configuration
ai-excel-extractor config set api.ssl.enabled true
ai-excel-extractor config set api.ssl.cert_file /etc/ssl/certs/ai-excel-extractor.crt
ai-excel-extractor config set api.ssl.key_file /etc/ssl/private/ai-excel-extractor.key

# 4. Use Let's Encrypt for production
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoring and Alerting

### Monitoring Problems

#### Issue 1: Metrics Collection Failures

**Symptoms:**
```
Prometheus scrape error: Connection refused
Metrics endpoint not responding
Grafana dashboard shows no data
```

**Diagnosis:**
```bash
# Check metrics endpoint
curl http://localhost:9090/metrics

# Check Prometheus configuration
curl http://localhost:9090/api/v1/targets

# Verify metrics format
ai-excel-extractor metrics validate
```

**Resolution:**
```yaml
# 1. Fix metrics configuration
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
    format: "prometheus"
    
    # Ensure proper metrics format
    validation:
      enabled: true
      strict_mode: true
      
# 2. Restart monitoring
sudo systemctl restart ai-excel-extractor
sudo systemctl restart prometheus

# 3. Test metrics collection
ai-excel-extractor metrics test
```

### Alerting Issues

#### Issue 1: Alert Delivery Failures

**Symptoms:**
```
Email alerts not received
Slack notifications failing
Webhook calls timing out
```

**Diagnosis:**
```bash
# Test email configuration
ai-excel-extractor alert test --type email --recipient test@example.com

# Test Slack integration
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test alert from AI Extractor"}' \
  your-slack-webhook-url

# Check alert logs
tail -f /var/log/ai-excel-extractor/alerts.log
```

**Resolution:**
```yaml
# 1. Fix email configuration
monitoring:
  alerting:
    email:
      enabled: true
      smtp_server: "smtp.company.com"
      smtp_port: 587
      username: "alerts@company.com"
      password_file: "/etc/ai-excel-extractor/email_password"
      use_tls: true
      
    # Test email configuration
    test:
      enabled: true
      recipients: ["admin@company.com"]
      
# 2. Fix Slack configuration
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
      channel: "#ai-extractor-alerts"
      
# 3. Fix webhook configuration
    webhook:
      enabled: true
      url: "https://alerts.company.com/webhook"
      timeout: 10
      retry_attempts: 3
```

## 🛠️ Advanced Troubleshooting

### System Investigation Tools

#### Comprehensive System Analysis
```bash
#!/bin/bash
# comprehensive_diagnostics.sh

echo "=== AI Excel Extractor Diagnostics Report ==="
echo "Generated: $(date)"
echo

echo "=== System Information ==="
uname -a
cat /etc/os-release
echo

echo "=== Resource Usage ==="
free -h
df -h
echo

echo "=== Service Status ==="
systemctl status ai-excel-extractor
systemctl status postgresql
systemctl status redis
echo

echo "=== Network Status ==="
netstat -tlnp | grep ai-excel
echo

echo "=== Recent Logs ==="
echo "=== Application Logs ==="
tail -20 /var/log/ai-excel-extractor/app.log
echo

echo "=== Error Logs ==="
grep -i error /var/log/ai-excel-extractor/app.log | tail -10
echo

echo "=== Configuration Check ==="
ai-excel-extractor config validate
echo

echo "=== Health Check ==="
ai-excel-extractor health-check --verbose
echo

echo "=== Performance Check ==="
ai-excel-extractor performance-check
echo

echo "=== Diagnostics Complete ==="
```

### Automated Recovery Scripts

#### Service Recovery Script
```bash
#!/bin/bash
# recovery_script.sh

set -e

echo "Starting AI Excel Extractor recovery..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

# Stop services
echo "Stopping services..."
systemctl stop ai-excel-extractor
systemctl stop ai-excel-extractor-api

# Clear temporary files
echo "Clearing temporary files..."
rm -rf /tmp/ai-excel-extractor/*
rm -f /var/log/ai-excel-extractor/*.tmp

# Restart services
echo "Restarting services..."
systemctl start postgresql
systemctl start redis
systemctl start ai-excel-extractor
systemctl start ai-excel-extractor-api

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 30

# Verify services
echo "Verifying services..."
if systemctl is-active --quiet ai-excel-extractor; then
    echo "✓ AI Extractor service is running"
else
    echo "✗ AI Extractor service failed to start"
    systemctl status ai-excel-extractor
fi

# Run health check
echo "Running health check..."
ai-excel-extractor health-check

echo "Recovery complete!"
```

### Performance Profiling

#### CPU and Memory Profiling
```python
# performance_profiler.py
import cProfile
import pstats
import io
import psutil
import time
from ai_excel_extractor import AIExcelExtractor

def profile_extraction():
    """Profile the extraction process"""
    
    # Memory monitoring
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # CPU profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Initialize extractor
        extractor = AIExcelExtractor()
        
        # Profile extraction
        result = extractor.extract("test_file.xlsx")
        
        print(f"Extraction completed successfully")
        print(f"Memory used: {process.memory_info().rss - initial_memory / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        
    finally:
        profiler.disable()
        
        # Save profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        with open('extraction_profile.txt', 'w') as f:
            f.write(s.getvalue())
        
        print("Profile saved to extraction_profile.txt")

if __name__ == "__main__":
    profile_extraction()
```

### Log Analysis Tools

#### Automated Log Analysis
```bash
#!/bin/bash
# log_analyzer.sh

LOG_FILE="/var/log/ai-excel-extractor/app.log"
OUTPUT_FILE="log_analysis_$(date +%Y%m%d_%H%M%S).txt"

echo "=== AI Excel Extractor Log Analysis ===" > $OUTPUT_FILE
echo "Analysis Date: $(date)" >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

# Error count by type
echo "=== Error Count by Type ===" >> $OUTPUT_FILE
grep -c "ERROR" $LOG_FILE | awk '{print "Total Errors: " $1}' >> $OUTPUT_FILE
grep -c "CRITICAL" $LOG_FILE | awk '{print "Critical Errors: " $1}' >> $OUTPUT_FILE
grep -c "WARNING" $LOG_FILE | awk '{print "Warnings: " $1}' >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

# Most common errors
echo "=== Most Common Error Messages ===" >> $OUTPUT_FILE
grep "ERROR" $LOG_FILE | sed 's/.*ERROR: //' | sort | uniq -c | sort -nr | head -10 >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

# Performance issues
echo "=== Performance Issues ===" >> $OUTPUT_FILE
grep -E "timeout|slow|high.*memory|cpu.*high" $LOG_FILE | wc -l | awk '{print "Performance-related log entries: " $1}' >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

# Recent errors
echo "=== Recent Errors (Last 24 Hours) ===" >> $OUTPUT_FILE
find $LOG_FILE -mtime -1 -exec grep "ERROR" {} \; | tail -20 >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

# Recommendations
echo "=== Recommendations ===" >> $OUTPUT_FILE
echo "1. Check error patterns identified above" >> $OUTPUT_FILE
echo "2. Review system resources if performance issues detected" >> $OUTPUT_FILE
echo "3. Monitor for recent error clusters" >> $OUTPUT_FILE
echo "4. Consider adjusting timeouts if timeout errors are frequent" >> $OUTPUT_FILE

echo "Analysis complete. Results saved to $OUTPUT_FILE"
```

## 📞 Getting Help

### Support Resources

#### Community Support
- **GitHub Issues**: [github.com/company/ai-excel-extractor/issues](https://github.com/company/ai-excel-extractor/issues)
- **Documentation**: [docs.ai-excel-extractor.com](https://docs.ai-excel-extractor.com)
- **Community Forum**: [community.ai-excel-extractor.com](https://community.ai-excel-extractor.com)

#### Commercial Support
- **Email Support**: support@ai-excel-extractor.com
- **Priority Support**: Available for enterprise customers
- **Phone Support**: +1-800-AI-EXTRACTOR (24/7 for critical issues)

### Gathering Information for Support

#### Support Information Template
```bash
#!/bin/bash
# gather_support_info.sh

echo "Gathering support information..."

# Create support directory
mkdir -p ai_extractor_support_$(date +%Y%m%d_%H%M%S)
cd ai_extractor_support_$(date +%Y%m%d_%H%M%S)

# System information
echo "=== System Information ===" > system_info.txt
uname -a >> system_info.txt
cat /etc/os-release >> system_info.txt
free -h >> system_info.txt
df -h >> system_info.txt

# Configuration
echo "=== Configuration ===" > configuration.txt
ai-excel-extractor config dump > configuration.txt

# Logs (last 1000 lines)
echo "=== Recent Logs ===" > logs.txt
tail -1000 /var/log/ai-excel-extractor/app.log > logs.txt

# Health check
echo "=== Health Check ===" > health_check.txt
ai-excel-extractor health-check --verbose > health_check.txt

# Version information
echo "=== Version Information ===" > version_info.txt
ai-excel-extractor version --detailed > version_info.txt

# Performance metrics
echo "=== Performance Metrics ===" > performance.txt
ai-excel-extractor performance-check > performance.txt

echo "Support information gathered in directory: $(pwd)"
echo "Please zip this directory and send to support team"
```

#### Support Ticket Template
```
# AI Excel Extractor Support Ticket

## Issue Summary
- **Problem**: [Brief description of the problem]
- **Impact**: [How this affects your operations]
- **Urgency**: [Critical/High/Medium/Low]

## Environment
- **Version**: [AI Excel Extractor version]
- **OS**: [Operating system and version]
- **Python**: [Python version]
- **Database**: [Database type and version]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Error Messages
```
[Paste relevant error messages here]
```

## Recent Changes
- [List any recent changes to system, configuration, or environment]

## Logs
[Paste relevant log entries here or attach log files]

## Additional Information
[Any other relevant information]
```

## 🎯 Prevention and Best Practices

### Preventive Maintenance

#### Regular Maintenance Tasks
```bash
#!/bin/bash
# maintenance_script.sh

echo "Starting regular maintenance..."

# Clean old log files
echo "Cleaning old log files..."
find /var/log/ai-excel-extractor/ -name "*.log.*" -mtime +30 -delete

# Clean temporary files
echo "Cleaning temporary files..."
rm -rf /tmp/ai-excel-extractor/*

# Optimize database
echo "Optimizing database..."
ai-excel-extractor database optimize

# Check disk space
echo "Checking disk space..."
df -h

# Verify integrity
echo "Verifying system integrity..."
ai-excel-extractor integrity-check

# Update models
echo "Checking for model updates..."
ai-excel-extractor models check-updates

echo "Maintenance completed!"
```

#### Monitoring Setup
```yaml
# monitoring_setup.yaml
monitoring:
  # Enable comprehensive monitoring
  health_checks:
    enabled: true
    interval: 60  # Check every minute
    
  metrics:
    enabled: true
    collection_interval: 30  # Collect every 30 seconds
    
  alerting:
    enabled: true
    email_notifications: true
    slack_notifications: true
    
  # Set up performance monitoring
  performance:
    enabled: true
    threshold_response_time: 5.0  # seconds
    threshold_memory_usage: 80  # percentage
    threshold_cpu_usage: 80  # percentage
    
  # Automated alerts
  automated_alerts:
    - condition: "health_check_failed"
      action: "restart_service"
      notification: true
      
    - condition: "memory_usage > 90%"
      action: "scale_up"
      notification: true
      
    - condition: "error_rate > 5%"
      action: "investigate"
      notification: true
```

This comprehensive troubleshooting guide covers the most common issues you may encounter with the AI Excel Extraction System. For additional support, please refer to the support resources section or contact our support team with the gathered diagnostic information.