# Installation Guide - AI Excel Extraction System

## Overview

This guide provides comprehensive step-by-step instructions for installing and configuring the AI-Powered Excel Extraction System across different environments and deployment scenarios.

## 📋 Prerequisites

### System Requirements

#### Minimum Hardware Requirements
```yaml
CPU: 4 cores, 2.5GHz+
RAM: 8GB (16GB recommended)
Storage: 50GB available space
Network: Broadband internet connection
```

#### Recommended Hardware
```yaml
CPU: 8+ cores, 3.0GHz+
RAM: 32GB
Storage: 200GB SSD
Network: High-speed internet (100Mbps+)
GPU: NVIDIA GPU with CUDA support (optional, for accelerated processing)
```

#### Operating System Support
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.15+ (Catalina or later)
- **Linux**: Ubuntu 18.04+, CentOS 7+, or equivalent
- **Cloud**: AWS, Azure, GCP (Linux-based)

### Software Dependencies

#### Required Software
```bash
# Python 3.8+ (Required)
python --version
# Should return 3.8.0 or higher

# Git (Required for source installation)
git --version

# Virtual Environment Tool (Recommended)
python -m venv --help  # or conda, virtualenv
```

#### Optional Software
```bash
# Docker (For containerized deployment)
docker --version

# Node.js (For web interface)
node --version
npm --version
```

## 🚀 Installation Methods

### Method 1: Quick Installation (Recommended for New Users)

#### Automated Installer

**Windows**
```powershell
# Download and run the installer
Invoke-WebRequest -Uri "https://releases.ai-excel-extractor.com/installer/windows/setup.exe" -OutFile "ai-excel-installer.exe"
.\ai-excel-installer.exe

# Follow the installation wizard
```

**macOS**
```bash
# Download and run the installer
curl -L https://releases.ai-excel-extractor.com/installer/macos/dmg -o ai-excel-installer.dmg
open ai-excel-installer.dmg

# Drag to Applications folder
```

**Linux**
```bash
# Download and run the installer
wget https://releases.ai-excel-extractor.com/installer/linux/ai-excel-installer.sh
chmod +x ai-excel-installer.sh
sudo ./ai-excel-installer.sh
```

#### Post-Installation Verification
```bash
# Test the installation
ai-excel-extractor --version
ai-excel-extractor --health-check

# Expected output:
# AI Excel Extractor v2.1.0
# Health check: PASSED
# All components: OPERATIONAL
```

### Method 2: Python Package Installation

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv ai-excel-env

# Activate virtual environment
# Windows
ai-excel-env\Scripts\activate
# macOS/Linux
source ai-excel-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Install Package
```bash
# Install from PyPI (recommended)
pip install ai-excel-extractor

# Or install from source
git clone https://github.com/company/ai-excel-extractor.git
cd ai-excel-extractor
pip install -e .
```

#### Verify Installation
```python
# Test installation in Python
import ai_excel_extractor
print(ai_excel_extractor.__version__)
# Should print: '2.1.0' or higher

# Test core functionality
from ai_excel_extractor import AIExcelExtractor
extractor = AIExcelExtractor()
print("Installation successful!")
```

### Method 3: Docker Installation

#### Pull Docker Image
```bash
# Pull the latest image
docker pull ai-excel-extractor:latest

# Verify image
docker images | grep ai-excel-extractor
```

#### Run Container
```bash
# Run in detached mode
docker run -d \
  --name ai-excel-extractor \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  ai-excel-extractor:latest

# Check container status
docker ps
```

#### Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-excel-extractor:
    image: ai-excel-extractor:latest
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - AI_EXTRACTOR_ENV=production
      - AI_EXTRACTOR_DB_URL=postgresql://user:pass@db:5432/ai_extractor
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: ai_extractor
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

```bash
# Start with Docker Compose
docker-compose up -d

# Check services
docker-compose ps
```

### Method 4: Source Installation (Advanced Users)

#### Clone Repository
```bash
# Clone the repository
git clone https://github.com/company/ai-excel-extractor.git
cd ai-excel-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

#### Install Dependencies
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

#### Build from Source
```bash
# Build the package
python setup.py build
python setup.py install

# Or install with extras
pip install .[all]
```

## ⚙️ Configuration

### Basic Configuration

#### Configuration File Setup
```bash
# Create configuration directory
mkdir -p ~/.ai-excel-extractor
cd ~/.ai-excel-extractor

# Copy default configuration
cp /opt/ai-excel-extractor/config/default.yaml ./config.yaml
```

#### Basic Configuration File
```yaml
# config.yaml
system:
  app_name: "AI Excel Extractor"
  version: "2.1.0"
  environment: "production"
  
processing:
  max_file_size: "100MB"
  concurrent_jobs: 4
  timeout: 300
  cache_enabled: true
  
ai:
  model_path: "/opt/ai-excel-extractor/models"
  confidence_threshold: 0.85
  auto_correction: true
  validation_enabled: true
  
logging:
  level: "INFO"
  file: "/var/log/ai-excel-extractor/app.log"
  max_size: "100MB"
  backup_count: 5
  
storage:
  input_directory: "/var/data/input"
  output_directory: "/var/data/output"
  temp_directory: "/var/tmp/ai-extractor"
  
api:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  enable_cors: true
  
database:
  url: "sqlite:///var/data/ai_extractor.db"
  pool_size: 10
  timeout: 30
```

#### Environment Variables
```bash
# Create environment file
cat > .env << EOF
AI_EXTRACTOR_ENV=production
AI_EXTRACTOR_LOG_LEVEL=INFO
AI_EXTRACTOR_DB_URL=postgresql://user:pass@localhost:5432/ai_extractor
AI_EXTRACTOR_REDIS_URL=redis://localhost:6379
AI_EXTRACTOR_API_KEY=your-secret-api-key
AI_EXTRACTOR_ENCRYPTION_KEY=your-encryption-key
EOF
```

### Advanced Configuration

#### Database Configuration
```yaml
# PostgreSQL Configuration
database:
  url: "postgresql://username:password@localhost:5432/ai_extractor"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  echo: false  # Set to true for SQL logging
  
# MySQL Configuration
database:
  url: "mysql://username:password@localhost:3306/ai_extractor"
  pool_size: 20
  pool_recycle: 3600
  
# SQLite Configuration (for development)
database:
  url: "sqlite:///var/data/ai_extractor.db"
  echo: true  # Enable for development
```

#### AI Model Configuration
```yaml
ai:
  models:
    primary:
      type: "neural_network"
      path: "/opt/ai-excel-extractor/models/v2.1"
      confidence_threshold: 0.90
    secondary:
      type: "ensemble"
      path: "/opt/ai-excel-extractor/models/ensemble"
      confidence_threshold: 0.85
      
  processing:
    batch_size: 32
    max_sequence_length: 2048
    temperature: 0.1
    top_p: 0.9
    
  validation:
    cross_validation: true
    validation_split: 0.2
    early_stopping: true
    epochs: 100
```

#### Performance Optimization
```yaml
performance:
  # Multi-threading
  worker_threads: 8
  io_threads: 4
  
  # Memory management
  max_memory_usage: "4GB"
  garbage_collection: "aggressive"
  
  # Caching
  cache:
    type: "redis"  # or "memory", "disk"
    ttl: 3600  # seconds
    max_size: "1GB"
    
  # GPU acceleration
  gpu:
    enabled: true
    device_id: 0
    memory_fraction: 0.8
    
  # Monitoring
  monitoring:
    enabled: true
    metrics_port: 9090
    health_check_interval: 30
```

## 🔧 Platform-Specific Installation

### Windows Installation

#### System Requirements Check
```powershell
# Check Windows version
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion

# Check available memory
Get-WmiObject -Class Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum

# Check available disk space
Get-WmiObject -Class Win32_LogicalDisk | Where-Object {$_.DriveType -eq 3} | Select-Object DeviceID, Size, FreeSpace
```

#### Windows Service Installation
```powershell
# Install as Windows Service
ai-excel-extractor install-service --name "AIExcelExtractor" --display-name "AI Excel Extraction Service"

# Start the service
Start-Service "AIExcelExtractor"

# Check service status
Get-Service "AIExcelExtractor"
```

#### Windows Firewall Configuration
```powershell
# Add firewall rule
New-NetFirewallRule -DisplayName "AI Excel Extractor" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow

# Or use the installer to configure firewall automatically
```

### macOS Installation

#### Homebrew Installation
```bash
# Add the tap
brew tap ai-excel-extractor/tap

# Install the package
brew install ai-excel-extractor

# Start the service
brew services start ai-excel-extractor
```

#### macOS LaunchAgent
```xml
<!-- ~/Library/LaunchAgents/com.ai-excel-extractor.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ai-excel-extractor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/ai-excel-extractor</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

```bash
# Load the launch agent
launchctl load ~/Library/LaunchAgents/com.ai-excel-extractor.plist
```

### Linux Installation

#### Package Manager Installation

**Ubuntu/Debian**
```bash
# Add the repository
wget -qO - https://apt.ai-excel-extractor.com/repo.key | sudo apt-key add -
echo "deb https://apt.ai-excel-extractor.com/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ai-excel-extractor.list

# Install the package
sudo apt update
sudo apt install ai-excel-extractor

# Start the service
sudo systemctl start ai-excel-extractor
sudo systemctl enable ai-excel-extractor
```

**Red Hat/CentOS**
```bash
# Add the repository
sudo tee /etc/yum.repos.d/ai-excel-extractor.repo << EOF
[ai-excel-extractor]
name=AI Excel Extractor
baseurl=https://rpm.ai-excel-extractor.com/centos/\$releasever/\$basearch
gpgcheck=1
gpgkey=https://rpm.ai-excel-extractor.com/repo.key
enabled=1
EOF

# Install the package
sudo yum install ai-excel-extractor

# Start the service
sudo systemctl start ai-excel-extractor
sudo systemctl enable ai-excel-extractor
```

#### Systemd Service Configuration
```ini
# /etc/systemd/system/ai-excel-extractor.service
[Unit]
Description=AI Excel Extraction Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=ai-excel
Group=ai-excel
WorkingDirectory=/opt/ai-excel-extractor
Environment=AI_EXTRACTOR_ENV=production
EnvironmentFile=/etc/ai-excel-extractor/environment
ExecStart=/usr/bin/ai-excel-extractor start
ExecStop=/usr/bin/ai-excel-extractor stop
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable ai-excel-extractor
sudo systemctl start ai-excel-extractor
sudo systemctl status ai-excel-extractor
```

## 🌐 Cloud Installation

### AWS Deployment

#### EC2 Instance Setup
```bash
# Launch EC2 instance (Ubuntu 20.04)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-excel-extractor}]'

# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip
```

#### Install on EC2
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Create user
sudo useradd -m -s /bin/bash ai-excel
sudo usermod -aG sudo ai-excel

# Install application
sudo su - ai-excel
git clone https://github.com/company/ai-excel-extractor.git
cd ai-excel-extractor
pip3 install -r requirements.txt
pip3 install .

# Configure as systemd service
sudo cp deployment/aws/ai-excel-extractor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-excel-extractor
sudo systemctl start ai-excel-extractor
```

#### RDS Database Setup
```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier ai-extractor-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username aiuser \
  --master-user-password your-secure-password \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-12345678

# Update configuration
sudo -u ai-excel cat > ~/.ai-excel-extractor/config.yaml << EOF
database:
  url: "postgresql://aiuser:your-secure-password@your-rds-endpoint:5432/aidb"
EOF
```

### Azure Deployment

#### App Service Setup
```bash
# Create resource group
az group create --name ai-extractor-rg --location eastus

# Create App Service plan
az appservice plan create \
  --name ai-extractor-plan \
  --resource-group ai-extractor-rg \
  --sku P1V2 \
  --is-linux

# Create web app
az webapp create \
  --resource-group ai-extractor-rg \
  --plan ai-extractor-plan \
  --name ai-excel-extractor-app \
  --runtime "PYTHON|3.9"

# Deploy application
az webapp deployment source config-zip \
  --resource-group ai-extractor-rg \
  --name ai-excel-extractor-app \
  --src deploy.zip
```

#### Azure Database Setup
```bash
# Create PostgreSQL server
az postgres server create \
  --resource-group ai-extractor-rg \
  --name ai-extractor-db \
  --location eastus \
  --admin-user aiuser \
  --admin-password your-secure-password \
  --sku-name B_Gen5_2

# Create database
az postgres db create \
  --resource-group ai-extractor-rg \
  --server-name ai-extractor-db \
  --name ai_extractor
```

### Google Cloud Platform

#### Cloud Run Deployment
```bash
# Create project
gcloud projects create ai-excel-extractor --name="AI Excel Extractor"

# Set project
gcloud config set project ai-excel-extractor

# Enable services
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com

# Create Cloud SQL instance
gcloud sql instances create ai-extractor-db \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create ai_extractor --instance=ai-extractor-db

# Build and deploy to Cloud Run
gcloud run deploy ai-excel-extractor \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL="postgresql://aiuser:password@/ai_extractor?host=/cloudsql/project:region:ai-extractor-db"
```

## 🔐 Security Configuration

### SSL/TLS Configuration

#### Certificate Installation
```bash
# Generate self-signed certificate (development)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/ai-excel-extractor.key \
  -out /etc/ssl/certs/ai-excel-extractor.crt

# For production, use Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

#### HTTPS Configuration
```yaml
# config.yaml
api:
  ssl:
    enabled: true
    cert_file: "/etc/ssl/certs/ai-excel-extractor.crt"
    key_file: "/etc/ssl/private/ai-excel-extractor.key"
    port: 8443
    redirect_http: true
    
security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_rotation_days: 90
    
  authentication:
    enabled: true
    method: "jwt"  # or "oauth2", "ldap"
    jwt_secret: "${JWT_SECRET}"
    jwt_expiry: "24h"
    
  access_control:
    enabled: true
    rate_limiting: true
    max_requests_per_minute: 100
    allowed_origins: ["https://your-domain.com"]
```

### User Management

#### Create Admin User
```bash
# Create initial admin user
ai-excel-extractor create-admin \
  --username admin \
  --email admin@company.com \
  --role administrator

# Create API key
ai-excel-extractor create-api-key \
  --name "Production API" \
  --permissions "read,write,admin" \
  --expiry "2025-12-31"
```

#### Role-Based Access Control
```yaml
# config.yaml
access_control:
  roles:
    administrator:
      permissions: ["read", "write", "delete", "admin", "users", "settings"]
      users: ["admin"]
      
    engineer:
      permissions: ["read", "write", "export"]
      users: ["engineering_team"]
      
    viewer:
      permissions: ["read", "export"]
      users: ["management", "clients"]
      
  policies:
    - name: "project_isolation"
      condition: "user.project_id == resource.project_id"
      permissions: ["read", "write"]
      
    - name: "admin_full_access"
      condition: "user.role == 'administrator'"
      permissions: ["*"]
```

## 📊 Monitoring Setup

### Health Checks

#### Health Check Endpoint
```bash
# Test health check endpoint
curl -X GET http://localhost:8080/health \
  -H "Accept: application/json"

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-11-02T00:54:03Z",
  "version": "2.1.0",
  "components": {
    "database": "healthy",
    "ai_models": "healthy",
    "storage": "healthy",
    "cache": "healthy"
  }
}
```

#### Monitoring Configuration
```yaml
# config.yaml
monitoring:
  health_check:
    enabled: true
    interval: 30  # seconds
    endpoints: ["/health", "/ready", "/metrics"]
    
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
    
  logging:
    level: "INFO"
    format: "json"
    file: "/var/log/ai-excel-extractor/app.log"
    
  alerting:
    enabled: true
    email: "admin@company.com"
    webhook: "https://hooks.slack.com/your-webhook"
```

### Log Management

#### Log Rotation
```bash
# Configure logrotate
sudo tee /etc/logrotate.d/ai-excel-extractor << EOF
/var/log/ai-excel-extractor/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 ai-excel ai-excel
    postrotate
        systemctl reload ai-excel-extractor
    endscript
}
EOF
```

## 🧪 Verification and Testing

### Installation Verification

#### Basic Verification
```bash
# Check installation
ai-excel-extractor --version

# Test configuration
ai-excel-extractor config validate

# Run health check
ai-excel-extractor health-check

# Test processing
ai-excel-extractor process --test-file test.xlsx --output test_output.json
```

#### Functional Testing
```python
# Python test script
import unittest
from ai_excel_extractor import AIExcelExtractor

class TestInstallation(unittest.TestCase):
    def setUp(self):
        self.extractor = AIExcelExtractor()
        
    def test_basic_functionality(self):
        # Test basic extraction
        result = self.extractor.extract("test.xlsx")
        self.assertIsNotNone(result)
        self.assertIn("loads", result)
        
    def test_api_health(self):
        import requests
        response = requests.get("http://localhost:8080/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

if __name__ == "__main__":
    unittest.main()
```

#### Performance Testing
```bash
# Run performance benchmarks
ai-excel-extractor benchmark \
  --test-files test_data/ \
  --iterations 100 \
  --output benchmark_results.json

# Expected output:
{
  "average_processing_time": "2.3s",
  "throughput": "26 files/hour",
  "memory_usage": "512MB",
  "cpu_utilization": "45%",
  "success_rate": "99.7%"
}
```

## 🔧 Troubleshooting Installation

### Common Issues

#### Issue 1: Python Version Incompatibility
```bash
# Error: Python 3.8+ required
python --version

# Solution: Upgrade Python
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9

# CentOS/RHEL
sudo yum install python39

# macOS (using Homebrew)
brew install python@3.9
```

#### Issue 2: Permission Denied
```bash
# Error: Permission denied when accessing directories
sudo chown -R ai-excel:ai-excel /opt/ai-excel-extractor
sudo chmod -R 755 /opt/ai-excel-extractor

# For data directories
sudo mkdir -p /var/data/ai-excel-extractor
sudo chown ai-excel:ai-excel /var/data/ai-excel-extractor
sudo chmod 755 /var/data/ai-excel-extractor
```

#### Issue 3: Database Connection Failed
```bash
# Error: Unable to connect to database
# Check database configuration
cat ~/.ai-excel-extractor/config.yaml | grep database

# Test database connection
ai-excel-extractor db test-connection

# Fix common issues:
# 1. Check if database service is running
sudo systemctl status postgresql

# 2. Verify credentials
ai-excel-extractor db verify-credentials

# 3. Check network connectivity
telnet localhost 5432
```

#### Issue 4: AI Model Loading Failed
```bash
# Error: AI model files not found
# Check model directory
ls -la /opt/ai-excel-extractor/models/

# Download models if missing
ai-excel-extractor models download

# Verify model integrity
ai-excel-extractor models verify
```

#### Issue 5: Port Already in Use
```bash
# Error: Port 8080 already in use
netstat -tlnp | grep :8080

# Find and kill process using port
sudo kill -9 $(sudo lsof -t -i:8080)

# Or configure different port
ai-excel-extractor config set api.port 8081
```

### Diagnostic Tools

#### System Diagnostics
```bash
# Run comprehensive diagnostics
ai-excel-extractor doctor

# Detailed system check
ai-excel-extractor system-check --verbose

# Component testing
ai-excel-extractor test components
```

#### Log Analysis
```bash
# View recent logs
tail -f /var/log/ai-excel-extractor/app.log

# Search for errors
grep -i error /var/log/ai-excel-extractor/app.log

# Analyze log patterns
ai-excel-extractor log-analyze --pattern "ERROR" --timeframe "1h"
```

## 📋 Next Steps

### Post-Installation Checklist

- [ ] Verify installation with health check
- [ ] Configure database connection
- [ ] Set up SSL/TLS certificates
- [ ] Create admin user account
- [ ] Configure monitoring and logging
- [ ] Test basic functionality
- [ ] Set up backup procedures
- [ ] Configure security policies
- [ ] Test performance benchmarks
- [ ] Document configuration changes

### Initial Configuration

1. **System Configuration**
   - Update configuration files
   - Set environment variables
   - Configure database connections
   - Set up monitoring

2. **Security Setup**
   - Generate SSL certificates
   - Configure authentication
   - Set up user roles and permissions
   - Configure access controls

3. **Performance Tuning**
   - Optimize worker processes
   - Configure caching
   - Set memory limits
   - Enable GPU acceleration (if available)

4. **Monitoring and Alerts**
   - Configure health checks
   - Set up log rotation
   - Configure alerting
   - Set up performance monitoring

### Production Readiness

For production deployment, ensure:
- SSL/TLS certificates are properly configured
- Database backups are automated
- Monitoring and alerting are active
- Security policies are implemented
- Performance is optimized for your workload
- Documentation is updated with production-specific settings

### Support Resources

- **Documentation**: [docs.ai-excel-extractor.com](https://docs.ai-excel-extractor.com)
- **Support**: support@ai-excel-extractor.com
- **Community**: [github.com/company/ai-excel-extractor](https://github.com/company/ai-excel-extractor)
- **Issues**: [github.com/company/ai-excel-extractor/issues](https://github.com/company/ai-excel-extractor/issues)

The installation is now complete! You can proceed to the [Configuration Guide](configuration_guide.md) for detailed system configuration instructions.