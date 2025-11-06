# Deployment Guide - AI Excel Extraction System

## Overview

This comprehensive deployment guide provides detailed instructions for deploying the AI-Powered Excel Extraction System across different environments, from development to production. It covers containerized deployments, cloud platforms, on-premises installations, and enterprise-grade configurations.

## 🚀 Deployment Architecture

### Deployment Models

The AI Excel Extraction System supports multiple deployment models:

1. **Single-Node Deployment** - Standalone installation for small organizations
2. **Multi-Node Cluster** - Distributed deployment for high availability and scalability
3. **Containerized Deployment** - Docker/Kubernetes for cloud and microservices environments
4. **Hybrid Deployment** - Combination of on-premises and cloud components
5. **Edge Deployment** - Lightweight deployment for remote locations

### Infrastructure Requirements

#### Hardware Specifications

**Minimum Requirements**
```
CPU: 4 cores, 2.5GHz+
RAM: 8GB
Storage: 100GB SSD
Network: 100Mbps
GPU: Optional (NVIDIA GPU for acceleration)
```

**Recommended Production Requirements**
```
CPU: 16+ cores, 3.0GHz+
RAM: 32GB
Storage: 500GB NVMe SSD
Network: 1Gbps
GPU: NVIDIA Tesla V100 or better (optional)
```

**High-Availability Requirements**
```
Load Balancer: 2x (HA pair)
Application Servers: 3x minimum
Database Cluster: 3x (master + 2 replicas)
Cache Cluster: 3x Redis nodes
Storage: Shared storage or distributed file system
```

## 🐳 Containerized Deployment

### Docker Deployment

#### Single Container Deployment
```bash
# Create deployment directory
mkdir -p ai-excel-extractor-deployment
cd ai-excel-extractor-deployment

# Create Docker Compose file
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  ai-excel-extractor:
    image: ai-excel-extractor:latest
    container_name: ai-extractor
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "9090:9090"  # Metrics port
    environment:
      - AI_EXTRACTOR_ENV=production
      - AI_EXTRACTOR_DB_URL=postgresql://postgres:password@db:5432/ai_extractor
      - AI_EXTRACTOR_REDIS_URL=redis://redis:6379/0
      - AI_EXTRACTOR_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/var/log/ai-excel-extractor
      - ./config:/app/config
      - ./models:/opt/ai-excel-extractor/models
    depends_on:
      - db
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:13
    container_name: ai-extractor-db
    restart: unless-stopped
    environment:
      - POSTGRES_DB=ai_extractor
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    container_name: ai-extractor-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    container_name: ai-extractor-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - ai-excel-extractor

volumes:
  postgres_data:
  redis_data:
EOF

# Create environment file
cat > .env << 'EOF'
# Database Configuration
POSTGRES_DB=ai_extractor
POSTGRES_USER=postgres
POSTGRES_PASSWORD=SecurePassword123!
POSTGRES_HOST=db
POSTGRES_PORT=5432

# AI Extractor Configuration
AI_EXTRACTOR_ENV=production
AI_EXTRACTOR_LOG_LEVEL=INFO
AI_EXTRACTOR_API_HOST=0.0.0.0
AI_EXTRACTOR_API_PORT=8080
AI_EXTRACTOR_WORKERS=4
AI_EXTRACTOR_MAX_FILE_SIZE=100MB
AI_EXTRACTOR_CONCURRENT_JOBS=4

# Security Configuration
AI_EXTRACTOR_JWT_SECRET_KEY=your-jwt-secret-key-here
AI_EXTRACTOR_ENCRYPTION_KEY=your-encryption-key-here
EOF

# Start the deployment
docker-compose up -d
```

### Kubernetes Deployment

#### Production Kubernetes Manifest
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-excel-extractor
  labels:
    name: ai-excel-extractor

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-extractor
  namespace: ai-excel-extractor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-extractor
  template:
    metadata:
      labels:
        app: ai-extractor
    spec:
      containers:
      - name: ai-extractor
        image: ai-excel-extractor:latest
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 9090
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: api
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: api
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-extractor-service
  namespace: ai-excel-extractor
spec:
  selector:
    app: ai-extractor
  ports:
  - name: api
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-extractor-ingress
  namespace: ai-excel-extractor
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: ai-extractor.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-extractor-service
            port:
              number: 8080
```

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 and RDS Deployment Script
```bash
#!/bin/bash
# deploy-aws.sh

set -e

# Configuration
REGION="us-east-1"
VPC_NAME="ai-extractor-vpc"
KEY_PAIR="ai-extractor-key"
SECURITY_GROUP="ai-extractor-sg"
INSTANCE_TYPE="t3.large"
RDS_INSTANCE_CLASS="db.t3.medium"

echo "Deploying AI Excel Extractor to AWS..."

# Create VPC
echo "Creating VPC..."
VPC_ID=$(aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=$VPC_NAME}]" \
  --query 'Vpc.VpcId' --output text)

# Create security group
echo "Creating security group..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
  --group-name ai-extractor-sg \
  --description "AI Excel Extractor Security Group" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)

# Add security group rules
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8080 --cidr 0.0.0.0/0

# Create RDS instance
echo "Creating RDS instance..."
aws rds create-db-instance \
  --db-instance-identifier ai-extractor-db \
  --db-instance-class $RDS_INSTANCE_CLASS \
  --engine postgres \
  --master-username postgres \
  --master-user-password "SecurePassword123!" \
  --allocated-storage 100 \
  --vpc-security-group-ids $SECURITY_GROUP_ID \
  --backup-retention-period 7 \
  --storage-encrypted

echo "Deployment completed!"
echo "Next steps:"
echo "1. Wait for RDS instance to be available (5-10 minutes)"
echo "2. Launch EC2 instance with the application"
echo "3. Configure load balancer and SSL certificates"
```

### Azure Deployment

#### App Service and Azure Database
```bash
#!/bin/bash
# deploy-azure.sh

set -e

# Configuration
RESOURCE_GROUP="ai-extractor-rg"
LOCATION="eastus"
APP_NAME="ai-extractor-app"
DB_SERVER="ai-extractor-db-server"
DB_NAME="ai_extractor"

echo "Deploying AI Excel Extractor to Azure..."

# Create resource group
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create PostgreSQL database
echo "Creating PostgreSQL database..."
az postgres server create \
  --resource-group $RESOURCE_GROUP \
  --name $DB_SERVER \
  --location $LOCATION \
  --admin-user aiuser \
  --admin-password "SecurePassword123!" \
  --sku-name B_Gen5_2

# Create database
echo "Creating database..."
az postgres db create \
  --resource-group $RESOURCE_GROUP \
  --server-name $DB_SERVER \
  --name $DB_NAME

# Configure firewall rules
echo "Configuring firewall rules..."
az postgres server firewall-rule create \
  --resource-group $RESOURCE_GROUP \
  --server $DB_SERVER \
  --name AllowAll \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 255.255.255.255

# Create App Service plan
echo "Creating App Service plan..."
az appservice plan create \
  --name ai-extractor-plan \
  --resource-group $RESOURCE_GROUP \
  --sku P1V2 \
  --is-linux

# Create web app
echo "Creating web app..."
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan ai-extractor-plan \
  --name $APP_NAME \
  --deployment-container-image-name ai-excel-extractor:latest

echo "Deployment completed!"
echo "App Service URL: https://${APP_NAME}.azurewebsites.net"
```

### Google Cloud Platform

#### Cloud Run and Cloud SQL
```bash
#!/bin/bash
# deploy-gcp.sh

set -e

# Configuration
PROJECT_ID="ai-extractor-project"
REGION="us-central1"
SERVICE_NAME="ai-extractor-service"
DB_INSTANCE="ai-extractor-db"
DB_NAME="ai_extractor"

echo "Deploying AI Excel Extractor to Google Cloud Platform..."

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com

# Create Cloud SQL instance
echo "Creating Cloud SQL instance..."
gcloud sql instances create $DB_INSTANCE \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=$REGION

# Create database
echo "Creating database..."
gcloud sql databases create $DB_NAME --instance=$DB_INSTANCE

# Create database user
echo "Creating database user..."
gcloud sql users create aiuser \
  --instance=$DB_INSTANCE \
  --password="SecurePassword123!"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/ai-excel-extractor \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 1

echo "Deployment completed!"
```

## 🏢 On-Premises Deployment

### High-Availability On-Premises Setup

#### Network Configuration
```bash
#!/bin/bash
# network-setup.sh

set -e

# Network configuration for on-premises deployment
NETWORK_CIDR="192.168.10.0/24"
LOAD_BALANCER_IP="192.168.10.10"
APP_SERVERS=("192.168.10.20" "192.168.10.21" "192.168.10.22" "192.168.10.23")
DB_SERVERS=("192.168.10.30" "192.168.10.31" "192.168.10.32")

echo "Setting up network configuration..."

# Configure firewall rules
setup_firewall() {
    local server_ip=$1
    local server_name=$2
    
    echo "Configuring firewall for $server_name ($server_ip)..."
    
    # SSH access
    iptables -A INPUT -p tcp --dport 22 -s $NETWORK_CIDR -j ACCEPT
    
    # Application ports
    iptables -A INPUT -p tcp --dport 8080 -s $NETWORK_CIDR -j ACCEPT
    iptables -A INPUT -p tcp --dport 9090 -s $NETWORK_CIDR -j ACCEPT
    
    # Database ports
    if [[ "$server_name" == db-* ]]; then
        iptables -A INPUT -p tcp --dport 5432 -s $NETWORK_CIDR -j ACCEPT
    fi
    
    # Internal communication
    iptables -A INPUT -p tcp --dport 8000-9000 -s $NETWORK_CIDR -j ACCEPT
    
    # Save rules
    iptables-save > /etc/iptables/rules.v4
}

# Configure each server type
setup_firewall $LOAD_BALANCER_IP "lb-01"
for server in "${APP_SERVERS[@]}"; do
    setup_firewall $server "app-01"
done
for server in "${DB_SERVERS[@]}"; do
    setup_firewall $server "db-01"
done

echo "Network configuration completed!"
```

#### Load Balancer Configuration (HAProxy)
```haproxy
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

# Frontend for HTTP
frontend http_frontend
    bind *:80
    redirect scheme https if !{ ssl_fc }

# Frontend for HTTPS
frontend https_frontend
    bind *:443 ssl crt /etc/ssl/certs/ai-extractor.pem
    
    # API endpoints
    acl is_api path_beg /api/
    use_backend api_backend if is_api
    
    # Health check
    acl is_health path /health
    use_backend health_backend if is_health
    
    default_backend api_backend

# API backend
backend api_backend
    balance roundrobin
    option httpchk GET /health
    server app-01 192.168.10.20:8080 check inter 30s
    server app-02 192.168.10.21:8080 check inter 30s
    server app-03 192.168.10.22:8080 check inter 30s
    server app-04 192.168.10.23:8080 check inter 30s
```

## 🔧 Deployment Automation

### Ansible Deployment

#### Main Playbook
```yaml
# deploy-ai-extractor.yml
---
- name: Deploy AI Excel Extractor
  hosts: all
  become: yes
  vars:
    app_version: "2.1.0"
    app_user: "ai-excel"
    app_dir: "/opt/ai-excel-extractor"
    
  tasks:
    - name: Create application user
      user:
        name: "{{ app_user }}"
        system: yes
        shell: /bin/bash
        
    - name: Create directories
      file:
        path: "{{ item }}"
        state: directory
        owner: "{{ app_user }}"
        mode: '0755'
      loop:
        - "{{ app_dir }}"
        - "{{ app_dir }}/config"
        - "{{ app_dir }}/models"
        
    - name: Install system dependencies
      package:
        name:
          - python3
          - python3-pip
          - postgresql-client
          - nginx
        state: present
        
    - name: Install application
      pip:
        name: ai-excel-extractor
        version: "{{ app_version }}"
        
    - name: Copy configuration files
      template:
        src: config.yaml.j2
        dest: "{{ app_dir }}/config/config.yaml"
        owner: "{{ app_user }}"
        
    - name: Enable and start services
      systemd:
        name: "{{ item }}"
        daemon_reload: yes
        enabled: yes
        state: started
      loop:
        - ai-extractor
        - nginx
```

## 📊 Deployment Monitoring

### Health Check Implementation

```python
# health_check.py
from fastapi import FastAPI
from sqlalchemy import text
from redis import Redis
import psutil

app = FastAPI()

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "checks": {}
    }
    
    # Database check
    try:
        with app.state.db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            checks["checks"]["database"] = "healthy" if result else "unhealthy"
    except Exception as e:
        checks["checks"]["database"] = f"unhealthy: {str(e)}"
        checks["status"] = "degraded"
    
    # Redis check
    try:
        redis_client = Redis.from_url(app.state.redis_url)
        redis_client.ping()
        checks["checks"]["redis"] = "healthy"
    except Exception as e:
        checks["checks"]["redis"] = f"unhealthy: {str(e)}"
        checks["status"] = "degraded"
    
    # Disk space check
    disk_usage = psutil.disk_usage('/')
    usage_percent = (disk_usage.used / disk_usage.total) * 100
    if usage_percent > 90:
        checks["checks"]["disk_space"] = f"critical: {usage_percent:.1f}%"
        checks["status"] = "unhealthy"
    elif usage_percent > 80:
        checks["checks"]["disk_space"] = f"warning: {usage_percent:.1f}%"
        if checks["status"] == "healthy":
            checks["status"] = "degraded"
    else:
        checks["checks"]["disk_space"] = f"healthy: {usage_percent:.1f}%"
    
    return checks
```

## 📋 Deployment Checklist

### Pre-Deployment Checklist
- [ ] Infrastructure requirements verified
- [ ] Security configuration completed
- [ ] Database setup and migration scripts tested
- [ ] SSL/TLS certificates configured
- [ ] Monitoring and alerting configured
- [ ] Backup procedures implemented
- [ ] Load balancer configuration verified
- [ ] Health check endpoints tested

### Post-Deployment Checklist
- [ ] Application starts successfully
- [ ] Health checks pass
- [ ] Database connectivity verified
- [ ] API endpoints respond correctly
- [ ] File upload functionality tested
- [ ] Performance benchmarks met
- [ ] Monitoring dashboards functional
- [ ] Backup procedures tested
- [ ] Security scan completed
- [ ] Documentation updated

### Production Readiness Checklist
- [ ] All components deployed across multiple servers
- [ ] Load balancing configured and tested
- [ ] Database clustering setup for high availability
- [ ] Monitoring and alerting fully configured
- [ ] SSL/TLS certificates installed and verified
- [ ] Security policies implemented
- [ ] Backup and disaster recovery procedures tested
- [ ] Performance monitoring and optimization configured
- [ ] Documentation and runbooks prepared
- [ ] Team training completed

## 🎯 Next Steps

After successful deployment:

1. **Monitor System Performance** - Set up comprehensive monitoring
2. **Configure Backups** - Implement automated backup procedures
3. **Security Hardening** - Apply security best practices
4. **Performance Tuning** - Optimize for your specific workload
5. **Disaster Recovery** - Test recovery procedures
6. **Documentation** - Keep deployment documentation current

For ongoing maintenance and troubleshooting, refer to the [Troubleshooting Guide](troubleshooting_guide.md).