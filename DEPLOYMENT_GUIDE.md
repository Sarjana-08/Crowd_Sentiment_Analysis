# Deploy Crowd Monitoring Dashboard to Cloud

## Quick Deployment Options

### 🚀 Option 1: PythonAnywhere (Easiest - Recommended)

**PythonAnywhere** is the fastest way to deploy Python Flask apps.

#### Step 1: Create Account
1. Go to https://www.pythonanywhere.com
2. Sign up for free account
3. Verify email

#### Step 2: Upload Files
```bash
# In PythonAnywhere Files section, upload:
- dashboard_app_production.py
- crowd_detector.py
- email_alerts.py
- requirements_deployment.txt
- templates/ (folder)
- static/ (folder)
```

#### Step 3: Create Web App
1. Click "Web" in PythonAnywhere
2. Click "Add a new web app"
3. Choose "Python 3.12" and "Flask"
4. Enter app name (e.g., "crowd-monitor")

#### Step 4: Configure
1. Go to Web tab
2. Set source code to: `/home/yourusername/crowd_monitor/dashboard_app_production.py`
3. Set working directory to: `/home/yourusername/crowd_monitor/`

#### Step 5: Install Dependencies
```bash
mkvirtualenv --python=/usr/bin/python3.12 crowd-monitor
pip install -r requirements_deployment.txt
```

#### Step 6: Environment Variables
In Web tab, add to `.env`:
```
DATABASE_URL=sqlite:////home/yourusername/crowd_monitor/monitor.db
CROWD_THRESHOLD=500
ALERTS_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

#### Step 7: Run
Click "Reload" on the Web tab

**URL:** `https://yourusername.pythonanywhere.com`

---

### 🌐 Option 2: Heroku (Free Alternative Ending)

**Note:** Heroku removed free tier in Nov 2022, but here's the process:

#### Step 1: Install Heroku CLI
```bash
# Windows
choco install heroku-cli

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

#### Step 2: Create Heroku App
```bash
heroku login
heroku create crowd-monitoring-app
```

#### Step 3: Add PostgreSQL
```bash
heroku addons:create heroku-postgresql:basic
```

#### Step 4: Set Environment Variables
```bash
heroku config:set DATABASE_URL=postgresql://...
heroku config:set CROWD_THRESHOLD=500
heroku config:set ALERTS_ENABLED=true
heroku config:set SMTP_SERVER=smtp.gmail.com
heroku config:set SMTP_PORT=587
heroku config:set SMTP_USER=your-email@gmail.com
heroku config:set SMTP_PASSWORD=your-app-password
```

#### Step 5: Deploy
```bash
git push heroku main
heroku open
```

**URL:** `https://crowd-monitoring-app.herokuapp.com`

---

### 🖥️ Option 3: AWS EC2 (Most Control)

#### Step 1: Launch EC2 Instance
1. AWS Console → EC2
2. Launch Ubuntu 22.04 LTS instance (t2.micro free tier)
3. Configure security groups (allow 80, 443, 22)
4. Create key pair and download

#### Step 2: Connect & Setup
```bash
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & dependencies
sudo apt install -y python3.12 python3.12-venv python3-pip
sudo apt install -y nginx

# Install FFmpeg (for video)
sudo apt install -y ffmpeg
```

#### Step 3: Setup App
```bash
cd /home/ubuntu
git clone your-repo
cd crowd-monitor

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements_deployment.txt
```

#### Step 4: Configure Environment
```bash
cat > .env << EOF
DATABASE_URL=postgresql://user:pass@localhost/crowddb
CROWD_THRESHOLD=500
ALERTS_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EOF
```

#### Step 5: Setup Nginx Reverse Proxy
```bash
sudo cat > /etc/nginx/sites-available/crowd-monitor << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/crowd-monitor /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

#### Step 6: Setup Systemd Service
```bash
sudo cat > /etc/systemd/system/crowd-monitor.service << EOF
[Unit]
Description=Crowd Monitoring Dashboard
After=network.target

[Service]
Type=notify
User=ubuntu
WorkingDirectory=/home/ubuntu/crowd-monitor
ExecStart=/home/ubuntu/crowd-monitor/venv/bin/gunicorn --bind 127.0.0.1:5000 dashboard_app_production:app
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable crowd-monitor
sudo systemctl start crowd-monitor
```

#### Step 7: Setup SSL (Let's Encrypt)
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

**URL:** `https://your-domain.com`

---

### 📦 Option 4: DigitalOcean (Best Balance)

#### Step 1: Create Droplet
1. DigitalOcean Console → Create Droplet
2. Choose Ubuntu 22.04 LTS
3. Size: $6/month (2GB RAM)
4. Add SSH key

#### Step 2: SSH & Setup
```bash
ssh root@your-droplet-ip

# Update & install
apt update && apt upgrade -y
apt install -y python3.12 python3.12-venv python3-pip nginx postgresql postgresql-contrib

# Create app user
adduser appuser
usermod -aG sudo appuser
```

#### Step 3: Deploy App
```bash
su - appuser
cd /home/appuser
git clone your-repo
cd crowd-monitor

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements_deployment.txt
```

#### Step 4: Setup Database
```bash
# As postgres user
sudo -u postgres createdb crowdmonitor
sudo -u postgres createuser appuser
```

#### Step 5: Setup Environment & Run
(Same as AWS steps 4-6)

**URL:** `https://your-droplet-ip` or `https://your-domain.com`

---

### 🐳 Option 5: Docker + Cloud Run (Google Cloud - $0 to start)

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_deployment.txt .
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 5000

# Run app
CMD exec gunicorn --bind :$PORT --workers 2 dashboard_app_production:app
```

#### Step 2: Create Cloud Run Service
```bash
# Install Google Cloud CLI
# From: https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build & deploy
gcloud run deploy crowd-monitor \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=postgresql://...,CROWD_THRESHOLD=500
```

**URL:** `https://crowd-monitor-xxxxx.run.app`

---

## Configuration for Production

### Environment Variables Template

Create `.env` file with:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/crowddb

# Crowd Detection
CROWD_THRESHOLD=500
ALERTS_ENABLED=true
FPS_TARGET=5
SMOOTH_WINDOW=5
MAX_HISTORY=1000

# Email Alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Flask
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Server
PORT=5000
```

### Email Configuration

**For Gmail:**
1. Enable 2-factor authentication on Gmail
2. Generate app-specific password at: https://myaccount.google.com/apppasswords
3. Use that password in `SMTP_PASSWORD`

**For Other Services:**
- Outlook: `smtp.office365.com:587`
- SendGrid: `smtp.sendgrid.net:587` (use `apikey` as username)

### Database Setup

**PostgreSQL (Production):**
```bash
# Connect to database
psql -U postgres

# Create database and user
CREATE DATABASE crowdmonitor;
CREATE USER appuser WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE crowdmonitor TO appuser;
```

**SQLite (Development):**
Automatically created at `crowd_monitor.db`

---

## Monitoring & Maintenance

### Check Health
```bash
# From browser
https://your-domain.com/api/health

# Response:
{
  "status": "healthy",
  "timestamp": "2025-12-27T10:30:00",
  "uptime_seconds": 3600,
  "frame_count": 12000
}
```

### View Logs (PythonAnywhere)
Dashboard → Logs → Web

### View Logs (Linux/Docker)
```bash
# Systemd
sudo journalctl -u crowd-monitor -f

# Docker
docker logs crowd-monitor -f
```

### Database Backup
```bash
# PostgreSQL
pg_dump crowdmonitor > backup_$(date +%Y%m%d).sql

# SQLite
cp crowd_monitor.db backup_$(date +%Y%m%d).db
```

---

## SSL/HTTPS Setup

### Auto with Let's Encrypt
```bash
# On Linux with Nginx
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com --auto
```

### Manual Certificate
1. Go to: https://letsencrypt.org
2. Follow "Getting Started" guide
3. Upload certificate to your server

---

## Performance Optimization

### Database Indexes
```sql
CREATE INDEX idx_detection_timestamp ON detection_records(timestamp);
CREATE INDEX idx_alert_timestamp ON alert_history(timestamp);
CREATE INDEX idx_alert_email ON alert_recipients(email);
```

### Caching
The dashboard uses in-memory cache for:
- Latest frame (updated at FPS target)
- Latest heatmap (updated at FPS target)
- Latest detection (updated at FPS target)

### Scaling
For high traffic:
1. Use load balancer (AWS ELB, Nginx, HAProxy)
2. Run multiple Flask instances
3. Use Redis for session management
4. Enable database connection pooling

---

## Troubleshooting

### "Module not found" Error
```bash
source venv/bin/activate
pip install -r requirements_deployment.txt
```

### Database Connection Failed
```bash
# Check environment variable
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL
```

### High Memory Usage
```bash
# Reduce MAX_HISTORY
export MAX_HISTORY=500

# Or reduce FPS
export FPS_TARGET=2
```

### Camera Not Working in Cloud
Cloud servers don't have cameras. Options:
1. Use video file: `cv2.VideoCapture('video.mp4')`
2. Use RTSP stream: `cv2.VideoCapture('rtsp://camera-ip:554/stream')`
3. Use WebRTC: Deploy with Janus gateway

---

## Quick Deployment Checklist

- [ ] Create account on hosting platform
- [ ] Upload files
- [ ] Install dependencies
- [ ] Configure environment variables
- [ ] Set up database
- [ ] Configure email (SMTP)
- [ ] Test API endpoints
- [ ] Add SSL certificate
- [ ] Enable monitoring
- [ ] Setup backups
- [ ] Document domain/credentials

---

## Support

For issues:
1. Check logs: `/api/health`
2. Test endpoints manually
3. Verify environment variables
4. Check database connection
5. Review platform-specific docs
