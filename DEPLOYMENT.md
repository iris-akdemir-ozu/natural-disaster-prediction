# Deployment Guide

Complete guide for deploying the Flood Segmentation System to production.

## System Architecture

\`\`\`
┌─────────────────┐
│   Frontend      │  (SvelteKit)
│   Port: 5173    │
└────────┬────────┘
         │
         │ HTTP API Calls
         │
┌────────▼────────┐
│   Backend API   │  (Flask)
│   Port: 5000    │
└────────┬────────┘
         │
         │ Model Inference
         │
┌────────▼────────┐
│  Trained Model  │  (U-Net .h5 file)
│  models/        │
└─────────────────┘
\`\`\`

## Prerequisites

1. Trained U-Net model at \`models/flood_segmentation_model.h5\`
2. Python 3.8+ with all dependencies installed
3. Node.js 18+ for frontend

## Local Development Setup

### 1. Setup Backend

\`\`\`bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt

# Start Flask server
python app.py
\`\`\`

Backend will be available at \`http://localhost:5000\`

### 2. Setup Frontend

\`\`\`bash
# In project root directory
npm install

# Start development server
npm run dev
\`\`\`

Frontend will be available at \`http://localhost:5173\`

## Production Deployment

### Option 1: Docker Deployment (Recommended)

Create \`Dockerfile\` for backend:

\`\`\`dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/

WORKDIR /app/backend

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
\`\`\`

Create \`docker-compose.yml\`:

\`\`\`yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/flood_segmentation_model.h5
    restart: unless-stopped

  frontend:
    image: node:18-alpine
    working_dir: /app
    volumes:
      - ./:/app
      - /app/node_modules
    ports:
      - "5173:5173"
    command: sh -c "npm install && npm run dev -- --host"
    environment:
      - VITE_API_URL=http://localhost:5000
    depends_on:
      - backend
    restart: unless-stopped
\`\`\`

Deploy with Docker:

\`\`\`bash
docker-compose up -d
\`\`\`

### Option 2: Cloud Deployment (Vercel + Cloud Run)

#### Deploy Frontend to Vercel

\`\`\`bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
\`\`\`

#### Deploy Backend to Google Cloud Run

\`\`\`bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/flood-api

# Deploy to Cloud Run
gcloud run deploy flood-api \
  --image gcr.io/PROJECT_ID/flood-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 300 \
  --allow-unauthenticated
\`\`\`

### Option 3: Traditional VPS Deployment

#### Backend Setup (using Gunicorn + Nginx)

1. Install dependencies on server:

\`\`\`bash
sudo apt update
sudo apt install python3-pip python3-venv nginx
\`\`\`

2. Setup application:

\`\`\`bash
cd /var/www/flood-segmentation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
\`\`\`

3. Create systemd service (\`/etc/systemd/system/flood-api.service\`):

\`\`\`ini
[Unit]
Description=Flood Segmentation API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/flood-segmentation/backend
Environment="PATH=/var/www/flood-segmentation/venv/bin"
ExecStart=/var/www/flood-segmentation/venv/bin/gunicorn --workers 2 --bind 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
\`\`\`

4. Configure Nginx:

\`\`\`nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
\`\`\`

5. Start services:

\`\`\`bash
sudo systemctl start flood-api
sudo systemctl enable flood-api
sudo systemctl restart nginx
\`\`\`

#### Frontend Setup

\`\`\`bash
# Build frontend
npm run build

# Serve with Nginx
sudo cp -r build/* /var/www/html/
\`\`\`

## Environment Variables

### Backend

- \`MODEL_PATH\`: Path to trained model (default: \`../models/flood_segmentation_model.h5\`)
- \`FLASK_ENV\`: Set to \`production\` for production deployment

### Frontend

Create \`.env\` file:

\`\`\`
VITE_API_URL=http://localhost:5000
\`\`\`

For production:

\`\`\`
VITE_API_URL=https://api.yourdomain.com
\`\`\`

## Performance Optimization

### Backend

1. **Use Gunicorn with multiple workers:**
   \`\`\`bash
   gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
   \`\`\`

2. **Enable model caching** (already implemented in app.py)

3. **Add Redis caching** for repeated predictions:
   \`\`\`python
   import redis
   import hashlib
   
   redis_client = redis.Redis(host='localhost', port=6379, db=0)
   
   def get_cached_prediction(image_hash):
       return redis_client.get(f"pred:{image_hash}")
   \`\`\`

### Frontend

1. **Enable compression in Nginx:**
   \`\`\`nginx
   gzip on;
   gzip_types text/plain text/css application/json application/javascript;
   \`\`\`

2. **Add caching headers:**
   \`\`\`nginx
   location /assets {
       expires 1y;
       add_header Cache-Control "public, immutable";
   }
   \`\`\`

## Monitoring

### Backend Health Check

\`\`\`bash
curl http://localhost:5000/api/health
\`\`\`

### Logging

Backend logs are available in console output. For production, configure logging to file:

\`\`\`python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
\`\`\`

## Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple backend instances behind a load balancer
2. **GPU Support**: Use GPU-enabled instances for faster inference
3. **Model Optimization**: Convert model to TensorFlow Lite or ONNX for faster inference
4. **Caching**: Implement Redis caching for frequently processed images

## Security

1. **API Rate Limiting**:
   \`\`\`python
   from flask_limiter import Limiter
   
   limiter = Limiter(app, key_func=lambda: request.remote_addr)
   
   @app.route('/api/predict')
   @limiter.limit("10 per minute")
   def predict():
       # ...
   \`\`\`

2. **HTTPS**: Always use HTTPS in production
3. **Input Validation**: Validate file size and type
4. **CORS**: Configure specific origins instead of \`*\`

## Troubleshooting

### Backend Issues

- **Model not loading**: Ensure model file exists and path is correct
- **Out of memory**: Reduce batch size or image dimensions
- **Slow predictions**: Consider using GPU or optimizing model

### Frontend Issues

- **CORS errors**: Ensure backend CORS is properly configured
- **API connection failed**: Check backend URL in environment variables
- **Map not loading**: Verify Leaflet.js is properly imported

## Backup and Recovery

1. **Model Backup**: Regularly backup trained model file
2. **Database Backup**: If storing predictions, backup database regularly
3. **Configuration Backup**: Version control all configuration files

## Cost Optimization

1. **Use smaller instance types** when traffic is low
2. **Implement auto-scaling** based on demand
3. **Cache predictions** to reduce compute costs
4. **Use spot instances** for training (if retraining regularly)
\`\`\`
