# UC-2 Flask Application - Docker Setup

This repository contains a Flask application that provides XAI (Explainable AI) endpoints for UC-2 driver fatigue analysis. The Docker setup allows you to run the application in a containerized environment.

## 🐳 Docker Setup

### Prerequisites
- Docker
- Docker Compose

### Quick Start

1. **Build the Docker image:**
   ```bash
   docker build -t uc2-flask-app .
   ```

2. **Run the application:**
   ```bash
   # Simple run
   docker run --rm -p 5000:5000 -v $(pwd)/csv-pkl-json:/app/csv-pkl-json:ro -v $(pwd)/output:/app/output uc2-flask-app
   
   # With docker-compose
   docker-compose up uc2-flask-app
   ```

3. **Development mode:**
   ```bash
   docker-compose --profile dev up uc2-flask-dev
   ```

### 📁 Directory Structure

```
main-data-retrieval-endpoint/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── csv-pkl-json/          # Model and data files
│   ├── model.pkl
│   └── label_encoders.pkl
├── scripts/               # Flask application
│   └── main-explain-uc2.py
├── output/               # Generated results (created automatically)
└── build-and-run.bat    # Windows build script
```

### 🔧 Available Services

| Service | Description | Port | Command |
|---------|-------------|------|---------|
| `uc2-flask-app` | Production Flask app | 5000 | `python scripts/main-explain-uc2.py` |
| `uc2-flask-dev` | Development Flask app | 5001 | `flask run --reload` |

### 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/explain-uc2-model` | POST | Model explanation analysis |
| `/explain-uc2-data` | POST | Driver data analysis |

### 📊 API Usage Examples

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Model Explanation (Expert):**
```bash
curl -X POST http://localhost:5000/explain-uc2-model \
  -F "user_level=expert" \
  -F "model_file=@csv-pkl-json/model.pkl" \
  -F "encoder_file=@csv-pkl-json/label_encoders.pkl" \
  -F "data_file=@csv-pkl-json/alert-data-uc2-demographics.csv"
```

**Model Explanation (Beginner):**
```bash
curl -X POST http://localhost:5000/explain-uc2-model \
  -F "user_level=beginner" \
  -F "model_file=@csv-pkl-json/model.pkl" \
  -F "encoder_file=@csv-pkl-json/label_encoders.pkl" \
  -F "data_file=@csv-pkl-json/alert-data-uc2-demographics.csv"
```

**Driver Data Analysis:**
```bash
curl -X POST http://localhost:5000/explain-uc2-data \
  -F "user_level=expert" \
  -F "frame_file=@csv-pkl-json/frames-cleaned.csv" \
  -F "hr_file=@csv-pkl-json/heart_rate.csv"
```

### 🚀 Usage Examples

**Run complete application:**
```bash
docker-compose up uc2-flask-app
```

**Run in development mode:**
```bash
docker-compose --profile dev up uc2-flask-dev
```

**Windows users:**
```cmd
# Build and run
build-and-run.bat run

# Development mode
build-and-run.bat dev

# Using docker-compose
build-and-run.bat compose
```

**Interactive shell for debugging:**
```bash
docker run -it --rm -p 5000:5000 -v $(pwd)/csv-pkl-json:/app/csv-pkl-json:ro -v $(pwd)/output:/app/output uc2-flask-app bash
```

### 📋 Environment Variables

- `PYTHONPATH=/app`: Python path configuration
- `FLASK_APP=scripts/main-explain-uc2.py`: Flask application entry point
- `FLASK_ENV=production/development`: Flask environment
- `MPLBACKEND=Agg`: Matplotlib backend for headless operation

### 🔍 Features

- **Model Explanation**: SHAP analysis, fairness metrics, demographic breakdowns
- **Driver Analysis**: Heart rate analysis, anomaly detection, clustering
- **User Levels**: Beginner and expert report formats
- **Health Monitoring**: Built-in health check endpoint
- **Production Ready**: Optimized for containerized deployment

### 🛠️ Troubleshooting

**Permission issues:**
```bash
sudo chown -R $USER:$USER output/
```

**View container logs:**
```bash
docker-compose logs -f uc2-flask-app
```

**Check health:**
```bash
curl http://localhost:5000/health
```

**Clean up:**
```bash
docker-compose down
docker system prune -f
```

### 📊 Output

The application generates HTML reports with:
- **Model Performance**: Classification metrics and SHAP analysis
- **Feature Importance**: Which factors most influence predictions
- **Fairness Analysis**: Performance across demographic groups
- **Driver Patterns**: Real-time analysis of heart rate and behavior
- **Anomaly Detection**: Unusual driver behavior patterns
- **Clustering Analysis**: Classification of different driver states
