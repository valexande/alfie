# Data Warehouse API

A comprehensive Python-based Data Warehouse application for storing and managing both structured (CSV, XLS, tabular) and unstructured data (images, videos, audio files, AI models) with automatic metadata extraction and user-based organization.

## Features

- **Multi-format File Support**: CSV, Excel, JSON, images, videos, audio files, and AI model weights
- **User Separation**: Organized file storage with user-based directories
- **Versioning**: Simple versioning system (v1, v2, etc.)
- **Automatic Metadata Extraction**: Extracts relevant metadata from uploaded files
- **MongoDB Integration**: Stores metadata with efficient indexing
- **MinIO Integration**: Scalable file storage with S3-compatible API
- **RESTful API**: FastAPI-based REST endpoints
- **Docker Support**: Easy deployment with Docker Compose

## Architecture

```
├── FastAPI Application (Python)
├── MongoDB (Metadata Storage)
└── MinIO (File Storage)
```

### File Organization Structure
```
datasets/
├── user1/
│   ├── drowsiness/
│   │   ├── v1/
│   │   │   └── heart_rate.csv
│   │   └── v2/
│   │       └── heart_rate_updated.csv
│   └── another_dataset/
│       └── v1/
│           └── data.xlsx
└── user2/
    └── project_x/
        └── v1/
            └── model_weights.pkl
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for infrastructure)
- Git

### 1. Clone and Setup

```bash
git clone <repository-url>
cd data-warehouse-app
```

### 2. Environment Configuration

Copy and modify the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=data_warehouse

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=False
MINIO_BUCKET_NAME=data-warehouse

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-change-this-in-production
```

### 3. Start Infrastructure Services

```bash
# Start MongoDB and MinIO
docker-compose up -d mongodb minio
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the API

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m app.main
```

### 6. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

## API Endpoints

### Dataset Management

#### Upload Dataset
```http
POST /datasets/upload
Content-Type: multipart/form-data

Form fields:
- file: The file to upload
- user_id: User identifier
- dataset_id: Dataset identifier
- name: Dataset name
- description: Optional description
- version: Version (default: v1)
- tags: Comma-separated tags
```

#### Get Dataset
```http
GET /datasets/{user_id}/{dataset_id}
```

#### Get User Datasets
```http
GET /datasets/{user_id}?skip=0&limit=100
```

#### Update Dataset Metadata
```http
PUT /datasets/{user_id}/{dataset_id}
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "Updated description",
  "tags": ["tag1", "tag2"]
}
```

#### Delete Dataset
```http
DELETE /datasets/{user_id}/{dataset_id}
```

#### Download Dataset
```http
GET /datasets/{user_id}/{dataset_id}/download
```

#### Search Datasets
```http
GET /datasets/search/{user_id}?query=heart&tags=medical&file_type=csv
```

### User Management

#### Create User
```http
POST /users/
Content-Type: application/json

{
  "user_id": "user1",
  "username": "john_doe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "password": "secure_password"
}
```

#### Get User
```http
GET /users/{user_id}
GET /users/username/{username}
```

## Usage Examples

### 1. Upload a CSV File

```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@heart_rate.csv" \
  -F "user_id=user1" \
  -F "dataset_id=drowsiness" \
  -F "name=Heart Rate Data" \
  -F "description=Heart rate measurements for drowsiness detection" \
  -F "version=v1" \
  -F "tags=medical,heart-rate,drowsiness"
```

### 2. Get Dataset Information

```bash
curl -X GET "http://localhost:8000/datasets/user1/drowsiness" \
  -H "accept: application/json"
```

### 3. Search Datasets

```bash
curl -X GET "http://localhost:8000/datasets/search/user1?query=heart&tags=medical" \
  -H "accept: application/json"
```

## Development

### Project Structure

```
data-warehouse-app/
├── app/
│   ├── api/
│   │   ├── datasets.py      # Dataset endpoints
│   │   └── users.py         # User endpoints
│   ├── core/
│   │   ├── config.py        # Configuration
│   │   ├── database.py      # MongoDB connection
│   │   └── minio_client.py  # MinIO connection
│   ├── models/
│   │   ├── dataset.py       # Dataset models
│   │   └── user.py          # User models
│   ├── services/
│   │   ├── file_service.py      # File operations
│   │   └── metadata_service.py  # Metadata operations
│   └── main.py              # FastAPI application
├── docker-compose.yml       # Infrastructure services
├── Dockerfile              # API container
├── requirements.txt        # Python dependencies
└── README.md
```

### Adding New File Type Support

To add support for a new file type, modify the `_extract_file_metadata` method in `app/services/file_service.py`:

```python
async def _extract_file_metadata(self, file_data: bytes, filename: str, content_type: Optional[str]) -> Dict[str, Any]:
    metadata = {}
    file_extension = self._get_file_extension(filename)
    
    if file_extension in ['your_new_extension']:
        # Add your metadata extraction logic here
        pass
    
    return metadata
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## Deployment

### Production Deployment with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale API instances
docker-compose up -d --scale api=3
```

### Environment Variables for Production

```env
SECRET_KEY=your-very-secure-secret-key
MONGODB_URL=mongodb://username:password@mongodb-host:27017/data_warehouse
MINIO_ENDPOINT=minio-host:9000
MINIO_ACCESS_KEY=your-minio-access-key
MINIO_SECRET_KEY=your-minio-secret-key
```

## Monitoring and Maintenance

### Health Check

```bash
curl http://localhost:8000/health
```

### Database Indexes

The application automatically creates necessary indexes for optimal performance:

- Dataset collection: `dataset_id`, `user_id`, `created_at`, `tags`, `file_type`
- Users collection: `user_id`, `username`, `email`

### Backup Strategies

1. **MongoDB**: Use `mongodump` for regular backups
2. **MinIO**: Configure bucket replication or use `mc mirror`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for debugging information
