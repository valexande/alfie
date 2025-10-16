# XAI Reports API Documentation

## Overview

The XAI Reports system provides endpoints for storing and retrieving Explainable AI (XAI) reports. These reports explain:
- **Model explanations**: How the AI model makes decisions (SHAP values, feature importance, etc.)
- **Data explanations**: Insights about the dataset used for training

Each report comes in two expertise levels:
- **Beginner**: Simplified explanations for non-technical users
- **Expert**: Detailed technical analysis with statistics and metrics

## Architecture

### Storage Structure
```
xai_reports/
  └── {user_id}/
      └── {dataset_id}/
          └── {model_id}/
              ├── model_explanation/
              │   ├── beginner/
              │   │   └── report.html
              │   └── expert/
              │       └── report.html
              └── data_explanation/
                  ├── beginner/
                  │   └── report.html
                  └── expert/
                      └── report.html
```

### Components

1. **Models** (`app/models/xai_report.py`):
   - `ReportType`: Enum for report types (model_explanation, data_explanation)
   - `ExpertiseLevel`: Enum for expertise levels (beginner, expert)
   - `XAIReportMetadata`: Database model
   - `XAIReportCreate`: Creation schema
   - `XAIReportResponse`: Response schema

2. **Service** (`app/services/xai_report_service.py`):
   - Handles file upload to MinIO
   - Stores metadata in MongoDB
   - Manages report retrieval and deletion

3. **API** (`app/api/xai_reports.py`):
   - RESTful endpoints for CRUD operations

## API Endpoints

### 1. Upload XAI Report

**Endpoint**: `POST /xai-reports/upload`

**Description**: Upload an HTML file containing XAI explanations

**Parameters**:
- `file` (file): HTML file to upload
- `user_id` (string): User ID
- `dataset_id` (string): Dataset ID
- `model_id` (string): AI Model ID
- `report_type` (enum): Either "model_explanation" or "data_explanation"
- `level` (enum): Either "beginner" or "expert"

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/xai-reports/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@model-beginner.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=drowsiness_model_v1" \
  -F "report_type=model_explanation" \
  -F "level=beginner"
```

**Response**:
```json
{
  "user_id": "user1",
  "dataset_id": "TUC3",
  "model_id": "drowsiness_model_v1",
  "report_type": "model_explanation",
  "level": "beginner",
  "file_path": "xai_reports/user1/TUC3/drowsiness_model_v1/model_explanation/beginner/report.html",
  "file_size": 45678,
  "file_hash": "a1b2c3d4e5f6...",
  "custom_metadata": {},
  "created_at": "2025-10-10T10:30:00",
  "updated_at": "2025-10-10T10:30:00"
}
```

### 2. Get All XAI Reports

**Endpoint**: `GET /xai-reports/{user_id}/{dataset_id}/{model_id}`

**Description**: Retrieve all XAI reports for a specific model and dataset

**Example**:
```bash
curl -X GET "http://localhost:8000/xai-reports/user1/TUC3/drowsiness_model_v1" \
  -H "accept: application/json"
```

**Response**: Array of XAI report metadata

### 3. Get Specific XAI Report

**Endpoint**: `GET /xai-reports/{user_id}/{dataset_id}/{model_id}/{report_type}/{level}`

**Description**: Get metadata for a specific XAI report

**Example**:
```bash
curl -X GET "http://localhost:8000/xai-reports/user1/TUC3/drowsiness_model_v1/model_explanation/beginner" \
  -H "accept: application/json"
```

### 4. View XAI Report (HTML)

**Endpoint**: `GET /xai-reports/{user_id}/{dataset_id}/{model_id}/{report_type}/{level}/view`

**Description**: View the XAI report HTML directly in browser

**Example**:
```bash
# Open in browser:
http://localhost:8000/xai-reports/user1/TUC3/drowsiness_model_v1/model_explanation/beginner/view

# Or download with curl:
curl -X GET "http://localhost:8000/xai-reports/user1/TUC3/drowsiness_model_v1/model_explanation/beginner/view" \
  --output report.html
```

### 5. Delete XAI Report

**Endpoint**: `DELETE /xai-reports/{user_id}/{dataset_id}/{model_id}/{report_type}/{level}`

**Description**: Delete a specific XAI report and its file

**Example**:
```bash
curl -X DELETE "http://localhost:8000/xai-reports/user1/TUC3/drowsiness_model_v1/model_explanation/beginner" \
  -H "accept: application/json"
```

**Response**:
```json
{
  "message": "XAI report deleted successfully",
  "user_id": "user1",
  "dataset_id": "TUC3",
  "model_id": "drowsiness_model_v1",
  "report_type": "model_explanation",
  "level": "beginner"
}
```

## Usage Examples

### Complete Workflow

1. **Upload model explanation reports**:
```bash
# Beginner level
curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@model-beginner.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=drowsiness_model_v1" \
  -F "report_type=model_explanation" \
  -F "level=beginner"

# Expert level
curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@model-expert.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=drowsiness_model_v1" \
  -F "report_type=model_explanation" \
  -F "level=expert"
```

2. **Upload data explanation reports**:
```bash
# Beginner level
curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@data-beginner.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=drowsiness_model_v1" \
  -F "report_type=data_explanation" \
  -F "level=beginner"

# Expert level
curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@data-expert.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=drowsiness_model_v1" \
  -F "report_type=data_explanation" \
  -F "level=expert"
```

3. **List all reports**:
```bash
curl -X GET "http://localhost:8000/xai-reports/user1/TUC3/drowsiness_model_v1"
```

4. **View a specific report in browser**:
```
http://localhost:8000/xai-reports/user1/TUC3/drowsiness_model_v1/data_explanation/beginner/view
```

## Report Types Explained

### Model Explanation Reports

These reports explain how your AI model makes decisions:

**Beginner Level**:
- Simple language explanations
- Visual charts (SHAP graphs, feature importance)
- "What does this mean?" sections
- Real-world analogies

**Expert Level**:
- Detailed SHAP (SHapley Additive exPlanations) analysis
- Feature importance scores
- Decision trees visualization
- Statistical confidence intervals
- Model architecture details

### Data Explanation Reports

These reports provide insights about your dataset:

**Beginner Level**:
- Summary statistics in plain language
- Distribution charts
- Pattern identification
- Data quality indicators

**Expert Level**:
- Correlation matrices
- Anomaly detection analysis
- Statistical tests results
- Clustering analysis
- Outlier detection details

## Database Schema

### MongoDB Collection: `xai_reports`

```json
{
  "_id": "ObjectId(...)",
  "user_id": "user1",
  "dataset_id": "TUC3",
  "model_id": "drowsiness_model_v1",
  "report_type": "model_explanation",
  "level": "beginner",
  "file_path": "xai_reports/user1/TUC3/drowsiness_model_v1/model_explanation/beginner/report.html",
  "file_size": 45678,
  "file_hash": "a1b2c3d4e5f6...",
  "custom_metadata": {},
  "created_at": "2025-10-10T10:30:00Z",
  "updated_at": "2025-10-10T10:30:00Z"
}
```

### Indexes

The system should create the following indexes for optimal performance:

```javascript
db.xai_reports.createIndex({ 
  "user_id": 1, 
  "dataset_id": 1, 
  "model_id": 1, 
  "report_type": 1, 
  "level": 1 
}, { unique: true })
```

## Integration with Existing Systems

### With Dataset Management
```bash
# Get dataset
GET /datasets/user1/TUC3

# Upload XAI report for this dataset
POST /xai-reports/upload
```

### With AI Models
```bash
# Get AI model
GET /ai-models/user1/drowsiness_model_v1

# Upload XAI report explaining this model
POST /xai-reports/upload
```

## Future Enhancements

Potential future features:
1. **JSON Storage**: Parse HTML and store structured data in MongoDB
2. **Comparison Tool**: Compare reports across model versions
3. **Export Options**: PDF, JSON, or other formats
4. **Interactive Reports**: Real-time chart interactions
5. **Report Templates**: Standardized report generation
6. **Versioning**: Track report versions over time

## Troubleshooting

### Common Issues

**Issue**: "Failed to upload XAI report"
- **Solution**: Ensure file is valid HTML and MinIO is accessible

**Issue**: "XAI report not found"
- **Solution**: Check that user_id, dataset_id, and model_id match exactly

**Issue**: "Only HTML files are supported"
- **Solution**: Ensure your file has `.html` extension

## Testing

To test the XAI reports system with the provided example files:

```bash
# Test with the example files in the root directory
cd /path/to/data-warehouse-app

# Upload all 4 example reports
curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@model-beginner.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=test_model" \
  -F "report_type=model_explanation" \
  -F "level=beginner"

curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@model-expert.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=test_model" \
  -F "report_type=model_explanation" \
  -F "level=expert"

curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@data-beginner.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=test_model" \
  -F "report_type=data_explanation" \
  -F "level=beginner"

curl -X POST "http://localhost:8000/xai-reports/upload" \
  -F "file=@data-expert.html" \
  -F "user_id=user1" \
  -F "dataset_id=TUC3" \
  -F "model_id=test_model" \
  -F "report_type=data_explanation" \
  -F "level=expert"

# List all reports
curl -X GET "http://localhost:8000/xai-reports/user1/TUC3/test_model"

# View beginner data explanation in browser
# Open: http://localhost:8000/xai-reports/user1/TUC3/test_model/data_explanation/beginner/view
```

## Support

For issues or questions, please refer to the main API documentation at `/docs` when the server is running.

