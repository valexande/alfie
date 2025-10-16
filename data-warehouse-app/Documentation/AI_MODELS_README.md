# AI Model Management API

This document describes the AI Model Management endpoints that allow you to store, manage, and retrieve AI/ML models in the data warehouse.

## Features

- **Multi-format Model Support**: PKL, ONNX, H5, PyTorch (.pth), TensorFlow (.pb), HuggingFace models, and more
- **Folder Upload**: Upload entire model directories with preserved structure
- **Model Metadata**: Comprehensive metadata storage including training metrics, framework info, and custom fields
- **Version Control**: Simple versioning system (v1, v2, etc.)
- **User Separation**: Organized model storage with user-based directories
- **Search & Filtering**: Advanced search capabilities by framework, type, tags, etc.
- **Download Options**: Download individual files or entire models as ZIP archives

## File Organization Structure

```
models/
├── user1/
│   ├── classification_model/
│   │   ├── v1/
│   │   │   ├── model.pkl
│   │   │   ├── weights.h5
│   │   │   ├── config.json
│   │   │   └── requirements.txt
│   │   └── v2/
│   │       ├── model.onnx
│   │       └── config.json
│   └── nlp_model/
│       └── v1/
│           ├── model.bin
│           ├── config.json
│           └── tokenizer.json
└── user2/
    └── cv_model/
        └── v1/
            ├── model.pth
            ├── weights/
            │   ├── layer1.pth
            │   └── layer2.pth
            └── config.yaml
```

## API Endpoints

### Upload Single Model File

```http
POST /ai-models/upload/single
Content-Type: multipart/form-data

Form fields:
- file: The model file to upload
- user_id: User identifier
- model_id: Model identifier
- name: Model name
- description: Optional description
- version: Version (default: v1)
- framework: ML framework (pytorch, tensorflow, sklearn, onnx, keras, huggingface, other)
- model_type: Type of model (classification, regression, clustering, nlp, computer_vision, recommendation, time_series, reinforcement_learning, other)
- algorithm: Optional specific algorithm name
- is_primary: Whether this is the primary model file (default: false)
- tags: Comma-separated tags
- training_accuracy: Optional training accuracy
- validation_accuracy: Optional validation accuracy
- test_accuracy: Optional test accuracy
- training_loss: Optional training loss
- python_version: Optional required Python version
- hardware_requirements: Optional hardware requirements
```

### Upload Model Folder

```http
POST /ai-models/upload/folder
Content-Type: multipart/form-data

Form fields:
- zip_file: ZIP file containing model files
- user_id: User identifier
- model_id: Model identifier
- name: Model name
- description: Optional description
- version: Version (default: v1)
- framework: ML framework
- model_type: Type of model
- algorithm: Optional specific algorithm name
- preserve_structure: Whether to preserve folder structure (default: true)
- tags: Comma-separated tags
- training_accuracy: Optional training accuracy
- validation_accuracy: Optional validation accuracy
- test_accuracy: Optional test accuracy
- training_loss: Optional training loss
- python_version: Optional required Python version
- hardware_requirements: Optional hardware requirements
```

### Get Model Metadata

```http
GET /ai-models/{user_id}/{model_id}?version=v1
```

### Get User Models

```http
GET /ai-models/{user_id}?skip=0&limit=100&framework=pytorch&model_type=classification&tags=production
```

### Download Model File(s)

```http
# Download specific file
GET /ai-models/{user_id}/{model_id}/download?filename=model.pkl

# Download all files as ZIP
GET /ai-models/{user_id}/{model_id}/download
```

### Update Model Metadata

```http
PUT /ai-models/{user_id}/{model_id}?version=v1
Content-Type: application/json

{
  "name": "Updated Model Name",
  "description": "Updated description",
  "tags": ["updated", "production"],
  "is_production_ready": true,
  "test_accuracy": 0.95
}
```

### Delete Model

```http
DELETE /ai-models/{user_id}/{model_id}?version=v1
```

### List Model Files

```http
GET /ai-models/{user_id}/{model_id}/files?version=v1
```

### Search Models

```http
GET /ai-models/search/{user_id}?query=classification&framework=sklearn&model_type=classification&tags=production
```

## Supported Model Formats

### File Extensions
- **PKL/JOBLIB**: `.pkl`, `.joblib` (Scikit-learn, custom models)
- **ONNX**: `.onnx` (Universal model format)
- **Keras/TensorFlow**: `.h5`, `.hdf5` (Keras models)
- **TensorFlow**: `.pb` (SavedModel format)
- **PyTorch**: `.pth`, `.pt` (PyTorch models)
- **HuggingFace**: `.bin`, `.safetensors` (Transformer models)
- **Configuration**: `.json`, `.yaml`, `.yml` (Model configs)
- **Requirements**: `.txt` (Dependencies)
- **Documentation**: `.md`, `.txt` (Model docs)

### Framework Detection
The system automatically detects the model framework based on file extensions:
- `.pkl`, `.joblib` → SKLEARN
- `.onnx` → ONNX
- `.h5`, `.hdf5` → KERAS
- `.pb` → TENSORFLOW
- `.pth`, `.pt` → PYTORCH
- `.bin`, `.safetensors` → HUGGINGFACE

## Model Metadata Schema

```json
{
  "model_id": "string",
  "user_id": "string",
  "name": "string",
  "description": "string",
  "version": "string",
  "framework": "pytorch|tensorflow|sklearn|onnx|keras|huggingface|other",
  "model_type": "classification|regression|clustering|nlp|computer_vision|recommendation|time_series|reinforcement_learning|other",
  "algorithm": "string",
  "files": [
    {
      "filename": "string",
      "file_path": "string",
      "file_size": "integer",
      "file_type": "string",
      "file_hash": "string",
      "content_type": "string",
      "is_primary": "boolean",
      "description": "string"
    }
  ],
  "primary_file_path": "string",
  "input_shape": ["integer"],
  "output_shape": ["integer"],
  "num_parameters": "integer",
  "model_size_mb": "float",
  "training_dataset": "string",
  "training_accuracy": "float",
  "validation_accuracy": "float",
  "test_accuracy": "float",
  "training_loss": "float",
  "python_version": "string",
  "dependencies": ["string"],
  "hardware_requirements": "string",
  "tags": ["string"],
  "custom_metadata": {},
  "created_at": "datetime",
  "updated_at": "datetime",
  "is_active": "boolean",
  "is_production_ready": "boolean"
}
```

## Usage Examples

### 1. Upload a Scikit-learn Model

```bash
curl -X POST "http://localhost:8000/ai-models/upload/single" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@model.pkl" \
  -F "user_id=user1" \
  -F "model_id=iris_classifier" \
  -F "name=Iris Classification Model" \
  -F "description=Random Forest classifier for iris dataset" \
  -F "framework=sklearn" \
  -F "model_type=classification" \
  -F "algorithm=RandomForest" \
  -F "is_primary=true" \
  -F "tags=sklearn,classification,iris" \
  -F "training_accuracy=0.98" \
  -F "test_accuracy=0.96"
```

### 2. Upload a PyTorch Model Folder

```bash
# First, create a ZIP file with your model files
zip -r pytorch_model.zip model.pth config.json requirements.txt

curl -X POST "http://localhost:8000/ai-models/upload/folder" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "zip_file=@pytorch_model.zip" \
  -F "user_id=user1" \
  -F "model_id=resnet_classifier" \
  -F "name=ResNet Image Classifier" \
  -F "description=ResNet model for image classification" \
  -F "framework=pytorch" \
  -F "model_type=computer_vision" \
  -F "algorithm=ResNet" \
  -F "preserve_structure=true" \
  -F "tags=pytorch,resnet,computer_vision" \
  -F "training_accuracy=0.95" \
  -F "python_version=3.9"
```

### 3. Download Model Files

```bash
# Download entire model as ZIP
curl -X GET "http://localhost:8000/ai-models/user1/iris_classifier/download" \
  -H "accept: application/octet-stream" \
  --output iris_classifier.zip

# Download specific file
curl -X GET "http://localhost:8000/ai-models/user1/iris_classifier/download?filename=model.pkl" \
  -H "accept: application/octet-stream" \
  --output model.pkl
```

### 4. Search Models

```bash
curl -X GET "http://localhost:8000/ai-models/search/user1?query=classification&framework=sklearn&tags=production" \
  -H "accept: application/json"
```

### 5. Update Model Metadata

```bash
curl -X PUT "http://localhost:8000/ai-models/user1/iris_classifier" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Iris Classifier",
    "description": "Updated model with better performance",
    "tags": ["sklearn", "classification", "iris", "production"],
    "is_production_ready": true,
    "test_accuracy": 0.97
  }'
```

## Python Client Example

```python
import requests

# Upload a model file
def upload_model(file_path, user_id, model_id, name, framework, model_type):
    url = "http://localhost:8000/ai-models/upload/single"
    
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'application/octet-stream')}
        data = {
            'user_id': user_id,
            'model_id': model_id,
            'name': name,
            'framework': framework,
            'model_type': model_type,
            'is_primary': True
        }
        
        response = requests.post(url, files=files, data=data)
        return response.json()

# Download a model
def download_model(user_id, model_id, output_path):
    url = f"http://localhost:8000/ai-models/{user_id}/{model_id}/download"
    
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    return output_path

# Usage
model_info = upload_model(
    "model.pkl", 
    "user1", 
    "my_model", 
    "My Classification Model",
    "sklearn",
    "classification"
)

download_model("user1", "my_model", "downloaded_model.zip")
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (model/file doesn't exist)
- `409`: Conflict (model already exists)
- `500`: Internal Server Error

Error responses include a `detail` field with the error message:

```json
{
  "detail": "Model with this ID and version already exists"
}
```

## Best Practices

1. **File Naming**: Use descriptive filenames that indicate the file's purpose
2. **Versioning**: Use semantic versioning (v1.0, v1.1, v2.0) for model versions
3. **Metadata**: Provide comprehensive metadata including training metrics
4. **Tags**: Use consistent tagging for easy searching and filtering
5. **Folder Structure**: When uploading folders, organize files logically
6. **Documentation**: Include README files and configuration files in your model packages
7. **Dependencies**: Always include requirements.txt or equivalent dependency files
8. **Testing**: Test model downloads to ensure all files are properly uploaded

## Integration with Existing Data Warehouse

The AI model endpoints integrate seamlessly with the existing data warehouse:

- **User Management**: Uses the same user system as datasets
- **Storage**: Uses the same MinIO instance with organized bucket structure
- **Database**: Uses the same MongoDB instance with dedicated collections
- **Authentication**: Inherits the same authentication mechanisms
- **API Structure**: Follows the same RESTful patterns as other endpoints

The models are stored separately from datasets but follow the same organizational principles and access patterns.
