// MongoDB initialization script
db = db.getSiblingDB('data_warehouse');

// Create collections
db.createCollection('datasets');
db.createCollection('users');

// Create indexes for better performance
// Allow multiple versions per (user_id, dataset_id)
db.datasets.createIndex({ "user_id": 1 });
db.datasets.createIndex({ "user_id": 1, "dataset_id": 1 });
db.datasets.createIndex({ "user_id": 1, "dataset_id": 1, "version": 1 }, { name: "user_dataset_version_unique", unique: true });
db.datasets.createIndex({ "created_at": 1 });
db.datasets.createIndex({ "tags": 1 });
db.datasets.createIndex({ "file_type": 1 });

db.users.createIndex({ "user_id": 1 }, { unique: true });
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });

print('Database initialized successfully');
