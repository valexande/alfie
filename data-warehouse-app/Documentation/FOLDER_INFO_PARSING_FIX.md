# Folder Info Parsing Fix

## Problem

Consumers were not correctly detecting folder datasets because they were looking for `is_folder` in the wrong location within the Kafka message.

### The Issue

**Kafka Message Structure:**
```json
{
  "event_type": "bias-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "is_folder": true,           // ✅ Top level - CORRECT location
  "file_count": 3,
  "metadata": {
    "dataset_id": "dataset123",
    "is_folder": false,        // ❌ Nested - WRONG (from auto-extraction)
    // ... other fields
  }
}
```

**Problem Code:**
```python
# WRONG - looked in metadata section
dataset = value.get("metadata", {})
is_folder = dataset.get("is_folder", False)  # ❌ Gets false from metadata
```

**Result**: Consumers always thought datasets were single files, even for folders.

---

## Root Cause

When datasets are uploaded, the DW's automatic metadata extraction creates an `is_folder` field in the file metadata section. This nested `is_folder` in the `metadata` object was being read instead of the correct top-level `is_folder` that indicates the actual dataset type.

---

## Solution

Changed all consumers to read `is_folder` from the **top level** of the message:

**Correct Code:**
```python
# Get is_folder from top level of message
is_folder = value.get("is_folder", False)  # ✅ Correct
file_count = value.get("file_count", 1)
```

---

## Files Fixed

### Bias Detector Consumer
**File**: `kafka_bias_detector_consumer_example.py`

**Before:**
```python
dataset = value.get("metadata", {})
is_folder = dataset.get("is_folder", False)  # ❌ Wrong location
```

**After:**
```python
dataset = value.get("metadata", {})
# Get is_folder from top level of message (not from metadata section)
is_folder = value.get("is_folder", False)  # ✅ Correct location
file_count = value.get("file_count", 1)
```

### AutoML Consumer
**File**: `kafka_automl_consumer_example.py`

**Before:**
```python
metadata = fetch_dataset_metadata(user_id, dataset_id)
is_folder = metadata.get("is_folder", False)  # ❌ From API call
```

**After:**
```python
metadata = fetch_dataset_metadata(user_id, dataset_id)
# Get is_folder from trigger event (not from fetched metadata)
is_folder = event.get("is_folder", False)  # ✅ From trigger event
file_count = event.get("file_count", 1)
```

### XAI Consumer
**File**: `kafka_xai_consumer_example.py`

**Before:**
```python
dataset_meta = fetch_dataset_metadata(user_id, dataset_id)
is_folder = dataset_meta.get("is_folder", False)  # ❌ From API call
```

**After:**
```python
dataset_meta = fetch_dataset_metadata(user_id, dataset_id)
# Get is_folder from trigger event (not from fetched metadata)
is_folder = event.get("is_folder", False)  # ✅ From trigger event
file_count = event.get("file_count", 1)
```

---

## Why This Matters

### Correct Parsing Locations

**For bias-trigger-events:**
```python
# Top level (correct)
is_folder = value.get("is_folder", False)

# NOT from nested metadata (wrong)
# is_folder = value.get("metadata", {}).get("is_folder", False)
```

**For automl-trigger-events:**
```python
# Top level (correct)
is_folder = event.get("is_folder", False)

# NOT from fetched API metadata (wrong)
# is_folder = fetch_dataset_metadata(...).get("is_folder", False)
```

**For xai-trigger-events:**
```python
# Top level (correct)
is_folder = event.get("is_folder", False)

# NOT from fetched API metadata (wrong)
# is_folder = fetch_dataset_metadata(...).get("is_folder", False)
```

---

## Message Structure Clarification

### bias-trigger-events
```json
{
  "event_type": "bias-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "is_folder": true,        // ✅ Read this
  "file_count": 3,          // ✅ Read this
  "target_column_name": "target",
  "task_type": "classification",
  "metadata": {             // Don't read is_folder from here
    "dataset_id": "dataset123",
    "is_folder": false,     // ❌ Ignore this (auto-extracted)
    // ...
  }
}
```

### automl-trigger-events
```json
{
  "event_type": "automl-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "is_folder": true,        // ✅ Read this
  "file_count": 3,          // ✅ Read this
  "target_column_name": "target",
  "task_type": "classification",
  // ...
}
```

### xai-trigger-events
```json
{
  "event_type": "xai-trigger.reported",
  "user_id": "user123",
  "dataset_id": "dataset123",
  "model_id": "model123",
  "is_folder": true,        // ✅ Read this
  "file_count": 3,          // ✅ Read this
  "level": "beginner",
  // ...
}
```

---

## Testing

After restarting consumers, you should now see correct output:

### Before Fix
```
Dataset type: SINGLE FILE  ❌ (Wrong - it was a folder!)
```

### After Fix
```
Dataset type: FOLDER       ✅ (Correct!)
File count: 3
Extracting folder dataset...
Extracted 3 files
```

---

## Files Modified

- ✅ `kafka_bias_detector_consumer_example.py` - Fixed parsing location
- ✅ `kafka_automl_consumer_example.py` - Fixed parsing location
- ✅ `kafka_xai_consumer_example.py` - Fixed parsing location

---

## Summary

**Problem:** Consumers read `is_folder` from wrong location (nested metadata instead of top level)

**Solution:** Changed to read from top level: `value.get("is_folder", False)`

**Result:** All consumers now correctly detect folder datasets! 🎉

Just restart the consumers and test - folder datasets should now be properly recognized and extracted! 🚀

