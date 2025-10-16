# Bug Fix: Bias Events Not Being Published

## Problem

The Kafka orchestration flow was stopping at the bias-trigger stage. Symptoms:
- ✅ `bias-trigger-events` message visible in Kafka UI
- ✅ Bias report successfully saved to Data Warehouse
- ❌ `bias-events` message NOT appearing in Kafka UI
- ❌ Flow stopped - no automl-trigger-events produced

## Root Cause

The `BiasReportCreate` model was missing the `target_column_name` and `task_type` fields. 

**What was happening:**
1. Bias detector consumer received `bias-trigger-events` with target_column_name and task_type ✓
2. Bias detector processed and saved report to DW ✓
3. DW tried to send `bias-events` with `payload.target_column_name` and `payload.task_type`
4. **BUT** these fields didn't exist in the `BiasReportCreate` model!
5. The Kafka send failed silently (caught by `except Exception: pass`)
6. No `bias-events` message was published ✗
7. Agentic Core never received the message to trigger AutoML ✗

## Solution

### 1. Added Missing Fields to Model
**File**: `app/models/bias_report.py`

```python
class BiasReportCreate(BaseModel):
    user_id: str
    dataset_id: str
    report: Any
    target_column_name: Optional[str] = None  # ✅ Added
    task_type: Optional[str] = None            # ✅ Added
```

### 2. Updated Bias Detector to Pass These Fields
**File**: `kafka_bias_detector_consumer_example.py`

```python
def post_bias_report(user_id: str, dataset_id: str, report: dict, 
                     target_column_name: str = None, task_type: str = None) -> dict:
    payload = {
        "user_id": user_id,
        "dataset_id": dataset_id,
        "report": report,
        "target_column_name": target_column_name,  # ✅ Now included
        "task_type": task_type,                    # ✅ Now included
    }
    # ...
```

And when calling it:
```python
saved = post_bias_report(
    user_id=user_id, 
    dataset_id=dataset_id, 
    report=bias_report,
    target_column_name=target_column_name,  # ✅ Pass from trigger event
    task_type=task_type                      # ✅ Pass from trigger event
)
```

### 3. Improved Error Logging
**File**: `app/api/bias_reports.py`

Changed from silent failure:
```python
except Exception:
    pass  # ❌ Silent failure - no way to debug
```

To logged errors:
```python
except Exception as e:
    logger.error(f"Failed to send bias Kafka event: {e}", exc_info=True)  # ✅ Now logged
```

## Testing

### Before Fix
```
Dataset Upload → dataset-events ✓
     ↓
Agentic Core → bias-trigger-events ✓
     ↓
Bias Detector → saves report ✓
     ↓
DW tries to send bias-events ✗ (FAILED SILENTLY)
     ↓
❌ FLOW STOPPED HERE
```

### After Fix
```
Dataset Upload → dataset-events ✓
     ↓
Agentic Core → bias-trigger-events ✓
     ↓
Bias Detector → saves report (with target_column_name, task_type) ✓
     ↓
DW sends bias-events ✓
     ↓
Agentic Core → automl-trigger-events ✓
     ↓
AutoML Consumer → trains model ✓
     ↓
... (continues to XAI)
```

## How to Verify the Fix

1. **Restart the Data Warehouse API** (to reload the model changes)
   ```bash
   # Stop the API (Ctrl+C)
   # Start again
   python run.py
   ```

2. **Check the logs** - You should now see:
   ```
   INFO: Bias Kafka event sent for dataset_id=test2
   ```
   
   If there's still an error, you'll see:
   ```
   ERROR: Failed to send bias Kafka event: <error details>
   ```

3. **Check Kafka UI** - You should now see messages in the `bias-events` topic

4. **Verify flow continues** - The Agentic Core should produce `automl-trigger-events`

## Files Modified

- ✅ `app/models/bias_report.py` - Added target_column_name and task_type fields
- ✅ `app/api/bias_reports.py` - Improved error logging
- ✅ `kafka_bias_detector_consumer_example.py` - Pass target_column_name and task_type when posting report

## Key Takeaway

**Always include all required context fields in your data models!**

When events need to carry information through multiple stages of a pipeline, ensure that:
1. The data model includes all necessary fields
2. Consumers pass those fields when posting to the API
3. Error handling logs issues instead of silently failing
4. Test the complete end-to-end flow

## Next Steps

After restarting the API, try uploading a dataset again and verify that:
1. bias-trigger-events appears in Kafka UI ✓
2. Bias report is saved to DW ✓
3. **bias-events appears in Kafka UI** ✓ (This was the bug)
4. automl-trigger-events appears in Kafka UI ✓
5. Flow continues through AutoML and XAI ✓

The flow should now work end-to-end! 🎉

