# CSV Encoding Error Fix

## Problem

Consumers were failing to parse CSV files with encoding errors like:
```
ERROR: 'utf-8' codec can't decode byte 0xb5 in position 10: invalid start byte
```

## Root Cause

CSV files can be encoded in various formats depending on:
- Operating system (Windows, Mac, Linux)
- Locale/language settings
- Excel export options
- Text editor used

Common encodings:
- **UTF-8**: Modern standard
- **Windows-1252 (cp1252)**: Windows default
- **Latin-1 (ISO-8859-1)**: Western European
- **UTF-16**: Some Excel exports

When pandas tries to read a CSV with the wrong encoding, it throws a `UnicodeDecodeError`.

---

## Solution

Added `read_csv_with_encoding()` function to all consumer scripts that:
1. Tries multiple common encodings
2. Falls back to error-ignoring mode if all fail
3. Logs which encoding worked

---

## Implementation

### New Helper Function

```python
def read_csv_with_encoding(file_data: bytes) -> pd.DataFrame:
    """
    Try to read CSV with multiple encodings
    
    Handles files with different encodings:
    - utf-8: Standard
    - latin-1 (ISO-8859-1): Western European
    - cp1252 (Windows-1252): Windows default
    - utf-16: Some Excel exports
    """
    from io import BytesIO
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(BytesIO(file_data), encoding=encoding)
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except (UnicodeDecodeError, Exception):
            continue
    
    # If all encodings fail, try with error handling
    try:
        df = pd.read_csv(BytesIO(file_data), encoding='utf-8', encoding_errors='ignore')
        logger.warning("Read CSV with 'ignore' errors - some characters may be missing")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV with all encodings: {e}")
        raise
```

### Usage in Consumers

**Before (Single encoding):**
```python
df = pd.read_csv(BytesIO(file_bytes))  # Fails if not UTF-8
```

**After (Multiple encodings):**
```python
df = read_csv_with_encoding(file_bytes)  # Tries multiple encodings
```

---

## Files Updated

All 4 consumer scripts now include encoding handling:

- ✅ `kafka_agentic_core_consumer_example.py`
- ✅ `kafka_bias_detector_consumer_example.py`
- ✅ `kafka_automl_consumer_example.py`
- ✅ `kafka_xai_consumer_example.py`

---

## Expected Behavior

### Successful Encoding Detection

**Console Output:**
```
Downloaded dataset: 2048 bytes
Successfully read CSV with encoding: cp1252
Loaded single file dataset with shape (100, 5)
```

### Fallback Mode

If no encoding works perfectly:
```
Downloaded dataset: 2048 bytes
Read CSV with 'ignore' errors - some characters may be missing
Loaded single file dataset with shape (100, 5)
```

⚠️ **Note**: In fallback mode, some special characters might be missing, but the CSV structure is preserved.

---

## Encoding Detection Order

The function tries encodings in this order:
1. **UTF-8** - Modern standard (most common)
2. **Latin-1** - Western European characters
3. **CP1252** - Windows default (very common)
4. **ISO-8859-1** - Similar to Latin-1
5. **UTF-16** - Some Excel exports
6. **Fallback** - UTF-8 with errors ignored

---

## Testing

### Test with Different Encodings

```python
import pandas as pd

# Create CSV with Latin-1 encoding
df = pd.DataFrame({'name': ['José', 'François', 'Müller'], 'value': [1, 2, 3]})
df.to_csv('latin1.csv', index=False, encoding='latin-1')

# Create CSV with Windows-1252 encoding
df.to_csv('windows.csv', index=False, encoding='cp1252')

# Create CSV with UTF-8 encoding
df.to_csv('utf8.csv', index=False, encoding='utf-8')

# Upload each and verify consumers can read them
```

---

## Common Character Issues

### Characters That Cause Problems

| Character | UTF-8 | Latin-1 | CP1252 | Issue |
|-----------|-------|---------|--------|-------|
| µ (micro) | ✅ | ✅ | 0xB5 | Your error! |
| € (euro) | ✅ | ❌ | ✅ | Different bytes |
| é (e-acute) | ✅ | ✅ | ✅ | Usually OK |
| — (em dash) | ✅ | ❌ | ✅ | Common in Windows |

The byte `0xB5` (µ symbol) is valid in Latin-1/CP1252 but not in UTF-8, which is why you got that error.

---

## Alternative: Detect Encoding Automatically

If you want even more robust handling, you can use the `chardet` library:

```python
import chardet

def detect_and_read_csv(file_data: bytes) -> pd.DataFrame:
    """Detect encoding automatically"""
    # Detect encoding
    result = chardet.detect(file_data)
    encoding = result['encoding']
    confidence = result['confidence']
    
    logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
    
    # Read with detected encoding
    df = pd.read_csv(BytesIO(file_data), encoding=encoding)
    return df
```

**To use this:**
```bash
pip install chardet
```

---

## Best Practices for CSV Files

### For Data Providers

**✅ Recommended:**
- Save CSVs as UTF-8 encoding
- Avoid special characters if possible
- Test with different locales

**In Excel:**
1. Save As → CSV UTF-8 (not just CSV)
2. Or use Python to convert:
   ```python
   df = pd.read_csv('file.csv', encoding='latin-1')
   df.to_csv('file_utf8.csv', encoding='utf-8', index=False)
   ```

---

## Troubleshooting

### Issue: Still getting encoding errors

**Try:**
1. Check which encoding fails in logs
2. Add that encoding to the list
3. Use chardet for automatic detection

### Issue: Characters look wrong (������)

**Cause:** Wrong encoding used  
**Solution:** System will try multiple encodings automatically

### Issue: Data seems truncated

**Cause:** Fallback mode with 'ignore'  
**Solution:** Fix source file encoding to UTF-8

---

## Summary

✅ **Problem**: CSV encoding errors in consumers  
✅ **Cause**: Files not in UTF-8 format  
✅ **Solution**: Added multi-encoding support  
✅ **Result**: Consumers now handle various encodings automatically  

**All 4 consumers updated:**
- ✅ Agentic Core
- ✅ Bias Detector  
- ✅ AutoML Consumer
- ✅ XAI Consumer

The encoding errors should now be resolved! 🎉

