"""
ICD Code Lookup - Get disease names from diagnosis codes
Supports both ICD-9 and ICD-10 codes
"""
import pandas as pd
import os

# File paths for ICD code CSV files
# Use relative paths for portability (Vercel deployment)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICD9_FILE_PATH = os.path.join(BASE_DIR, "Dataset", "Synthetic Dataset", "ICD9codes.csv")
ICD10_FILE_PATH = os.path.join(BASE_DIR, "Dataset", "Synthetic Dataset", "ICD10codes.csv")

# Global lookup dictionary
icd_lookup = {}
_loaded = False

def load_icd_codes():
    """Load ICD-9 and ICD-10 codes into memory"""
    global icd_lookup, _loaded
    
    if _loaded:
        return  # Already loaded
    
    try:
        # LOAD ICD-10 FIRST (so ICD-9 can override if there are conflicts)
        # Our dataset uses ICD-9 codes, so we want ICD-9 to take priority
        # ICD-10 CSV format: Col0=base, Col1=suffix, Col2=full_code, Col3=long_desc, Col4=short_desc
        if os.path.exists(ICD10_FILE_PATH):
            df10 = pd.read_csv(ICD10_FILE_PATH, header=None, dtype=str)
            for _, row in df10.iterrows():
                # Full code is in column 2 (e.g., "J449")
                full_code = str(row[2]).strip() if len(row) > 2 and pd.notna(row[2]) else str(row[0]).strip()
                # Long description is in column 3
                long_desc = str(row[3]).strip() if len(row) > 3 and pd.notna(row[3]) else 'Unknown'
                # Short description is in column 4 (or fall back to long)
                short_desc = str(row[4]).strip() if len(row) > 4 and pd.notna(row[4]) else long_desc
                
                if full_code and full_code != 'nan' and len(full_code) >= 3:
                    entry = {
                        'short_desc': short_desc,
                        'long_desc': long_desc,
                        'category_desc': short_desc
                    }
                    # Store with full code
                    icd_lookup[full_code] = entry
                    # Also store with dot format (e.g., J44.9 for J449)
                    if len(full_code) > 3 and full_code[0].isalpha():
                        dotted = full_code[:3] + '.' + full_code[3:]
                        icd_lookup[dotted] = entry
            
            print(f"Loaded {len(df10)} ICD-10 codes")
        
        # LOAD ICD-9 SECOND (so it takes priority - our dataset uses ICD-9)
        if os.path.exists(ICD9_FILE_PATH):
            df9 = pd.read_csv(ICD9_FILE_PATH, header=None, dtype=str)
            # Columns: 0=code, 1=?, 2=full_code, 3=short_desc, 4=long_desc, 5=category
            for _, row in df9.iterrows():
                code = str(row[2]).strip() if pd.notna(row[2]) else str(row[0]).strip()
                short_desc = str(row[3]).strip() if pd.notna(row[3]) else 'Unknown'
                long_desc = str(row[4]).strip() if pd.notna(row[4]) else short_desc
                
                if code and code != 'nan':
                    entry = {
                        'short_desc': short_desc,
                        'long_desc': long_desc,
                        'category_desc': short_desc
                    }
                    # ICD-9 codes take priority (overwrite any ICD-10 conflicts)
                    icd_lookup[code] = entry
                    # Also add without leading zeros
                    icd_lookup[code.lstrip('0')] = entry
                    # Add with padding for various formats
                    icd_lookup[code.zfill(4)] = entry
                    icd_lookup[code.zfill(5)] = entry
            
            print(f"Loaded {len(df9)} ICD-9 codes (priority)")
        
        _loaded = True
        print(f"Total lookup entries: {len(icd_lookup)}")
        
    except Exception as e:
        print(f"Error loading ICD codes: {e}")
        import traceback
        traceback.print_exc()

def get_disease_info(code: str) -> dict:
    """
    Get disease information for a diagnosis code
    
    Returns dict with:
        - short_desc: Short description
        - long_desc: Long description
        - category_desc: Category description
    """
    load_icd_codes()
    
    if not code:
        return {
            'short_desc': 'Unknown Diagnosis',
            'long_desc': 'No diagnosis code provided',
            'category_desc': 'Unknown'
        }
    
    code = str(code).strip().upper()
    
    # Try exact match
    if code in icd_lookup:
        return icd_lookup[code]
    
    # Try without dots
    code_no_dot = code.replace('.', '')
    if code_no_dot in icd_lookup:
        return icd_lookup[code_no_dot]
    
    # Try with padding
    for padded in [code.zfill(4), code.zfill(5), code.lstrip('0')]:
        if padded in icd_lookup:
            return icd_lookup[padded]
    
    # Try prefix match (first 3-4 chars)
    for prefix_len in [4, 3]:
        prefix = code_no_dot[:prefix_len]
        if prefix in icd_lookup:
            return icd_lookup[prefix]
    
    # Not found - return code as name
    return {
        'short_desc': f'Diagnosis {code}',
        'long_desc': f'Diagnosis code {code}',
        'category_desc': 'Unknown category'
    }

# Alias for backwards compatibility
def get_disease_name(code: str) -> str:
    """Get disease name for a diagnosis code"""
    return get_disease_info(code)['short_desc']
