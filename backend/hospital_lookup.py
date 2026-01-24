"""
Hospital Lookup Module for Healthcare Fraud Detection System

Provides functions to search and filter hospitals from the 
indian_hospitals_classified.csv dataset (30,273 hospitals).
"""

import pandas as pd
import os
from typing import List, Dict, Optional

# Get absolute path to the CSV file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSPITAL_CSV = os.path.join(BASE_DIR, "Dataset", "Synthetic Dataset", "indian_hospitals_classified.csv")

# Load hospitals into memory lazily
hospitals_df = pd.DataFrame()

def load_hospitals_data():
    """Load hospital data if not already loaded."""
    global hospitals_df
    if hospitals_df.empty:
        print(f"📊 Loading hospitals from: {HOSPITAL_CSV}")
        try:
            hospitals_df = pd.read_csv(HOSPITAL_CSV, low_memory=False)
            print(f"✅ Loaded {len(hospitals_df):,} hospitals")
        except Exception as e:
            print(f"❌ Error loading hospitals: {e}")
            hospitals_df = pd.DataFrame()

# Call this function at start of each public function


# Available hospital types
HOSPITAL_TYPES = ["Government", "Clinic", "Private"]


def get_hospital_types() -> List[str]:
    """Return list of available hospital types."""
    return HOSPITAL_TYPES


def get_hospitals_by_type(hospital_type: str, limit: int = 100) -> List[Dict]:
    """
    Get hospitals filtered by type.
    
    Parameters:
    -----------
    hospital_type : str
        One of: "Government", "Clinic", "Private"
    limit : int
        Maximum number of results (default 100)
    
    Returns:
    --------
    List of hospital dictionaries
    """
    if hospitals_df.empty:
        load_hospitals_data()
    if hospitals_df.empty:
        return []
    
    filtered = hospitals_df[hospitals_df['Hospital_Type'] == hospital_type]
    
    results = []
    for _, row in filtered.head(limit).iterrows():
        results.append({
            "name": row['Hospital_Name'],
            "type": row['Hospital_Type'],
            "state": row['State'] if pd.notna(row['State']) else "",
            "district": row['District'] if pd.notna(row['District']) else "",
            "pincode": str(row['Pincode']) if pd.notna(row['Pincode']) else "",
            "specialties": row['Specialties'] if pd.notna(row['Specialties']) else "",
            "facilities": row['Facilities'] if pd.notna(row['Facilities']) else ""
        })
    
    return results


def search_hospitals(query: str, hospital_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """
    Search hospitals by name with optional type filter.
    
    Parameters:
    -----------
    query : str
        Search query (hospital name or partial name)
    hospital_type : str, optional
        Filter by type: "Government", "Clinic", "Private"
    limit : int
        Maximum number of results (default 20)
    
    Returns:
    --------
    List of matching hospital dictionaries
    """
    if hospitals_df.empty:
        load_hospitals_data()
    if hospitals_df.empty or not query:
        return []
    
    query_lower = query.lower().strip()
    
    # Filter by type first if specified
    if hospital_type and hospital_type in HOSPITAL_TYPES:
        df = hospitals_df[hospitals_df['Hospital_Type'] == hospital_type]
    else:
        df = hospitals_df
    
    # Search by name (case-insensitive partial match)
    mask = df['Hospital_Name'].str.lower().str.contains(query_lower, na=False)
    matches = df[mask]
    
    results = []
    for _, row in matches.head(limit).iterrows():
        results.append({
            "name": row['Hospital_Name'],
            "type": row['Hospital_Type'],
            "state": row['State'] if pd.notna(row['State']) else "",
            "district": row['District'] if pd.notna(row['District']) else "",
            "pincode": str(row['Pincode']) if pd.notna(row['Pincode']) else "",
            "address": row['Address_Original_First_Line'] if pd.notna(row['Address_Original_First_Line']) else "",
            "phone": row['Telephone'] if pd.notna(row['Telephone']) else ""
        })
    
    return results


def get_hospital_details(hospital_name: str) -> Optional[Dict]:
    """
    Get full details of a specific hospital.
    
    Parameters:
    -----------
    hospital_name : str
        Exact hospital name
    
    Returns:
    --------
    Hospital details dictionary or None if not found
    """
    if hospitals_df.empty:
        load_hospitals_data()
    if hospitals_df.empty:
        return None
    
    match = hospitals_df[hospitals_df['Hospital_Name'] == hospital_name]
    
    if match.empty:
        return None
    
    row = match.iloc[0]
    return {
        "name": row['Hospital_Name'],
        "type": row['Hospital_Type'],
        "category": row['Hospital_Category'] if pd.notna(row['Hospital_Category']) else "",
        "state": row['State'] if pd.notna(row['State']) else "",
        "district": row['District'] if pd.notna(row['District']) else "",
        "pincode": str(row['Pincode']) if pd.notna(row['Pincode']) else "",
        "address": row['Address_Original_First_Line'] if pd.notna(row['Address_Original_First_Line']) else "",
        "phone": row['Telephone'] if pd.notna(row['Telephone']) else "",
        "specialties": row['Specialties'] if pd.notna(row['Specialties']) else "",
        "facilities": row['Facilities'] if pd.notna(row['Facilities']) else ""
    }


def get_hospital_stats() -> Dict:
    """Get statistics about loaded hospitals."""
    if hospitals_df.empty:
        load_hospitals_data()
    if hospitals_df.empty:
        return {"total": 0, "by_type": {}}
    
    stats = hospitals_df['Hospital_Type'].value_counts().to_dict()
    return {
        "total": len(hospitals_df),
        "by_type": stats
    }
