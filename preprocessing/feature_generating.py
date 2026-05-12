import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(json_path: str) -> pd.DataFrame:
    """Load JSON data into a pandas DataFrame."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def load_company_rates(company_rates_path: str) -> pd.DataFrame:
    """Load company rates from CSV."""
    if not os.path.exists(company_rates_path):
        raise FileNotFoundError(f"Company rates file not found: {company_rates_path}")
    return pd.read_csv(company_rates_path)

def company_popularity(data: pd.DataFrame) -> pd.Series:
    """Calculate total applications per company."""
    return data.groupby('client_name').size()

def staff_turnover(data: pd.DataFrame) -> pd.Series:
    """Calculate confirmed applications (staff turnover proxy) per company."""
    confirmed = data[data['grade_proof'] == 'подтверждён']
    return confirmed.groupby('client_name').size()

def competition_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate competition ratio per company and position."""
    total = data.groupby(['client_name', 'position']).size()
    confirmed = data[data['grade_proof'] == 'подтверждён'].groupby(['client_name', 'position']).size()
    
    comp_df = total.to_frame(name='total_applicants').join(
        confirmed.to_frame(name='confirmed_applicants'), how='left'
    )
    comp_df['confirmed_applicants'] = comp_df['confirmed_applicants'].fillna(0).astype(int)
    
    # Avoid division by zero
    mask = comp_df['confirmed_applicants'] > 0
    comp_df.loc[mask, 'competition_ratio'] = (
        comp_df.loc[mask, 'total_applicants'] / comp_df.loc[mask, 'confirmed_applicants']
    )
    comp_df['competition_ratio'] = comp_df['competition_ratio'].replace(np.inf, np.nan)
    
    return comp_df

def enrich_with_rates(df: pd.DataFrame, company_rates_path: str) -> pd.DataFrame:
    """Merge main data with company rates."""
    rates = load_company_rates(company_rates_path)
    return df.merge(rates, on='client_name', how='inner')

def main():
    base_dir = Path(__file__).parent.parent / 'data'
    json_path = base_dir / 'client_dataset.json'
    rates_path = base_dir / 'company_rate.csv'
    
    try:
        df = load_data(json_path)
        df = enrich_with_rates(df, rates_path)
        print(df.head(10))
        logger.info("Data processing completed successfully.")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main()
