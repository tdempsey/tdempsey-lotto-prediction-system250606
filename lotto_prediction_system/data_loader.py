"""Module for loading and processing lottery data."""

import pandas as pd
import os
import csv
from datetime import datetime


def load_data(file_path):
    """
    Load lottery data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing lottery data
        
    Returns:
        pandas.DataFrame: DataFrame containing the lottery data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() != '.csv':
        raise ValueError(f"Unsupported file format: {ext}. Only CSV files are supported.")
    
    # Try to determine the format and load the data
    try:
        df = pd.read_csv(file_path)
        return _process_data(df)
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def _process_data(df):
    """
    Process and clean the loaded lottery data.
    
    Args:
        df (pandas.DataFrame): Raw DataFrame with lottery data
        
    Returns:
        pandas.DataFrame: Processed and cleaned DataFrame
    """
    # Check for required columns
    required_cols = ['date', 'numbers']
    
    # Try to infer columns if they don't match exactly
    cols = df.columns.str.lower()
    date_col = next((col for col in cols if 'date' in col), None)
    
    # Look for number columns - could be separate columns or one column with all numbers
    number_cols = [col for col in cols if ('number' in col or 'num' in col) and 'bonus' not in col]
    bonus_cols = [col for col in cols if ('bonus' in col or 'extra' in col or 'powerball' in col)]
    
    # If we don't have clear number columns, try to infer from the structure
    if not number_cols:
        # If we have columns named like num1, num2, etc. or 1, 2, 3, etc.
        try:
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            if len(numeric_cols) >= 5:  # Assume at least 5 numbers in a lottery draw
                number_cols = numeric_cols[:6]  # Take the first 6 as main numbers
                if len(numeric_cols) > 6:
                    bonus_cols = [numeric_cols[6]]
        except:
            pass
    
    # Create a standardized DataFrame
    if date_col and (number_cols or bonus_cols):
        # Convert date to standard format
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            # If we can't parse the date, create a dummy date
            df[date_col] = pd.date_range(start='2000-01-01', periods=len(df))
        
        # Combine number columns into a list
        if len(number_cols) > 1:
            # Numbers are in separate columns
            df['numbers'] = df[number_cols].apply(lambda x: sorted(x.dropna().astype(int).tolist()), axis=1)
        elif len(number_cols) == 1:
            # Numbers might be in a single column as string
            try:
                df['numbers'] = df[number_cols[0]].apply(lambda x: sorted([int(n.strip()) for n in str(x).split(',')]))
            except:
                df['numbers'] = df[number_cols[0]]
        
        # Process bonus numbers if available
        if bonus_cols:
            if len(bonus_cols) > 1:
                df['bonus'] = df[bonus_cols].apply(lambda x: x.dropna().astype(int).tolist(), axis=1)
            else:
                df['bonus'] = df[bonus_cols[0]]
        
        # Create a cleaned DataFrame with standardized columns
        cleaned_df = pd.DataFrame({
            'date': df[date_col],
            'numbers': df['numbers'] if 'numbers' in df else [],
            'bonus': df['bonus'] if 'bonus' in df else None
        })
        
        return cleaned_df.sort_values('date')
    
    # If we can't infer the structure, return the original DataFrame with a warning
    print("Warning: Could not determine the structure of the lottery data. Returning original data.")
    return df


def save_data(df, file_path):
    """
    Save processed lottery data to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        file_path (str): Path to save the CSV file
        
    Returns:
        str: Path to the saved file
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    
    return file_path