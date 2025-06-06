"""Module for generating seed data for testing and development."""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os


def generate_seed_data(num_draws=1000, num_numbers=5, max_number=42, include_columns=True,
                      output_path=None):
    """
    Generate seed data for lottery analysis and prediction testing.
    
    Args:
        num_draws (int): Number of draws to generate
        num_numbers (int): Number of numbers per draw
        max_number (int): Maximum possible lottery number
        include_columns (bool): Whether to include column-based analysis
        output_path (str): Path to save the generated data
        
    Returns:
        pandas.DataFrame: DataFrame containing the generated data
    """
    # Generate dates for the draws, starting from today and going backward
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=num_draws*7)
    
    # Create draws twice a week (Wednesday and Saturday)
    draw_dates = []
    current_date = start_date
    while current_date <= end_date and len(draw_dates) < num_draws:
        # If Wednesday (2) or Saturday (5)
        if current_date.weekday() in [2, 5]:
            draw_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Take only the required number of draws
    draw_dates = draw_dates[:num_draws]
    
    # Generate random numbers for each draw
    numbers = []
    for _ in range(num_draws):
        draw_numbers = sorted(random.sample(range(1, max_number + 1), num_numbers))
        numbers.append(draw_numbers)
    
    # Create a DataFrame
    data = {
        'Date': draw_dates,
        'Numbers': numbers,
        'Bonus': [random.randint(1, max_number) for _ in range(num_draws)]
    }
    
    # If we want column-based data (for database-style storage)
    if include_columns:
        for i in range(1, num_numbers + 1):
            column_name = f'Number{i}'
            data[column_name] = [nums[i-1] if i <= len(nums) else None for nums in numbers]
    
    df = pd.DataFrame(data)
    
    # Save to file if path is provided
    if output_path:
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(output_path, index=False)
    
    return df


def generate_column_balanced_data(num_draws=1000, num_numbers=5, max_number=42, 
                                  num_columns=5, output_path=None):
    """
    Generate seed data where numbers are balanced across columns.
    
    This is useful for lottery systems where numbers are drawn from separate 
    physical columns or groups.
    
    Args:
        num_draws (int): Number of draws to generate
        num_numbers (int): Number of numbers per draw
        max_number (int): Maximum possible lottery number
        num_columns (int): Number of columns to balance numbers across
        output_path (str): Path to save the generated data
        
    Returns:
        pandas.DataFrame: DataFrame containing the generated data
    """
    # Generate dates for the draws
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=num_draws*7)
    
    # Create draws twice a week (Wednesday and Saturday)
    draw_dates = []
    current_date = start_date
    while current_date <= end_date and len(draw_dates) < num_draws:
        if current_date.weekday() in [2, 5]:
            draw_dates.append(current_date)
        current_date += timedelta(days=1)
    
    draw_dates = draw_dates[:num_draws]
    
    # Calculate numbers per column
    numbers_per_column = max_number // num_columns
    remainder = max_number % num_columns
    
    # Create column ranges
    column_ranges = []
    start = 1
    for i in range(num_columns):
        count = numbers_per_column + (1 if i < remainder else 0)
        end = start + count - 1
        column_ranges.append((start, end))
        start = end + 1
    
    # Generate balanced draws
    all_numbers = []
    columns_data = {}
    
    for i in range(1, num_columns + 1):
        columns_data[f'Col{i}'] = []
    
    for _ in range(num_draws):
        # Generate numbers from each column
        draw_numbers = []
        for col_idx, (start, end) in enumerate(column_ranges):
            # Pick 1 number from each column
            col_number = random.randint(start, end)
            draw_numbers.append(col_number)
            columns_data[f'Col{col_idx+1}'].append(col_number)
        
        # If we need more numbers than columns, add random ones
        while len(draw_numbers) < num_numbers:
            col_idx = random.randint(0, num_columns-1)
            start, end = column_ranges[col_idx]
            col_number = random.randint(start, end)
            if col_number not in draw_numbers:  # avoid duplicates
                draw_numbers.append(col_number)
        
        draw_numbers.sort()
        all_numbers.append(draw_numbers)
    
    # Create DataFrame
    data = {
        'Date': draw_dates,
        'Numbers': all_numbers,
        'Bonus': [random.randint(1, max_number) for _ in range(num_draws)]
    }
    
    # Add column data
    for col_name, col_data in columns_data.items():
        data[col_name] = col_data
    
    # Add individual number columns
    for i in range(1, num_numbers + 1):
        column_name = f'Number{i}'
        data[column_name] = [nums[i-1] if i <= len(nums) else None for nums in all_numbers]
    
    df = pd.DataFrame(data)
    
    # Save to file if path is provided
    if output_path:
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(output_path, index=False)
    
    return df


if __name__ == "__main__":
    # Example usage
    seed_data = generate_seed_data(
        num_draws=1000, 
        num_numbers=5,
        max_number=42,
        output_path="data/seed_data_1000.csv"
    )
    
    column_data = generate_column_balanced_data(
        num_draws=1000,
        num_numbers=5,
        max_number=42,
        num_columns=5,
        output_path="data/column_balanced_1000.csv"
    )
    
    print(f"Generated {len(seed_data)} random draws")
    print(f"Generated {len(column_data)} column-balanced draws")