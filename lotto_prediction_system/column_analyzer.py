"""Module for analyzing lottery data by column distributions."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def analyze_columns(data, num_columns=5, max_number=42):
    """
    Analyze lottery data by assigning numbers to columns and analyzing patterns.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        num_columns (int): Number of columns to divide numbers into
        max_number (int): Maximum possible lottery number
        
    Returns:
        dict: Dictionary containing column analysis results
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if 'numbers' not in data.columns:
        raise ValueError("Data must contain a 'numbers' column")
    
    # Create a dictionary to store analysis results
    analysis = {}
    
    # Determine column ranges
    numbers_per_column = max_number // num_columns
    remainder = max_number % num_columns
    
    column_ranges = []
    start = 1
    for i in range(num_columns):
        count = numbers_per_column + (1 if i < remainder else 0)
        end = start + count - 1
        column_ranges.append((start, end))
        start = end + 1
    
    analysis['column_ranges'] = column_ranges
    
    # Analyze each draw to determine column distribution
    column_counts = []
    for draw_numbers in data['numbers']:
        counts = [0] * num_columns
        for num in draw_numbers:
            for col_idx, (start, end) in enumerate(column_ranges):
                if start <= num <= end:
                    counts[col_idx] += 1
                    break
        column_counts.append(counts)
    
    # Convert to DataFrame for easier analysis
    col_names = [f'Column{i+1}' for i in range(num_columns)]
    column_df = pd.DataFrame(column_counts, columns=col_names)
    
    # Calculate statistics for each column
    analysis['column_stats'] = {}
    for col in col_names:
        analysis['column_stats'][col] = {
            'mean': column_df[col].mean(),
            'median': column_df[col].median(),
            'std_dev': column_df[col].std(),
            'min': column_df[col].min(),
            'max': column_df[col].max(),
        }
    
    # Calculate common column patterns
    patterns = [tuple(counts) for counts in column_counts]
    pattern_counts = Counter(patterns)
    analysis['common_patterns'] = pattern_counts.most_common(10)
    
    # Analyze frequency of individual numbers within each column
    column_number_freq = {}
    for col_idx, (start, end) in enumerate(column_ranges):
        column_name = col_names[col_idx]
        number_freq = {}
        for num in range(start, end + 1):
            count = sum(1 for draw in data['numbers'] if num in draw)
            number_freq[num] = count
        column_number_freq[column_name] = number_freq
    
    analysis['column_number_frequency'] = column_number_freq
    
    return analysis


def get_column_recommendations(analysis, numbers_per_draw=5):
    """
    Generate recommendations for lottery number selection based on column analysis.
    
    Args:
        analysis (dict): Dictionary containing column analysis results
        numbers_per_draw (int): Number of numbers to pick per draw
        
    Returns:
        dict: Dictionary containing column distribution recommendations
    """
    recommendations = {}
    
    # Extract column ranges
    column_ranges = analysis['column_ranges']
    num_columns = len(column_ranges)
    
    # Determine optimal distribution of numbers across columns
    # Start with minimum 1 number from each column if possible
    base_distribution = [1] * num_columns
    remaining = numbers_per_draw - num_columns
    
    # If we have fewer numbers than columns, adjust
    if remaining < 0:
        base_distribution = [0] * num_columns
        remaining = numbers_per_draw
    
    # Find columns with highest frequency numbers
    column_stats = []
    for col_idx, col_name in enumerate([f'Column{i+1}' for i in range(num_columns)]):
        if col_name in analysis['column_stats']:
            stats = analysis['column_stats'][col_name]
            column_stats.append((col_idx, stats['mean']))
    
    # Sort columns by mean frequency (descending)
    sorted_columns = sorted(column_stats, key=lambda x: x[1], reverse=True)
    
    # Distribute remaining numbers to highest frequency columns
    for i in range(remaining):
        if i < len(sorted_columns):
            col_idx = sorted_columns[i][0]
            base_distribution[col_idx] += 1
    
    recommendations['suggested_distribution'] = base_distribution
    
    # Find hot numbers in each column
    hot_numbers_by_column = {}
    for col_idx, (start, end) in enumerate(column_ranges):
        col_name = f'Column{col_idx+1}'
        if col_name in analysis['column_number_frequency']:
            freq = analysis['column_number_frequency'][col_name]
            sorted_numbers = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            hot_numbers_by_column[col_name] = [num for num, _ in sorted_numbers[:5]]
    
    recommendations['hot_numbers_by_column'] = hot_numbers_by_column
    
    return recommendations


def plot_column_distribution(analysis, save_path=None):
    """
    Plot the distribution of numbers across columns.
    
    Args:
        analysis (dict): Dictionary containing column analysis results
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Extract column statistics
    column_stats = analysis['column_stats']
    col_names = sorted(column_stats.keys())
    
    means = [column_stats[col]['mean'] for col in col_names]
    std_devs = [column_stats[col]['std_dev'] for col in col_names]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Bar chart with error bars
    plt.bar(col_names, means, yerr=std_devs, alpha=0.7)
    plt.axhline(y=sum(means)/len(means), color='r', linestyle='--', label='Average')
    
    plt.title('Column Distribution in Lottery Draws')
    plt.xlabel('Column')
    plt.ylabel('Average Numbers per Draw')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()


def add_column_data(df, num_columns=5, max_number=42):
    """
    Add column data to an existing DataFrame with lottery draws.
    
    Args:
        df (pandas.DataFrame): DataFrame containing lottery data
        num_columns (int): Number of columns to divide numbers into
        max_number (int): Maximum possible lottery number
        
    Returns:
        pandas.DataFrame: DataFrame with added column data
    """
    # Determine column ranges
    numbers_per_column = max_number // num_columns
    remainder = max_number % num_columns
    
    column_ranges = []
    start = 1
    for i in range(num_columns):
        count = numbers_per_column + (1 if i < remainder else 0)
        end = start + count - 1
        column_ranges.append((start, end))
        start = end + 1
    
    # Create new columns for each draw's distribution
    for i in range(num_columns):
        col_name = f'Col{i+1}'
        start, end = column_ranges[i]
        
        # Count numbers in this column range for each draw
        df[col_name] = df['numbers'].apply(
            lambda nums: sum(1 for num in nums if start <= num <= end)
        )
    
    return df