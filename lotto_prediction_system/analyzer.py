"""Module for analyzing lottery data and detecting patterns."""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from lotto_prediction_system import column_analyzer


def analyze(data, include_column_analysis=True, num_columns=5, max_number=42):
    """
    Analyze lottery data to extract statistics and patterns.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        include_column_analysis (bool): Whether to include column-based analysis
        num_columns (int): Number of columns to analyze
        max_number (int): Maximum possible lottery number
        
    Returns:
        dict: Dictionary containing analysis results
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if 'numbers' not in data.columns:
        raise ValueError("Data must contain a 'numbers' column")
    
    # Create a dictionary to store analysis results
    analysis = {}
    
    # Frequency analysis of individual numbers
    analysis['frequency'] = frequency_analysis(data)
    
    # Hot and cold numbers
    analysis['hot_numbers'] = analysis['frequency'].head(10).index.tolist()
    analysis['cold_numbers'] = analysis['frequency'].tail(10).index.tolist()
    
    # Analyze gaps between occurrences
    analysis['gaps'] = gap_analysis(data)
    
    # Analyze pairs and triplets
    analysis['pairs'] = pair_analysis(data)
    analysis['triplets'] = triplet_analysis(data)
    
    # Time-based patterns
    analysis['time_patterns'] = time_analysis(data)
    
    # Statistical measures
    analysis['statistics'] = statistical_analysis(data)
    
    # Column-based analysis
    if include_column_analysis:
        analysis['column_analysis'] = column_analyzer.analyze_columns(
            data, num_columns=num_columns, max_number=max_number
        )
        analysis['column_recommendations'] = column_analyzer.get_column_recommendations(
            analysis['column_analysis'], numbers_per_draw=len(data['numbers'].iloc[0])
        )
    
    return analysis


def frequency_analysis(data):
    """
    Analyze the frequency of each number in the lottery data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        
    Returns:
        pandas.Series: Series with numbers as index and their frequencies as values
    """
    # Flatten the list of numbers
    all_numbers = [num for numbers in data['numbers'] for num in numbers]
    
    # Count occurrences of each number
    counter = Counter(all_numbers)
    
    # Convert to series and sort
    frequency = pd.Series(counter).sort_values(ascending=False)
    
    return frequency


def gap_analysis(data):
    """
    Analyze the gaps between occurrences of each number.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        
    Returns:
        dict: Dictionary with numbers as keys and gap statistics as values
    """
    # Get all unique numbers
    all_numbers = set([num for numbers in data['numbers'] for num in numbers])
    
    # Create a dictionary to store gap information
    gaps = {}
    
    # For each number, find the gaps between occurrences
    for num in all_numbers:
        # Get the draw indices where this number appeared
        appearances = [i for i, row in data.iterrows() if num in row['numbers']]
        
        # Calculate gaps between consecutive appearances
        number_gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
        
        if number_gaps:
            gaps[num] = {
                'min_gap': min(number_gaps),
                'max_gap': max(number_gaps),
                'avg_gap': sum(number_gaps) / len(number_gaps),
                'current_gap': len(data) - 1 - appearances[-1] if appearances else float('inf')
            }
        else:
            gaps[num] = {
                'min_gap': float('inf'),
                'max_gap': float('inf'),
                'avg_gap': float('inf'),
                'current_gap': float('inf')
            }
    
    return gaps


def pair_analysis(data):
    """
    Analyze the frequency of pairs of numbers appearing together.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        
    Returns:
        pandas.Series: Series with pairs as index and their frequencies as values
    """
    pairs = []
    
    # For each draw, generate all possible pairs
    for numbers in data['numbers']:
        numbers = sorted(numbers)
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                pairs.append((numbers[i], numbers[j]))
    
    # Count occurrences of each pair
    counter = Counter(pairs)
    
    # Convert to series and sort
    frequency = pd.Series(counter).sort_values(ascending=False)
    
    return frequency


def triplet_analysis(data):
    """
    Analyze the frequency of triplets of numbers appearing together.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        
    Returns:
        pandas.Series: Series with triplets as index and their frequencies as values
    """
    triplets = []
    
    # For each draw, generate all possible triplets
    for numbers in data['numbers']:
        numbers = sorted(numbers)
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                for k in range(j+1, len(numbers)):
                    triplets.append((numbers[i], numbers[j], numbers[k]))
    
    # Count occurrences of each triplet
    counter = Counter(triplets)
    
    # Convert to series and sort
    frequency = pd.Series(counter).sort_values(ascending=False)
    
    return frequency


def time_analysis(data):
    """
    Analyze patterns over time, such as seasonality or trends.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        
    Returns:
        dict: Dictionary containing time-based analysis results
    """
    if 'date' not in data.columns:
        return {}
    
    # Ensure date column is datetime type
    data['date'] = pd.to_datetime(data['date'])
    
    # Extract time components
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    
    # Analyze by month
    monthly_counts = data.groupby('month').size()
    
    # Analyze by day of week
    day_of_week_counts = data.groupby('day_of_week').size()
    
    return {
        'monthly': monthly_counts.to_dict(),
        'day_of_week': day_of_week_counts.to_dict()
    }


def statistical_analysis(data):
    """
    Perform statistical analysis on the lottery numbers.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        
    Returns:
        dict: Dictionary containing statistical analysis results
    """
    # Get all numbers
    all_numbers = [num for numbers in data['numbers'] for num in numbers]
    
    # Basic statistics
    stats = {
        'mean': np.mean(all_numbers),
        'median': np.median(all_numbers),
        'std_dev': np.std(all_numbers),
        'min': min(all_numbers),
        'max': max(all_numbers)
    }
    
    # Check for even/odd distribution
    evens = sum(1 for num in all_numbers if num % 2 == 0)
    odds = len(all_numbers) - evens
    stats['even_odd_ratio'] = evens / odds if odds > 0 else float('inf')
    
    # Check for high/low distribution (assuming numbers range from 1 to max_number)
    max_number = max(all_numbers)
    threshold = max_number / 2
    high = sum(1 for num in all_numbers if num > threshold)
    low = len(all_numbers) - high
    stats['high_low_ratio'] = high / low if low > 0 else float('inf')
    
    return stats


def plot_frequency(frequency, title="Number Frequency", save_path=None):
    """
    Plot the frequency of lottery numbers.
    
    Args:
        frequency (pandas.Series): Series with numbers as index and frequencies as values
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(frequency.index.astype(str), frequency.values)
    
    # Add a trend line
    plt.plot(range(len(frequency)), frequency.values, 'r--')
    
    plt.title(title)
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()


def plot_column_distribution(data, analysis, save_path=None):
    """
    Plot the distribution of numbers across columns.
    
    Args:
        data (pandas.DataFrame): DataFrame containing lottery data
        analysis (dict): Dictionary containing analysis results
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if 'column_analysis' not in analysis:
        raise ValueError("Column analysis results not found in analysis dictionary")
    
    return column_analyzer.plot_column_distribution(analysis['column_analysis'], save_path=save_path)