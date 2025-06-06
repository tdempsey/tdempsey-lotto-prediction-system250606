#!/usr/bin/env python3
"""Demo script to showcase column analysis functionality."""

import sys
import os
import json

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lotto_prediction_system import data_generator, analyzer


def main():
    """Run a demonstration of the column analysis features."""
    print("Lotto Prediction System - Column Analysis Demo")
    print("-" * 50)
    
    # Generate sample data with 5 columns
    print("Generating sample data...")
    data = data_generator.generate_column_balanced_data(
        num_draws=100,
        num_numbers=5,
        max_number=42,
        num_columns=5
    )
    
    print(f"Generated {len(data)} draws")
    
    # Display first few rows
    print("\nSample data (first 3 rows):")
    sample = data.head(3)
    for i, row in sample.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        print(f"  {date_str}: {row['Numbers']}")
        print(f"    Column distribution: Col1={row['Col1']} Col2={row['Col2']} Col3={row['Col3']} Col4={row['Col4']} Col5={row['Col5']}")
    
    # Analyze the data
    print("\nAnalyzing data...")
    results = analyzer.analyze(data, include_column_analysis=True, 
                              num_columns=5, max_number=42)
    
    # Print column analysis results
    print("\nColumn Analysis Results:")
    
    # Print column ranges
    print("\n1. Column Ranges:")
    col_ranges = results['column_analysis']['column_ranges']
    for i, (start, end) in enumerate(col_ranges):
        print(f"  Column {i+1}: Numbers {start}-{end}")
    
    # Print column statistics
    print("\n2. Column Statistics:")
    col_stats = results['column_analysis']['column_stats']
    for col, stats in col_stats.items():
        print(f"  {col}:")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Median: {stats['median']:.2f}")
        print(f"    Std Dev: {stats['std_dev']:.2f}")
        print(f"    Min: {stats['min']}")
        print(f"    Max: {stats['max']}")
    
    # Print common patterns
    print("\n3. Most Common Column Distribution Patterns:")
    patterns = results['column_analysis']['common_patterns']
    for pattern, count in patterns[:5]:
        print(f"  Pattern {pattern}: {count} occurrences")
    
    # Print hot numbers by column
    print("\n4. Hot Numbers by Column:")
    hot_by_col = results['column_recommendations']['hot_numbers_by_column']
    for col, numbers in hot_by_col.items():
        print(f"  {col}: {numbers}")
    
    # Print recommended distribution
    print("\n5. Recommended Distribution:")
    distribution = results['column_recommendations']['suggested_distribution']
    for i, count in enumerate(distribution):
        print(f"  Column {i+1}: {count} number(s)")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()