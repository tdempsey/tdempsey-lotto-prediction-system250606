#!/usr/bin/env python3
"""Main entry point for the Lotto Prediction System command line tool."""

import argparse
import sys
import os
import json
from datetime import datetime

from lotto_prediction_system import data_loader, analyzer, predictor, data_generator


def main():
    """Run the Lotto Prediction System command line tool."""
    parser = argparse.ArgumentParser(description="Lottery number prediction system")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse and process lottery data")
    parse_parser.add_argument("input_file", help="Path to the input CSV file")
    parse_parser.add_argument("--output-file", "-o", help="Path to the output file")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze lottery data")
    analyze_parser.add_argument("data_file", help="Path to the data file")
    analyze_parser.add_argument("--output-file", "-o", help="Path to save the analysis results")
    analyze_parser.add_argument("--plot", "-p", action="store_true", help="Generate plots")
    analyze_parser.add_argument("--plot-dir", default="plots", help="Directory to save plots")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate lottery number predictions")
    predict_parser.add_argument("data_file", help="Path to the data file")
    predict_parser.add_argument(
        "--method", "-m", 
        choices=["statistical", "random", "hot", "due", "ml"],
        default="statistical",
        help="Prediction method to use"
    )
    predict_parser.add_argument(
        "--count", "-c", 
        type=int, 
        default=1,
        help="Number of predictions to generate"
    )
    predict_parser.add_argument(
        "--number-count", "-n", 
        type=int, 
        default=6,
        help="Number of numbers per draw"
    )
    predict_parser.add_argument(
        "--max-number", "-x", 
        type=int, 
        default=49,
        help="Maximum possible lottery number"
    )
    predict_parser.add_argument(
        "--output-file", "-o",
        help="Path to save the predictions"
    )
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate seed data for testing")
    generate_parser.add_argument(
        "--draws", "-d", 
        type=int, 
        default=1000,
        help="Number of draws to generate"
    )
    generate_parser.add_argument(
        "--numbers", "-n", 
        type=int, 
        default=5,
        help="Number of numbers per draw"
    )
    generate_parser.add_argument(
        "--max-number", "-x", 
        type=int, 
        default=42,
        help="Maximum possible lottery number"
    )
    generate_parser.add_argument(
        "--columns", "-c", 
        type=int, 
        default=5,
        help="Number of columns to balance numbers across"
    )
    generate_parser.add_argument(
        "--type", "-t",
        choices=["random", "column-balanced"],
        default="column-balanced",
        help="Type of data generation"
    )
    generate_parser.add_argument(
        "--output-file", "-o",
        default="lotto_prediction_system/data/seed_data.csv",
        help="Path to save the generated data"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is provided, print help
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the requested command
    try:
        if args.command == "parse":
            return parse_data(args)
        elif args.command == "analyze":
            return analyze_data(args)
        elif args.command == "predict":
            return predict_numbers(args)
        elif args.command == "generate":
            return generate_seed_data(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def parse_data(args):
    """Parse and process lottery data."""
    print(f"Parsing data from {args.input_file}")
    
    # Load and process the data
    data = data_loader.load_data(args.input_file)
    
    # Determine output file path
    output_file = args.output_file
    if not output_file:
        # Generate a default output file path
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_file = f"data/{base_name}_processed.csv"
    
    # Save the processed data
    output_path = data_loader.save_data(data, output_file)
    print(f"Processed data saved to {output_path}")
    
    return 0


def analyze_data(args):
    """Analyze lottery data."""
    print(f"Analyzing data from {args.data_file}")
    
    # Load the data
    data = data_loader.load_data(args.data_file)
    
    # Determine max number from the data
    max_number = max(num for numbers in data['numbers'] for num in numbers)
    
    # Analyze the data
    results = analyzer.analyze(data, include_column_analysis=True, 
                              num_columns=5, max_number=max_number)
    
    # Print summary of the analysis
    print("\nAnalysis Results:")
    print(f"Total number of draws: {len(data)}")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    
    print("\nHot numbers (most frequent):")
    for num in results['hot_numbers'][:10]:
        freq = results['frequency'][num]
        print(f"  {num}: {freq} occurrences ({freq/len(data)*100:.1f}%)")
    
    print("\nCold numbers (least frequent):")
    for num in results['cold_numbers'][:10]:
        freq = results['frequency'][num]
        print(f"  {num}: {freq} occurrences ({freq/len(data)*100:.1f}%)")
    
    # Print column analysis
    if 'column_analysis' in results:
        print("\nColumn Analysis:")
        col_ranges = results['column_analysis']['column_ranges']
        for i, (start, end) in enumerate(col_ranges):
            print(f"  Column {i+1}: Numbers {start}-{end}")
        
        print("\nRecommended column distribution:")
        distribution = results['column_recommendations']['suggested_distribution']
        for i, count in enumerate(distribution):
            print(f"  Column {i+1}: {count} number(s)")
        
        print("\nHot numbers by column:")
        hot_by_col = results['column_recommendations']['hot_numbers_by_column']
        for col, numbers in hot_by_col.items():
            print(f"  {col}: {numbers}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # Frequency plot
        frequency_plot_path = os.path.join(args.plot_dir, "number_frequency.png")
        analyzer.plot_frequency(results['frequency'], save_path=frequency_plot_path)
        print(f"Frequency plot saved to {frequency_plot_path}")
        
        # Column distribution plot
        if 'column_analysis' in results:
            column_plot_path = os.path.join(args.plot_dir, "column_distribution.png")
            analyzer.plot_column_distribution(data, results, save_path=column_plot_path)
            print(f"Column distribution plot saved to {column_plot_path}")
    
    # Save results if an output file is specified
    if args.output_file:
        # Convert the results to a serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, pd.Series):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        # Save to JSON
        with open(args.output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Analysis results saved to {args.output_file}")
    
    return 0


def predict_numbers(args):
    """Generate lottery number predictions."""
    print(f"Generating predictions using method: {args.method}")
    
    # Load the data
    data = data_loader.load_data(args.data_file)
    
    # Generate predictions
    predictions = predictor.predict(
        data, 
        method=args.method,
        num_predictions=args.count,
        number_count=args.number_count,
        max_number=args.max_number
    )
    
    # Print the predictions
    print("\nPredictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  Prediction {i}: {pred}")
    
    # Save predictions if an output file is specified
    if args.output_file:
        # Create a dictionary with prediction information
        output_data = {
            "date": datetime.now().isoformat(),
            "method": args.method,
            "data_file": args.data_file,
            "predictions": predictions
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        # Save to JSON
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Predictions saved to {args.output_file}")
    
    return 0


def generate_seed_data(args):
    """Generate seed data for testing and development."""
    print(f"Generating {args.draws} draws of {args.numbers} numbers (max: {args.max_number})")
    
    if args.type == "random":
        print("Using random generation method")
        df = data_generator.generate_seed_data(
            num_draws=args.draws,
            num_numbers=args.numbers,
            max_number=args.max_number,
            output_path=args.output_file
        )
    else:
        print(f"Using column-balanced generation method with {args.columns} columns")
        df = data_generator.generate_column_balanced_data(
            num_draws=args.draws,
            num_numbers=args.numbers,
            max_number=args.max_number,
            num_columns=args.columns,
            output_path=args.output_file
        )
    
    print(f"\nGenerated {len(df)} draws")
    print(f"Data saved to {args.output_file}")
    
    # Display first few rows
    print("\nSample data (first 5 rows):")
    sample = df.head(5)
    for i, row in sample.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], datetime) else row['Date']
        print(f"  {date_str}: {row['Numbers']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())