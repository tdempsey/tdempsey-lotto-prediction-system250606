"""Web-based GUI for the Lotto Prediction System."""

import os
import json
from flask import Flask, render_template, request, redirect, url_for, send_file
from datetime import datetime

from lotto_prediction_system import data_loader, analyzer, predictor, data_generator


app = Flask(__name__)

# Ensure templates and static directories exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'plots'), exist_ok=True)


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    """Generate seed data based on form inputs."""
    if request.method == 'POST':
        # Get form data
        draws = int(request.form.get('draws', 1000))
        numbers = int(request.form.get('numbers', 5))
        max_number = int(request.form.get('max_number', 42))
        columns = int(request.form.get('columns', 5))
        generation_type = request.form.get('type', 'column-balanced')
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"seed_data_{timestamp}.csv"
        output_path = os.path.join('lotto_prediction_system', 'data', filename)
        
        # Generate the data
        if generation_type == 'random':
            df = data_generator.generate_seed_data(
                num_draws=draws,
                num_numbers=numbers,
                max_number=max_number,
                output_path=output_path
            )
        else:
            df = data_generator.generate_column_balanced_data(
                num_draws=draws,
                num_numbers=numbers,
                max_number=max_number,
                num_columns=columns,
                output_path=output_path
            )
        
        # Sample data for display
        sample_data = []
        for i, row in df.head(10).iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            sample_data.append({
                'date': date_str,
                'numbers': row['Numbers'],
                'columns': [row.get(f'Col{j+1}', 0) for j in range(columns)]
            })
        
        return render_template('generate_result.html', 
                              draws=draws,
                              numbers=numbers, 
                              max_number=max_number,
                              columns=columns,
                              type=generation_type,
                              filename=filename,
                              sample_data=sample_data)
    
    # GET request - show the form
    return render_template('generate.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_data():
    """Analyze lottery data based on form inputs."""
    if request.method == 'POST':
        # Get the data file
        if 'data_file' not in request.files:
            return render_template('analyze.html', error="No file selected")
        
        file = request.files['data_file']
        if file.filename == '':
            return render_template('analyze.html', error="No file selected")
        
        # Save the uploaded file
        file_path = os.path.join('lotto_prediction_system', 'data', file.filename)
        file.save(file_path)
        
        # Load the data
        try:
            data = data_loader.load_data(file_path)
        except Exception as e:
            return render_template('analyze.html', error=f"Error loading data: {str(e)}")
        
        # Determine max number from the data
        max_number = max(num for numbers in data['numbers'] for num in numbers)
        
        # Analyze the data
        num_columns = int(request.form.get('columns', 5))
        include_plots = 'generate_plots' in request.form
        
        results = analyzer.analyze(data, include_column_analysis=True, 
                                 num_columns=num_columns, max_number=max_number)
        
        # Generate plots if requested
        plots = {}
        if include_plots:
            # Create plots directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_dir = os.path.join('lotto_prediction_system', 'plots', timestamp)
            os.makedirs(plot_dir, exist_ok=True)
            
            # Frequency plot
            freq_plot_path = os.path.join(plot_dir, "frequency.png")
            analyzer.plot_frequency(results['frequency'], save_path=freq_plot_path)
            plots['frequency'] = os.path.basename(freq_plot_path)
            
            # Column distribution plot
            if 'column_analysis' in results:
                col_plot_path = os.path.join(plot_dir, "columns.png")
                analyzer.plot_column_distribution(data, results, save_path=col_plot_path)
                plots['columns'] = os.path.basename(col_plot_path)
        
        # Prepare analysis summary
        summary = {
            'draws': len(data),
            'date_range': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
            'hot_numbers': [
                {'number': num, 'frequency': results['frequency'][num]} 
                for num in results['hot_numbers'][:10]
            ],
            'cold_numbers': [
                {'number': num, 'frequency': results['frequency'][num]} 
                for num in results['cold_numbers'][:10]
            ],
            'plots': plots
        }
        
        # Add column analysis if available
        if 'column_analysis' in results:
            col_ranges = results['column_analysis']['column_ranges']
            summary['columns'] = [
                {'id': i+1, 'range': f"{start}-{end}"} 
                for i, (start, end) in enumerate(col_ranges)
            ]
            
            summary['column_distribution'] = [
                {'column': i+1, 'count': count} 
                for i, count in enumerate(results['column_recommendations']['suggested_distribution'])
            ]
            
            summary['hot_by_column'] = {}
            for col, numbers in results['column_recommendations']['hot_numbers_by_column'].items():
                summary['hot_by_column'][col] = numbers
        
        return render_template('analyze_result.html', summary=summary, timestamp=timestamp)
    
    # GET request - show the form
    return render_template('analyze.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_numbers():
    """Generate lottery number predictions based on form inputs."""
    if request.method == 'POST':
        # Get the data file
        if 'data_file' not in request.files:
            return render_template('predict.html', error="No file selected")
        
        file = request.files['data_file']
        if file.filename == '':
            return render_template('predict.html', error="No file selected")
        
        # Save the uploaded file
        file_path = os.path.join('lotto_prediction_system', 'data', file.filename)
        file.save(file_path)
        
        # Load the data
        try:
            data = data_loader.load_data(file_path)
        except Exception as e:
            return render_template('predict.html', error=f"Error loading data: {str(e)}")
        
        # Get prediction parameters
        method = request.form.get('method', 'statistical')
        count = int(request.form.get('count', 5))
        number_count = int(request.form.get('numbers', 5))
        max_number = int(request.form.get('max_number', 42))
        
        # Generate predictions
        predictions = predictor.predict(
            data, 
            method=method,
            num_predictions=count,
            number_count=number_count,
            max_number=max_number
        )
        
        # Format predictions for display
        formatted_predictions = []
        for i, pred in enumerate(predictions, 1):
            formatted_predictions.append({
                'id': i,
                'numbers': pred
            })
        
        return render_template('predict_result.html', 
                             method=method,
                             predictions=formatted_predictions)
    
    # GET request - show the form
    return render_template('predict.html')


@app.route('/download/<path:filename>')
def download_file(filename):
    """Download a generated file."""
    directory = os.path.join('lotto_prediction_system', 'data')
    return send_file(os.path.join(directory, filename), as_attachment=True)


def run_app(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask web application."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app()