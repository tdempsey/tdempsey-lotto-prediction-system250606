# Lotto Prediction System

A Python-based system for analyzing lottery data and generating predictions.

## Installation

Basic installation:
```bash
pip install -e .
```

With web GUI support:
```bash
pip install -e ".[web]"
```

## Requirements
- Python 3.6+
- NumPy
- pandas
- scikit-learn
- matplotlib
- Flask (for web GUI)

## Usage

### Web Interface

```bash
# Install with web support
pip install -e ".[web]"

# Start the web GUI
python run_web_gui.py

# Then open a browser and navigate to:
# http://localhost:5000
```

Using the web interface, you can:
1. Generate 1000 draws with column-balanced distribution
2. Analyze historical lottery data
3. Generate predictions using various methods
4. Visualize number frequencies and column distributions

### Command Line Interface

```bash
# Generate test data with column distribution
python main.py generate --draws 1000 --numbers 5 --max-number 42 --columns 5

# Analyze lottery data
python main.py analyze data/seed_data.csv --plot

# Generate predictions
python main.py predict data/seed_data.csv --method statistical
```

### Python API

```python
from lotto_prediction_system import data_loader, analyzer, predictor, data_generator

# Generate seed data
data = data_generator.generate_column_balanced_data(
    num_draws=1000,
    num_numbers=5,
    max_number=42,
    num_columns=5
)

# Analyze the data with column analysis
analysis = analyzer.analyze(data, include_column_analysis=True)

# Generate predictions
predictions = predictor.predict(data, method="statistical")
```

## Features

- Historical lottery data analysis
- Statistical pattern detection
- Multiple prediction methods:
  - Statistical (frequency-based)
  - Random selection
  - Hot numbers
  - Due numbers
  - Machine learning
- Column-based analysis
  - Distribution of numbers across columns
  - Column pattern detection
  - Optimal column distribution recommendations
- Performance evaluation
- Data visualization

## Column Analysis

The system can analyze lottery draws based on a column model, where numbers are divided into groups or "columns" (e.g., numbers 1-8 in Column 1, 9-16 in Column 2, etc.). This helps identify patterns in number selection across different ranges.

Example column analysis output:
```
Column Analysis:
  Column 1: Numbers 1-8
  Column 2: Numbers 9-17
  Column 3: Numbers 18-25
  Column 4: Numbers 26-34
  Column 5: Numbers 35-42

Recommended column distribution:
  Column 1: 1 number(s)
  Column 2: 1 number(s)
  Column 3: 1 number(s)
  Column 4: 1 number(s)
  Column 5: 1 number(s)
```

## Disclaimer

This system is for educational purposes only. There is no guaranteed way to predict lottery numbers, as lottery drawings are designed to be random events. Use at your own risk.