"""Module for predicting lottery numbers based on historical data."""

import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

from lotto_prediction_system import analyzer


def predict(data, method="statistical", num_predictions=1, number_count=6, max_number=49):
    """
    Generate lottery number predictions based on historical data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing historical lottery data
        method (str): Prediction method to use ('statistical', 'random', 'hot', 'due', 'ml')
        num_predictions (int): Number of predictions to generate
        number_count (int): Number of numbers to predict per draw
        max_number (int): Maximum possible lottery number
        
    Returns:
        list: List of predictions, where each prediction is a list of lottery numbers
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if 'numbers' not in data.columns:
        raise ValueError("Data must contain a 'numbers' column")
    
    # Choose prediction method
    if method == "statistical":
        return statistical_prediction(data, num_predictions, number_count, max_number)
    elif method == "random":
        return random_prediction(num_predictions, number_count, max_number)
    elif method == "hot":
        return hot_numbers_prediction(data, num_predictions, number_count)
    elif method == "due":
        return due_numbers_prediction(data, num_predictions, number_count, max_number)
    elif method == "ml":
        return ml_prediction(data, num_predictions, number_count, max_number)
    else:
        raise ValueError(f"Unknown prediction method: {method}")


def statistical_prediction(data, num_predictions=1, number_count=6, max_number=49):
    """
    Generate predictions based on statistical analysis of historical data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing historical lottery data
        num_predictions (int): Number of predictions to generate
        number_count (int): Number of numbers to predict per draw
        max_number (int): Maximum possible lottery number
        
    Returns:
        list: List of predictions, where each prediction is a list of lottery numbers
    """
    # Analyze the data
    analysis = analyzer.analyze(data)
    
    # Get frequency of each number
    frequency = analysis['frequency']
    
    # Create a probability distribution based on frequency
    probs = {}
    total_draws = len(data)
    
    # Ensure all possible numbers are included with at least some probability
    for num in range(1, max_number + 1):
        if num in frequency:
            # Higher frequency means higher probability
            probs[num] = frequency[num] / total_draws
        else:
            # Numbers that have never appeared get a small non-zero probability
            probs[num] = 0.01 / total_draws
    
    # Adjust probabilities to account for hot/cold and due numbers
    for num in probs:
        # Adjust for current gap (due numbers)
        if num in analysis['gaps']:
            gap_ratio = analysis['gaps'][num]['current_gap'] / analysis['gaps'][num]['avg_gap']
            if gap_ratio > 1:
                # Number is overdue, increase its probability
                probs[num] *= (1 + min(gap_ratio, 2) * 0.2)
    
    # Normalize probabilities
    total_prob = sum(probs.values())
    for num in probs:
        probs[num] /= total_prob
    
    # Generate predictions
    predictions = []
    for _ in range(num_predictions):
        # Use weighted random sampling without replacement
        prediction = []
        remaining_probs = probs.copy()
        
        for _ in range(number_count):
            # Normalize remaining probabilities
            total_remaining = sum(remaining_probs.values())
            if total_remaining == 0:
                break
                
            norm_probs = {num: p / total_remaining for num, p in remaining_probs.items()}
            
            # Select a number based on the probability distribution
            nums = list(norm_probs.keys())
            probs_list = list(norm_probs.values())
            selected = np.random.choice(nums, p=probs_list)
            
            prediction.append(selected)
            remaining_probs.pop(selected)
        
        predictions.append(sorted(prediction))
    
    return predictions


def random_prediction(num_predictions=1, number_count=6, max_number=49):
    """
    Generate completely random predictions.
    
    Args:
        num_predictions (int): Number of predictions to generate
        number_count (int): Number of numbers to predict per draw
        max_number (int): Maximum possible lottery number
        
    Returns:
        list: List of predictions, where each prediction is a list of lottery numbers
    """
    predictions = []
    for _ in range(num_predictions):
        prediction = sorted(random.sample(range(1, max_number + 1), number_count))
        predictions.append(prediction)
    
    return predictions


def hot_numbers_prediction(data, num_predictions=1, number_count=6):
    """
    Generate predictions using the most frequently drawn numbers.
    
    Args:
        data (pandas.DataFrame): DataFrame containing historical lottery data
        num_predictions (int): Number of predictions to generate
        number_count (int): Number of numbers to predict per draw
        
    Returns:
        list: List of predictions, where each prediction is a list of lottery numbers
    """
    # Analyze the data
    analysis = analyzer.analyze(data)
    
    # Get the most frequent numbers
    hot_numbers = list(analysis['frequency'].head(number_count * 2).index)
    
    # Generate predictions
    predictions = []
    for _ in range(num_predictions):
        prediction = sorted(random.sample(hot_numbers, number_count))
        predictions.append(prediction)
    
    return predictions


def due_numbers_prediction(data, num_predictions=1, number_count=6, max_number=49):
    """
    Generate predictions using numbers that are due to be drawn based on their gaps.
    
    Args:
        data (pandas.DataFrame): DataFrame containing historical lottery data
        num_predictions (int): Number of predictions to generate
        number_count (int): Number of numbers to predict per draw
        max_number (int): Maximum possible lottery number
        
    Returns:
        list: List of predictions, where each prediction is a list of lottery numbers
    """
    # Analyze the data
    analysis = analyzer.analyze(data)
    
    # Get the gaps between occurrences for each number
    gaps = analysis['gaps']
    
    # Calculate a 'due score' for each number based on current gap vs. average gap
    due_scores = {}
    for num in range(1, max_number + 1):
        if num in gaps:
            avg_gap = gaps[num]['avg_gap']
            current_gap = gaps[num]['current_gap']
            due_scores[num] = current_gap / avg_gap if avg_gap > 0 else 0
        else:
            # Numbers that have never appeared get a high due score
            due_scores[num] = 10
    
    # Sort numbers by due score (highest first)
    due_numbers = sorted(due_scores.keys(), key=lambda x: due_scores[x], reverse=True)
    
    # Take the top due numbers
    top_due = due_numbers[:number_count * 2]
    
    # Generate predictions
    predictions = []
    for _ in range(num_predictions):
        prediction = sorted(random.sample(top_due, number_count))
        predictions.append(prediction)
    
    return predictions


def ml_prediction(data, num_predictions=1, number_count=6, max_number=49):
    """
    Generate predictions using machine learning based on historical patterns.
    
    Args:
        data (pandas.DataFrame): DataFrame containing historical lottery data
        num_predictions (int): Number of predictions to generate
        number_count (int): Number of numbers to predict per draw
        max_number (int): Maximum possible lottery number
        
    Returns:
        list: List of predictions, where each prediction is a list of lottery numbers
    """
    # Check if we have enough data
    if len(data) < 50:
        return statistical_prediction(data, num_predictions, number_count, max_number)
    
    # Extract features and labels
    features, labels = extract_features(data, max_number)
    
    # Train a model
    model = train_model(features, labels)
    
    # Generate predictions
    predictions = []
    for _ in range(num_predictions):
        # Create feature vector for prediction
        latest_features = features.iloc[-1].values.reshape(1, -1)
        
        # Predict probabilities for each number
        probs = model.predict_proba(latest_features)[0]
        
        # Use probabilities to perform weighted sampling
        nums = list(range(1, max_number + 1))
        prediction = []
        
        # Select numbers based on probability, without replacement
        for _ in range(number_count):
            if not nums:
                break
                
            # Normalize probabilities for remaining numbers
            remaining_probs = [probs[num-1] for num in nums]
            total = sum(remaining_probs)
            if total > 0:
                norm_probs = [p / total for p in remaining_probs]
            else:
                norm_probs = [1/len(nums)] * len(nums)
            
            # Select a number
            selected_idx = np.random.choice(range(len(nums)), p=norm_probs)
            selected = nums.pop(selected_idx)
            prediction.append(selected)
        
        predictions.append(sorted(prediction))
    
    return predictions


def extract_features(data, max_number=49):
    """
    Extract features from historical lottery data for machine learning.
    
    Args:
        data (pandas.DataFrame): DataFrame containing historical lottery data
        max_number (int): Maximum possible lottery number
        
    Returns:
        tuple: (features, labels) as pandas DataFrame/Series
    """
    # Create a DataFrame to store features for each draw
    features = pd.DataFrame()
    
    # Track the frequency of each number in the last N draws
    window_sizes = [5, 10, 20, 50]
    for window in window_sizes:
        for num in range(1, max_number + 1):
            # Count occurrences of this number in a sliding window
            col_name = f'freq_{num}_last_{window}'
            features[col_name] = data['numbers'].rolling(window, min_periods=1).apply(
                lambda x: sum(1 for draw in x if num in draw) / len(x)
            )
    
    # Add gap features
    for num in range(1, max_number + 1):
        # Initialize gap counting
        gap = 0
        gaps = []
        
        # Calculate gap for each draw
        for numbers in data['numbers']:
            if num in numbers:
                gaps.append(gap)
                gap = 0
            else:
                gap += 1
                gaps.append(gap)
        
        features[f'gap_{num}'] = gaps
    
    # Add time-based features if date is available
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        features['day_of_week'] = data['date'].dt.dayofweek
        features['month'] = data['date'].dt.month
    
    # Create binary labels for each number (1 if drawn, 0 if not)
    labels = pd.DataFrame()
    for num in range(1, max_number + 1):
        labels[f'num_{num}'] = data['numbers'].apply(lambda x: 1 if num in x else 0)
    
    # Drop the first few rows where we don't have enough history
    features = features.dropna()
    labels = labels.loc[features.index]
    
    return features, labels


def train_model(features, labels):
    """
    Train a machine learning model on historical lottery data.
    
    Args:
        features (pandas.DataFrame): Feature matrix
        labels (pandas.DataFrame): Label matrix
        
    Returns:
        object: Trained machine learning model
    """
    # Use a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(features, labels)
    
    return model


def save_model(model, model_path="models/lottery_model.pkl"):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained machine learning model
        model_path (str): Path to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(model_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path


def load_model(model_path="models/lottery_model.pkl"):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded machine learning model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model