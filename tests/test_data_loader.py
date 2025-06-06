"""Tests for the data_loader module."""

import unittest
import os
import pandas as pd
import tempfile
from datetime import datetime

from lotto_prediction_system import data_loader


class TestDataLoader(unittest.TestCase):
    """Test cases for the data_loader module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a sample CSV file
        self.sample_csv_path = os.path.join(self.temp_dir.name, "sample_data.csv")
        self.create_sample_csv()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def create_sample_csv(self):
        """Create a sample CSV file for testing."""
        data = {
            'Date': ['2023-01-01', '2023-01-08', '2023-01-15'],
            'Number1': [1, 7, 13],
            'Number2': [10, 15, 20],
            'Number3': [23, 25, 30],
            'Number4': [35, 36, 40],
            'Number5': [42, 45, 47],
            'Number6': [49, 48, 49],
            'Bonus': [7, 8, 9]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.sample_csv_path, index=False)
    
    def test_load_data(self):
        """Test that load_data correctly loads and processes CSV data."""
        # Load the data
        data = data_loader.load_data(self.sample_csv_path)
        
        # Check that the data is a DataFrame
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check that required columns are present
        self.assertIn('date', data.columns)
        self.assertIn('numbers', data.columns)
        self.assertIn('bonus', data.columns)
        
        # Check that date is parsed correctly
        self.assertIsInstance(data['date'].iloc[0], pd.Timestamp)
        
        # Check that numbers are parsed as lists
        self.assertIsInstance(data['numbers'].iloc[0], list)
        
        # Check that the first row has expected values
        self.assertEqual(data['numbers'].iloc[0], [1, 10, 23, 35, 42, 49])
        self.assertEqual(data['bonus'].iloc[0], 7)
    
    def test_load_data_file_not_found(self):
        """Test that load_data raises an error when the file is not found."""
        with self.assertRaises(FileNotFoundError):
            data_loader.load_data("nonexistent_file.csv")
    
    def test_load_data_unsupported_format(self):
        """Test that load_data raises an error for unsupported file formats."""
        # Create a temporary file with unsupported extension
        temp_file = os.path.join(self.temp_dir.name, "data.txt")
        with open(temp_file, 'w') as f:
            f.write("test data")
        
        with self.assertRaises(ValueError):
            data_loader.load_data(temp_file)
    
    def test_save_data(self):
        """Test that save_data correctly saves data to a CSV file."""
        # Create a DataFrame to save
        data = {
            'date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-08')],
            'numbers': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            'bonus': [7, 8]
        }
        df = pd.DataFrame(data)
        
        # Define a path to save the data
        save_path = os.path.join(self.temp_dir.name, "saved_data.csv")
        
        # Save the data
        result_path = data_loader.save_data(df, save_path)
        
        # Check that the function returns the correct path
        self.assertEqual(result_path, save_path)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Load the saved data and check it matches the original
        loaded_df = pd.read_csv(save_path)
        self.assertEqual(len(loaded_df), len(df))


if __name__ == '__main__':
    unittest.main()