import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LotteryAnalyzer:
    """
    A class to analyze Australian lottery data (Oz Lotto and Powerball).
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the LotteryAnalyzer with data directory.
        
        Args:
            data_dir: Directory to store lottery data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.oz_lotto_file = self.data_dir / 'oz_lotto.csv'
        self.powerball_file = self.data_dir / 'powerball.csv'
        
        # Game configurations
        self.game_configs = {
            'oz_lotto': {
                'main_numbers': 7,
                'main_pool': 47,
                'supplementary_numbers': 2,
                'supp_pool': 45,
                'url': 'https://www.thelott.com/oz-lotto/results'
            },
            'powerball': {
                'main_numbers': 7,
                'main_pool': 35,
                'powerball_number': 1,
                'powerball_pool': 20,
                'url': 'https://www.thelott.com/powerball/results'
            }
        }
    
    def download_historical_data(self, game_type: str) -> pd.DataFrame:
        """
        Download historical lottery data.
        
        Args:
            game_type: Type of lottery game ('oz_lotto' or 'powerball')
            
        Returns:
            DataFrame containing historical draw data
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Scrape the official lottery website
        # 2. Or use their API if available
        # 3. Or download from a reliable data source
        
        print(f"Downloading historical data for {game_type}...")
        # TODO: Implement actual data download
        
        # Return sample data structure
        if game_type == 'oz_lotto':
            return pd.DataFrame({
                'draw_date': pd.date_range(end=datetime.today(), periods=100, freq='W-THU'),
                **{f'number_{i+1}': np.random.randint(1, 48, 100) for i in range(7)},
                **{f'supplementary_{i+1}': np.random.randint(1, 46, 100) for i in range(2)}
            })
        else:  # powerball
            return pd.DataFrame({
                'draw_date': pd.date_range(end=datetime.today(), periods=100, freq='W-THU'),
                **{f'number_{i+1}': np.random.randint(1, 36, 100) for i in range(7)},
                'powerball': np.random.randint(1, 21, 100)
            })
    
    def analyze_frequencies(self, df: pd.DataFrame, game_type: str) -> Dict:
        """
        Analyze number frequencies from historical data.
        
        Args:
            df: DataFrame containing historical draw data
            game_type: Type of lottery game
            
        Returns:
            Dictionary containing frequency analysis
        """
        config = self.game_configs[game_type]
        analysis = {}
        
        # Analyze main numbers
        main_cols = [col for col in df.columns if col.startswith('number_')]
        all_numbers = df[main_cols].values.flatten()
        unique, counts = np.unique(all_numbers, return_counts=True)
        analysis['main_numbers'] = dict(zip(unique, counts))
        
        # For Oz Lotto, analyze supplementary numbers
        if game_type == 'oz_lotto':
            supp_cols = [col for col in df.columns if col.startswith('supplementary_')]
            all_supp = df[supp_cols].values.flatten()
            unique_supp, counts_supp = np.unique(all_supp, return_counts=True)
            analysis['supplementary_numbers'] = dict(zip(unique_supp, counts_supp))
        
        # For Powerball, analyze powerball numbers
        elif game_type == 'powerball' and 'powerball' in df.columns:
            powerball_counts = df['powerball'].value_counts().to_dict()
            analysis['powerball'] = powerball_counts
        
        return analysis
    
    def predict_next_draw(self, df: pd.DataFrame, game_type: str) -> Dict:
        """
        Predict the next draw numbers based on historical data.
        
        Args:
            df: DataFrame containing historical draw data
            game_type: Type of lottery game
            
        Returns:
            Dictionary containing predicted numbers
        """
        config = self.game_configs[game_type]
        prediction = {}
        
        # Simple prediction based on frequency analysis
        main_cols = [col for col in df.columns if col.startswith('number_')]
        all_numbers = df[main_cols].values.flatten()
        unique, counts = np.unique(all_numbers, return_counts=True)
        
        # Get most common numbers
        sorted_indices = np.argsort(counts)[::-1]
        prediction['main_numbers'] = sorted(unique[sorted_indices][:config['main_numbers']])
        
        # For Oz Lotto, predict supplementary numbers
        if game_type == 'oz_lotto':
            supp_cols = [col for col in df.columns if col.startswith('supplementary_')]
            all_supp = df[supp_cols].values.flatten()
            unique_supp, counts_supp = np.unique(all_supp, return_counts=True)
            supp_sorted = np.argsort(counts_supp)[::-1]
            prediction['supplementary_numbers'] = sorted(unique_supp[supp_sorted][:2])
        
        # For Powerball, predict powerball number
        elif game_type == 'powerball' and 'powerball' in df.columns:
            powerball_counts = df['powerball'].value_counts()
            prediction['powerball'] = powerball_counts.idxmax()
        
        return prediction
    
    def save_analysis(self, analysis: Dict, filename: str):
        """Save analysis results to a JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            else:
                return obj

        with open(self.data_dir / filename, 'w') as f:
            json.dump(convert_numpy_types(analysis), f, indent=2)
    
    def plot_frequencies(self, frequencies: Dict, title: str):
        """Plot number frequencies."""
        plt.figure(figsize=(12, 6))
        numbers = list(frequencies.keys())
        counts = list(frequencies.values())
        
        # Sort by number for better visualization
        sorted_indices = np.argsort(numbers)
        numbers = [numbers[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        plt.bar(numbers, counts)
        plt.title(f'Number Frequencies - {title}')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.data_dir / f'frequencies_{title.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved frequency plot to {plot_path}")


def main():
    # Initialize the analyzer
    analyzer = LotteryAnalyzer()
    
    # Analyze Oz Lotto
    print("\n=== Analyzing Oz Lotto ===")
    oz_data = analyzer.download_historical_data('oz_lotto')
    oz_analysis = analyzer.analyze_frequencies(oz_data, 'oz_lotto')
    oz_prediction = analyzer.predict_next_draw(oz_data, 'oz_lotto')
    
    print("\nOz Lotto Prediction:")
    print(f"Main numbers: {oz_prediction['main_numbers']}")
    print(f"Supplementary numbers: {oz_prediction.get('supplementary_numbers', [])}")
    
    # Save and plot analysis
    analyzer.save_analysis(oz_analysis, 'oz_lotto_analysis.json')
    analyzer.plot_frequencies(oz_analysis['main_numbers'], 'Oz Lotto Main Numbers')
    
    # Analyze Powerball
    print("\n=== Analyzing Powerball ===")
    pb_data = analyzer.download_historical_data('powerball')
    pb_analysis = analyzer.analyze_frequencies(pb_data, 'powerball')
    pb_prediction = analyzer.predict_next_draw(pb_data, 'powerball')
    
    print("\nPowerball Prediction:")
    print(f"Main numbers: {pb_prediction['main_numbers']}")
    print(f"Powerball: {pb_prediction.get('powerball', 'N/A')}")
    
    # Save and plot analysis
    analyzer.save_analysis(pb_analysis, 'powerball_analysis.json')
    analyzer.plot_frequencies(pb_analysis['main_numbers'], 'Powerball Main Numbers')
    
    print("\nAnalysis complete! Check the 'data' directory for results.")


if __name__ == "__main__":
    main()
