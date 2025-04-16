import re
import pandas as pd
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class StatisticsModule:
    def __init__(self, stash_client=None):
        """Initialize the statistics module"""
        self.stash_client = stash_client
        self.performers_data = None
        self.scenes_data = None
        self.cup_size_pattern = re.compile(r'(\d{2,3})([A-K])')
    
    def _load_data(self):
        """Load data from Stash"""
        try:
            self.performers_data = self.stash_client.get_performers()
            self.scenes_data = self.stash_client.get_scenes()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.performers_data = []
            self.scenes_data = []
    
    def _extract_cup_size(self, measurements):
        """Extract cup size from measurements string"""
        if not measurements:
            return None, None
            
        match = self.cup_size_pattern.search(measurements)
        if match:
            band_size = match.group(1)
            cup_letter = match.group(2)
            return band_size, cup_letter
        return None, None
    
    def generate_all_stats(self):
        """Generate all statistics"""
        # Ensure data is loaded
        if not self.performers_data or not self.scenes_data:
            self._load_data()
        
        # Generate individual stat components
        stats = {
            'cup_size_stats': self._get_cup_size_stats(),
            'o_counter_stats': self._get_o_counter_stats(),
            'correlation_stats': self._get_cup_size_o_counter_correlation(),
            'ratio_stats': self._get_ratio_stats()
        }
        
        # Convert DataFrames to dictionaries for JSON serialization
        for key, value in stats.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, pd.DataFrame):
                        stats[key][subkey] = subvalue.to_dict(orient='records')
        
        return stats
    
    def _get_cup_size_stats(self):
        """Get statistics about cup sizes"""
        cup_sizes = []
        cup_size_data = []
        
        for performer in self.performers_data:
            measurements = performer.get('measurements')
            band_size, cup_letter = self._extract_cup_size(measurements)
            
            if band_size and cup_letter:
                cup_size = f"{band_size}{cup_letter}"
                cup_sizes.append(cup_size)
                
                cup_size_data.append({
                    'id': performer.get('id'),
                    'name': performer.get('name'),
                    'cup_size': cup_size,
                    'band_size': band_size,
                    'cup_letter': cup_letter,
                    'favorite': performer.get('favorite', False),
                    'height_cm': performer.get('height_cm'),
                    'weight': performer.get('weight'),
                    'measurements': measurements
                })
        
        # Create a DataFrame for easier analysis
        df = pd.DataFrame(cup_size_data)
        
        # Convert numeric columns
        if not df.empty:
            df['band_size'] = pd.to_numeric(df['band_size'], errors='coerce')
            df['height_cm'] = pd.to_numeric(df['height_cm'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        
        # Count frequencies
        cup_size_counts = Counter(cup_sizes)
        
        return {
            'cup_size_counts': dict(cup_size_counts),
            'cup_size_dataframe': df
        }
    
    def _get_o_counter_stats(self):
        """Get statistics about o-counter values"""
        o_counter_data = []
        
        for scene in self.scenes_data:
            o_counter = scene.get('o_counter', 0)
            performers = scene.get('performers', [])
            
            scene_data = {
                'id': scene.get('id'),
                'title': scene.get('title'),
                'o_counter': o_counter,
                'performers': [p.get('name') for p in performers],
                'performer_ids': [p.get('id') for p in performers],
                'tags': [t.get('name') for t in scene.get('tags', [])]
            }
            
            o_counter_data.append(scene_data)
        
        # Create a DataFrame
        df = pd.DataFrame(o_counter_data)
        
        return {
            'o_counter_dataframe': df
        }
    
    def _get_cup_size_o_counter_correlation(self):
        """Analyze correlation between cup size and o-counter"""
        # Get cup size and o-counter stats
        cup_stats = self._get_cup_size_stats()
        o_stats = self._get_o_counter_stats()
        
        cup_df = cup_stats['cup_size_dataframe']
        o_df = o_stats['o_counter_dataframe']
        
        # Calculate o-counter per performer
        performer_o_counts = {}
        for _, row in o_df.iterrows():
            o_count = row['o_counter']
            performer_ids = row['performer_ids']
            
            for performer_id in performer_ids:
                if performer_id not in performer_o_counts:
                    performer_o_counts[performer_id] = {
                        'total_o_count': 0,
                        'scene_count': 0
                    }
                
                performer_o_counts[performer_id]['total_o_count'] += o_count
                performer_o_counts[performer_id]['scene_count'] += 1
        
        # Add o-counter to cup size dataframe
        if not cup_df.empty:
            def get_o_count(performer_id):
                return performer_o_counts.get(performer_id, {}).get('total_o_count', 0)
            
            def get_o_scene_count(performer_id):
                return performer_o_counts.get(performer_id, {}).get('scene_count', 0)
            
            cup_df['total_o_count'] = cup_df['id'].apply(get_o_count)
            cup_df['o_scene_count'] = cup_df['id'].apply(get_o_scene_count)
            
            # Group by cup letter and calculate statistics
            cup_letter_stats = cup_df.groupby('cup_letter').agg({
                'total_o_count': 'mean',
                'id': 'count'
            }).reset_index()
            
            cup_letter_stats.columns = ['cup_letter', 'avg_o_count', 'performer_count']
            cup_letter_stats = cup_letter_stats.sort_values('cup_letter')
            
            return {
                'cup_size_o_counter_df': cup_df,
                'cup_letter_o_stats': cup_letter_stats
            }
        
        return {
            'cup_size_o_counter_df': pd.DataFrame(),
            'cup_letter_o_stats': []
        }
    
    def _get_ratio_stats(self):
        """Calculate various ratios like cup-to-bmi, cup-to-height, cup-to-weight"""
        # Get cup size statistics
        cup_stats = self._get_cup_size_stats()
        cup_df = cup_stats['cup_size_dataframe']
        
        if cup_df.empty:
            return {}
        
        # Numeric cup letter values 
        cup_letter_values = {letter: idx+1 for idx, letter in enumerate('ABCDEFGHIJK')}
        
        # Calculate ratios
        cup_df['cup_letter_value'] = cup_df['cup_letter'].map(cup_letter_values)
        
        # BMI calculation
        cup_df['bmi'] = np.where(
            (cup_df['height_cm'].notna() & cup_df['weight'].notna()),
            cup_df['weight'] / ((cup_df['height_cm'] / 100) ** 2),
            np.nan
        )
        
        # Cup size to various metrics ratios
        cup_df['cup_to_bmi'] = np.where(
            cup_df['bmi'].notna(), 
            cup_df['cup_letter_value'] / cup_df['bmi'],
            np.nan
        )
        cup_df['cup_to_height'] = np.where(
            cup_df['height_cm'].notna(),
            cup_df['cup_letter_value'] / cup_df['height_cm'],
            np.nan
        )
        cup_df['cup_to_weight'] = np.where(
            cup_df['weight'].notna(),
            cup_df['cup_letter_value'] / cup_df['weight'],
            np.nan
        )
        
        # Group by cup letter and calculate average ratios
        ratio_stats = cup_df.groupby('cup_letter').agg({
            'cup_to_bmi': 'mean',
            'cup_to_height': 'mean',
            'cup_to_weight': 'mean',
            'id': 'count'
        }).reset_index()
        
        ratio_stats.columns = ['cup_letter', 'avg_cup_to_bmi', 'avg_cup_to_height', 
                              'avg_cup_to_weight', 'performer_count']
        
        return {
            'ratio_dataframe': cup_df,
            'ratio_stats': ratio_stats
        }
