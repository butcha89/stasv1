import re
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Füge das übergeordnete Verzeichnis zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stash_api import StashClient

class StatisticsModule:
    def __init__(self, stash_client=None):
        """Initialize the statistics module"""
        self.stash_client = stash_client or StashClient()
        self.performers_data = None
        self.scenes_data = None
        self.cup_size_pattern = re.compile(r'(\d{2,3})([A-K])')
        
    def load_data(self):
        """Load data from Stash"""
        self.performers_data = self.stash_client.get_performers()
        self.scenes_data = self.stash_client.get_scenes()
        
    def extract_cup_size(self, measurements):
        """Extract cup size from measurements string"""
        if not measurements:
            return None, None
            
        match = self.cup_size_pattern.search(measurements)
        if match:
            band_size = match.group(1)
            cup_letter = match.group(2)
            return band_size, cup_letter
        return None, None
    
    def get_cup_size_stats(self):
        """Get statistics about cup sizes"""
        if not self.performers_data:
            self.load_data()
            
        cup_sizes = []
        cup_size_data = []
        
        for performer in self.performers_data:
            measurements = performer.get('measurements')
            band_size, cup_letter = self.extract_cup_size(measurements)
            
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
            
            # Calculate BMI where possible
            df['bmi'] = np.where(
                (df['height_cm'].notna() & df['weight'].notna()),
                df['weight'] / ((df['height_cm'] / 100) ** 2),
                np.nan
            )
        
        # Count frequencies
        cup_size_counts = Counter(cup_sizes)
        
        return {
            'cup_size_counts': dict(cup_size_counts),
            'cup_size_dataframe': df
        }
    
    def get_o_counter_stats(self):
        """Get statistics about o-counter values"""
        if not self.scenes_data:
            self.load_data()
            
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
                'favorite_performers': [p.get('name') for p in performers if p.get('favorite', False)],
                'tags': [t.get('name') for t in scene.get('tags', [])]
            }
            
            o_counter_data.append(scene_data)
        
        # Create a DataFrame
        df = pd.DataFrame(o_counter_data)
        
        # Get performers with highest o-counter sum
        performer_o_counts = {}
        
        if not df.empty:
            for _, row in df.iterrows():
                o_count = row['o_counter']
                if o_count > 0:
                    for performer_id, performer_name in zip(row['performer_ids'], row['performers']):
                        if performer_id not in performer_o_counts:
                            performer_o_counts[performer_id] = {
                                'name': performer_name,
                                'total_o_count': 0,
                                'scene_count': 0
                            }
                        
                        performer_o_counts[performer_id]['total_o_count'] += o_count
                        performer_o_counts[performer_id]['scene_count'] += 1
        
        return {
            'o_counter_dataframe': df,
            'performer_o_counts': performer_o_counts
        }
    
    def get_cup_size_o_counter_correlation(self):
        """Analyze correlation between cup size and o-counter"""
        cup_stats = self.get_cup_size_stats()
        o_stats = self.get_o_counter_stats()
        
        cup_df = cup_stats['cup_size_dataframe']
        performer_o_counts = o_stats['performer_o_counts']
        
        # Add o-counter data to cup size dataframe
        if not cup_df.empty:
            cup_df['total_o_count'] = cup_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('total_o_count', 0)
            )
            cup_df['o_scene_count'] = cup_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('scene_count', 0)
            )
            
            # Group by cup letter and calculate average o-count
            cup_letter_stats = cup_df.groupby('cup_letter').agg({
                'total_o_count': 'mean',
                'id': 'count'
            }).reset_index()
            
            cup_letter_stats.columns = ['cup_letter', 'avg_o_count', 'performer_count']
            cup_letter_stats = cup_letter_stats.sort_values('cup_letter')
            
            return {
                'cup_size_o_counter_df': cup_df,
                'cup_letter_o_stats': cup_letter_stats.to_dict('records')
            }
        
        return {
            'cup_size_o_counter_df': pd.DataFrame(),
            'cup_letter_o_stats': []
        }
    
    def get_ratio_stats(self):
        """Calculate various ratios like cup-to-bmi, cup-to-height, cup-to-weight"""
        cup_stats = self.get_cup_size_stats()
        cup_df = cup_stats['cup_size_dataframe']
        
        if cup_df.empty:
            return {}
            
        # Map cup letters to numeric values (A=1, B=2, etc.)
        cup_letter_values = {letter: idx+1 for idx, letter in enumerate('ABCDEFGHIJK')}
        
        cup_df['cup_letter_value'] = cup_df['cup_letter'].map(cup_letter_values)
        
        # Calculate ratios
        cup_df['cup_to_bmi'] = cup_df['cup_letter_value'] / cup_df['bmi']
        cup_df['cup_to_height'] = cup_df['cup_letter_value'] / cup_df['height_cm']
        cup_df['cup_to_weight'] = cup_df['cup_letter_value'] / cup_df['weight']
        
        # Group by cup letter
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
            'ratio_stats': ratio_stats.to_dict('records')
        }
    
    def generate_all_stats(self):
        """Generate all statistics"""
        cup_size_stats = self.get_cup_size_stats()
        o_counter_stats = self.get_o_counter_stats()
        correlation_stats = self.get_cup_size_o_counter_correlation()
        ratio_stats = self.get_ratio_stats()
        
        return {
            'cup_size_stats': cup_size_stats,
            'o_counter_stats': o_counter_stats,
            'correlation_stats': correlation_stats,
            'ratio_stats': ratio_stats
        }
    
    def plot_cup_size_distribution(self, save_path=None):
        """Plot cup size distribution"""
        cup_stats = self.get_cup_size_stats()
        cup_counts = cup_stats['cup_size_counts']
        
        if not cup_counts:
            return None
            
        plt.figure(figsize=(12, 6))
        
        # Sort by cup size
        sorted_cups = sorted(cup_counts.items(), 
                            key=lambda x: (int(x[0][:-1]), x[0][-1]))
        
        cups, counts = zip(*sorted_cups)
        
        plt.bar(cups, counts)
        plt.title('Cup Size Distribution')
        plt.xlabel('Cup Size')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        
        return plt
    
    def plot_o_counter_by_cup(self, save_path=None):
        """Plot o-counter by cup size"""
        corr_stats = self.get_cup_size_o_counter_correlation()
        cup_letter_stats = corr_stats.get('cup_letter_o_stats', [])
        
        if not cup_letter_stats:
            return None
            
        # Convert to DataFrame for plotting
        df = pd.DataFrame(cup_letter_stats)
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        ax = sns.barplot(x='cup_letter', y='avg_o_count', data=df)
        
        # Add performer count as text
        for i, row in enumerate(cup_letter_stats):
            ax.text(i, row['avg_o_count'] + 0.1, 
                   f"n={row['performer_count']}", 
                   ha='center')
        
        plt.title('Average O-Counter by Cup Size')
        plt.xlabel('Cup Letter')
        plt.ylabel('Average O-Counter')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        
        return plt
