import re
import logging
import pandas as pd
import numpy as np
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any, Union

# Sklearn Imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class StatisticsModule:
    """Module for analyzing performer statistics with optimized data processing."""
    
    # EU band size range constants
    MIN_EU_BAND = 60
    MAX_EU_BAND = 110
    
    # Cup size related mappings
    BAND_CONVERSION = {
        28: 60, 30: 65, 32: 70, 34: 75, 36: 80, 
        38: 85, 40: 90, 42: 95, 44: 100, 46: 105, 48: 110
    }
    
    CUP_CONVERSION = {
        "A": "A", "B": "B", "C": "C", "D": "D", 
        "DD": "E", "DDD": "F", "DDDD": "G",
        "E": "E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J"
    }
    
    CUP_NUMERIC = {
        "A": 1, "B": 2, "C": 3, "D": 4, 
        "E": 5, "DD": 5, "F": 6, "DDD": 6, 
        "G": 7, "H": 8, "I": 9, "J": 10
    }
    
    # Cup size regex pattern (compiled once)
    CUP_SIZE_PATTERN = re.compile(r'(\d{2,3})([A-KJ-Z]+)')
    
    def __init__(self, stash_client=None):
        """Initialize the statistics module with optional stash client.
        
        Args:
            stash_client: Client for accessing stash data
        """
        self.stash_client = stash_client
        self._performers_data = None
        self._scenes_data = None
        self._cup_size_df = None
        self._o_counter_df = None
        self._ratio_df = None
        
    @property
    def performers_data(self) -> List[Dict]:
        """Lazy-loaded performers data."""
        if self._performers_data is None:
            self._load_data()
        return self._performers_data
    
    @property
    def scenes_data(self) -> List[Dict]:
        """Lazy-loaded scenes data."""
        if self._scenes_data is None:
            self._load_data()
        return self._scenes_data
        
    def _load_data(self) -> None:
        """Load data from Stash client once and cache it."""
        try:
            self._performers_data = self.stash_client.get_performers() if self.stash_client else []
            self._scenes_data = self.stash_client.get_scenes() if self.stash_client else []
            # Reset cached dataframes when data is reloaded
            self._cup_size_df = None
            self._o_counter_df = None
            self._ratio_df = None
            
            # Clear the cache for generate_all_stats
            if hasattr(self, 'generate_all_stats'):
                self.generate_all_stats.cache_clear()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._performers_data = []
            self._scenes_data = []
    
    def _is_eu_band_size(self, band_size: Union[str, int]) -> bool:
        """Check if a band size is in EU format (60-110 cm).
        
        Args:
            band_size: Band size to check
            
        Returns:
            bool: True if the band size is in EU format
        """
        try:
            # Ensure band_size is not None before converting to int
            if band_size is None:
                return False
            band = int(band_size)
            return self.MIN_EU_BAND <= band <= self.MAX_EU_BAND
        except (ValueError, TypeError):
            return False
    
    def _convert_bra_size(self, measurements: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
        """Extract and convert bra sizes to EU (German) format.
        
        Args:
            measurements: Measurement string containing bra size
            
        Returns:
            Tuple[Optional[str], Optional[int]]: EU formatted bra size and numeric cup value
        """
        if not measurements:
            return None, None
        
        # Extract using the compiled pattern
        match = self.CUP_SIZE_PATTERN.search(measurements)
        if not match:
            return None, None
        
        # Ensure we have valid match groups before processing
        try:
            band_size = int(match.group(1))
            cup_letter = match.group(2)
        except (ValueError, IndexError):
            return None, None
        
        # If already in EU format, return as is
        if self._is_eu_band_size(band_size):
            eu_cup = self.CUP_CONVERSION.get(cup_letter, cup_letter)
            return f"{band_size}{eu_cup}", self.CUP_NUMERIC.get(eu_cup, 0)
        
        # Convert US/UK to EU
        eu_band = self.BAND_CONVERSION.get(band_size, round((band_size + 16) / 2) * 5)
        eu_cup = self.CUP_CONVERSION.get(cup_letter, cup_letter)
        
        return f"{eu_band}{eu_cup}", self.CUP_NUMERIC.get(eu_cup, 0)
    
    @property
    def cup_size_df(self) -> pd.DataFrame:
        """Get or create cup size DataFrame with caching."""
        if self._cup_size_df is None:
            self._create_cup_size_df()
        return self._cup_size_df
        
    def _create_cup_size_df(self) -> None:
        """Create the cup size DataFrame for analysis."""
        cup_size_data = []
        
        for performer in self.performers_data:
            measurements = performer.get('measurements')
            eu_bra_size, cup_numeric = self._convert_bra_size(measurements)
            
            if not eu_bra_size:
                continue
                
            # Extract band size and cup letter from EU format
            match = re.match(r'(\d+)([A-KJ-Z]+)', eu_bra_size)
            if not match:
                continue
                
            # Safely extract match groups
            try:
                band_size, cup_letter = match.groups()
            except (ValueError, IndexError):
                continue
            
            # Calculate BMI
            height_cm = performer.get('height_cm', 0)
            weight = performer.get('weight', 0)
            bmi = None
            # Ensure height_cm and weight are not None before calculations
            if height_cm and weight and height_cm > 0 and weight > 0:
                try:
                    height_m = height_cm / 100
                    bmi = round(weight / (height_m * height_m), 1)
                except (TypeError, ZeroDivisionError):
                    bmi = None
            
            # Ensure all values are initialized to safe defaults
            performer_data = {
                'id': performer.get('id', ''),
                'name': performer.get('name', 'Unknown'),
                'cup_size': eu_bra_size,
                'band_size': band_size,
                'cup_letter': cup_letter,
                'cup_numeric': cup_numeric or 0,
                'favorite': bool(performer.get('favorite', False)),
                'height_cm': height_cm or 0,
                'weight': weight or 0,
                'bmi': bmi,
                'measurements': measurements or '',
                'scene_count': performer.get('scene_count', 0) or 0,
                'rating100': performer.get('rating100', 0) or 0,
                'o_counter': performer.get('o_counter', 0) or 0
            }
            
            cup_size_data.append(performer_data)
        
        # Create a DataFrame
        df = pd.DataFrame(cup_size_data)
        
        # Convert numeric columns at once
        if not df.empty:
            numeric_columns = ['band_size', 'cup_numeric', 'height_cm', 'weight', 
                               'scene_count', 'rating100', 'o_counter']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # Replace NaNs with 0 for numeric columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Special handling for BMI which can be None
            df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
            
        self._cup_size_df = df
    
    def get_cup_size_stats(self) -> Dict[str, Any]:
        """Get statistics about cup sizes using EU format.
        
        Returns:
            Dict with cup size statistics
        """
        df = self.cup_size_df
        
        # Count frequencies
        cup_size_counts = Counter(df['cup_size'].tolist()) if not df.empty else Counter()
        
        # Convert DataFrame to dict for JSON serialization
        df_dict = df.to_dict('records') if not df.empty else []
        
        return {
            'cup_size_counts': dict(cup_size_counts),
            'cup_size_dataframe': df_dict
        }
    
    @property
    def o_counter_df(self) -> pd.DataFrame:
        """Get or create o-counter DataFrame with caching."""
        if self._o_counter_df is None:
            self._create_o_counter_df()
        return self._o_counter_df
    
    def _create_o_counter_df(self) -> None:
        """Create the o-counter DataFrame for analysis."""
        o_counter_data = []
        
        for scene in self.scenes_data:
            # Ensure o_counter is not None
            o_counter = scene.get('o_counter', 0)
            if o_counter is None:
                o_counter = 0
                
            performers = scene.get('performers', [])
            
            scene_data = {
                'id': scene.get('id', ''),
                'title': scene.get('title', ''),
                'o_counter': o_counter,
                'performers': [p.get('name', '') for p in performers],
                'performer_ids': [p.get('id', '') for p in performers],
                'favorite_performers': [p.get('name', '') for p in performers if p.get('favorite', False)],
                'tags': [t.get('name', '') for t in scene.get('tags', [])]
            }
            
            o_counter_data.append(scene_data)
        
        self._o_counter_df = pd.DataFrame(o_counter_data)
    
    def get_o_counter_stats(self) -> Dict[str, Any]:
        """Get statistics about o-counter values.
        
        Returns:
            Dict with o-counter statistics
        """
        df = self.o_counter_df
        
        # Get performers with highest o-counter sum
        performer_o_counts = {}
        
        if not df.empty:
            # Create a copy to avoid warnings
            df_copy = df.copy()
            # Filter for rows with o_counter > 0 to improve efficiency
            # Ensure o_counter is numeric before comparison
            df_copy.loc[:, 'o_counter'] = pd.to_numeric(df_copy['o_counter'], errors='coerce').fillna(0)
            o_counter_rows = df_copy[df_copy['o_counter'] > 0]
            
            for _, row in o_counter_rows.iterrows():
                o_count = row['o_counter']
                for performer_id, performer_name in zip(row['performer_ids'], row['performers']):
                    if not performer_id:  # Skip empty IDs
                        continue
                        
                    if performer_id not in performer_o_counts:
                        performer_o_counts[performer_id] = {
                            'name': performer_name,
                            'total_o_count': 0,
                            'scene_count': 0
                        }
                    
                    performer_o_counts[performer_id]['total_o_count'] += o_count
                    performer_o_counts[performer_id]['scene_count'] += 1
        
        # Calculate overall O-counter statistics - only for non-zero values
        o_counter_values = df['o_counter'][df['o_counter'] > 0].tolist() if not df.empty else []
        
        if o_counter_values:
            avg_o_counter = np.mean(o_counter_values)
            median_o_counter = np.median(o_counter_values)
            max_o_counter = max(o_counter_values)
        else:
            avg_o_counter = 0
            median_o_counter = 0
            max_o_counter = 0
            
        # Convert DataFrame to dict for JSON serialization
        df_dict = df.to_dict('records') if not df.empty else []
            
        return {
            'o_counter_dataframe': df_dict,
            'performer_o_counts': performer_o_counts,
            'average_o_counter': avg_o_counter,
            'median_o_counter': median_o_counter,
            'max_o_counter': max_o_counter,
            'total_performers': len(performer_o_counts)
        }
    
    def get_favorite_o_counter_stats(self) -> Dict[str, Any]:
        """Analyze the relationship between favorite status and o-counter values.
        
        Returns:
            Dict with detailed statistics about favorites vs non-favorites
        """
        # Extract performers with o-counter > 0
        # Ensure o_counter is not None before comparison
        performers_with_o = [
            p for p in self.performers_data 
            if p.get('o_counter', 0) is not None and p.get('o_counter', 0) > 0
        ]
        
        # Separate favorites and non-favorites
        favorites = [p for p in performers_with_o if p.get('favorite', False)]
        non_favorites = [p for p in performers_with_o if not p.get('favorite', False)]
        
        # Helper function to calculate stats
        def calculate_stats(performers):
            o_values = [p.get('o_counter', 0) for p in performers]
            return {
                'count': len(performers),
                'avg_o_counter': np.mean(o_values) if o_values else 0,
                'median_o_counter': np.median(o_values) if o_values else 0,
                'max_o_counter': max(o_values) if o_values else 0,
                'performers': [
                    {
                        'name': p.get('name', 'Unknown'),
                        'o_counter': p.get('o_counter', 0),
                        'rating100': p.get('rating100', 0),
                        'scene_count': p.get('scene_count', 0),
                        'measurements': p.get('measurements', 'N/A')
                    }
                    for p in sorted(performers, key=lambda x: x.get('o_counter', 0), reverse=True)
                ]
            }
        
        # Calculate statistics for both groups
        favorite_stats = calculate_stats(favorites)
        non_favorite_stats = calculate_stats(non_favorites)
        
        # Overall statistics
        total_performers = len(performers_with_o)
        favorite_percentage = (len(favorites) / total_performers * 100) if total_performers else 0
        non_favorite_percentage = (len(non_favorites) / total_performers * 100) if total_performers else 0
        
        return {
            'favorite_stats': favorite_stats,
            'non_favorite_stats': non_favorite_stats,
            'overall_stats': {
                'total_performers': total_performers,
                'favorite_percentage': favorite_percentage,
                'non_favorite_percentage': non_favorite_percentage
            }
        }
    
    def get_rating_o_counter_correlation(self) -> Dict[str, Any]:
        """Analyze the correlation between performer ratings and o-counter values.
        
        Returns:
            Dict with correlation statistics
        """
        # Extract performers with rating100 > 0
        # Ensure rating100 is not None before comparison
        performers_with_data = [
            p for p in self.performers_data 
            if p.get('rating100', 0) is not None and p.get('rating100', 0) > 0
        ]
        
        if not performers_with_data:
            return {
                'correlation': 0,
                'high_rated_high_o': [],
                'high_rated_low_o': [],
                'low_rated_high_o': [],
                'rating_o_counter_data': []
            }
        
        # Create DataFrame for analysis - use pandas constructor more efficiently
        # Ensure all values are not None with safe defaults
        rating_data = [{
            'id': p.get('id', ''),
            'name': p.get('name', 'Unknown'),
            'rating100': float(p.get('rating100', 0) or 0),
            'o_counter': float(p.get('o_counter', 0) or 0),
            'favorite': bool(p.get('favorite', False)),
            'scene_count': int(p.get('scene_count', 0) or 0),
            'measurements': p.get('measurements', '')
        } for p in performers_with_data]
        
        df = pd.DataFrame(rating_data)
        
        # Calculate correlation if data exists
        if not df.empty and len(df) > 1:
            correlation = df['rating100'].corr(df['o_counter'])
            correlation = correlation if pd.notna(correlation) else 0
        else:
            correlation = 0
        
        # Identify performers in different categories
        if not df.empty:
            # Use median as threshold for high/low
            rating_median = df['rating100'].median()
            
            # Get median for performers with o_counter > 0
            o_counter_positive = df[df['o_counter'] > 0]
            o_counter_median = o_counter_positive['o_counter'].median() if not o_counter_positive.empty else 1
            
            # Categorize performers
            high_rated_high_o = df[
                (df['rating100'] >= rating_median) & 
                (df['o_counter'] >= o_counter_median)
            ].sort_values('o_counter', ascending=False).to_dict('records')
            
            high_rated_low_o = df[
                (df['rating100'] >= rating_median) & 
                (df['o_counter'] < o_counter_median)
            ].sort_values('rating100', ascending=False).to_dict('records')
            
            low_rated_high_o = df[
                (df['rating100'] < rating_median) & 
                (df['o_counter'] >= o_counter_median)
            ].sort_values('o_counter', ascending=False).to_dict('records')
        else:
            high_rated_high_o = []
            high_rated_low_o = []
            low_rated_high_o = []
        
        return {
            'correlation': correlation,
            'high_rated_high_o': high_rated_high_o,
            'high_rated_low_o': high_rated_low_o,
            'low_rated_high_o': low_rated_high_o,
            'rating_o_counter_data': df.to_dict('records') if not df.empty else []
        }
    
    @property
    def ratio_df(self) -> pd.DataFrame:
        """Get or create ratio DataFrame with caching."""
        if self._ratio_df is None:
            self._create_ratio_df()
        return self._ratio_df
    
    def _create_ratio_df(self) -> None:
        """Create ratio DataFrame for analysis."""
        df = self.cup_size_df.copy()
        
        if df.empty:
            self._ratio_df = pd.DataFrame()
            return
            
        # Cup letter to numeric mapping
        cup_letter_values = {letter: idx+1 for idx, letter in enumerate('ABCDEFGHIJK')}
        
        # Add cup letter numeric value
        df['cup_letter_value'] = df['cup_letter'].map(cup_letter_values)
        
        # Calculate ratios using vectorized operations with safe handling of nulls
        # More robust handling of potential None/NaN values
        
        # Convert to numeric first to handle any string values
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['height_cm'] = pd.to_numeric(df['height_cm'], errors='coerce')
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        df['cup_letter_value'] = pd.to_numeric(df['cup_letter_value'], errors='coerce')
        
        # Fill NaN values with 0 for safer division
        for col in ['bmi', 'height_cm', 'weight', 'cup_letter_value']:
            df[col] = df[col].fillna(0)
        
        # Calculate ratios with safe division
        df['cup_to_bmi'] = np.where(
            (df['bmi'] > 0) & (df['cup_letter_value'] > 0),
            df['cup_letter_value'] / df['bmi'],
            np.nan
        )
        df['cup_to_height'] = np.where(
            (df['height_cm'] > 0) & (df['cup_letter_value'] > 0),
            df['cup_letter_value'] / df['height_cm'],
            np.nan
        )
        df['cup_to_weight'] = np.where(
            (df['weight'] > 0) & (df['cup_letter_value'] > 0),
            df['cup_letter_value'] / df['weight'],
            np.nan
        )
        
        self._ratio_df = df
    
    def get_ratio_stats(self) -> Dict[str, Any]:
        """Calculate various ratios like cup-to-bmi, cup-to-height, cup-to-weight.
        
        Returns:
            Dict with ratio statistics
        """
        ratio_df = self.ratio_df
        
        if ratio_df.empty:
            return {'ratio_dataframe': [], 'ratio_stats': []}
        
        # Make a copy to avoid warnings
        ratio_df_copy = ratio_df.copy()
        
        # Filter to only include performers with o_counter > 0 for ratio statistics
        non_zero_o_count_df = ratio_df_copy[ratio_df_copy['o_counter'] > 0]
        
        # If we have no performers with non-zero o-count, fall back to using all performers
        analysis_df = non_zero_o_count_df if not non_zero_o_count_df.empty else ratio_df_copy
        
        # Calculate all statistics in a single groupby operation
        try:
            # Handle potential errors in groupby operation
            ratio_stats = (analysis_df.groupby('cup_letter')
                .agg({
                    'cup_to_bmi': ['mean', 'median'],
                    'cup_to_height': ['mean', 'median'],
                    'cup_to_weight': ['mean', 'median'],
                    'id': ['count']
                })
            )
            
            # Flatten the MultiIndex columns
            ratio_stats.columns = ['_'.join(col).strip('_') for col in ratio_stats.columns.values]
            ratio_stats = ratio_stats.reset_index()
            
            # Rename columns for clarity
            ratio_stats = ratio_stats.rename(columns={
                'cup_to_bmi_mean': 'avg_cup_to_bmi',
                'cup_to_height_mean': 'avg_cup_to_height',
                'cup_to_weight_mean': 'avg_cup_to_weight',
                'id_count': 'performer_count'
            })
            
            ratio_stats_dict = ratio_stats.to_dict('records')
        except Exception as e:
            logger.error(f"Error in ratio_stats calculation: {e}")
            ratio_stats_dict = []
        
        # Convert DataFrame to dict for JSON serialization
        ratio_df_dict = ratio_df.to_dict('records')
        
        return {
            'ratio_dataframe': ratio_df_dict,
            'ratio_stats': ratio_stats_dict
        }
    
    def get_cup_size_o_counter_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between cup size and o-counter.
        
        Returns:
            Dict with cup size and o-counter correlation statistics
        """
        cup_df = self.cup_size_df
        performer_o_counts = self.get_o_counter_stats()['performer_o_counts']
        ratio_df = self.ratio_df
    
        if cup_df.empty:
            return {
                'cup_size_o_counter_df': [],
                'cup_letter_o_stats': []
            }
            
        # Create a copy to avoid modifying the cached dataframe
        analysis_df = cup_df.copy()
        
        # Merge with ratio_df to include cup_to_bmi if available
        if not ratio_df.empty and 'cup_to_bmi' in ratio_df.columns:
            analysis_df = analysis_df.merge(
                ratio_df[['id', 'cup_to_bmi']], 
                on='id', 
                how='left'
            )
        
        # Add o-counter data using vectorized operations with safe handling for None values
        # Handle None values in the mapping function
        analysis_df['total_o_count'] = analysis_df['id'].map(
            lambda pid: performer_o_counts.get(pid, {}).get('total_o_count', 0) or 0
        )
        analysis_df['o_scene_count'] = analysis_df['id'].map(
            lambda pid: performer_o_counts.get(pid, {}).get('scene_count', 0) or 0
        )
        
        # Fill NaN values with 0 to ensure safe comparisons
        analysis_df['total_o_count'] = analysis_df['total_o_count'].fillna(0)
        
        # Filter to only include performers with o_counter > 0 for statistics
        non_zero_o_count_df = analysis_df[analysis_df['total_o_count'] > 0]
        
        if non_zero_o_count_df.empty:
            return {
                'cup_size_o_counter_df': analysis_df.to_dict('records'),
                'cup_letter_o_stats': []
            }
        
        try:
            # Calculate statistics with groupby
            # For non-zero values only
            stats = non_zero_o_count_df.groupby('cup_letter').agg({
                'total_o_count': ['mean', 'median'],
                'id': 'count'
            })
            
            # Flatten MultiIndex columns
            stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
            stats = stats.reset_index()
            
            # Rename columns for clarity
            stats = stats.rename(columns={
                'total_o_count_mean': 'avg_o_count',
                'total_o_count_median': 'median_o_count',
                'id_count': 'performer_count'
            })
            
            # Calculate cup_to_bmi median if the column exists
            if 'cup_to_bmi' in non_zero_o_count_df.columns:
                cup_to_bmi_stats = non_zero_o_count_df.groupby('cup_letter').agg({
                    'cup_to_bmi': 'median'
                }).reset_index()
                
                cup_to_bmi_stats = cup_to_bmi_stats.rename(columns={
                    'cup_to_bmi': 'median_cup_to_bmi'
                })
                
                # Merge with main stats
                stats = stats.merge(cup_to_bmi_stats, on='cup_letter', how='left')
            
            # Calculate count and percentage of performers with o-counter > 0
            o_count_stats = analysis_df.groupby('cup_letter').apply(
                lambda x: (x['total_o_count'] > 0).sum()
            ).reset_index(name='performers_with_o_count')
            
            total_performers_by_cup = analysis_df.groupby('cup_letter').size().reset_index(name='total_performers')
            
            # Merge all stats
            cup_letter_stats = stats.merge(o_count_stats, on='cup_letter')
            cup_letter_stats = cup_letter_stats.merge(total_performers_by_cup, on='cup_letter')
            
            # Calculate percentage with vectorized operation and safe division
            cup_letter_stats['pct_with_o_count'] = np.where(
                cup_letter_stats['total_performers'] > 0,
                cup_letter_stats['performers_with_o_count'] / cup_letter_stats['total_performers'] * 100,
                0
            )
        
            # Sort by average o-count for consistent output
            cup_letter_stats = cup_letter_stats.sort_values('avg_o_count', ascending=False)
            cup_letter_stats_dict = cup_letter_stats.to_dict('records')
        except Exception as e:
            logger.error(f"Error in cup size correlation calculation: {e}")
            cup_letter_stats_dict = []
        
        # Convert DataFrame to dict for JSON serialization
        analysis_df_dict = analysis_df.to_dict('records')
        
        return {
            'cup_size_o_counter_df': analysis_df_dict,
            'cup_letter_o_stats': cup_letter_stats_dict
        }
    
    def get_top_o_counter_performers(self, top_n=10) -> List[Dict]:
        """Get top performers based on o-counter.
        
        Args:
            top_n: Number of top performers to return
            
        Returns:
            List of top performer details
        """
        # Filter performers with O-Counter > 0
        # Ensure o_counter is not None before comparison
        performers_with_o_counter = [
            p for p in self.performers_data 
            if p.get('o_counter', 0) is not None and p.get('o_counter', 0) > 0
        ]
        
        # Sort by O-Counter descending
        top_performers = sorted(
            performers_with_o_counter, 
            key=lambda x: x.get('o_counter', 0), 
            reverse=True
        )[:top_n]
        
        # Prepare details
        top_o_counter_details = []
        for performer in top_performers:
            # Get EU cup size
            eu_cup_size, _ = self._convert_bra_size(performer.get('measurements'))
            
            top_o_counter_details.append({
                'name': performer.get('name', 'Unknown'),
                'o_counter': performer.get('o_counter', 0),
                'measurements': performer.get('measurements', 'N/A'),
                'eu_cup_size': eu_cup_size,
                'scene_count': performer.get('scene_count', 0),
                'id': performer.get('id')
            })
        
        return top_o_counter_details
    
    def create_preference_profile(self, feature_weights=None) -> Dict[str, Any]:
        """Create a detailed profile of user preferences with configurable feature weights.
        
        Args:
            feature_weights: Optional dictionary of feature weights to override defaults
            
        Returns:
            Dict with preference profile details
        """
        # Default weights
        default_weights = {
            'o_counter': 2.0,
            'rating100': 1.5,
            'height_cm': 0.5,
            'weight': 0.5,
            'eu_cup_numeric': 1.0
        }
        
        # Update with custom weights if provided
        if feature_weights:
            default_weights.update(feature_weights)
        
        # Combine favorites and performers with O-Counter > 1
        # Ensure o_counter is not None before comparison
        relevant_performers = [
            p for p in self.performers_data 
            if p.get('favorite', False) or (p.get('o_counter', 0) is not None and p.get('o_counter', 0) > 1)
        ]
        
        if not relevant_performers:
            return {
                'feature_weights': default_weights,
                'preference_profile': {
                    'total_relevant_performers': 0,
                    'avg_o_counter': 0,
                    'avg_rating': 0,
                    'most_common_cup_sizes': []
                },
                'cluster_analysis': {'clusters': {}, 'cluster_centroids': []},
                'cup_size_distribution': {'total_cup_sizes': {}, 'relevant_cup_size_distribution': {}},
                'top_performers_by_cluster': {}
            }
        
        # Use cup_size_df for cup size data
        cup_df = self.cup_size_df
        
        # Create performer data for clustering efficiently
        performer_data = []
        relevant_cup_sizes = []
        
        for p in relevant_performers:
            p_id = p.get('id')
            o_counter = float(p.get('o_counter', 0) or 0)
            
            # Skip if no id
            if not p_id:
                continue
                
            # Find cup size data in cup_df
            cup_data = cup_df[cup_df['id'] == p_id]
            cup_numeric = float(cup_data['cup_numeric'].values[0] if not cup_data.empty else 0)
            
            # For cup size distribution
            if o_counter > 0:
                eu_cup_size, _ = self._convert_bra_size(p.get('measurements'))
                if eu_cup_size:
                    relevant_cup_sizes.append(eu_cup_size)
            
            performer_data.append({
                'o_counter': o_counter,
                'rating100': float(p.get('rating100', 0) or 0),
                'height_cm': float(p.get('height_cm', 0) or 0),
                'weight': float(p.get('weight', 0) or 0),
                'eu_cup_numeric': cup_numeric,
                'name': p.get('name', 'Unknown')
            })
        
        # Create DataFrame from collected data
        df = pd.DataFrame(performer_data)
        
        # Skip clustering if not enough data
        if len(df) < 3:
            return {
                'feature_weights': default_weights,
                'preference_profile': {
                    'total_relevant_performers': 0,
                    'avg_o_counter': 0,
                    'avg_rating': 0,
                    'most_common_cup_sizes': []
                },
                'cluster_analysis': {'clusters': {}, 'cluster_centroids': []},
                'cup_size_distribution': {'total_cup_sizes': {}, 'relevant_cup_size_distribution': {}},
                'top_performers_by_cluster': {}
            }
        
        # Features for clustering
        features = list(default_weights.keys())
        
        # Only use features that exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        # Preprocessing Pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Filter rows with insufficient data
        X = df[features].dropna(thresh=len(features)//2)
        
        # If insufficient data after filtering, return empty results
        if len(X) < 3:
            return {
                'feature_weights': default_weights,
                'preference_profile': {
                    'total_relevant_performers': len(relevant_performers),
                    'avg_o_counter': df['o_counter'].mean() if not df.empty else 0,
                    'avg_rating': df['rating100'].mean() if not df.empty else 0,
                    'most_common_cup_sizes': Counter(relevant_cup_sizes).most_common(3)
                },
                'cluster_analysis': {'clusters': {}, 'cluster_centroids': []},
                'cup_size_distribution': {'total_cup_sizes': {}, 'relevant_cup_size_distribution': Counter(relevant_cup_sizes)},
                'top_performers_by_cluster': {}
            }
        
        try:
            # Preprocessing
            X_processed = preprocessor.fit_transform(X)
            
            # Apply feature weights
            weighted_features = np.copy(X_processed)
            for i, feature in enumerate(features):
                weighted_features[:, i] *= default_weights[feature]
            
            # K-Means Clustering
            kmeans = KMeans(n_clusters=min(3, len(X)), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(weighted_features)
            
            # Add cluster assignments back to dataframe - create a copy to avoid warnings
            df_with_index = df.reset_index().copy()
            indices = X.index
            df_with_index.loc[indices, 'cluster'] = clusters
            
            # Cluster details with performer names
            cluster_details = {}
            for cluster in range(kmeans.n_clusters):
                cluster_performers = df_with_index[df_with_index['cluster'] == cluster]['name'].tolist()
                cluster_details[cluster] = {
                    'performers': cluster_performers,
                    'count': len(cluster_performers)
                }
            
            # Calculate average values for performers with o_counter > 0
            # Ensure o_counter is numeric before filtering - use a copy and .loc
            df_copy = df.copy()
            df_copy.loc[:, 'o_counter'] = pd.to_numeric(df_copy['o_counter'], errors='coerce').fillna(0)
            o_count_df = df_copy[df_copy['o_counter'] > 0]
            avg_o_counter = o_count_df['o_counter'].mean() if not o_count_df.empty else 0
            avg_rating = o_count_df['rating100'].mean() if not o_count_df.empty else 0
            
            # Cup-Size frequencies using Counter
            cup_size_counter = Counter(relevant_cup_sizes)
            most_common_cup_sizes = cup_size_counter.most_common(3)
            
            # Get top performers by cluster
            top_performers_by_cluster = {}
            for cluster in range(kmeans.n_clusters):
                # Use .loc to avoid SettingWithCopyWarning
                cluster_df = df_with_index[df_with_index['cluster'] == cluster].copy()
                # Ensure o_counter is numeric before filtering
                cluster_df.loc[:, 'o_counter'] = pd.to_numeric(cluster_df['o_counter'], errors='coerce').fillna(0)
                o_counter_df = cluster_df[cluster_df['o_counter'] > 0]
                
                if not o_counter_df.empty:
                    top_performers = o_counter_df.nlargest(5, 'o_counter')[['name', 'o_counter', 'rating100']]
                    top_performers_by_cluster[cluster] = top_performers.to_dict('records')
                else:
                    top_performers_by_cluster[cluster] = []
            
            # Get cup size counts from the cached data
            cup_size_stats = self.get_cup_size_stats()
            cup_size_counts = cup_size_stats['cup_size_counts']
            
            return {
                'feature_weights': default_weights,
                'preference_profile': {
                    'total_relevant_performers': len([p for p in relevant_performers if p.get('o_counter', 0) is not None and p.get('o_counter', 0) > 0]),
                    'avg_o_counter': avg_o_counter,
                    'avg_rating': avg_rating,
                    'most_common_cup_sizes': [
                        {'size': size, 'count': count} 
                        for size, count in most_common_cup_sizes
                    ]
                },
                'cluster_analysis': {
                    'clusters': cluster_details,
                    'cluster_centroids': kmeans.cluster_centers_.tolist()
                },
                'cup_size_distribution': {
                    'total_cup_sizes': cup_size_counts,
                    'relevant_cup_size_distribution': dict(cup_size_counter)
                },
                'top_performers_by_cluster': top_performers_by_cluster
            }
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {
                'feature_weights': default_weights,
                'preference_profile': {
                    'total_relevant_performers': len(relevant_performers),
                    'avg_o_counter': df['o_counter'].mean() if not df.empty else 0,
                    'avg_rating': df['rating100'].mean() if not df.empty else 0,
                    'most_common_cup_sizes': [
                        {'size': size, 'count': count} 
                        for size, count in Counter(relevant_cup_sizes).most_common(3)
                    ]
                },
                'cluster_analysis': {'clusters': {}, 'cluster_centroids': []},
                'cup_size_distribution': {
                    'total_cup_sizes': cup_size_stats['cup_size_counts'] if 'cup_size_stats' in locals() else {},
                    'relevant_cup_size_distribution': dict(cup_size_counter)
                },
                'top_performers_by_cluster': {}
            }
    
    @lru_cache(maxsize=1)
    def generate_all_stats(self) -> Dict[str, Any]: 
        """Generate all statistics and return them in a single dictionary.
        
        Returns:
            Dict containing all statistics
        """
        # Collect all statistics in one call
        cup_size_stats = self.get_cup_size_stats()
        o_counter_stats = self.get_o_counter_stats()
        favorite_o_counter_stats = self.get_favorite_o_counter_stats()
        rating_o_counter_correlation = self.get_rating_o_counter_correlation()
        ratio_stats = self.get_ratio_stats()
        cup_size_o_counter_correlation = self.get_cup_size_o_counter_correlation()
        top_o_counter_performers = self.get_top_o_counter_performers()
        preference_profile = self.create_preference_profile()
        
        # Convert any DataFrame objects to dictionaries for JSON serialization
        if 'cup_size_dataframe' in cup_size_stats:
            if isinstance(cup_size_stats['cup_size_dataframe'], pd.DataFrame):
                cup_size_stats['cup_size_dataframe'] = cup_size_stats['cup_size_dataframe'].to_dict('records')
                
        if 'o_counter_dataframe' in o_counter_stats:
            if isinstance(o_counter_stats['o_counter_dataframe'], pd.DataFrame):
                o_counter_stats['o_counter_dataframe'] = o_counter_stats['o_counter_dataframe'].to_dict('records')
                
        if 'ratio_dataframe' in ratio_stats:
            if isinstance(ratio_stats['ratio_dataframe'], pd.DataFrame):
                ratio_stats['ratio_dataframe'] = ratio_stats['ratio_dataframe'].to_dict('records')
                
        if 'cup_size_o_counter_df' in cup_size_o_counter_correlation:
            if isinstance(cup_size_o_counter_correlation['cup_size_o_counter_df'], pd.DataFrame):
                cup_size_o_counter_correlation['cup_size_o_counter_df'] = cup_size_o_counter_correlation['cup_size_o_counter_df'].to_dict('records')
    
        # Return combined stats
        return {
            'cup_size_stats': cup_size_stats,
            'o_counter_stats': o_counter_stats,
            'favorite_o_counter_stats': favorite_o_counter_stats,
            'rating_o_counter_correlation': rating_o_counter_correlation,
            'ratio_stats': ratio_stats,
            'cup_size_o_counter_correlation': cup_size_o_counter_correlation,
            'top_o_counter_performers': top_o_counter_performers,
            'preference_profile': preference_profile
        }
