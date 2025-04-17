import re
import pandas as pd
import numpy as np
import logging
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any, Union

from modules.base_module import BaseModule
from modules import stats_utils

logger = logging.getLogger(__name__)

class CupSizeModule(BaseModule):
    """Module for analyzing cup sizes and bra measurements."""
    
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
        "G": 7, "DDDD": 7, "H": 8, "I": 9, "J": 10
    }
    
    def __init__(self, stash_client=None):
        """Initialize the cup size module.
        
        Args:
            stash_client: Client for accessing stash data
        """
        super().__init__(stash_client)
        self._cup_size_df = None
        # Compile the regex pattern once during initialization
        self.CUP_SIZE_PATTERN = re.compile(r'(\d{2,3})([A-KJ-Z]+)')
        
    def _is_eu_band_size(self, band_size: Union[str, int, None]) -> bool:
        """Check if a band size is in EU format (60-110 cm).
        
        Args:
            band_size: Band size to check
            
        Returns:
            bool: True if the band size is in EU format
        """
        # Handle None values explicitly
        if band_size is None:
            return False
            
        try:
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
        # Handle None or empty measurements
        if not measurements or not isinstance(measurements, str):
            return None, None
        
        # Extract using the compiled pattern
        match = self.CUP_SIZE_PATTERN.search(measurements)
        if not match:
            return None, None
        
        try:
            # Extract band and cup from regex match
            band_size = match.group(1)
            cup_letter = match.group(2)
            
            # Convert band to integer
            try:
                band_size_int = int(band_size)
            except (ValueError, TypeError):
                return None, None
                
            # If already in EU format, return as is
            if self._is_eu_band_size(band_size_int):
                eu_cup = self.CUP_CONVERSION.get(cup_letter, cup_letter)
                cup_numeric = self.CUP_NUMERIC.get(eu_cup, 0)
                return f"{band_size_int}{eu_cup}", cup_numeric
            
            # Convert US/UK to EU
            eu_band = self.BAND_CONVERSION.get(band_size_int)
            
            # If band size isn't in our mapping, calculate approximate EU size
            if eu_band is None:
                eu_band = round((band_size_int + 16) / 2) * 5
                
            eu_cup = self.CUP_CONVERSION.get(cup_letter, cup_letter)
            cup_numeric = self.CUP_NUMERIC.get(eu_cup, 0)
            
            return f"{eu_band}{eu_cup}", cup_numeric
            
        except (IndexError, AttributeError):
            # Handle any issues with match groups
            return None, None
    
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
            # Skip performers without ID
            if not performer.get('id'):
                continue
                
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
            except (ValueError, IndexError, AttributeError):
                continue
            
            # Calculate BMI with robust error handling
            height_cm = performer.get('height_cm')
            weight = performer.get('weight')
            bmi = None
            
            # Ensure height_cm and weight are valid numbers before calculations
            try:
                height_cm = float(height_cm) if height_cm is not None else 0
                weight = float(weight) if weight is not None else 0
                
                if height_cm > 0 and weight > 0:
                    height_m = height_cm / 100
                    bmi = round(weight / (height_m * height_m), 1)
            except (TypeError, ValueError, ZeroDivisionError):
                bmi = None
            
            # Safely handle potential None values for all fields
            def safe_value(value, default=0):
                if value is None:
                    return default
                return value
                
            # Create performer data dictionary with safe defaults
            performer_data = {
                'id': performer.get('id', ''),
                'name': performer.get('name', 'Unknown'),
                'cup_size': eu_bra_size,
                'band_size': band_size,
                'cup_letter': cup_letter,
                'cup_numeric': cup_numeric or 0,
                'favorite': bool(performer.get('favorite', False)),
                'height_cm': safe_value(height_cm),
                'weight': safe_value(weight),
                'bmi': bmi,
                'measurements': measurements or '',
                'scene_count': safe_value(performer.get('scene_count')),
                'rating100': safe_value(performer.get('rating100')),
                'o_counter': safe_value(performer.get('o_counter'))
            }
            
            cup_size_data.append(performer_data)
        
        # Create a DataFrame - handle empty case
        if not cup_size_data:
            self._cup_size_df = pd.DataFrame()
            return
            
        df = pd.DataFrame(cup_size_data)
        
        # Convert numeric columns with error handling
        try:
            numeric_columns = ['band_size', 'cup_numeric', 'height_cm', 'weight', 
                              'scene_count', 'rating100', 'o_counter']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Replace NaNs with 0 for numeric columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Special handling for BMI which can be None
            if 'bmi' in df.columns:
                df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting numeric columns: {e}")
            # If conversion fails, ensure we still have a valid DataFrame
            pass
            
        self._cup_size_df = df
    
    def get_cup_size_stats(self) -> Dict[str, Any]:
        """Get statistics about cup sizes using EU format.
        
        Returns:
            Dict with cup size statistics
        """
        df = self.cup_size_df
        
        # Handle empty DataFrame
        if df.empty:
            return {
                'cup_size_counts': {},
                'cup_size_dataframe': []
            }
        
        try:
            # Count frequencies with robust error handling
            cup_sizes = df['cup_size'].tolist()
            cup_sizes = [cs for cs in cup_sizes if cs]  # Filter out None/empty values
            cup_size_counts = Counter(cup_sizes)
        except Exception as e:
            logger.error(f"Error counting cup sizes: {e}")
            cup_size_counts = Counter()
        
        # Convert DataFrame to dict for JSON serialization with error handling
        try:
            df_dict = df.to_dict('records')
        except Exception as e:
            logger.error(f"Error converting DataFrame to dict: {e}")
            df_dict = []
        
        return {
            'cup_size_counts': dict(cup_size_counts),
            'cup_size_dataframe': df_dict
        }
    
    def get_cup_size_o_counter_correlation(self, o_counter_module) -> Dict[str, Any]:
        """Analyze correlation between cup size and o-counter.
        
        Args:
            o_counter_module: O-counter module instance for getting o-counter data
            
        Returns:
            Dict with cup size and o-counter correlation statistics
        """
        cup_df = self.cup_size_df
        
        # Handle case where o_counter_module is None or has no get_o_counter_stats method
        try:
            performer_o_counts = o_counter_module.get_o_counter_stats()['performer_o_counts']
        except (AttributeError, KeyError, TypeError) as e:
            logger.error(f"Error getting o-counter data: {e}")
            performer_o_counts = {}
    
        if cup_df.empty:
            return {
                'cup_size_o_counter_df': [],
                'cup_letter_o_stats': []
            }
            
        try:
            # Create a copy to avoid modifying the cached dataframe
            analysis_df = cup_df.copy()
            
            # Add o-counter data using vectorized operations with safe handling for None values
            analysis_df['total_o_count'] = analysis_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('total_o_count', 0) or 0
                if pid else 0
            )
            
            analysis_df['o_scene_count'] = analysis_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('scene_count', 0) or 0
                if pid else 0
            )
            
            # Fill NaN values with 0 to ensure safe comparisons
            analysis_df['total_o_count'] = analysis_df['total_o_count'].fillna(0)
            analysis_df['o_scene_count'] = analysis_df['o_scene_count'].fillna(0)
            
            # Filter to only include performers with o_counter > 0 for statistics
            non_zero_o_count_df = analysis_df[analysis_df['total_o_count'] > 0]
        except Exception as e:
            logger.error(f"Error preparing data for cup size correlation: {e}")
            return {
                'cup_size_o_counter_df': [],
                'cup_letter_o_stats': []
            }
        
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
        try:
            analysis_df_dict = analysis_df.to_dict('records')
        except Exception as e:
            logger.error(f"Error converting analysis DataFrame to dict: {e}")
            analysis_df_dict = []
        
        return {
            'cup_size_o_counter_df': analysis_df_dict,
            'cup_letter_o_stats': cup_letter_stats_dict
        }
