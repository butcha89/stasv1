import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

from base_module import BaseModule

logger = logging.getLogger(__name__)

class RatioModule(BaseModule):
    """Module for analyzing ratios and relationships between measurements."""
    
    def __init__(self, stash_client=None, cup_size_module=None):
        """Initialize the ratio module.
        
        Args:
            stash_client: Client for accessing stash data
            cup_size_module: Reference to cup size module for cup data
        """
        super().__init__(stash_client)
        self.cup_size_module = cup_size_module
        self._ratio_df = None
    
    @property
    def ratio_df(self) -> pd.DataFrame:
        """Get or create ratio DataFrame with caching."""
        if self._ratio_df is None:
            self._create_ratio_df()
        return self._ratio_df
    
    def _create_ratio_df(self) -> None:
        """Create ratio DataFrame for analysis."""
        # Handle missing cup_size_module
        if self.cup_size_module is None:
            logger.warning("Cup size module not available. Creating empty ratio DataFrame.")
            self._ratio_df = pd.DataFrame()
            return
        
        try:
            # Get cup size DataFrame
            df = self.cup_size_module.cup_size_df.copy()
        except Exception as e:
            logger.error(f"Error accessing cup_size_df: {e}")
            self._ratio_df = pd.DataFrame()
            return
            
        if df.empty:
            logger.info("Cup size DataFrame is empty. Creating empty ratio DataFrame.")
            self._ratio_df = pd.DataFrame()
            return
        
        try:
            # Cup letter to numeric mapping
            cup_letter_values = {letter: idx+1 for idx, letter in enumerate('ABCDEFGHIJK')}
            
            # Add cup letter numeric value with safe handling
            df['cup_letter_value'] = df['cup_letter'].map(lambda x: cup_letter_values.get(x, 0) if x else 0)
            
            # Make sure all required columns exist and are numeric
            required_columns = ['bmi', 'height_cm', 'weight', 'cup_letter_value']
            
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} missing from DataFrame. Adding with zeros.")
                    df[col] = 0
                else:
                    # Convert to numeric safely
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with 0 for safer division
            df[required_columns] = df[required_columns].fillna(0)
            
            # Calculate ratios with safe division using numpy.where
            # Only calculate ratio where both values are positive
            
            # Cup to BMI ratio
            df['cup_to_bmi'] = np.where(
                (df['bmi'] > 0) & (df['cup_letter_value'] > 0),
                df['cup_letter_value'] / df['bmi'],
                np.nan
            )
            
            # Cup to height ratio
            df['cup_to_height'] = np.where(
                (df['height_cm'] > 0) & (df['cup_letter_value'] > 0),
                df['cup_letter_value'] / df['height_cm'],
                np.nan
            )
            
            # Cup to weight ratio
            df['cup_to_weight'] = np.where(
                (df['weight'] > 0) & (df['cup_letter_value'] > 0),
                df['cup_letter_value'] / df['weight'],
                np.nan
            )
            
            self._ratio_df = df
            
        except Exception as e:
            logger.error(f"Error creating ratio DataFrame: {e}")
            # Return empty DataFrame if anything fails
            self._ratio_df = pd.DataFrame()
    
    def get_ratio_stats(self) -> Dict[str, Any]:
        """Calculate various ratios like cup-to-bmi, cup-to-height, cup-to-weight.
        
        Returns:
            Dict with ratio statistics
        """
        try:
            ratio_df = self.ratio_df
        except Exception as e:
            logger.error(f"Error accessing ratio_df: {e}")
            return {'ratio_dataframe': [], 'ratio_stats': []}
        
        if ratio_df.empty:
            logger.info("Ratio DataFrame is empty. Returning empty stats.")
            return {'ratio_dataframe': [], 'ratio_stats': []}
        
        try:
            # Make a copy to avoid warnings
            ratio_df_copy = ratio_df.copy()
            
            # Ensure o_counter is numeric
            if 'o_counter' in ratio_df_copy.columns:
                ratio_df_copy['o_counter'] = pd.to_numeric(ratio_df_copy['o_counter'], errors='coerce').fillna(0)
            else:
                logger.warning("o_counter column missing from ratio DataFrame.")
                ratio_df_copy['o_counter'] = 0
            
            # Filter to only include performers with o_counter > 0 for ratio statistics
            non_zero_o_count_df = ratio_df_copy[ratio_df_copy['o_counter'] > 0]
            
            # If we have no performers with non-zero o-count, fall back to using all performers
            analysis_df = non_zero_o_count_df if not non_zero_o_count_df.empty else ratio_df_copy
            
            # Check if cup_letter column exists
            if 'cup_letter' not in analysis_df.columns:
                logger.warning("cup_letter column missing from analysis DataFrame.")
                return {'ratio_dataframe': [], 'ratio_stats': []}
            
            # Check if we have ratio columns to analyze
            ratio_columns = ['cup_to_bmi', 'cup_to_height', 'cup_to_weight']
            missing_columns = [col for col in ratio_columns if col not in analysis_df.columns]
            
            if missing_columns:
                logger.warning(f"Missing ratio columns: {missing_columns}")
                # Add missing columns as NaN
                for col in missing_columns:
                    analysis_df[col] = np.nan
            
            # Calculate all statistics in a single groupby operation
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
        
        # Convert DataFrame to dict for JSON serialization with error handling
        try:
            ratio_df_dict = ratio_df.to_dict('records')
        except Exception as e:
            logger.error(f"Error converting ratio DataFrame to dict: {e}")
            ratio_df_dict = []
        
        return {
            'ratio_dataframe': ratio_df_dict,
            'ratio_stats': ratio_stats_dict
        }