import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

from base_module import BaseModule

class SisterSizeModule(BaseModule):
    """Module for analyzing sister sizes and breast volume."""
    
    # Constants for sister size calculations
    SISTER_SIZE_STEP = 5  # Band size difference in EU sizing (5 cm)
    
    # Breast volume constants (approximate values in cubic centimeters)
    # These are approximate values based on research
    CUP_VOLUME_BASE = {
        "A": 200,
        "B": 350,
        "C": 500,
        "D": 650,
        "E": 800,
        "F": 950,
        "G": 1100,
        "H": 1250,
        "I": 1400,
        "J": 1550
    }
    
    # Volume difference between cup sizes (approx)
    VOLUME_PER_CUP = 150  # cc or ml
    
    def __init__(self, stash_client=None, cup_size_module=None):
        """Initialize the sister size module.
        
        Args:
            stash_client: Client for accessing stash data
            cup_size_module: Reference to cup size module for cup data
        """
        super().__init__(stash_client)
        self.cup_size_module = cup_size_module
        self._sister_size_df = None
        self._volume_df = None
        
    def _calculate_band_cup_volume(self, band_size: Union[int, str, None], cup_letter: Optional[str]) -> float:
        """Calculate the approximate breast volume based on band size and cup.
        
        Research shows that the same cup letter on different band sizes represents
        different volumes. This calculates an approximate volume.
        
        Args:
            band_size: Band size in cm (EU format)
            cup_letter: Cup letter (A-J)
            
        Returns:
            float: Estimated breast volume in cubic centimeters
        """
        # Handle None or empty values
        if cup_letter is None or cup_letter == '' or band_size is None:
            return 0.0
            
        # If cup letter not in our mapping, return 0
        if cup_letter not in self.CUP_VOLUME_BASE:
            return 0.0
        
        # Convert band_size to int safely
        try:
            band_size_int = int(band_size)
        except (ValueError, TypeError):
            return 0.0
            
        # Get the base volume for this cup size
        base_volume = self.CUP_VOLUME_BASE[cup_letter]
        
        # Adjust for band size (larger bands = larger volume for same cup)
        # Use 75 (EU) as a reference band
        band_adjustment = (band_size_int - 75) / 5 * 30  # 30cc per 5cm band difference
        
        return base_volume + band_adjustment
    
    def _get_sister_sizes(self, band_size: Union[str, int, None], cup_letter: Optional[str], steps: int = 3) -> List[Dict[str, Any]]:
        """Get sister sizes for a given bra size.
        
        Args:
            band_size: Band size in cm (EU format)
            cup_letter: Cup letter (A-J)
            steps: Number of sister size steps to calculate in each direction
            
        Returns:
            List of sister size dictionaries with band, cup and volume
        """
        # Handle None values
        if cup_letter is None or cup_letter == '' or band_size is None:
            return []
            
        # Convert band_size to int safely
        try:
            band_size_int = int(band_size)
        except (ValueError, TypeError):
            return []
            
        # Cup letter to numeric mapping
        cup_letters = "ABCDEFGHIJ"
        
        # Find the index of the current cup letter
        try:
            cup_index = cup_letters.index(cup_letter)
        except ValueError:
            # Cup letter not found in our sequence
            return []
            
        sister_sizes = []
        
        # Original size
        original_volume = self._calculate_band_cup_volume(band_size_int, cup_letter)
        sister_sizes.append({
            'band_size': band_size_int,
            'cup_letter': cup_letter,
            'bra_size': f"{band_size_int}{cup_letter}",
            'volume_cc': original_volume,
            'relative_volume': 1.0,  # 100% of original
            'is_original': True
        })
        
        # Sister sizes with smaller bands (and larger cups)
        for i in range(1, steps + 1):
            new_band = band_size_int - (i * self.SISTER_SIZE_STEP)
            new_cup_index = min(cup_index + i, len(cup_letters) - 1)
            new_cup = cup_letters[new_cup_index]
            
            # Skip if band size is too small
            if new_band < self.cup_size_module.MIN_EU_BAND:
                continue
                
            new_volume = self._calculate_band_cup_volume(new_band, new_cup)
            sister_sizes.append({
                'band_size': new_band,
                'cup_letter': new_cup,
                'bra_size': f"{new_band}{new_cup}",
                'volume_cc': new_volume,
                'relative_volume': new_volume / original_volume if original_volume else 1.0,
                'is_original': False
            })
        
        # Sister sizes with larger bands (and smaller cups)
        for i in range(1, steps + 1):
            new_band = band_size_int + (i * self.SISTER_SIZE_STEP)
            new_cup_index = max(cup_index - i, 0)
            new_cup = cup_letters[new_cup_index]
            
            # Skip if band size is too large
            if new_band > self.cup_size_module.MAX_EU_BAND:
                continue
                
            new_volume = self._calculate_band_cup_volume(new_band, new_cup)
            sister_sizes.append({
                'band_size': new_band,
                'cup_letter': new_cup,
                'bra_size': f"{new_band}{new_cup}",
                'volume_cc': new_volume,
                'relative_volume': new_volume / original_volume if original_volume else 1.0,
                'is_original': False
            })
            
        # Sort by band size
        return sorted(sister_sizes, key=lambda x: x['band_size'])
    
    @property
    def sister_size_df(self) -> pd.DataFrame:
        """Get or create sister size DataFrame with caching."""
        if self._sister_size_df is None:
            self._create_sister_size_df()
        return self._sister_size_df
    
    def _create_sister_size_df(self) -> None:
        """Create the sister size DataFrame for analysis."""
        if not self.cup_size_module:
            self._sister_size_df = pd.DataFrame()
            return
            
        cup_df = self.cup_size_module.cup_size_df
        
        if cup_df.empty:
            self._sister_size_df = pd.DataFrame()
            return
            
        sister_size_data = []
        
        for _, performer in cup_df.iterrows():
            try:
                # Safely convert band_size to int
                try:
                    band_size = int(performer['band_size'])
                except (ValueError, TypeError):
                    continue
                
                # Check for valid cup letter
                cup_letter = performer['cup_letter']
                if not cup_letter or not isinstance(cup_letter, str):
                    continue
                
                # Get sister sizes
                sister_sizes = self._get_sister_sizes(band_size, cup_letter)
                
                # Add performer info to each sister size entry
                for size in sister_sizes:
                    # Ensure all required fields have default values
                    size_data = {
                        'performer_id': performer.get('id', ''),
                        'performer_name': performer.get('name', 'Unknown'),
                        'original_size': performer.get('cup_size', ''),
                        'sister_size': size.get('bra_size', ''),
                        'band_size': size.get('band_size', 0),
                        'cup_letter': size.get('cup_letter', ''),
                        'is_original': size.get('is_original', False),
                        'volume_cc': size.get('volume_cc', 0.0),
                        'relative_volume': size.get('relative_volume', 1.0),
                        'o_counter': performer.get('o_counter', 0),
                        'rating100': performer.get('rating100', 0),
                        'favorite': performer.get('favorite', False)
                    }
                    sister_size_data.append(size_data)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Error processing performer {performer.get('name', 'unknown')}: {e}")
                continue
                
        # Create DataFrame, or empty DataFrame if no data
        self._sister_size_df = pd.DataFrame(sister_size_data) if sister_size_data else pd.DataFrame()
    
    @property
    def volume_df(self) -> pd.DataFrame:
        """Get or create volume DataFrame with caching."""
        if self._volume_df is None:
            self._create_volume_df()
        return self._volume_df
    
    def _create_volume_df(self) -> None:
        """Create breast volume DataFrame for analysis."""
        if not self.cup_size_module:
            self._volume_df = pd.DataFrame()
            return
            
        cup_df = self.cup_size_module.cup_size_df
        
        if cup_df.empty:
            self._volume_df = pd.DataFrame()
            return
            
        # Copy relevant columns and add volume
        volume_df = cup_df.copy()
        
        # Convert band_size to numeric safely
        volume_df['band_size'] = pd.to_numeric(volume_df['band_size'], errors='coerce')
        
        # Calculate volume for each performer safely
        def safe_calculate_volume(row):
            # Check if necessary values exist
            if pd.isna(row['band_size']) or pd.isna(row['cup_letter']):
                return 0.0
            return self._calculate_band_cup_volume(int(row['band_size']), row['cup_letter'])
        
        volume_df['volume_cc'] = volume_df.apply(safe_calculate_volume, axis=1)
        
        # Add volume categories
        volume_df['volume_category'] = pd.cut(
            volume_df['volume_cc'],
            bins=[0, 300, 500, 700, 900, 1100, 1500, 2000, float('inf')],
            labels=['Very Small', 'Small', 'Medium-Small', 'Medium', 
                   'Medium-Large', 'Large', 'Very Large', 'Extremely Large']
        )
        
        self._volume_df = volume_df
        
    def get_volume_stats(self) -> Dict[str, Any]:
        """Get statistics about breast volumes.
        
        Returns:
            Dict with volume statistics
        """
        df = self.volume_df
        
        if df.empty:
            return {
                'volume_dataframe': [],
                'volume_category_stats': [],
                'volume_o_counter_correlation': 0,
                'top_volume_performers': []
            }
        
        # Overall volume stats - handle errors
        try:
            volume_stats = df['volume_cc'].describe().to_dict()
        except Exception:
            volume_stats = {}
        
        # Stats by volume category
        try:
            # Ensure categorical data exists
            if 'volume_category' not in df.columns or df['volume_category'].isna().all():
                return {
                    'volume_dataframe': df.to_dict('records'),
                    'volume_stats': volume_stats,
                    'volume_category_stats': [],
                    'volume_o_counter_correlation': 0,
                    'top_volume_performers': []
                }
                
            # Convert numeric columns to ensure they're numeric
            numeric_cols = ['o_counter', 'rating100', 'volume_cc']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            category_stats = df.groupby('volume_category').agg({
                'id': 'count',
                'o_counter': ['mean', 'sum'],
                'rating100': ['mean', 'median'],
                'volume_cc': ['mean', 'median']
            })
            
            # Flatten multi-index
            category_stats.columns = ['_'.join(col).strip('_') for col in category_stats.columns.values]
            category_stats = category_stats.reset_index()
            
            # Rename columns for clarity
            category_stats = category_stats.rename(columns={
                'id_count': 'performer_count',
                'o_counter_mean': 'avg_o_counter',
                'o_counter_sum': 'total_o_counter',
                'rating100_mean': 'avg_rating',
                'rating100_median': 'median_rating',
                'volume_cc_mean': 'avg_volume_cc',
                'volume_cc_median': 'median_volume_cc'
            })
            
            # Sort by volume order
            order = ['Very Small', 'Small', 'Medium-Small', 'Medium', 
                    'Medium-Large', 'Large', 'Very Large', 'Extremely Large']
            category_stats['order'] = category_stats['volume_category'].map({cat: i for i, cat in enumerate(order)})
            category_stats = category_stats.sort_values('order')
            category_stats = category_stats.drop('order', axis=1)
            
            category_stats_dict = category_stats.to_dict('records')
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in volume category stats: {e}")
            category_stats_dict = []
            
        # Calculate correlation between volume and o-counter
        # Filter to only include performers with o_counter > 0
        try:
            o_counter_df = df[df['o_counter'] > 0]
            
            if not o_counter_df.empty and len(o_counter_df) > 1:
                correlation = o_counter_df['volume_cc'].corr(o_counter_df['o_counter'])
                correlation = correlation if pd.notna(correlation) else 0
            else:
                correlation = 0
        except Exception:
            correlation = 0
            
        # Top performers by volume
        try:
            top_volume_performers = df.nlargest(10, 'volume_cc')[
                ['name', 'cup_size', 'volume_cc', 'volume_category', 'o_counter']
            ].to_dict('records')
        except Exception:
            top_volume_performers = []
        
        return {
            'volume_dataframe': df.to_dict('records'),
            'volume_stats': volume_stats,
            'volume_category_stats': category_stats_dict,
            'volume_o_counter_correlation': correlation,
            'top_volume_performers': top_volume_performers
        }
    
    def get_sister_size_stats(self) -> Dict[str, Any]:
        """Get statistics about sister sizes.
        
        Returns:
            Dict with sister size statistics
        """
        df = self.sister_size_df
        
        if df.empty:
            return {
                'sister_size_dataframe': [],
                'common_sister_sizes': [],
                'original_vs_sister_stats': {}
            }
            
        # Count most common sister sizes (excluding originals)
        try:
            sister_only_df = df[~df['is_original']]
            common_sizes = sister_only_df['sister_size'].value_counts().head(10).to_dict()
        except Exception:
            common_sizes = {}
        
        # Compare stats between original and sister sizes
        # For performers with o_counter > 0
        try:
            # Ensure o_counter is numeric
            df['o_counter'] = pd.to_numeric(df['o_counter'], errors='coerce').fillna(0)
            stats_df = df[df['o_counter'] > 0].copy()
            
            if not stats_df.empty:
                # Calculate average o_counter and rating for original vs sister sizes
                original_stats = stats_df[stats_df['is_original']].agg({
                    'o_counter': 'mean',
                    'rating100': 'mean',
                    'volume_cc': 'mean'
                }).to_dict()
                
                sister_stats = stats_df[~stats_df['is_original']].agg({
                    'o_counter': 'mean',
                    'rating100': 'mean',
                    'volume_cc': 'mean'
                }).to_dict()
                
                # Group sister sizes by band/cup combination
                group_stats = stats_df.groupby(['sister_size']).agg({
                    'o_counter': 'mean',
                    'rating100': 'mean',
                    'volume_cc': 'mean',
                    'performer_id': 'count'
                }).reset_index()
                
                # Rename columns
                group_stats = group_stats.rename(columns={
                    'performer_id': 'count'
                })
                
                # Sort by o_counter
                top_o_counter_sizes = group_stats.sort_values(
                    'o_counter', ascending=False
                ).head(10).to_dict('records')
                
                comparison_stats = {
                    'original_stats': original_stats,
                    'sister_stats': sister_stats,
                    'top_o_counter_sizes': top_o_counter_sizes
                }
            else:
                comparison_stats = {
                    'original_stats': {},
                    'sister_stats': {},
                    'top_o_counter_sizes': []
                }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in sister size stats comparison: {e}")
            comparison_stats = {
                'original_stats': {},
                'sister_stats': {},
                'top_o_counter_sizes': []
            }
        
        return {
            'sister_size_dataframe': df.to_dict('records') if not df.empty else [],
            'common_sister_sizes': common_sizes,
            'original_vs_sister_stats': comparison_stats
        }