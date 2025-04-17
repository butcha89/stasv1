import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union

from base_module import BaseModule

logger = logging.getLogger(__name__)

class OCounterModule(BaseModule):
    """Module for analyzing O-Counter statistics."""
    
    def __init__(self, stash_client=None):
        """Initialize the O-Counter module.
        
        Args:
            stash_client: Client for accessing stash data
        """
        super().__init__(stash_client)
        self._o_counter_df = None
    
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
            try:
                # Ensure o_counter is a valid number
                o_counter = scene.get('o_counter', 0)
                o_counter = 0 if o_counter is None else int(o_counter)
            except (ValueError, TypeError):
                # If conversion fails, default to 0
                o_counter = 0
                
            # Safely extract performer data with validation
            performers = scene.get('performers', [])
            if not isinstance(performers, list):
                performers = []
                
            # Safely extract performer names and IDs
            performer_names = []
            performer_ids = []
            favorite_performers = []
            
            for p in performers:
                if not isinstance(p, dict):
                    continue
                    
                # Extract performer info with fallbacks
                name = p.get('name', '')
                performer_id = p.get('id', '')
                is_favorite = p.get('favorite', False)
                
                if name:
                    performer_names.append(name)
                if performer_id:
                    performer_ids.append(performer_id)
                if is_favorite and name:
                    favorite_performers.append(name)
            
            # Extract tags safely
            tags = scene.get('tags', [])
            tag_names = []
            
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, dict) and 'name' in t:
                        tag_names.append(t.get('name', ''))
            
            scene_data = {
                'id': scene.get('id', ''),
                'title': scene.get('title', ''),
                'o_counter': o_counter,
                'performers': performer_names,
                'performer_ids': performer_ids,
                'favorite_performers': favorite_performers,
                'tags': tag_names
            }
            
            o_counter_data.append(scene_data)
        
        # Create DataFrame or empty DataFrame if no data
        if not o_counter_data:
            self._o_counter_df = pd.DataFrame()
        else:
            self._o_counter_df = pd.DataFrame(o_counter_data)
    
    def get_o_counter_stats(self) -> Dict[str, Any]:
        """Get statistics about o-counter values.
        
        Returns:
            Dict with o-counter statistics
        """
        df = self.o_counter_df
        
        # Initialize with empty results
        result = {
            'o_counter_dataframe': [],
            'performer_o_counts': {},
            'average_o_counter': 0,
            'median_o_counter': 0,
            'max_o_counter': 0,
            'total_performers': 0
        }
        
        if df.empty:
            return result
        
        # Get performers with highest o-counter sum
        performer_o_counts = {}
        
        try:
            # Create a copy to avoid warnings
            df_copy = df.copy()
            
            # Ensure o_counter is numeric
            df_copy['o_counter'] = pd.to_numeric(df_copy['o_counter'], errors='coerce').fillna(0)
            
            # Filter for rows with o_counter > 0 to improve efficiency
            o_counter_rows = df_copy[df_copy['o_counter'] > 0]
            
            for _, row in o_counter_rows.iterrows():
                o_count = row['o_counter']
                # Validate performer_ids and performers are lists and have matching length
                performer_ids = row.get('performer_ids', [])
                performers = row.get('performers', [])
                
                if not isinstance(performer_ids, list) or not isinstance(performers, list):
                    continue
                
                # Use zip with the shorter list to avoid index errors
                for performer_id, performer_name in zip(performer_ids, performers):
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
        except Exception as e:
            logger.error(f"Error calculating performer o-counts: {e}")
        
        # Calculate overall O-counter statistics - only for non-zero values
        try:
            # Ensure o_counter is numeric
            if 'o_counter' in df.columns:
                df_numeric = df.copy()
                df_numeric['o_counter'] = pd.to_numeric(df_numeric['o_counter'], errors='coerce').fillna(0)
                o_counter_values = df_numeric['o_counter'][df_numeric['o_counter'] > 0].tolist()
            else:
                o_counter_values = []
            
            if o_counter_values:
                avg_o_counter = np.mean(o_counter_values)
                median_o_counter = np.median(o_counter_values)
                max_o_counter = max(o_counter_values)
            else:
                avg_o_counter = 0
                median_o_counter = 0
                max_o_counter = 0
        except Exception as e:
            logger.error(f"Error calculating o-counter statistics: {e}")
            avg_o_counter = 0
            median_o_counter = 0
            max_o_counter = 0
            
        # Convert DataFrame to dict for JSON serialization with error handling
        try:
            df_dict = df.to_dict('records')
        except Exception as e:
            logger.error(f"Error converting DataFrame to dict: {e}")
            df_dict = []
            
        result = {
            'o_counter_dataframe': df_dict,
            'performer_o_counts': performer_o_counts,
            'average_o_counter': avg_o_counter,
            'median_o_counter': median_o_counter,
            'max_o_counter': max_o_counter,
            'total_performers': len(performer_o_counts)
        }
        
        return result
    
    def get_favorite_o_counter_stats(self) -> Dict[str, Any]:
        """Analyze the relationship between favorite status and o-counter values.
        
        Returns:
            Dict with detailed statistics about favorites vs non-favorites
        """
        # Extract performers with o-counter > 0 with robust error handling
        try:
            performers_with_o = []
            for p in self.performers_data:
                if not isinstance(p, dict):
                    continue
                    
                o_counter = p.get('o_counter')
                
                # Ensure o_counter is a valid number > 0
                try:
                    o_counter = 0 if o_counter is None else float(o_counter)
                    if o_counter > 0:
                        performers_with_o.append(p)
                except (ValueError, TypeError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error filtering performers with o-counter: {e}")
            performers_with_o = []
        
        # Separate favorites and non-favorites with validation
        try:
            favorites = [p for p in performers_with_o if p.get('favorite', False)]
            non_favorites = [p for p in performers_with_o if not p.get('favorite', False)]
        except Exception as e:
            logger.error(f"Error separating favorites: {e}")
            favorites = []
            non_favorites = []
        
        # Helper function to calculate stats with comprehensive error handling
        def calculate_stats(performers):
            try:
                if not performers:
                    return {
                        'count': 0,
                        'avg_o_counter': 0,
                        'median_o_counter': 0,
                        'max_o_counter': 0,
                        'performers': []
                    }
                
                # Safely extract o_counter values
                o_values = []
                for p in performers:
                    try:
                        o_counter = p.get('o_counter', 0)
                        if o_counter is not None:
                            o_values.append(float(o_counter))
                    except (ValueError, TypeError):
                        continue
                
                # Calculate statistics with validation
                avg_value = np.mean(o_values) if o_values else 0
                median_value = np.median(o_values) if o_values else 0
                max_value = max(o_values) if o_values else 0
                
                # Build performer details with careful extraction
                performer_details = []
                for p in sorted(performers, key=lambda x: x.get('o_counter', 0) or 0, reverse=True):
                    # Safe extraction of all values
                    try:
                        performer_details.append({
                            'name': p.get('name', 'Unknown'),
                            'o_counter': float(p.get('o_counter', 0) or 0),
                            'rating100': float(p.get('rating100', 0) or 0),
                            'scene_count': int(p.get('scene_count', 0) or 0),
                            'measurements': p.get('measurements', 'N/A') or 'N/A'
                        })
                    except (ValueError, TypeError):
                        # Skip this performer if values can't be converted
                        continue
                
                return {
                    'count': len(performers),
                    'avg_o_counter': avg_value,
                    'median_o_counter': median_value,
                    'max_o_counter': max_value,
                    'performers': performer_details
                }
            except Exception as e:
                logger.error(f"Error calculating performer stats: {e}")
                return {
                    'count': 0,
                    'avg_o_counter': 0,
                    'median_o_counter': 0,
                    'max_o_counter': 0,
                    'performers': []
                }
        
        # Calculate statistics for both groups
        favorite_stats = calculate_stats(favorites)
        non_favorite_stats = calculate_stats(non_favorites)
        
        # Overall statistics with safe division
        total_performers = len(performers_with_o)
        
        if total_performers > 0:
            favorite_percentage = (len(favorites) / total_performers * 100)
            non_favorite_percentage = (len(non_favorites) / total_performers * 100)
        else:
            favorite_percentage = 0
            non_favorite_percentage = 0
        
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
        # Initialize with empty results
        result = {
            'correlation': 0,
            'high_rated_high_o': [],
            'high_rated_low_o': [],
            'low_rated_high_o': [],
            'rating_o_counter_data': []
        }
        
        # Extract performers with rating100 > 0 with robust error handling
        try:
            performers_with_data = []
            for p in self.performers_data:
                if not isinstance(p, dict):
                    continue
                    
                rating100 = p.get('rating100')
                
                # Ensure rating100 is a valid number > 0
                try:
                    rating100 = 0 if rating100 is None else float(rating100)
                    if rating100 > 0:
                        performers_with_data.append(p)
                except (ValueError, TypeError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error filtering performers with rating: {e}")
            performers_with_data = []
        
        if not performers_with_data:
            return result
        
        # Create DataFrame for analysis with safe extraction
        try:
            # Use list comprehension with error handling for each performer
            rating_data = []
            for p in performers_with_data:
                try:
                    # Safe extraction with type conversion
                    rating_data.append({
                        'id': p.get('id', ''),
                        'name': p.get('name', 'Unknown'),
                        'rating100': float(p.get('rating100', 0) or 0),
                        'o_counter': float(p.get('o_counter', 0) or 0),
                        'favorite': bool(p.get('favorite', False)),
                        'scene_count': int(p.get('scene_count', 0) or 0),
                        'measurements': p.get('measurements', '') or ''
                    })
                except (ValueError, TypeError):
                    # Skip this performer if conversion fails
                    continue
            
            df = pd.DataFrame(rating_data)
        except Exception as e:
            logger.error(f"Error creating correlation DataFrame: {e}")
            return result
        
        # Calculate correlation if data exists
        try:
            if not df.empty and len(df) > 1:
                correlation = df['rating100'].corr(df['o_counter'])
                correlation = correlation if pd.notna(correlation) else 0
            else:
                correlation = 0
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            correlation = 0
        
        # Identify performers in different categories
        try:
            if not df.empty:
                # Use median as threshold for high/low with error handling
                try:
                    rating_median = df['rating100'].median()
                    
                    # Get median for performers with o_counter > 0
                    o_counter_positive = df[df['o_counter'] > 0]
                    o_counter_median = o_counter_positive['o_counter'].median() if not o_counter_positive.empty else 1
                    
                    # Categorize performers with safe comparisons
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
                except Exception as e:
                    logger.error(f"Error categorizing performers: {e}")
                    high_rated_high_o = []
                    high_rated_low_o = []
                    low_rated_high_o = []
            else:
                high_rated_high_o = []
                high_rated_low_o = []
                low_rated_high_o = []
        except Exception as e:
            logger.error(f"Error in performer categorization: {e}")
            high_rated_high_o = []
            high_rated_low_o = []
            low_rated_high_o = []
        
        # Safe DataFrame to dict conversion
        try:
            df_dict = df.to_dict('records') if not df.empty else []
        except Exception as e:
            logger.error(f"Error converting DataFrame to dict: {e}")
            df_dict = []
        
        return {
            'correlation': correlation,
            'high_rated_high_o': high_rated_high_o,
            'high_rated_low_o': high_rated_low_o,
            'low_rated_high_o': low_rated_high_o,
            'rating_o_counter_data': df_dict
        }
    
    def get_top_o_counter_performers(self, cup_size_module, top_n=10) -> List[Dict]:
        """Get top performers based on o-counter.
        
        Args:
            cup_size_module: Cup size module instance for getting bra size data
            top_n: Number of top performers to return
            
        Returns:
            List of top performer details
        """
        # Handle case where cup_size_module is None or invalid
        if cup_size_module is None:
            cup_size_convert = lambda x: (None, None)
        else:
            try:
                # Check if the module has the expected method
                if hasattr(cup_size_module, '_convert_bra_size'):
                    cup_size_convert = cup_size_module._convert_bra_size
                else:
                    cup_size_convert = lambda x: (None, None)
            except Exception:
                cup_size_convert = lambda x: (None, None)
        
        # Filter performers with O-Counter > 0 with robust validation
        try:
            performers_with_o_counter = []
            for p in self.performers_data:
                if not isinstance(p, dict):
                    continue
                    
                try:
                    o_counter = p.get('o_counter')
                    o_counter = 0 if o_counter is None else float(o_counter)
                    if o_counter > 0:
                        performers_with_o_counter.append(p)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            logger.error(f"Error filtering performers with o-counter: {e}")
            performers_with_o_counter = []
        
        # Sort by O-Counter descending with error handling for the sort key
        try:
            def safe_get_o_counter(p):
                try:
                    return float(p.get('o_counter', 0) or 0)
                except (ValueError, TypeError):
                    return 0
                    
            top_performers = sorted(
                performers_with_o_counter, 
                key=safe_get_o_counter, 
                reverse=True
            )[:top_n]
        except Exception as e:
            logger.error(f"Error sorting performers: {e}")
            top_performers = performers_with_o_counter[:top_n] if performers_with_o_counter else []
        
        # Prepare details with safe extraction
        top_o_counter_details = []
        for performer in top_performers:
            try:
                # Get EU cup size with error handling
                measurements = performer.get('measurements')
                try:
                    eu_cup_size, _ = cup_size_convert(measurements)
                except Exception:
                    eu_cup_size = None
                
                # Safe extraction of values with type checking
                performer_id = performer.get('id', '')
                name = performer.get('name', 'Unknown')
                
                try:
                    o_counter = float(performer.get('o_counter', 0) or 0)
                    scene_count = int(performer.get('scene_count', 0) or 0)
                except (ValueError, TypeError):
                    o_counter = 0
                    scene_count = 0
                
                top_o_counter_details.append({
                    'name': name,
                    'o_counter': o_counter,
                    'measurements': measurements or 'N/A',
                    'eu_cup_size': eu_cup_size or 'N/A',
                    'scene_count': scene_count,
                    'id': performer_id
                })
            except Exception as e:
                logger.error(f"Error processing performer details: {e}")
                continue
        
        return top_o_counter_details