import pandas as pd
import numpy as np
import logging
from collections import Counter
from typing import Dict, List, Any, Optional

# Ensure sklearn dependency is handled gracefully
try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create dummy classes for type annotations
    class SimpleImputer: pass
    class StandardScaler: pass
    class KMeans: pass
    class Pipeline: pass

from base_module import BaseModule

logger = logging.getLogger(__name__)

class PreferenceModule(BaseModule):
    """Module for analyzing performer preferences and identifying patterns."""
    
    def __init__(self, stash_client=None, cup_size_module=None):
        """Initialize the preference module.
        
        Args:
            stash_client: Client for accessing stash data
            cup_size_module: Reference to cup size module for cup data
        """
        super().__init__(stash_client)
        self.cup_size_module = cup_size_module
    
    def create_preference_profile(self, feature_weights=None) -> Dict[str, Any]:
        """Create a detailed profile of user preferences with configurable feature weights.
        
        Args:
            feature_weights: Optional dictionary of feature weights to override defaults
            
        Returns:
            Dict with preference profile details
        """
        # Check if sklearn is available
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is not available. Clustering cannot be performed.")
            return self._create_fallback_profile(feature_weights)
        
        # Default weights
        default_weights = {
            'o_counter': 2.0,
            'rating100': 1.5,
            'height_cm': 0.5,
            'weight': 0.5,
            'eu_cup_numeric': 1.0
        }
        
        # Update with custom weights if provided and valid
        if feature_weights and isinstance(feature_weights, dict):
            for key, value in feature_weights.items():
                try:
                    # Ensure values are numeric
                    default_weights[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid feature weight for {key}: {value}. Using default.")
        
        # Initialize empty results structure
        empty_result = {
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
        
        # Combine favorites and performers with O-Counter > 1 with robust validation
        try:
            relevant_performers = []
            for p in self.performers_data:
                if not isinstance(p, dict):
                    continue
                
                # Get performer attributes with safe extraction
                is_favorite = p.get('favorite', False)
                
                # Safely convert o_counter to float
                try:
                    o_counter = p.get('o_counter')
                    o_counter = 0 if o_counter is None else float(o_counter)
                except (ValueError, TypeError):
                    o_counter = 0
                
                # Include if favorite or o_counter > 1
                if is_favorite or o_counter > 1:
                    relevant_performers.append(p)
        except Exception as e:
            logger.error(f"Error filtering relevant performers: {e}")
            return empty_result
        
        if not relevant_performers:
            return empty_result
        
        # Get cup size data safely
        cup_df = pd.DataFrame()
        if self.cup_size_module:
            try:
                cup_df = self.cup_size_module.cup_size_df
            except Exception as e:
                logger.error(f"Error getting cup size data: {e}")
        
        # Create performer data for clustering efficiently with error handling
        try:
            performer_data = []
            relevant_cup_sizes = []
            
            for p in relevant_performers:
                p_id = p.get('id')
                
                # Skip if no id
                if not p_id:
                    continue
                
                # Safely extract and convert numeric values
                try:
                    o_counter = float(p.get('o_counter', 0) or 0)
                    rating100 = float(p.get('rating100', 0) or 0)
                    height_cm = float(p.get('height_cm', 0) or 0)
                    weight = float(p.get('weight', 0) or 0)
                except (ValueError, TypeError):
                    o_counter = 0
                    rating100 = 0
                    height_cm = 0
                    weight = 0
                
                # Find cup size data in cup_df
                cup_numeric = 0
                if not cup_df.empty:
                    try:
                        cup_data = cup_df[cup_df['id'] == p_id]
                        if not cup_data.empty:
                            cup_numeric = float(cup_data['cup_numeric'].values[0])
                    except (IndexError, ValueError, TypeError):
                        cup_numeric = 0
                
                # For cup size distribution with error handling
                if o_counter > 0 and self.cup_size_module:
                    try:
                        measurements = p.get('measurements')
                        eu_cup_size, _ = self.cup_size_module._convert_bra_size(measurements)
                        if eu_cup_size:
                            relevant_cup_sizes.append(eu_cup_size)
                    except Exception:
                        pass
                
                performer_data.append({
                    'o_counter': o_counter,
                    'rating100': rating100,
                    'height_cm': height_cm,
                    'weight': weight,
                    'eu_cup_numeric': cup_numeric,
                    'name': p.get('name', 'Unknown')
                })
        except Exception as e:
            logger.error(f"Error creating performer data: {e}")
            return empty_result
        
        # Create DataFrame from collected data
        try:
            df = pd.DataFrame(performer_data)
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return empty_result
        
        # Skip clustering if not enough data
        if len(df) < 3:
            return self._create_basic_profile(df, relevant_performers, relevant_cup_sizes, default_weights)
        
        # Features for clustering
        features = list(default_weights.keys())
        
        # Only use features that exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        # Preprocessing Pipeline
        try:
            preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Filter rows with insufficient data
            X = df[features].dropna(thresh=len(features)//2)
            
            # If insufficient data after filtering, return basic profile
            if len(X) < 3:
                return self._create_basic_profile(df, relevant_performers, relevant_cup_sizes, default_weights)
                
            # Preprocessing
            X_processed = preprocessor.fit_transform(X)
            
            # Apply feature weights
            weighted_features = np.copy(X_processed)
            for i, feature in enumerate(features):
                weighted_features[:, i] *= default_weights[feature]
            
            # K-Means Clustering
            kmeans = KMeans(n_clusters=min(3, len(X)), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(weighted_features)
            
            # Add cluster assignments back to dataframe
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
            df_copy = df.copy()
            df_copy['o_counter'] = pd.to_numeric(df_copy['o_counter'], errors='coerce').fillna(0)
            o_count_df = df_copy[df_copy['o_counter'] > 0]
            avg_o_counter = o_count_df['o_counter'].mean() if not o_count_df.empty else 0
            avg_rating = o_count_df['rating100'].mean() if not o_count_df.empty else 0
            
            # Cup-Size frequencies using Counter
            cup_size_counter = Counter(relevant_cup_sizes)
            most_common_cup_sizes = cup_size_counter.most_common(3)
            
            # Get top performers by cluster
            top_performers_by_cluster = {}
            for cluster in range(kmeans.n_clusters):
                try:
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
                except Exception as e:
                    logger.error(f"Error getting top performers for cluster {cluster}: {e}")
                    top_performers_by_cluster[cluster] = []
            
            # Get cup size counts from the cached data
            cup_size_stats = {}
            if self.cup_size_module:
                try:
                    cup_size_stats = self.cup_size_module.get_cup_size_stats()
                except Exception as e:
                    logger.error(f"Error getting cup size stats: {e}")
            
            cup_size_counts = cup_size_stats.get('cup_size_counts', {})
            
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
            return self._create_basic_profile(df, relevant_performers, relevant_cup_sizes, default_weights)
    
    def _create_basic_profile(self, df, relevant_performers, relevant_cup_sizes, default_weights):
        """Create a basic profile without clustering when there's insufficient data.
        
        Args:
            df: DataFrame with performer data
            relevant_performers: List of relevant performer dictionaries
            relevant_cup_sizes: List of cup sizes for relevant performers
            default_weights: Dictionary of feature weights
            
        Returns:
            Dict with basic preference profile details
        """
        try:
            # Calculate basic statistics if DataFrame is not empty
            avg_o_counter = 0
            avg_rating = 0
            if not df.empty:
                try:
                    # Ensure numeric columns
                    df['o_counter'] = pd.to_numeric(df['o_counter'], errors='coerce').fillna(0)
                    df['rating100'] = pd.to_numeric(df['rating100'], errors='coerce').fillna(0)
                    
                    # Calculate averages
                    avg_o_counter = df['o_counter'].mean()
                    avg_rating = df['rating100'].mean()
                except Exception as e:
                    logger.error(f"Error calculating basic statistics: {e}")
            
            # Get cup size stats if available
            cup_size_stats = {}
            if self.cup_size_module:
                try:
                    cup_size_stats = self.cup_size_module.get_cup_size_stats()
                except Exception:
                    pass
            
            # Create a basic profile
            return {
                'feature_weights': default_weights,
                'preference_profile': {
                    'total_relevant_performers': len(relevant_performers),
                    'avg_o_counter': avg_o_counter,
                    'avg_rating': avg_rating,
                    'most_common_cup_sizes': [
                        {'size': size, 'count': count} 
                        for size, count in Counter(relevant_cup_sizes).most_common(3)
                    ]
                },
                'cluster_analysis': {'clusters': {}, 'cluster_centroids': []},
                'cup_size_distribution': {
                    'total_cup_sizes': cup_size_stats.get('cup_size_counts', {}),
                    'relevant_cup_size_distribution': dict(Counter(relevant_cup_sizes))
                },
                'top_performers_by_cluster': {}
            }
        except Exception as e:
            logger.error(f"Error creating basic profile: {e}")
            return self._create_fallback_profile(default_weights)
    
    def _create_fallback_profile(self, feature_weights=None):
        """Create a minimal fallback profile when all else fails.
        
        Args:
            feature_weights: Dictionary of feature weights (or None)
            
        Returns:
            Dict with minimal preference profile
        """
        # Default weights
        default_weights = {
            'o_counter': 2.0,
            'rating100': 1.5,
            'height_cm': 0.5,
            'weight': 0.5,
            'eu_cup_numeric': 1.0
        }
        
        # Update with custom weights if provided and valid
        if feature_weights and isinstance(feature_weights, dict):
            for key, value in feature_weights.items():
                try:
                    default_weights[key] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Return minimal structure
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