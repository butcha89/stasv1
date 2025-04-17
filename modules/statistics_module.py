import logging
from functools import lru_cache
from typing import Dict, List, Any, Optional, Union

from base_module import BaseModule

# Import sub-modules with error handling
try:
    from cup_size_module import CupSizeModule
except ImportError as e:
    logging.error(f"Failed to import CupSizeModule: {e}")
    CupSizeModule = None

try:
    from o_counter_module import OCounterModule
except ImportError as e:
    logging.error(f"Failed to import OCounterModule: {e}")
    OCounterModule = None

try:
    from ratio_module import RatioModule
except ImportError as e:
    logging.error(f"Failed to import RatioModule: {e}")
    RatioModule = None

try:
    from sister_size_module import SisterSizeModule
except ImportError as e:
    logging.error(f"Failed to import SisterSizeModule: {e}")
    SisterSizeModule = None

try:
    from preference_module import PreferenceModule
except ImportError as e:
    logging.error(f"Failed to import PreferenceModule: {e}")
    PreferenceModule = None

logger = logging.getLogger(__name__)

class StatisticsModule:
    """Module for analyzing performer statistics with optimized modular design."""
    
    def __init__(self, stash_client=None):
        """Initialize the statistics module with optional stash client.
        
        Args:
            stash_client: Client for accessing stash data
        """
        self.stash_client = stash_client
        
        # Initialize sub-modules with error handling
        self.cup_size_module = self._init_module(CupSizeModule, "CupSizeModule", [stash_client])
        self.o_counter_module = self._init_module(OCounterModule, "OCounterModule", [stash_client])
        self.ratio_module = self._init_module(RatioModule, "RatioModule", [stash_client, self.cup_size_module])
        self.sister_size_module = self._init_module(SisterSizeModule, "SisterSizeModule", [stash_client, self.cup_size_module])
        self.preference_module = self._init_module(PreferenceModule, "PreferenceModule", [stash_client, self.cup_size_module])
    
    def _init_module(self, module_class, module_name: str, args: List[Any]) -> Optional[Any]:
        """Safely initialize a module with error handling.
        
        Args:
            module_class: The module class to initialize
            module_name: Name of the module for logging
            args: Arguments to pass to the module constructor
            
        Returns:
            Initialized module instance or None if initialization fails
        """
        # Check if module class is available
        if module_class is None:
            logger.error(f"{module_name} class is not available.")
            return None
            
        try:
            # Instantiate the module
            instance = module_class(*args)
            logger.info(f"Successfully initialized {module_name}")
            return instance
        except Exception as e:
            logger.error(f"Error initializing {module_name}: {e}")
            return None
    
    def reload_data(self) -> None:
        """Force reload of data from stash client."""
        # Reset the cache for generate_all_stats
        if hasattr(self, 'generate_all_stats'):
            try:
                # Clear the lru_cache
                self.generate_all_stats.cache_clear()
                logger.info("Cache cleared for generate_all_stats")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
        
        # Reload data in each module
        self._reload_module_data(self.cup_size_module, "CupSizeModule")
        self._reload_module_data(self.o_counter_module, "OCounterModule")
        self._reload_module_data(self.ratio_module, "RatioModule")
        self._reload_module_data(self.sister_size_module, "SisterSizeModule")
        self._reload_module_data(self.preference_module, "PreferenceModule")
    
    def _reload_module_data(self, module, module_name: str) -> None:
        """Safely reload data for a module with error handling.
        
        Args:
            module: Module instance to reload data for
            module_name: Name of the module for logging
        """
        if module is None:
            logger.warning(f"{module_name} is not available for reload.")
            return
            
        try:
            if hasattr(module, 'reload_data'):
                module.reload_data()
                logger.info(f"Successfully reloaded data for {module_name}")
            else:
                logger.warning(f"{module_name} does not have a reload_data method.")
        except Exception as e:
            logger.error(f"Error reloading data for {module_name}: {e}")
    
    def get_cup_size_stats(self) -> Dict[str, Any]:
        """Get statistics about cup sizes.
        
        Returns:
            Dict with cup size statistics
        """
        return self._safe_module_call(
            self.cup_size_module, 
            'get_cup_size_stats', 
            "Error getting cup size statistics", 
            {'cup_size_counts': {}, 'cup_size_dataframe': []}
        )
    
    def get_o_counter_stats(self) -> Dict[str, Any]:
        """Get statistics about o-counter values.
        
        Returns:
            Dict with o-counter statistics
        """
        default_result = {
            'o_counter_dataframe': [],
            'performer_o_counts': {},
            'average_o_counter': 0,
            'median_o_counter': 0,
            'max_o_counter': 0,
            'total_performers': 0
        }
        
        return self._safe_module_call(
            self.o_counter_module, 
            'get_o_counter_stats', 
            "Error getting o-counter statistics", 
            default_result
        )
    
    def get_favorite_o_counter_stats(self) -> Dict[str, Any]:
        """Get statistics about favorite status and o-counter values.
        
        Returns:
            Dict with favorite o-counter statistics
        """
        default_result = {
            'favorite_stats': {'count': 0, 'avg_o_counter': 0, 'median_o_counter': 0, 'max_o_counter': 0, 'performers': []},
            'non_favorite_stats': {'count': 0, 'avg_o_counter': 0, 'median_o_counter': 0, 'max_o_counter': 0, 'performers': []},
            'overall_stats': {'total_performers': 0, 'favorite_percentage': 0, 'non_favorite_percentage': 0}
        }
        
        return self._safe_module_call(
            self.o_counter_module, 
            'get_favorite_o_counter_stats', 
            "Error getting favorite o-counter statistics", 
            default_result
        )
    
    def get_rating_o_counter_correlation(self) -> Dict[str, Any]:
        """Get correlation between ratings and o-counter values.
        
        Returns:
            Dict with rating o-counter correlation
        """
        default_result = {
            'correlation': 0,
            'high_rated_high_o': [],
            'high_rated_low_o': [],
            'low_rated_high_o': [],
            'rating_o_counter_data': []
        }
        
        return self._safe_module_call(
            self.o_counter_module, 
            'get_rating_o_counter_correlation', 
            "Error getting rating o-counter correlation", 
            default_result
        )
    
    def get_ratio_stats(self) -> Dict[str, Any]:
        """Get cup size ratio statistics.
        
        Returns:
            Dict with ratio statistics
        """
        default_result = {'ratio_dataframe': [], 'ratio_stats': []}
        
        return self._safe_module_call(
            self.ratio_module, 
            'get_ratio_stats', 
            "Error getting ratio statistics", 
            default_result
        )
    
    def get_cup_size_o_counter_correlation(self) -> Dict[str, Any]:
        """Get correlation between cup size and o-counter.
        
        Returns:
            Dict with cup size o-counter correlation
        """
        default_result = {'cup_size_o_counter_df': [], 'cup_letter_o_stats': []}
        
        # This method requires passing the o_counter_module as an argument
        if self.cup_size_module is None or self.o_counter_module is None:
            logger.error("Cannot get cup size o-counter correlation: required modules not available.")
            return default_result
            
        try:
            return self.cup_size_module.get_cup_size_o_counter_correlation(self.o_counter_module)
        except Exception as e:
            logger.error(f"Error getting cup size o-counter correlation: {e}")
            return default_result
    
    def get_top_o_counter_performers(self, top_n=10) -> List[Dict]:
        """Get top performers based on o-counter.
        
        Args:
            top_n: Number of top performers to return
            
        Returns:
            List of top performer details
        """
        # This method requires passing the cup_size_module as an argument
        if self.o_counter_module is None or self.cup_size_module is None:
            logger.error("Cannot get top o-counter performers: required modules not available.")
            return []
            
        try:
            return self.o_counter_module.get_top_o_counter_performers(self.cup_size_module, top_n)
        except Exception as e:
            logger.error(f"Error getting top o-counter performers: {e}")
            return []
    
    def create_preference_profile(self, feature_weights=None) -> Dict[str, Any]:
        """Create a detailed profile of user preferences.
        
        Args:
            feature_weights: Optional dictionary of feature weights
            
        Returns:
            Dict with preference profile details
        """
        default_result = {
            'feature_weights': {'o_counter': 2.0, 'rating100': 1.5, 'height_cm': 0.5, 'weight': 0.5, 'eu_cup_numeric': 1.0},
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
        
        if self.preference_module is None:
            logger.error("PreferenceModule is not available.")
            return default_result
            
        try:
            return self.preference_module.create_preference_profile(feature_weights)
        except Exception as e:
            logger.error(f"Error creating preference profile: {e}")
            return default_result
    
    def get_sister_size_stats(self) -> Dict[str, Any]:
        """Get statistics about sister sizes.
        
        Returns:
            Dict with sister size statistics
        """
        default_result = {
            'sister_size_dataframe': [],
            'common_sister_sizes': [],
            'original_vs_sister_stats': {}
        }
        
        return self._safe_module_call(
            self.sister_size_module, 
            'get_sister_size_stats', 
            "Error getting sister size statistics", 
            default_result
        )
    
    def get_volume_stats(self) -> Dict[str, Any]:
        """Get statistics about breast volumes.
        
        Returns:
            Dict with volume statistics
        """
        default_result = {
            'volume_dataframe': [],
            'volume_stats': {},
            'volume_category_stats': [],
            'volume_o_counter_correlation': 0,
            'top_volume_performers': []
        }
        
        return self._safe_module_call(
            self.sister_size_module, 
            'get_volume_stats', 
            "Error getting volume statistics", 
            default_result
        )
    
    def _safe_module_call(
        self, 
        module: Any, 
        method_name: str, 
        error_message: str, 
        default_result: Union[Dict[str, Any], List[Any]]
    ) -> Union[Dict[str, Any], List[Any]]:
        """Safely call a module method with error handling.
        
        Args:
            module: Module instance to call the method on
            method_name: Name of the method to call
            error_message: Error message to log if the call fails
            default_result: Default result to return if the call fails
            
        Returns:
            Result of the method call or default_result if call fails
        """
        if module is None:
            logger.error(f"{error_message}: module not available.")
            return default_result
            
        try:
            method = getattr(module, method_name, None)
            if method is None:
                logger.error(f"{error_message}: method {method_name} not found.")
                return default_result
                
            result = method()
            return result
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            return default_result
    
    @lru_cache(maxsize=1)
    def generate_all_stats(self) -> Dict[str, Any]: 
        """Generate all statistics and return them in a single dictionary.
        
        Returns:
            Dict containing all statistics
        """
        logger.info("Generating all statistics...")
        
        # Initialize result with empty values for each statistic
        result = {
            'cup_size_stats': {'cup_size_counts': {}, 'cup_size_dataframe': []},
            'o_counter_stats': {
                'o_counter_dataframe': [],
                'performer_o_counts': {},
                'average_o_counter': 0,
                'median_o_counter': 0,
                'max_o_counter': 0,
                'total_performers': 0
            },
            'favorite_o_counter_stats': {
                'favorite_stats': {'count': 0, 'avg_o_counter': 0, 'median_o_counter': 0, 'max_o_counter': 0, 'performers': []},
                'non_favorite_stats': {'count': 0, 'avg_o_counter': 0, 'median_o_counter': 0, 'max_o_counter': 0, 'performers': []},
                'overall_stats': {'total_performers': 0, 'favorite_percentage': 0, 'non_favorite_percentage': 0}
            },
            'rating_o_counter_correlation': {
                'correlation': 0,
                'high_rated_high_o': [],
                'high_rated_low_o': [],
                'low_rated_high_o': [],
                'rating_o_counter_data': []
            },
            'ratio_stats': {'ratio_dataframe': [], 'ratio_stats': []},
            'cup_size_o_counter_correlation': {'cup_size_o_counter_df': [], 'cup_letter_o_stats': []},
            'top_o_counter_performers': [],
            'preference_profile': {
                'feature_weights': {'o_counter': 2.0, 'rating100': 1.5, 'height_cm': 0.5, 'weight': 0.5, 'eu_cup_numeric': 1.0},
                'preference_profile': {
                    'total_relevant_performers': 0,
                    'avg_o_counter': 0,
                    'avg_rating': 0,
                    'most_common_cup_sizes': []
                },
                'cluster_analysis': {'clusters': {}, 'cluster_centroids': []},
                'cup_size_distribution': {'total_cup_sizes': {}, 'relevant_cup_size_distribution': {}},
                'top_performers_by_cluster': {}
            },
            'sister_size_stats': {
                'sister_size_dataframe': [],
                'common_sister_sizes': [],
                'original_vs_sister_stats': {}
            },
            'volume_stats': {
                'volume_dataframe': [],
                'volume_stats': {},
                'volume_category_stats': [],
                'volume_o_counter_correlation': 0,
                'top_volume_performers': []
            }
        }
        
        # Collect all statistics with error handling for each
        try:
            result['cup_size_stats'] = self.get_cup_size_stats()
            logger.info("Cup size statistics collected.")
        except Exception as e:
            logger.error(f"Error collecting cup size statistics: {e}")
        
        try:
            result['o_counter_stats'] = self.get_o_counter_stats()
            logger.info("O-counter statistics collected.")
        except Exception as e:
            logger.error(f"Error collecting o-counter statistics: {e}")
        
        try:
            result['favorite_o_counter_stats'] = self.get_favorite_o_counter_stats()
            logger.info("Favorite o-counter statistics collected.")
        except Exception as e:
            logger.error(f"Error collecting favorite o-counter statistics: {e}")
        
        try:
            result['rating_o_counter_correlation'] = self.get_rating_o_counter_correlation()
            logger.info("Rating o-counter correlation collected.")
        except Exception as e:
            logger.error(f"Error collecting rating o-counter correlation: {e}")
        
        try:
            result['ratio_stats'] = self.get_ratio_stats()
            logger.info("Ratio statistics collected.")
        except Exception as e:
            logger.error(f"Error collecting ratio statistics: {e}")
        
        try:
            result['cup_size_o_counter_correlation'] = self.get_cup_size_o_counter_correlation()
            logger.info("Cup size o-counter correlation collected.")
        except Exception as e:
            logger.error(f"Error collecting cup size o-counter correlation: {e}")
        
        try:
            result['top_o_counter_performers'] = self.get_top_o_counter_performers()
            logger.info("Top o-counter performers collected.")
        except Exception as e:
            logger.error(f"Error collecting top o-counter performers: {e}")
        
        try:
            result['preference_profile'] = self.create_preference_profile()
            logger.info("Preference profile collected.")
        except Exception as e:
            logger.error(f"Error collecting preference profile: {e}")
        
        try:
            result['sister_size_stats'] = self.get_sister_size_stats()
            logger.info("Sister size statistics collected.")
        except Exception as e:
            logger.error(f"Error collecting sister size statistics: {e}")
        
        try:
            result['volume_stats'] = self.get_volume_stats()
            logger.info("Volume statistics collected.")
        except Exception as e:
            logger.error(f"Error collecting volume statistics: {e}")
    
        logger.info("All statistics collected.")
        return result