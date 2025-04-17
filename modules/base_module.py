import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class BaseModule:
    """Base module with common functionality for statistics modules."""
    
    def __init__(self, stash_client=None):
        """Initialize the base module with optional stash client.
        
        Args:
            stash_client: Client for accessing stash data
        """
        self.stash_client = stash_client
        self._performers_data = None
        self._scenes_data = None
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @property
    def performers_data(self) -> List[Dict]:
        """Lazy-loaded performers data.
        
        Returns:
            List of performer dictionaries or empty list if data not available
        """
        if self._performers_data is None:
            self._load_data()
        return self._performers_data or []
    
    @property
    def scenes_data(self) -> List[Dict]:
        """Lazy-loaded scenes data.
        
        Returns:
            List of scene dictionaries or empty list if data not available
        """
        if self._scenes_data is None:
            self._load_data()
        return self._scenes_data or []
        
    def _load_data(self) -> None:
        """Load data from Stash client once and cache it."""
        try:
            if self.stash_client is None:
                logger.warning("No stash_client provided. Using empty data.")
                self._performers_data = []
                self._scenes_data = []
                return
                
            logger.info("Loading data from stash client...")
            
            # Load performers with error handling
            try:
                self._performers_data = self.stash_client.get_performers()
                # Validate the response
                if not isinstance(self._performers_data, list):
                    logger.error("get_performers() did not return a list. Using empty list.")
                    self._performers_data = []
            except Exception as e:
                logger.error(f"Error loading performers data: {e}")
                self._performers_data = []
            
            # Load scenes with error handling
            try:
                self._scenes_data = self.stash_client.get_scenes()
                # Validate the response
                if not isinstance(self._scenes_data, list):
                    logger.error("get_scenes() did not return a list. Using empty list.")
                    self._scenes_data = []
            except Exception as e:
                logger.error(f"Error loading scenes data: {e}")
                self._scenes_data = []
                
            logger.info(f"Data loaded: {len(self._performers_data)} performers, {len(self._scenes_data)} scenes")
            
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            self._performers_data = []
            self._scenes_data = []
    
    def reload_data(self) -> None:
        """Force reload of data from stash client."""
        logger.info(f"Reloading data for {self.__class__.__name__}...")
        # Reset cached data
        self._performers_data = None
        self._scenes_data = None
        
        # Access properties to trigger lazy loading
        try:
            performers_count = len(self.performers_data)
            scenes_count = len(self.scenes_data)
            logger.info(f"Data reloaded: {performers_count} performers, {scenes_count} scenes")
        except Exception as e:
            logger.error(f"Error during data reload: {e}")

    def get_performer_by_id(self, performer_id: str) -> Optional[Dict]:
        """Get performer data by ID.
        
        Args:
            performer_id: ID of the performer to retrieve
            
        Returns:
            Performer data dictionary or None if not found
        """
        if not performer_id:
            logger.warning("Empty performer_id provided to get_performer_by_id")
            return None
            
        try:
            # Ensure we have performers_data loaded
            performers = self.performers_data
            
            # Search for performer with matching ID
            for performer in performers:
                if not isinstance(performer, dict):
                    continue
                    
                if performer.get('id') == performer_id:
                    return performer
                    
            # If we get here, no matching performer was found
            logger.debug(f"No performer found with ID: {performer_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_performer_by_id: {e}")
            return None
    
    def validate_performer_data(self, performer: Dict) -> Dict:
        """Validate and normalize performer data.
        
        Args:
            performer: Raw performer data dictionary
            
        Returns:
            Normalized performer data with defaults for missing values
        """
        if not isinstance(performer, dict):
            logger.warning("Non-dictionary performer data provided")
            return {}
            
        # Create normalized performer with default values
        normalized = {
            'id': performer.get('id', ''),
            'name': performer.get('name', 'Unknown'),
            'measurements': performer.get('measurements', ''),
            'height_cm': self._safe_numeric(performer.get('height_cm'), 0),
            'weight': self._safe_numeric(performer.get('weight'), 0),
            'favorite': bool(performer.get('favorite', False)),
            'rating100': self._safe_numeric(performer.get('rating100'), 0),
            'scene_count': self._safe_numeric(performer.get('scene_count'), 0),
            'o_counter': self._safe_numeric(performer.get('o_counter'), 0)
        }
        
        return normalized
    
    def validate_scene_data(self, scene: Dict) -> Dict:
        """Validate and normalize scene data.
        
        Args:
            scene: Raw scene data dictionary
            
        Returns:
            Normalized scene data with defaults for missing values
        """
        if not isinstance(scene, dict):
            logger.warning("Non-dictionary scene data provided")
            return {}
            
        # Extract performers safely
        performers = []
        if 'performers' in scene and isinstance(scene['performers'], list):
            for p in scene['performers']:
                if isinstance(p, dict):
                    performers.append({
                        'id': p.get('id', ''),
                        'name': p.get('name', 'Unknown'),
                        'favorite': bool(p.get('favorite', False))
                    })
        
        # Extract tags safely
        tags = []
        if 'tags' in scene and isinstance(scene['tags'], list):
            for t in scene['tags']:
                if isinstance(t, dict):
                    tags.append({
                        'id': t.get('id', ''),
                        'name': t.get('name', '')
                    })
        
        # Create normalized scene with default values
        normalized = {
            'id': scene.get('id', ''),
            'title': scene.get('title', 'Unknown'),
            'o_counter': self._safe_numeric(scene.get('o_counter'), 0),
            'performers': performers,
            'tags': tags
        }
        
        return normalized
    
    def _safe_numeric(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to numeric type with fallback.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Numeric value or default
        """
        if value is None:
            return default
            
        try:
            return float(value)
        except (ValueError, TypeError):
            return default