import logging
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

class RecommendationModule:
    def __init__(self, stash_client=None, stats_module=None):
        """Initialize the recommendation module"""
        self.stash_client = stash_client
        self.stats_module = stats_module
        self.performers_data = []
        self.scenes_data = []
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load performers and scenes data"""
        try:
            self.performers_data = self.stash_client.get_performers() if self.stash_client else []
            self.scenes_data = self.stash_client.get_scenes() if self.stash_client else []
            logger.info(f"Loaded {len(self.performers_data)} performers and {len(self.scenes_data)} scenes")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.performers_data = []
            self.scenes_data = []
    
    def recommend_performers(self):
        """Generate performer recommendations"""
        if not self.performers_data:
            logger.warning("No performers data available for recommendations")
            return []
        
        try:
            # Basic recommendation logic using available data
            recommendations = []
            
            # Prioritize performers with specific characteristics
            for performer in self.performers_data:
                # Basic recommendation criteria
                if (performer.get('scene_count', 0) > 0 and 
                    not performer.get('favorite', False)):
                    
                    similar_performers = self._find_similar_performers(performer)
                    
                    if similar_performers:
                        recommendations.append({
                            'performer': {
                                'id': performer.get('id'),
                                'name': performer.get('name', 'Unknown'),
                                'scene_count': performer.get('scene_count', 0),
                                'measurements': performer.get('measurements', ''),
                                'favorite': performer.get('favorite', False)
                            },
                            'similar_performers': similar_performers
                        })
            
            # Sort recommendations
            recommendations.sort(
                key=lambda x: (
                    -x['performer'].get('scene_count', 0),
                    x['performer'].get('name', '')
                )
            )
            
            return recommendations[:10]  # Limit to top 10
        
        except Exception as e:
            logger.error(f"Error in performer recommendations: {e}")
            return []
    
    def _find_similar_performers(self, base_performer, top_n=3):
        """Find similar performers based on basic criteria"""
        similar = []
        
        for performer in self.performers_data:
            # Skip the base performer and already favorite performers
            if (performer.get('id') == base_performer.get('id') or 
                performer.get('favorite', False)):
                continue
            
            # Basic similarity criteria
            similarity_score = self._calculate_similarity(base_performer, performer)
            
            if similarity_score > 0:
                similar.append({
                    'id': performer.get('id'),
                    'name': performer.get('name', 'Unknown'),
                    'similarity': similarity_score,
                    'scene_count': performer.get('scene_count', 0),
                    'measurements': performer.get('measurements', '')
                })
        
        # Sort and return top N similar performers
        return sorted(
            similar, 
            key=lambda x: x['similarity'], 
            reverse=True
        )[:top_n]
    
    def _calculate_similarity(self, performer1, performer2):
        """Calculate basic similarity between two performers"""
        similarity = 0
        
        # Compare scene count
        if performer1.get('scene_count', 0) > 0 and performer2.get('scene_count', 0) > 0:
            similarity += min(performer1['scene_count'], performer2['scene_count']) / 10
        
        # Compare measurements if available
        if performer1.get('measurements') and performer2.get('measurements'):
            if performer1['measurements'] == performer2['measurements']:
                similarity += 1
        
        return similarity
    
    def recommend_scenes(self):
        """Generate scene recommendations"""
        if not self.scenes_data:
            logger.warning("No scenes data available for recommendations")
            return {}
        
        try:
            # Basic scene recommendation logic
            recommendations = {
                'favorite_performer_scenes': [],
                'non_favorite_performer_scenes': [],
                'recommended_performer_scenes': []
            }
            
            # Process scenes
            for scene in self.scenes_data:
                # Basic recommendation criteria
                if scene.get('o_counter', 0) == 0:  # Unwatched scenes
                    performers = scene.get('performers', [])
                    tags = scene.get('tags', [])
                    
                    # Check for favorite performers
                    has_favorite_performer = any(p.get('favorite', False) for p in performers)
                    
                    # Prepare recommendation
                    recommendation = {
                        'id': scene.get('id'),
                        'title': scene.get('title', 'Unknown'),
                        'performers': [p.get('name', 'Unknown') for p in performers],
                        'tags': [t.get('name', '') for t in tags],
                        'similarity': len(tags)  # Simple similarity metric
                    }
                    
                    # Categorize recommendations
                    if has_favorite_performer:
                        recommendations['favorite_performer_scenes'].append(recommendation)
                    else:
                        recommendations['non_favorite_performer_scenes'].append(recommendation)
            
            # Sort and limit recommendations
            for category in recommendations:
                recommendations[category] = sorted(
                    recommendations[category], 
                    key=lambda x: x['similarity'], 
                    reverse=True
                )[:10]
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error in scene recommendations: {e}")
            return {}
