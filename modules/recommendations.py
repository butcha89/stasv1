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
    
    def _calculate_similarity(self, base_performer, target_performer):
        """
        Calculate comprehensive similarity between two performers
        Considers multiple factors with weighted importance
        """
        similarity = 0
        weights = {
            'cup_to_bmi': 0.3,
            'o_counter': 0.2,
            'scene_count': 0.1,
            'band_size': 0.2,
            'cup_letter': 0.2
        }
        
        # Cup-to-BMI similarity (normalized)
        if base_performer.get('bmi') and target_performer.get('bmi'):
            cup_to_bmi_base = base_performer.get('cup_to_bmi', 0)
            cup_to_bmi_target = target_performer.get('cup_to_bmi', 0)
            
            if cup_to_bmi_base and cup_to_bmi_target:
                bmi_similarity = 1 - abs(cup_to_bmi_base - cup_to_bmi_target) / max(cup_to_bmi_base, cup_to_bmi_target)
                similarity += bmi_similarity * weights['cup_to_bmi']
        
        # O-Counter similarity
        base_o_counter = base_performer.get('o_counter', 0)
        target_o_counter = target_performer.get('o_counter', 0)
        if base_o_counter > 0 and target_o_counter > 0:
            o_counter_similarity = 1 - abs(base_o_counter - target_o_counter) / max(base_o_counter, target_o_counter)
            similarity += o_counter_similarity * weights['o_counter']
        
        # Scene count similarity
        base_scene_count = base_performer.get('scene_count', 0)
        target_scene_count = target_performer.get('scene_count', 0)
        if base_scene_count > 0 and target_scene_count > 0:
            scene_count_similarity = 1 - abs(base_scene_count - target_scene_count) / max(base_scene_count, target_scene_count)
            similarity += scene_count_similarity * weights['scene_count']
        
        # Band size similarity
        base_band = base_performer.get('band_size')
        target_band = target_performer.get('band_size')
        if base_band and target_band:
            band_similarity = 1 - abs(int(base_band) - int(target_band)) / max(int(base_band), int(target_band))
            similarity += band_similarity * weights['band_size']
        
        # Cup letter similarity
        base_cup = base_performer.get('cup_letter')
        target_cup = target_performer.get('cup_letter')
        if base_cup and target_cup:
            cup_letters = 'ABCDEFGHIJK'
            base_cup_index = cup_letters.index(base_cup)
            target_cup_index = cup_letters.index(target_cup)
            cup_similarity = 1 - abs(base_cup_index - target_cup_index) / len(cup_letters)
            similarity += cup_similarity * weights['cup_letter']
        
        return similarity
    
    def recommend_performers(self):
        """Generate performer recommendations based on top O-Counter performers"""
        if not self.performers_data:
            logger.warning("No performers data available for recommendations")
            return []
        
        try:
            # Get top O-Counter performers from statistics
            stats = self.stats_module.generate_all_stats()
            top_o_counter_performers = stats.get('top_o_counter_performers', [])
            
            recommendations = []
            
            for base_performer in top_o_counter_performers:
                # Find similar performers
                similar_performers = []
                
                for target_performer in self.performers_data:
                    # Skip the base performer itself and performers with no o-counter
                    if (target_performer.get('id') == base_performer.get('id') or 
                        target_performer.get('o_counter', 0) == 0):
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(base_performer, target_performer)
                    
                    if similarity > 0.5:  # Threshold for recommendation
                        similar_performers.append({
                            'id': target_performer.get('id'),
                            'name': target_performer.get('name', 'Unknown'),
                            'cup_size': f"{target_performer.get('band_size', 'N/A')}{target_performer.get('cup_letter', '')}",
                            'cup_to_bmi': target_performer.get('cup_to_bmi'),
                            'o_count': target_performer.get('o_counter', 0),
                            'similarity': similarity
                        })
                
                # Sort similar performers by similarity
                similar_performers.sort(key=lambda x: x['similarity'], reverse=True)
                
                recommendations.append({
                    'performer': {
                        'name': base_performer.get('name', 'Unknown'),
                        'measurements': base_performer.get('measurements', 'N/A'),
                        'o_count': base_performer.get('o_counter', 0),
                        'cup_to_bmi': base_performer.get('cup_to_bmi')
                    },
                    'similar_performers': similar_performers[:5]  # Top 5 similar performers
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error in performer recommendations: {e}")
            return []
    
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
