import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os
import math

# Füge das übergeordnete Verzeichnis zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stash_api import StashClient
from modules.statistics import StatisticsModule

# Eigene Implementierung der Kosinus-Ähnlichkeit ohne scikit-learn
def cosine_similarity_manual(X):
    """
    Berechnet die Kosinus-Ähnlichkeitsmatrix für eine Eingabematrix X
    ohne scikit-learn zu verwenden.
    
    Args:
        X: Eine Matrix (2D numpy array) mit Samples x Features
        
    Returns:
        Eine quadratische Matrix mit den Kosinus-Ähnlichkeiten zwischen allen Paaren von Samples
    """
    # Normalisiere jeden Vektor
    norms = np.sqrt(np.sum(X * X, axis=1))
    norms[norms == 0] = 1.0  # Vermeide Division durch Null
    
    # Normalisierte Matrix
    X_normalized = X / norms[:, np.newaxis]
    
    # Berechne Ähnlichkeitsmatrix
    similarity_matrix = np.dot(X_normalized, X_normalized.T)
    
    return similarity_matrix

class RecommendationModule:
    def __init__(self, stash_client=None, stats_module=None):
        """Initialize the recommendation module"""
        self.stash_client = stash_client or StashClient()
        self.stats_module = stats_module or StatisticsModule(self.stash_client)
        self.similarity_threshold = 0.7
        self.max_recommendations = 10
        
    def recommend_performers(self):
        """Recommend performers based on cup size, ratios, and other metrics"""
        # Get statistics data
        stats = self.stats_module.generate_all_stats()
        
        # Get the dataframe with cup sizes and ratios
        cup_df = stats['ratio_stats'].get('ratio_dataframe', pd.DataFrame())
        
        if cup_df.empty:
            return []
            
        # Add o-counter data
        corr_stats = stats['correlation_stats']
        if corr_stats:
            cup_o_df = corr_stats.get('cup_size_o_counter_df', pd.DataFrame())
            if not cup_o_df.empty:
                cup_df = cup_df.merge(cup_o_df[['id', 'total_o_count', 'o_scene_count']], 
                                     on='id', how='left')
        
        # Fill missing values
        cup_df['total_o_count'] = cup_df.get('total_o_count', 0).fillna(0)
        cup_df['o_scene_count'] = cup_df.get('o_scene_count', 0).fillna(0)
        
        # Create feature matrix for similarity calculation
        features = ['cup_letter_value', 'band_size', 'height_cm', 'weight', 'bmi',
                   'cup_to_bmi', 'cup_to_height', 'cup_to_weight']
        
        # Keep only rows with sufficient data
        feature_df = cup_df.dropna(subset=['cup_letter_value', 'band_size'])
        
        if feature_df.empty:
            return []
            
        # Normalize features
        for feature in features:
            if feature in feature_df.columns:
                feature_df[feature] = feature_df[feature].fillna(feature_df[feature].mean())
                feature_df[f'{feature}_norm'] = (feature_df[feature] - feature_df[feature].min()) / \
                                              (feature_df[feature].max() - feature_df[feature].min())
        
        norm_features = [f'{feature}_norm' for feature in features if f'{feature}_norm' in feature_df.columns]
        
        if not norm_features:
            return []
            
        # Calculate similarity matrix
        X = feature_df[norm_features].values
        similarity = cosine_similarity_manual(X)
        
        # Find recommendations
        recommendations = []
        
        for i, performer_id in enumerate(feature_df['id']):
            performer_name = feature_df.iloc[i]['name']
            favorite = feature_df.iloc[i]['favorite']
            o_count = feature_df.iloc[i]['total_o_count']
            
            # Skip performers with high o-count as they're already watched
            if o_count > 3:
                continue
                
            # Find similar performers
            similar_indices = np.where(similarity[i] >= self.similarity_threshold)[0]
            similar_performers = []
            
            for idx in similar_indices:
                if idx != i:  # Don't include self
                    similar_id = feature_df.iloc[idx]['id']
                    similar_name = feature_df.iloc[idx]['name']
                    similar_o_count = feature_df.iloc[idx]['total_o_count']
                    similar_favorite = feature_df.iloc[idx]['favorite']
                    sim_score = similarity[i][idx]
                    
                    # Prioritize performers with low o-count
                    if similar_o_count == 0:
                        score_boost = 0.2
                    else:
                        score_boost = 0
                        
                    similar_performers.append({
                        'id': similar_id,
                        'name': similar_name,
                        'similarity': sim_score + score_boost,
                        'o_count': similar_o_count,
                        'favorite': similar_favorite,
                        'cup_size': f"{feature_df.iloc[idx]['band_size']}{feature_df.iloc[idx]['cup_letter']}",
                        'height_cm': feature_df.iloc[idx]['height_cm'],
                        'weight': feature_df.iloc[idx]['weight'],
                        'bmi': feature_df.iloc[idx]['bmi']
                    })
            
            # Sort by similarity score
            similar_performers.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Take top recommendations
            top_similar = similar_performers[:self.max_recommendations]
            
            if top_similar:
                recommendations.append({
                    'performer': {
                        'id': performer_id,
                        'name': performer_name,
                        'favorite': favorite,
                        'o_count': o_count,
                        'cup_size': f"{feature_df.iloc[i]['band_size']}{feature_df.iloc[i]['cup_letter']}",
                        'height_cm': feature_df.iloc[i]['height_cm'],
                        'weight': feature_df.iloc[i]['weight'],
                        'bmi': feature_df.iloc[i]['bmi']
                    },
                    'similar_performers': top_similar
                })
        
        # Sort recommendations to prioritize favorites with low o-count
        recommendations.sort(key=lambda x: (
            not x['performer']['favorite'],  # Favorites first
            x['performer']['o_count'],       # Then by lowest o-count
            -len(x['similar_performers'])    # Then by most recommendations
        ))
        
        return recommendations
    
    def recommend_scenes(self):
        """Recommend scenes based on tags and performers"""
        # Get all scenes
        if not hasattr(self, 'scenes_data') or not self.scenes_data:
            self.scenes_data = self.stash_client.get_scenes()
        
        # Get performer recommendations
        performer_recs = self.recommend_performers()
        recommended_performer_ids = set()
        
        for rec in performer_recs:
            for similar in rec['similar_performers']:
                recommended_performer_ids.add(similar['id'])
        
        # Process scenes
        scene_tags = {}
        scene_performers = {}
        scene_o_counts = {}
        favorite_performer_scenes = set()
        
        for scene in self.scenes_data:
            scene_id = scene.get('id')
            
            # Extract tags
            tags = [tag.get('name') for tag in scene.get('tags', [])]
            scene_tags[scene_id] = set(tags)
            
            # Extract performers
            performers = scene.get('performers', [])
            performer_ids = [p.get('id') for p in performers]
            scene_performers[scene_id] = set(performer_ids)
            
            # Check if scene has favorite performers
            has_favorite = any(p.get('favorite', False) for p in performers)
            if has_favorite:
                favorite_performer_scenes.add(scene_id)
            
            # Get o-counter
            scene_o_counts[scene_id] = scene.get('o_counter', 0)
        
        # Find scenes with positive o-counter
        watched_scenes = {scene_id for scene_id, count in scene_o_counts.items() if count > 0}
        
        # Calculate tag similarity between scenes
        tag_similarity = defaultdict(dict)
        
        for watched_id in watched_scenes:
            watched_tags = scene_tags.get(watched_id, set())
            
            if not watched_tags:
                continue
                
            for scene_id in scene_tags:
                # Skip comparing to itself or other watched scenes
                if scene_id == watched_id or scene_id in watched_scenes:
                    continue
                    
                scene_tag_set = scene_tags.get(scene_id, set())
                
                if not scene_tag_set:
                    continue
                    
                # Calculate Jaccard similarity
                intersection = len(watched_tags.intersection(scene_tag_set))
                union = len(watched_tags.union(scene_tag_set))
                
                if union > 0:
                    similarity = intersection / union
                    tag_similarity[watched_id][scene_id] = similarity
        
        # Generate recommendations
        scene_recommendations = {
            'favorite_performer_scenes': [],
            'non_favorite_performer_scenes': [],
            'recommended_performer_scenes': []
        }
        
        # Process each watched scene
        for watched_id in watched_scenes:
            similar_scenes = tag_similarity.get(watched_id, {})
            
            # Sort by similarity
            sorted_scenes = sorted(similar_scenes.items(), key=lambda x: x[1], reverse=True)
            
            # Take top matches
            for scene_id, similarity in sorted_scenes[:self.max_recommendations]:
                if similarity < self.similarity_threshold:
                    continue
                    
                # Get scene details
                scene_details = next((s for s in self.scenes_data if s.get('id') == scene_id), None)
                
                if not scene_details:
                    continue
                    
                recommendation = {
                    'id': scene_id,
                    'title': scene_details.get('title', ''),
                    'similarity': similarity,
                    'tags': list(scene_tags.get(scene_id, set())),
                    'performers': [p.get('name') for p in scene_details.get('performers', [])]
                }
                
                # Check if scene has recommended performers
                scene_performer_ids = scene_performers.get(scene_id, set())
                has_recommended_performer = bool(scene_performer_ids.intersection(recommended_performer_ids))
                
                if has_recommended_performer:
                    scene_recommendations['recommended_performer_scenes'].append(recommendation)
                elif scene_id in favorite_performer_scenes:
                    scene_recommendations['favorite_performer_scenes'].append(recommendation)
                else:
                    scene_recommendations['non_favorite_performer_scenes'].append(recommendation)
        
        # Remove duplicates and sort by similarity
        for key in scene_recommendations:
            # Create a set of scene IDs to track duplicates
            seen_ids = set()
            unique_recommendations = []
            
            for rec in scene_recommendations[key]:
                if rec['id'] not in seen_ids:
                    seen_ids.add(rec['id'])
                    unique_recommendations.append(rec)
            
            # Sort by similarity
            unique_recommendations.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Limit to max recommendations
            scene_recommendations[key] = unique_recommendations[:self.max_recommendations]
        
        return scene_recommendations
