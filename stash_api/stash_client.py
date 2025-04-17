import re
import logging
import pandas as pd
import numpy as np
from collections import Counter

# Sklearn Imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class StatisticsModule:
    def __init__(self, stash_client=None):
        """Initialize the statistics module"""
        self.stash_client = stash_client
        self.performers_data = None
        self.scenes_data = None
        self.cup_size_pattern = re.compile(r'(\d{2,3})([A-KJ-Z]+)')
    
    def _load_data(self):
        """Load data from Stash"""
        try:
            self.performers_data = self.stash_client.get_performers()
            self.scenes_data = self.stash_client.get_scenes()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.performers_data = []
            self.scenes_data = []
    
    def _convert_bra_size(self, measurements):
        """Konvertiert US/UK BH-Größen in deutsche Größen"""
        if not measurements:
            return None, None
        
        # Regex-Muster für BH-Größen
        match = re.search(r'(\d{2,3})([A-KJ-Z]+)', measurements)
        if not match:
            return None, None
        
        us_band = int(match.group(1))
        us_cup = match.group(2)
        
        # Umrechnungstabellen
        band_conversion = {
            28: 60, 30: 65, 32: 70, 34: 75, 36: 80, 
            38: 85, 40: 90, 42: 95, 44: 100, 46: 105
        }
        
        cup_conversion = {
            "A": "A", "B": "B", "C": "C", "D": "D", 
            "DD": "E", "DDD": "F", "E": "E", "F": "F", 
            "G": "G", "H": "H", "I": "I", "J": "J"
        }
        
        cup_numeric = {
            "A": 1, "B": 2, "C": 3, "D": 4, 
            "E": 5, "DD": 5, "F": 6, "DDD": 6, 
            "G": 7, "H": 8, "I": 9, "J": 10
        }
        
        # Umrechnungen speichern
        de_band = band_conversion.get(us_band, round((us_band + 16) / 2) * 5)
        de_cup = cup_conversion.get(us_cup, us_cup)
        
        return f"{de_band}{de_cup}", cup_numeric.get(us_cup, 0)
    
    def get_cup_size_stats(self):
        """Get statistics about cup sizes"""
        if not self.performers_data:
            self._load_data()
            
        cup_sizes = []
        cup_size_data = []
        
        for performer in self.performers_data:
            measurements = performer.get('measurements')
            
            # Convert bra size
            german_bra_size, cup_numeric = self._convert_bra_size(measurements)
            
            if german_bra_size:
                cup_sizes.append(german_bra_size)
                
                band_size, cup_letter = re.match(r'(\d+)([A-KJ-Z]+)', german_bra_size).groups()
                
                performer_data = {
                    'id': performer.get('id'),
                    'name': performer.get('name'),
                    'cup_size': german_bra_size,
                    'band_size': band_size,
                    'cup_letter': cup_letter,
                    'cup_numeric': cup_numeric,
                    'favorite': performer.get('favorite', False),
                    'height_cm': performer.get('height_cm'),
                    'weight': performer.get('weight'),
                    'measurements': measurements,
                    'scene_count': performer.get('scene_count', 0),
                    'rating100': performer.get('rating100'),
                    'o_counter': performer.get('o_counter', 0)
                }
                
                # Calculate BMI
                height = performer_data['height_cm']
                weight = performer_data['weight']
                if height and weight and height > 0 and weight > 0:
                    height_m = height / 100
                    performer_data['bmi'] = round(weight / (height_m * height_m), 1)
                else:
                    performer_data['bmi'] = None
                
                cup_size_data.append(performer_data)
        
        # Create a DataFrame
        df = pd.DataFrame(cup_size_data)
        
        # Convert numeric columns
        if not df.empty:
            numeric_columns = ['band_size', 'height_cm', 'weight', 'bmi', 'scene_count', 'rating100', 'o_counter']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Count frequencies
        cup_size_counts = Counter(cup_sizes)
        
        return {
            'cup_size_counts': dict(cup_size_counts),
            'cup_size_dataframe': df
        }
    
    def get_o_counter_stats(self):
        """Get statistics about o-counter values"""
        if not self.scenes_data:
            self._load_data()
            
        o_counter_data = []
        
        for scene in self.scenes_data:
            o_counter = scene.get('o_counter', 0)
            performers = scene.get('performers', [])
            
            scene_data = {
                'id': scene.get('id'),
                'title': scene.get('title'),
                'o_counter': o_counter,
                'performers': [p.get('name') for p in performers],
                'performer_ids': [p.get('id') for p in performers],
                'favorite_performers': [p.get('name') for p in performers if p.get('favorite', False)],
                'tags': [t.get('name') for t in scene.get('tags', [])]
            }
            
            o_counter_data.append(scene_data)
        
        # Create a DataFrame
        df = pd.DataFrame(o_counter_data)
        
        # Get performers with highest o-counter sum
        performer_o_counts = {}
        
        if not df.empty:
            for _, row in df.iterrows():
                o_count = row['o_counter']
                if o_count > 0:
                    for performer_id, performer_name in zip(row['performer_ids'], row['performers']):
                        if performer_id not in performer_o_counts:
                            performer_o_counts[performer_id] = {
                                'name': performer_name,
                                'total_o_count': 0,
                                'scene_count': 0
                            }
                        
                        performer_o_counts[performer_id]['total_o_count'] += o_count
                        performer_o_counts[performer_id]['scene_count'] += 1
        
        return {
            'o_counter_dataframe': df,
            'performer_o_counts': performer_o_counts
        }
    
    def get_cup_size_o_counter_correlation(self):
        """Analyze correlation between cup size and o-counter"""
        cup_stats = self.get_cup_size_stats()
        o_stats = self.get_o_counter_stats()
        
        cup_df = cup_stats['cup_size_dataframe']
        performer_o_counts = o_stats['performer_o_counts']
        
        # Add o-counter data to cup size dataframe
        if not cup_df.empty:
            # Prepare o-counter data
            cup_df['total_o_count'] = cup_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('total_o_count', 0)
            )
            cup_df['o_scene_count'] = cup_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('scene_count', 0)
            )
            
            # Group by cup letter and calculate average o-count
            cup_letter_stats = cup_df.groupby('cup_letter').agg({
                'total_o_count': 'mean',
                'id': 'count'
            }).reset_index()
            
            cup_letter_stats.columns = ['cup_letter', 'avg_o_count', 'performer_count']
            cup_letter_stats = cup_letter_stats.sort_values('cup_letter')
            
            return {
                'cup_size_o_counter_df': cup_df,
                'cup_letter_o_stats': cup_letter_stats.to_dict('records')
            }
        
        return {
            'cup_size_o_counter_df': pd.DataFrame(),
            'cup_letter_o_stats': []
        }
    
    def get_top_o_counter_performers(self, top_n=10):
        """Ermittelt die Top-Performer basierend auf O-Counter"""
        if not self.performers_data:
            self._load_data()
        
        # Berechne Ratio-Statistiken
        ratio_stats = self.get_ratio_stats()
        ratio_df = ratio_stats.get('ratio_dataframe', pd.DataFrame())
        
        # Filtere Performer mit O-Counter
        performers_with_o_counter = [
            p for p in self.performers_data 
            if p.get('o_counter', 0) > 0
        ]
        
        # Sortiere nach O-Counter absteigend
        top_performers = sorted(
            performers_with_o_counter, 
            key=lambda x: x.get('o_counter', 0), 
            reverse=True
        )[:top_n]
        
        # Bereite detaillierte Informationen vor
        top_o_counter_details = []
        for performer in top_performers:
            # Finde den Cup-to-BMI für diesen Performer
            performer_ratio = ratio_df[ratio_df['id'] == performer.get('id')]
            cup_to_bmi = performer_ratio['cup_to_bmi'].values[0] if not performer_ratio.empty else None
            
            top_o_counter_details.append({
                'name': performer.get('name', 'Unbekannt'),
                'o_counter': performer.get('o_counter', 0),
                'measurements': performer.get('measurements', 'N/A'),
                'scene_count': performer.get('scene_count', 0),
                'cup_to_bmi': cup_to_bmi,
                'id': performer.get('id')
            })
        
        return top_o_counter_details
    
    def get_ratio_stats(self):
        """Calculate various ratios like cup-to-bmi, cup-to-height, cup-to-weight"""
        cup_stats = self.get_cup_size_stats()
        cup_df = cup_stats['cup_size_dataframe']
        
        if cup_df.empty:
            return {}
            
        # Cup letter to numeric mapping
        cup_letter_values = {letter: idx+1 for idx, letter in enumerate('ABCDEFGHIJK')}
        
        # Add cup letter numeric value
        cup_df['cup_letter_value'] = cup_df['cup_letter'].map(cup_letter_values)
        
        # Calculate ratios
        # Use numpy for safe division and handling of NaN values
        cup_df['cup_to_bmi'] = np.where(
            cup_df['bmi'].notna() & (cup_df['bmi'] > 0),
            cup_df['cup_letter_value'] / cup_df['bmi'],
            np.nan
        )
        cup_df['cup_to_height'] = np.where(
            cup_df['height_cm'].notna() & (cup_df['height_cm'] > 0),
            cup_df['cup_letter_value'] / cup_df['height_cm'],
            np.nan
        )
        cup_df['cup_to_weight'] = np.where(
            cup_df['weight'].notna() & (cup_df['weight'] > 0),
            cup_df['cup_letter_value'] / cup_df['weight'],
            np.nan
        )
        
        # Group by cup letter
        ratio_stats = cup_df.groupby('cup_letter').agg({
            'cup_to_bmi': 'mean',
            'cup_to_height': 'mean',
            'cup_to_weight': 'mean',
            'id': 'count'
        }).reset_index()
        
        ratio_stats.columns = ['cup_letter', 'avg_cup_to_bmi', 'avg_cup_to_height', 
                              'avg_cup_to_weight', 'performer_count']
        
        return {
            'ratio_dataframe': cup_df,
            'ratio_stats': ratio_stats.to_dict('records')
        }
    
    def create_preference_profile(self, feature_weights=None):
        """
        Erstellt ein detailliertes Profil der Nutzer-Präferenzen 
        mit konfigurierbaren Feature-Gewichtungen

        Parameters:
        -----------
        feature_weights : dict, optional
            Benutzerdefinierte Gewichtungen für Features.
            Standardmäßig werden sinnvolle Gewichtungen verwendet.
            Beispiel: {'o_counter': 2.0, 'rating100': 1.5, 'height_cm': 0.5, 'weight': 0.5, 'eu_cup_numeric': 1.0}
        
        Returns:
        --------
        dict
            Detaillierte Präferenzanalyse mit Clustering-Ergebnissen
        """
        # Standard-Gewichtungen mit sinnvollen Defaultwerten
        default_weights = {
            'o_counter': 2.0,    # Höhere Gewichtung, da direkter Interessensindikator
            'rating100': 1.5,    # Wichtig, aber nicht überbetonen
            'height_cm': 0.5,    # Geringere Bedeutung
            'weight': 0.5,       # Geringere Bedeutung
            'eu_cup_numeric': 1.0  # Moderate Gewichtung
        }
        
        # Benutzerdefinierte Gewichtungen überschreiben Standardwerte
        if feature_weights is not None:
            default_weights.update(feature_weights)
        
        # Kombiniere Favoriten und Performer mit O-Counter > 1
        relevant_performers = [
            p for p in self.performers_data 
            if p.get('favorite', False) or p.get('o_counter', 0) > 1
        ]
        
        # Berechne Durchschnittswerte
        avg_o_counter = np.mean([p.get('o_counter', 0) for p in relevant_performers])
        avg_rating = np.mean([p.get('rating100', 0) for p in relevant_performers])
        
        # Cup-Sizes analysieren
        relevant_cup_sizes = [
            self._convert_bra_size(p.get('measurements'))[0] 
            for p in relevant_performers 
            if self._convert_bra_size(p.get('measurements'))[0]
        ]
        
        # Cup-Size Häufigkeiten 
        cup_size_counter = Counter(relevant_cup_sizes)
        most_common_cup_sizes = cup_size_counter.most_common(3)
        
        # Cup-Size Stats vorbereiten
        cup_stats = self.get_cup_size_stats()
        cup_df = cup_stats['cup_size_dataframe']
        
        # DataFrame mit relevanten Performern erstellen
        df = pd.DataFrame([
            {
                'o_counter': p.get('o_counter', 0),
                'rating100': p.get('rating100', 0),
                'height_cm': p.get('height_cm', 0),
                'weight': p.get('weight', 0),
                'eu_cup_numeric': cup_df[cup_df['id'] == p.get('id')]['cup_numeric'].values[0] 
                if not cup_df[cup_df['id'] == p.get('id')].empty 
                else 0,
                'name': p.get('name', 'Unbekannt')
            }
            for p in relevant_performers
        ])
        
        # Features für Clustering
        features = list(default_weights.keys())
        
        # Skaliere Features mit Gewichtungen
        # Kopiere DataFrame für Skalierung
        scaled_df = df.copy()
        
        # Skalierung und Gewichtung
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(scaled_df[features])
        
        # Manuelle Gewichtung anwenden
        weighted_features = scaled_features.copy()
        for i, feature in enumerate(features):
            weighted_features[:, i] *= default_weights[feature]
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=3)
        df['cluster'] = kmeans.fit_predict(weighted_features)
        
        # Cluster-Details mit Performer-Namen
        cluster_details = {}
        for cluster in range(3):
            cluster_performers = df[df['cluster'] == cluster]['name'].tolist()
            cluster_details[cluster] = {
                'performers': cluster_performers,
                'count': len(cluster_performers)
            }
        
        return {
            'feature_weights': default_weights,
            'preference_profile': {
                'total_relevant_performers': len(relevant_performers),
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
                'total_cup_sizes': cup_stats['cup_size_counts'],
                'relevant_cup_size_distribution': dict(cup_size_counter)
            },
            'top_performers_by_cluster': {
                cluster: df[df['cluster'] == cluster]
                .nlargest(5, 'o_counter')[['name', 'o_counter', 'rating100']]
                .to_dict('records')
                for cluster in range(3)
            }
        }
    
    def generate_all_stats(self):
        """Generate all statistics"""
        cup_size_stats = self.get_cup_size_stats()
        o_counter_stats = self.get_o_counter_stats()
        correlation_stats = self.get_cup_size_o_counter_correlation()
        ratio_stats = self.get_ratio_stats()
        top_o_counter_performers = self.get_top_o_counter_performers()
        preference_profile = self.create_preference_profile()
        
        return {
            'cup_size_stats': cup_size_stats,
            'o_counter_stats': o_counter_stats,
            'correlation_stats': correlation_stats,
            'ratio_stats': ratio_stats,
            'top_o_counter_performers': top_o_counter_performers,
            'preference_profile': preference_profile
        }
