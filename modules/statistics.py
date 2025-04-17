import re
import logging
import pandas as pd
import numpy as np
from collections import Counter

# Sklearn Imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class StatisticsModule:
    def __init__(self, stash_client=None):
        """Initialize the statistics module"""
        self.stash_client = stash_client
        self.performers_data = None
        self.scenes_data = None
        self.cup_size_pattern = re.compile(r'(\d{2,3})([A-KJ-Z]+)')
        
        # EU band size ranges
        self.min_eu_band = 60
        self.max_eu_band = 110
        
        # Band size conversion mappings (US/UK to EU)
        self.band_conversion = {
            28: 60, 30: 65, 32: 70, 34: 75, 36: 80, 
            38: 85, 40: 90, 42: 95, 44: 100, 46: 105, 48: 110
        }
        
        # Cup conversion mappings
        self.cup_conversion = {
            "A": "A", "B": "B", "C": "C", "D": "D", 
            "DD": "E", "DDD": "F", "DDDD": "G",
            "E": "E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J"
        }
        
        # Cup numeric mapping for calculations
        self.cup_numeric = {
            "A": 1, "B": 2, "C": 3, "D": 4, 
            "E": 5, "DD": 5, "F": 6, "DDD": 6, 
            "G": 7, "H": 8, "I": 9, "J": 10
        }
    
    def _load_data(self):
        """Load data from Stash"""
        try:
            self.performers_data = self.stash_client.get_performers()
            self.scenes_data = self.stash_client.get_scenes()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.performers_data = []
            self.scenes_data = []
    
    def _is_eu_band_size(self, band_size):
        """Check if a band size is in EU format (60-110 cm)"""
        try:
            band = int(band_size)
            return self.min_eu_band <= band <= self.max_eu_band
        except (ValueError, TypeError):
            return False
    
    def _ensure_eu_format(self, size_str):
        """Ensure a cup size string is in EU format"""
        if not size_str:
            return None, None
            
        # Try to parse the size string
        match = re.match(r'(\d+)([A-KJ-Z]+)', size_str)
        if not match:
            return None, None
            
        band = int(match.group(1))
        cup = match.group(2)
        
        # If already in EU format, return as is
        if self._is_eu_band_size(band):
            return f"{band}{cup}", self.cup_numeric.get(cup, 0)
        
        # Convert to EU
        eu_band = self.band_conversion.get(band, round((band + 16) / 2) * 5)
        eu_cup = self.cup_conversion.get(cup, cup)
        
        return f"{eu_band}{eu_cup}", self.cup_numeric.get(eu_cup, 0)
    
    def _convert_bra_size(self, measurements):
        """Extract and convert bra sizes to EU (German) format"""
        if not measurements:
            return None, None
        
        # First check if the measurements string contains EU-style sizing
        eu_pattern = re.compile(r'(\d{2,3})([A-KJ-Z]+)')
        eu_match = eu_pattern.search(measurements)
        
        if not eu_match:
            return None, None
        
        band_size = int(eu_match.group(1))
        cup_letter = eu_match.group(2)
        
        # If already in EU format (60-110), return as is
        if self._is_eu_band_size(band_size):
            eu_cup = self.cup_conversion.get(cup_letter, cup_letter)
            return f"{band_size}{eu_cup}", self.cup_numeric.get(eu_cup, 0)
        
        # Convert US/UK to EU
        eu_band = self.band_conversion.get(band_size, round((band_size + 16) / 2) * 5)
        eu_cup = self.cup_conversion.get(cup_letter, cup_letter)
        
        return f"{eu_band}{eu_cup}", self.cup_numeric.get(eu_cup, 0)
    
    def get_cup_size_stats(self):
        """Get statistics about cup sizes using EU format"""
        if not self.performers_data:
            self._load_data()
            
        cup_sizes = []
        cup_size_data = []
        
        for performer in self.performers_data:
            measurements = performer.get('measurements')
            
            # Convert bra size to EU format
            eu_bra_size, cup_numeric = self._convert_bra_size(measurements)
            
            if eu_bra_size:
                cup_sizes.append(eu_bra_size)
                
                # Extract band size and cup letter from EU format
                band_size, cup_letter = re.match(r'(\d+)([A-KJ-Z]+)', eu_bra_size).groups()
                
                performer_data = {
                    'id': performer.get('id'),
                    'name': performer.get('name'),
                    'cup_size': eu_bra_size,
                    'band_size': band_size,
                    'cup_letter': cup_letter,
                    'cup_numeric': cup_numeric,
                    'favorite': performer.get('favorite', False),
                    'height_cm': performer.get('height_cm', 0),
                    'weight': performer.get('weight', 0),
                    'measurements': measurements,
                    'scene_count': performer.get('scene_count', 0),
                    'rating100': performer.get('rating100', 0),
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
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
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
                if o_count > 0:  # Only consider scenes with o_counter > 0
                    for performer_id, performer_name in zip(row['performer_ids'], row['performers']):
                        if performer_id not in performer_o_counts:
                            performer_o_counts[performer_id] = {
                                'name': performer_name,
                                'total_o_count': 0,
                                'scene_count': 0
                            }
                        
                        performer_o_counts[performer_id]['total_o_count'] += o_count
                        performer_o_counts[performer_id]['scene_count'] += 1
        
        # Calculate overall O-counter statistics - only for non-zero values
        o_counter_values = [scene.get('o_counter', 0) for scene in self.scenes_data if scene.get('o_counter', 0) > 0]
        
        if o_counter_values:
            avg_o_counter = np.mean(o_counter_values)
            median_o_counter = np.median(o_counter_values)
            max_o_counter = max(o_counter_values)
        else:
            avg_o_counter = 0
            median_o_counter = 0
            max_o_counter = 0
            
        return {
            'o_counter_dataframe': df,
            'performer_o_counts': performer_o_counts,
            'average_o_counter': avg_o_counter,
            'median_o_counter': median_o_counter,
            'max_o_counter': max_o_counter,
            'total_performers': len(performer_o_counts)
        }
    
    def get_favorite_o_counter_stats(self):
        """
        Analyze the relationship between favorite status and o-counter values
        
        Returns detailed statistics about favorites vs non-favorites with o-counter > 0
        """
        if not self.performers_data:
            self._load_data()
            
        # Extract performers with o-counter > 0
        # Stellen sicher, dass wir immer einen Default-Wert haben und keine None-Werte
        performers_with_o = [p for p in self.performers_data if p.get('o_counter', 0) > 0]
        
        # Separate favorites and non-favorites
        favorites = [p for p in performers_with_o if p.get('favorite', False)]
        non_favorites = [p for p in performers_with_o if not p.get('favorite', False)]
        
        # Calculate statistics for favorites
        favorite_o_values = [p.get('o_counter', 0) for p in favorites]
        favorite_stats = {
            'count': len(favorites),
            'avg_o_counter': np.mean(favorite_o_values) if favorite_o_values else 0,
            'median_o_counter': np.median(favorite_o_values) if favorite_o_values else 0,
            'max_o_counter': max(favorite_o_values) if favorite_o_values else 0,
            'performers': [
                {
                    'name': p.get('name', 'Unknown'),
                    'o_counter': p.get('o_counter', 0),
                    'rating100': p.get('rating100', 0),
                    'scene_count': p.get('scene_count', 0),
                    'measurements': p.get('measurements', 'N/A')
                }
                for p in sorted(favorites, key=lambda x: x.get('o_counter', 0), reverse=True)
            ]
        }
        
        # Calculate statistics for non-favorites
        non_favorite_o_values = [p.get('o_counter', 0) for p in non_favorites]
        non_favorite_stats = {
            'count': len(non_favorites),
            'avg_o_counter': np.mean(non_favorite_o_values) if non_favorite_o_values else 0,
            'median_o_counter': np.median(non_favorite_o_values) if non_favorite_o_values else 0,
            'max_o_counter': max(non_favorite_o_values) if non_favorite_o_values else 0,
            'performers': [
                {
                    'name': p.get('name', 'Unknown'),
                    'o_counter': p.get('o_counter', 0),
                    'rating100': p.get('rating100', 0),
                    'scene_count': p.get('scene_count', 0),
                    'measurements': p.get('measurements', 'N/A')
                }
                for p in sorted(non_favorites, key=lambda x: x.get('o_counter', 0), reverse=True)
            ]
        }
        
        # Overall statistics
        total_stats = {
            'total_performers': len(performers_with_o),
            'favorite_percentage': (len(favorites) / len(performers_with_o) * 100) if performers_with_o else 0,
            'non_favorite_percentage': (len(non_favorites) / len(performers_with_o) * 100) if performers_with_o else 0
        }
        
        return {
            'favorite_stats': favorite_stats,
            'non_favorite_stats': non_favorite_stats,
            'overall_stats': total_stats
        }
    
    def get_rating_o_counter_correlation(self):
        """
        Analyze the correlation between performer ratings and o-counter values
        """
        if not self.performers_data:
            self._load_data()
        
        # Extract performers with both rating and o-counter
        # Stelle sicher, dass wir Default-Werte verwenden
        performers_with_data = [
            p for p in self.performers_data 
            if p.get('rating100', 0) > 0
        ]
        
        if not performers_with_data:
            return {
                'correlation': 0,
                'high_rated_high_o': [],
                'high_rated_low_o': [],
                'low_rated_high_o': [],
                'rating_o_counter_data': []
            }
        
        # Create DataFrame for analysis
        rating_data = []
        for performer in performers_with_data:
            # Stelle sicher, dass wir Default-Werte für alle Felder angeben
            rating_data.append({
                'id': performer.get('id', ''),
                'name': performer.get('name', 'Unknown'),
                'rating100': performer.get('rating100', 0),
                'o_counter': performer.get('o_counter', 0),
                'favorite': performer.get('favorite', False),
                'scene_count': performer.get('scene_count', 0),
                'measurements': performer.get('measurements', '')
            })
        
        df = pd.DataFrame(rating_data)
        
        # Stelle sicher, dass wir keine None-Werte in numerischen Spalten haben
        df['rating100'] = pd.to_numeric(df['rating100'], errors='coerce').fillna(0)
        df['o_counter'] = pd.to_numeric(df['o_counter'], errors='coerce').fillna(0)
        df['scene_count'] = pd.to_numeric(df['scene_count'], errors='coerce').fillna(0)
        
        # Calculate correlation if data exists
        if not df.empty and len(df) > 1:
            correlation = df['rating100'].corr(df['o_counter'])
            # Stelle sicher, dass die Korrelation nie None ist
            correlation = correlation if pd.notna(correlation) else 0
        else:
            correlation = 0
        
        # Identify performers in different categories
        if not df.empty:
            # Use median as threshold for high/low
            rating_median = df['rating100'].median()
            
            # Vorsichtig bei der Berechnung des o_counter_median
            o_counter_positive = df[df['o_counter'] > 0]
            if not o_counter_positive.empty:
                o_counter_median = o_counter_positive['o_counter'].median()
            else:
                o_counter_median = 1  # Default-Wert
            
            # Kategorisiere Performer
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
        else:
            high_rated_high_o = []
            high_rated_low_o = []
            low_rated_high_o = []
        
        return {
            'correlation': correlation,
            'high_rated_high_o': high_rated_high_o,
            'high_rated_low_o': high_rated_low_o,
            'low_rated_high_o': low_rated_high_o,
            'rating_o_counter_data': df.to_dict('records') if not df.empty else []
        }
    
    def get_cup_size_o_counter_correlation(self):
        """Analyze correlation between cup size and o-counter"""
        cup_stats = self.get_cup_size_stats()
        o_stats = self.get_o_counter_stats()
        
        # Get ratio data to include cup_to_bmi values
        ratio_stats = self.get_ratio_stats()
        
        cup_df = cup_stats['cup_size_dataframe']
        performer_o_counts = o_stats['performer_o_counts']
        
        # Get ratio dataframe with cup_to_bmi values
        ratio_df = ratio_stats.get('ratio_dataframe', pd.DataFrame())
    
        # Add o-counter data to cup size dataframe
        if not cup_df.empty:
            # Merge cup_df with ratio_df to include cup_to_bmi if ratio_df exists and is not empty
            if not ratio_df.empty and 'id' in ratio_df.columns and 'cup_to_bmi' in ratio_df.columns:
                cup_df = cup_df.merge(
                    ratio_df[['id', 'cup_to_bmi']], 
                    on='id', 
                    how='left'
                )
            
            # Prepare o-counter data
            cup_df['total_o_count'] = cup_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('total_o_count', 0)
            )
            cup_df['o_scene_count'] = cup_df['id'].apply(
                lambda pid: performer_o_counts.get(pid, {}).get('scene_count', 0)
            )
            
            # Filter to only include performers with o_counter > 0 for statistics
            non_zero_o_count_df = cup_df[cup_df['total_o_count'] > 0]
            
            # If there are no performers with non-zero o-count, create an empty result
            if non_zero_o_count_df.empty:
                return {
                    'cup_size_o_counter_df': cup_df,
                    'cup_letter_o_stats': []
                }
            
            # First, calculate basic stats including o_count mean using only non-zero values
            basic_stats = non_zero_o_count_df.groupby('cup_letter').agg({
                'total_o_count': 'mean',
                'id': 'count'
            }).reset_index()
            
            basic_stats.columns = ['cup_letter', 'avg_o_count', 'performer_count']
            
            # Now calculate o_count median in a separate operation using only non-zero values
            median_stats = non_zero_o_count_df.groupby('cup_letter').agg({
                'total_o_count': 'median'
            }).reset_index()
            
            median_stats.columns = ['cup_letter', 'median_o_count']
            
            # Calculate cup_to_bmi median if the column exists using only non-zero o-count performers
            if 'cup_to_bmi' in non_zero_o_count_df.columns:
                cup_to_bmi_stats = non_zero_o_count_df.groupby('cup_letter').agg({
                    'cup_to_bmi': 'median'
                }).reset_index()
                
                cup_to_bmi_stats.columns = ['cup_letter', 'median_cup_to_bmi']
            else:
                # Create an empty DataFrame with the right structure
                cup_to_bmi_stats = pd.DataFrame(columns=['cup_letter', 'median_cup_to_bmi'])
            
            # Calculate count of performers with o-counter > 0
            o_count_stats = cup_df.groupby('cup_letter').apply(
                lambda x: (x['total_o_count'] > 0).sum()
            ).reset_index(name='performers_with_o_count')
            
            # Merge all stats together
            cup_letter_stats = basic_stats.merge(median_stats, on='cup_letter')
            
            # Only merge cup_to_bmi_stats if it's not empty
            if not cup_to_bmi_stats.empty:
                cup_letter_stats = cup_letter_stats.merge(cup_to_bmi_stats, on='cup_letter', how='left')
            
            cup_letter_stats = cup_letter_stats.merge(o_count_stats, on='cup_letter')
            
            # Calculate percentage of performers with o-counter > 0
            # This needs to use the total count from cup_df, not just from non_zero_o_count_df
            total_performers_by_cup = cup_df.groupby('cup_letter').size().reset_index(name='total_performers')
            cup_letter_stats = cup_letter_stats.merge(total_performers_by_cup, on='cup_letter')
            
            cup_letter_stats['pct_with_o_count'] = (
                cup_letter_stats['performers_with_o_count'] / 
                cup_letter_stats['total_performers'] * 100
            )
        
            # Sort by average o-count for consistent output
            cup_letter_stats = cup_letter_stats.sort_values('avg_o_count', ascending=False)
            
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
        
        # Filtere Performer mit O-Counter > 0
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
            # Ensure EU cup size format
            eu_cup_size, _ = self._convert_bra_size(performer.get('measurements'))
            
            top_o_counter_details.append({
                'name': performer.get('name', 'Unbekannt'),
                'o_counter': performer.get('o_counter', 0),
                'measurements': performer.get('measurements', 'N/A'),
                'eu_cup_size': eu_cup_size,
                'scene_count': performer.get('scene_count', 0),
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
            pd.notna(cup_df['bmi']) & (cup_df['bmi'] > 0) & pd.notna(cup_df['cup_letter_value']),
            cup_df['cup_letter_value'] / cup_df['bmi'],
            np.nan
        )
        cup_df['cup_to_height'] = np.where(
            pd.notna(cup_df['height_cm']) & (cup_df['height_cm'] > 0) & pd.notna(cup_df['cup_letter_value']),
            cup_df['cup_letter_value'] / cup_df['height_cm'],
            np.nan
        )
        cup_df['cup_to_weight'] = np.where(
            pd.notna(cup_df['weight']) & (cup_df['weight'] > 0) & pd.notna(cup_df['cup_letter_value']),
            cup_df['cup_letter_value'] / cup_df['weight'],
            np.nan
        )
        
        # Filter to only include performers with o_counter > 0 for ratio statistics
        non_zero_o_count_df = cup_df[cup_df['o_counter'] > 0]
        
        # If we have no performers with non-zero o-count, fall back to using all performers
        # This ensures ratio stats still work even if we're filtering o-counter stats
        ratio_df = non_zero_o_count_df if not non_zero_o_count_df.empty else cup_df
        
        # Group by cup letter for mean values
        mean_ratio_stats = ratio_df.groupby('cup_letter').agg({
            'cup_to_bmi': 'mean',
            'cup_to_height': 'mean',
            'cup_to_weight': 'mean',
            'id': 'count'
        }).reset_index()
        
        mean_ratio_stats.columns = ['cup_letter', 'avg_cup_to_bmi', 'avg_cup_to_height', 
                                   'avg_cup_to_weight', 'performer_count']
        
        # Group by cup letter for median values
        median_ratio_stats = ratio_df.groupby('cup_letter').agg({
            'cup_to_bmi': 'median',
            'cup_to_height': 'median',
            'cup_to_weight': 'median'
        }).reset_index()
        
        median_ratio_stats.columns = ['cup_letter', 'median_cup_to_bmi', 'median_cup_to_height', 
                                     'median_cup_to_weight']
        
        # Merge mean and median stats
        ratio_stats = mean_ratio_stats.merge(median_ratio_stats, on='cup_letter', how='left')
        
        return {
            'ratio_dataframe': cup_df,  # Keep the full dataframe for other methods to use
            'ratio_stats': ratio_stats.to_dict('records')
        }
    
    def create_preference_profile(self, feature_weights=None):
        """
        Erstellt ein detailliertes Profil der Nutzer-Präferenzen 
        mit konfigurierbaren Feature-Gewichtungen
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
        
        # Cup-Size Stats vorbereiten
        cup_stats = self.get_cup_size_stats()
        cup_df = cup_stats['cup_size_dataframe']
        
        # DataFrame mit relevanten Performern erstellen
        df = pd.DataFrame([
            {
                'o_counter': float(p.get('o_counter', 0) or 0),
                'rating100': float(p.get('rating100', 0) or 0),
                'height_cm': float(p.get('height_cm', 0) or 0),
                'weight': float(p.get('weight', 0) or 0),
                'eu_cup_numeric': float(
                    cup_df[cup_df['id'] == p.get('id')]['cup_numeric'].values[0] 
                    if len(cup_df[cup_df['id'] == p.get('id')]['cup_numeric'].values) > 0 
                    else 0
                ),
                'name': p.get('name', 'Unbekannt')
            }
            for p in relevant_performers
        ])
        
        # Features für Clustering
        features = list(default_weights.keys())
        
        # Preprocessing Pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Ersetze NaNs durch Median
            ('scaler', StandardScaler())  # Skaliere Features
        ])
        
        # Vorbereitung der Features für Clustering
        X = df[features]
        
        # Preprocessing
        X_processed = preprocessor.fit_transform(X)
        
        # Manuelle Gewichtung anwenden
        weighted_features = X_processed.copy()
        for i, feature in enumerate(features):
            weighted_features[:, i] *= default_weights[feature]
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(weighted_features)
        
        # Cluster-Details mit Performer-Namen
        cluster_details = {}
        for cluster in range(3):
            cluster_performers = df[df['cluster'] == cluster]['name'].tolist()
            cluster_details[cluster] = {
                'performers': cluster_performers,
                'count': len(cluster_performers)
            }
        
        # Berechne Durchschnittswerte - only for those with o_counter > 0
        o_count_df = df[df['o_counter'] > 0]
        avg_o_counter = o_count_df['o_counter'].mean() if not o_count_df.empty else 0
        avg_rating = o_count_df['rating100'].mean() if not o_count_df.empty else 0
        
        # Cup-Sizes analysieren - make sure to use EU format
        relevant_cup_sizes = []
        for p in relevant_performers:
            if p.get('o_counter', 0) > 0:
                eu_cup_size, _ = self._convert_bra_size(p.get('measurements'))
                if eu_cup_size:
                    relevant_cup_sizes.append(eu_cup_size)
        
        # Cup-Size Häufigkeiten 
        cup_size_counter = Counter(relevant_cup_sizes)
        most_common_cup_sizes = cup_size_counter.most_common(3)
        
        return {
            'feature_weights': default_weights,
            'preference_profile': {
                'total_relevant_performers': len([p for p in relevant_performers if p.get('o_counter', 0) > 0]),
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
                cluster: df[(df['cluster'] == cluster) & (df['o_counter'] > 0)]
                .nlargest(5, 'o_counter')[['name', 'o_counter', 'rating100']]
                .to_dict('records')
                for cluster in range(3)
            }
        }
    
    def generate_all_stats(self):
        """Generate all statistics with JSON-serializable output"""
        cup_size_stats = self.get_cup_size_stats()
        o_counter_stats = self.get_o_counter_stats()
        correlation_stats = self.get_cup_size_o_counter_correlation()
        ratio_stats = self.get_ratio_stats()
        top_o_counter_performers = self.get_top_o_counter_performers()
        preference_profile = self.create_preference_profile()
        
        # Add the new statistics
        favorite_o_counter_stats = self.get_favorite_o_counter_stats()
        rating_o_counter_correlation = self.get_rating_o_counter_correlation()
        
        # Convert DataFrames to dictionaries
        def convert_dataframe(df):
            return df.to_dict(orient='records') if not df.empty else []
        
        return {
            'cup_size_stats': {
                'cup_size_counts': cup_size_stats['cup_size_counts'],
                'cup_size_dataframe': convert_dataframe(cup_size_stats['cup_size_dataframe'])
            },
            'o_counter_stats': {
                'o_counter_dataframe': convert_dataframe(o_counter_stats['o_counter_dataframe']),
                'performer_o_counts': o_counter_stats['performer_o_counts'],
                'average_o_counter': o_counter_stats.get('average_o_counter', 0),
                'median_o_counter': o_counter_stats.get('median_o_counter', 0),
                'max_o_counter': o_counter_stats.get('max_o_counter', 0),
                'total_performers': o_counter_stats.get('total_performers', 0)
            },
            'correlation_stats': {
                'cup_size_o_counter_df': convert_dataframe(correlation_stats['cup_size_o_counter_df']),
                'cup_letter_o_stats': correlation_stats['cup_letter_o_stats']
            },
            'ratio_stats': {
                'ratio_dataframe': convert_dataframe(ratio_stats['ratio_dataframe']),
                'ratio_stats': ratio_stats['ratio_stats']
            },
            'top_o_counter_performers': top_o_counter_performers,
            'preference_profile': preference_profile,
            # Add the new statistic sections
            'favorite_o_counter_stats': favorite_o_counter_stats,
            'rating_o_counter_correlation': {
                'correlation': rating_o_counter_correlation['correlation'],
                'high_rated_high_o': rating_o_counter_correlation['high_rated_high_o'],
                'high_rated_low_o': rating_o_counter_correlation['high_rated_low_o'],
                'low_rated_high_o': rating_o_counter_correlation['low_rated_high_o'],
                'rating_o_counter_data': convert_dataframe(pd.DataFrame(rating_o_counter_correlation['rating_o_counter_data']))
            }
        }
