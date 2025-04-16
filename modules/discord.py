import requests
import json
import configparser
import os
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import sys

# Füge das übergeordnete Verzeichnis zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.statistics import StatisticsModule
from modules.recommendations import RecommendationModule

class DiscordModule:
    def __init__(self, stats_module=None, recommendation_module=None, config_path=None):
        """Initialize the Discord module"""
        if config_path is None:
            # Default to the config in the project directory
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'config', 'configuration.ini')
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.webhook_url = config.get('discord', 'webhook_url')
        self.enable_stats = config.getboolean('discord', 'enable_stats_posting')
        self.enable_performer_recs = config.getboolean('discord', 'enable_performer_recommendations')
        self.enable_scene_recs = config.getboolean('discord', 'enable_scene_recommendations')
        
        self.stats_module = stats_module or StatisticsModule()
        self.recommendation_module = recommendation_module or RecommendationModule(
            stats_module=self.stats_module
        )
    
    def send_webhook(self, content=None, embeds=None, username="Stash Bot"):
        """Send a message to Discord via webhook"""
        if not self.webhook_url:
            return {'success': False, 'message': 'No webhook URL configured'}
            
        payload = {
            'username': username
        }
        
        if content:
            payload['content'] = content
            
        if embeds:
            payload['embeds'] = embeds
            
        response = requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        
        return {
            'success': response.status_code == 204,
            'status_code': response.status_code,
            'response': response.text
        }
    
    def post_statistics(self):
        """Post statistics to Discord"""
        if not self.enable_stats:
            return {'success': False, 'message': 'Statistics posting is disabled'}
            
        # Generate statistics
        stats = self.stats_module.generate_all_stats()
        
        # Create cup size distribution chart
        cup_size_chart = self.stats_module.plot_cup_size_distribution()
        cup_size_image = None
        
        if cup_size_chart:
            buf = io.BytesIO()
            cup_size_chart.savefig(buf, format='png')
            buf.seek(0)
            cup_size_image = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(cup_size_chart)
        
        # Create o-counter by cup chart
        o_counter_chart = self.stats_module.plot_o_counter_by_cup()
        o_counter_image = None
        
        if o_counter_chart:
            buf = io.BytesIO()
            o_counter_chart.savefig(buf, format='png')
            buf.seek(0)
            o_counter_image = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(o_counter_chart)
        
        # Create embeds
        embeds = [{
            'title': 'Stash Statistics Update',
            'description': f'Statistics generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'color': 3447003,  # Blue color
            'fields': []
        }]
        
        # Add cup size stats
        cup_stats = stats['cup_size_stats']
        cup_counts = cup_stats['cup_size_counts']
        
        if cup_counts:
            top_cup_sizes = sorted(cup_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            cup_size_text = '\n'.join([f"{cup}: {count}" for cup, count in top_cup_sizes])
            
            embeds[0]['fields'].append({
                'name': 'Top Cup Sizes',
                'value': cup_size_text,
                'inline': True
            })
        
        # Add o-counter stats
        corr_stats = stats['correlation_stats']
        if corr_stats:
            cup_letter_stats = corr_stats.get('cup_letter_o_stats', [])
            
            if cup_letter_stats:
                # Sort by average o-count
                sorted_stats = sorted(cup_letter_stats, key=lambda x: x['avg_o_count'], reverse=True)
                
                o_count_text = '\n'.join([
                    f"{stat['cup_letter']}: {stat['avg_o_count']:.2f} (n={stat['performer_count']})"
                    for stat in sorted_stats[:5]
                ])
                
                embeds[0]['fields'].append({
                    'name': 'Top Cup Letters by O-Count',
                    'value': o_count_text,
                    'inline': True
                })
        
        # Add ratio stats
        ratio_stats = stats['ratio_stats']
        if ratio_stats:
            ratio_data = ratio_stats.get('ratio_stats', [])
            
            if ratio_data:
                ratio_text = '\n'.join([
                    f"{stat['cup_letter']}: BMI={stat.get('avg_cup_to_bmi', 0):.3f}, " +
                    f"Height={stat.get('avg_cup_to_height', 0):.5f}, " +
                    f"Weight={stat.get('avg_cup_to_weight', 0):.4f}"
                    for stat in ratio_data[:3]
                ])
                
                embeds[0]['fields'].append({
                    'name': 'Sample Ratio Stats',
                    'value': ratio_text,
                    'inline': False
                })
        
        # Add images if available
        if cup_size_image:
            embeds.append({
                'title': 'Cup Size Distribution',
                'image': {
                    'url': f'attachment://cup_size_chart.png'
                }
            })
        
        if o_counter_image:
            embeds.append({
                'title': 'O-Counter by Cup Size',
                'image': {
                    'url': f'attachment://o_counter_chart.png'
                }
            })
        
        # Send webhook
        return self.send_webhook(
            content="Here are the latest statistics from your Stash collection:",
            embeds=embeds,
            username="Stash Statistics"
        )
    
    def post_performer_recommendations(self):
        """Post performer recommendations to Discord"""
        if not self.enable_performer_recs:
            return {'success': False, 'message': 'Performer recommendations posting is disabled'}
            
        # Get recommendations
        recommendations = self.recommendation_module.recommend_performers()
        
        if not recommendations:
            return {'success': False, 'message': 'No performer recommendations available'}
            
        # Create embeds
        embeds = [{
            'title': 'Performer Recommendations',
            'description': f'Based on your favorites and viewing habits',
            'color': 15105570,  # Orange color
            'fields': []
        }]
        
        # Add top recommendations
        for i, rec in enumerate(recommendations[:5]):
            performer = rec['performer']
            similar_performers = rec['similar_performers'][:3]  # Top 3 similar performers
            
            # Create field for this recommendation
            field_text = f"**Cup Size:** {performer['cup_size']}\n"
            field_text += f"**O-Count:** {performer['o_count']}\n"
            field_text += f"**Height:** {performer['height_cm']}cm\n"
            field_text += f"**Weight:** {performer['weight']}kg\n\n"
            
            field_text += "**Similar Performers:**\n"
            for sp in similar_performers:
                field_text += f"- {sp['name']} ({sp['cup_size']}) - Similarity: {sp['similarity']:.2f}\n"
            
            embeds[0]['fields'].append({
                'name': f"{i+1}. {performer['name']}",
                'value': field_text,
                'inline': False
            })
        
        # Send webhook
        return self.send_webhook(
            content="Here are some performer recommendations based on your preferences:",
            embeds=embeds,
            username="Stash Recommendations"
        )
    
    def post_scene_recommendations(self):
        """Post scene recommendations to Discord"""
        if not self.enable_scene_recs:
            return {'success': False, 'message': 'Scene recommendations posting is disabled'}
            
        # Get recommendations
        scene_recs = self.recommendation_module.recommend_scenes()
        
        if not scene_recs:
            return {'success': False, 'message': 'No scene recommendations available'}
            
        # Create embeds
        embeds = []
        
        # Add favorite performer scenes
        fav_scenes = scene_recs.get('favorite_performer_scenes', [])
        if fav_scenes:
            fav_embed = {
                'title': 'Recommended Scenes with Favorite Performers',
                'color': 15158332,  # Red color
                'fields': []
            }
            
            for i, scene in enumerate(fav_scenes[:3]):
                field_text = f"**Similarity:** {scene['similarity']:.2f}\n"
                field_text += f"**Performers:** {', '.join(scene['performers'])}\n"
                field_text += f"**Tags:** {', '.join(scene['tags'][:10])}..."
                
                fav_embed['fields'].append({
                    'name': f"{i+1}. {scene['title']}",
                    'value': field_text,
                    'inline': False
                })
            
            embeds.append(fav_embed)
        
        # Add recommended performer scenes
        rec_performer_scenes = scene_recs.get('recommended_performer_scenes', [])
        if rec_performer_scenes:
            rec_embed = {
                'title': 'Recommended Scenes with Recommended Performers',
                'color': 3066993,  # Green color
                'fields': []
            }
            
            for i, scene in enumerate(rec_performer_scenes[:3]):
                field_text = f"**Similarity:** {scene['similarity']:.2f}\n"
                field_text += f"**Performers:** {', '.join(scene['performers'])}\n"
                field_text += f"**Tags:** {', '.join(scene['tags'][:10])}..."
                
                rec_embed['fields'].append({
                    'name': f"{i+1}. {scene['title']}",
                    'value': field_text,
                    'inline': False
                })
            
            embeds.append(rec_embed)
        
        # Send webhook
        return self.send_webhook(
            content="Here are some scene recommendations based on your viewing habits:",
            embeds=embeds,
            username="Stash Scene Recommendations"
        )
    
    def post_all_updates(self):
        """Post all updates to Discord"""
        results = {}
        
        if self.enable_stats:
            results['statistics'] = self.post_statistics()
            
        if self.enable_performer_recs:
            results['performer_recommendations'] = self.post_performer_recommendations()
            
        if self.enable_scene_recs:
            results['scene_recommendations'] = self.post_scene_recommendations()
            
        return results
