import logging
import requests
import json
import configparser
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DiscordModule:
    def __init__(self, stats_module=None, recommendation_module=None, config_path=None):
        """Initialize the Discord module"""
        # Default config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'config', 'configuration.ini'
            )
        
        # Read configuration
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Get Discord settings
        self.webhook_url = config.get('discord', 'webhook_url', fallback='')
        self.enable_stats = config.getboolean('discord', 'enable_stats_posting', fallback=False)
        self.enable_performer_recs = config.getboolean('discord', 'enable_performer_recommendations', fallback=False)
        self.enable_scene_recs = config.getboolean('discord', 'enable_scene_recommendations', fallback=False)
        
        # Store modules
        self.stats_module = stats_module
        self.recommendation_module = recommendation_module
    
    def format_performer_recommendations(self):
        """Format performer recommendations for output"""
        try:
            # Get performer recommendations
            performer_recs = self.recommendation_module.recommend_performers()
            
            # Build recommendation message
            message = "🌟 Performer Recommendations 🌟\n\n"
            
            if not performer_recs:
                message += "No performer recommendations found."
                return message
            
            for i, rec in enumerate(performer_recs[:5], 1):
                performer = rec['performer']
                similar = rec.get('similar_performers', [])
                
                message += f"**{i}. {performer['name']}**\n"
                message += f"Scene Count: {performer.get('scene_count', 0)}\n"
                message += f"Measurements: {performer.get('measurements', 'N/A')}\n"
                
                if similar:
                    message += "Similar Performers:\n"
                    for j, sp in enumerate(similar, 1):
                        message += f"  {j}. {sp['name']} (Scene Count: {sp.get('scene_count', 0)})\n"
                
                message += "\n"
            
            return message
        
        except Exception as e:
            logger.error(f"Error formatting performer recommendations: {e}")
            return "Error generating performer recommendations."
    
    def format_scene_recommendations(self):
        """Format scene recommendations for output"""
        try:
            # Get scene recommendations
            scene_recs = self.recommendation_module.recommend_scenes()
            
            # Build recommendation message
            message = "🎬 Scene Recommendations 🎬\n\n"
            
            # Process different recommendation categories
            categories = {
                'favorite_performer_scenes': "Scenes with Favorite Performers",
                'non_favorite_performer_scenes': "Recommended Scenes",
                'recommended_performer_scenes': "Scenes with Recommended Performers"
            }
            
            for category_key, category_name in categories.items():
                scenes = scene_recs.get(category_key, [])
                
                message += f"**{category_name}**:\n"
                
                if not scenes:
                    message += "  No recommendations found.\n\n"
                    continue
                
                for i, scene in enumerate(scenes[:3], 1):
                    message += f"{i}. **{scene.get('title', 'Untitled')}**\n"
                    message += f"   Performers: {', '.join(scene.get('performers', ['N/A']))}\n"
                    message += f"   Tags: {', '.join(scene.get('tags', ['N/A']))[:100]}...\n\n"
            
            return message
        
        except Exception as e:
            logger.error(f"Error formatting scene recommendations: {e}")
            return "Error generating scene recommendations."
    
    def send_recommendations(self):
        """Send recommendations to output (console and Discord)"""
        # Format recommendations
        performer_recs = self.format_performer_recommendations()
        scene_recs = self.format_scene_recommendations()
        
        # Combine messages
        full_message = performer_recs + "\n" + scene_recs
        
        # Print to console
        print("\n" + full_message)
        
        # Send to Discord
        if self.webhook_url:
            self._send_to_discord(full_message)
        else:
            logger.warning("No Discord webhook URL configured. Skipping Discord message.")
    
    def _send_to_discord(self, message):
        """Send message to Discord webhook"""
        try:
            payload = {
                "content": message,
                "username": "Stash Recommendations Bot"
            }
            
            response = requests.post(
                self.webhook_url, 
                json=payload, 
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 204:
                logger.info("Successfully sent recommendations to Discord")
            else:
                logger.error(f"Failed to send message to Discord. Status code: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error sending message to Discord: {e}")

def send_recommendations(recommendation_module, config_path=None):
    """Helper function to send recommendations"""
    discord_module = DiscordModule(recommendation_module=recommendation_module, config_path=config_path)
    discord_module.send_recommendations()
