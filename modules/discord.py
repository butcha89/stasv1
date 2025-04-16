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
    
    def format_statistics(self):
        """Format statistics for output"""
        try:
            # Generate statistics
            stats = self.stats_module.generate_all_stats()
            
            # Build stats message
            message = "📊 Stash Statistics 📊\n\n"
            
            # Cup Size Distribution
            cup_stats = stats.get('cup_size_stats', {})
            cup_counts = cup_stats.get('cup_size_counts', {})
            
            message += "**Cup Size Distribution:**\n"
            if cup_counts:
                for cup, count in sorted(cup_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    message += f"- {cup}: {count} performers\n"
            else:
                message += "No cup size data available\n"
            
            # O-Counter Statistics
            o_counter_stats = stats.get('o_counter_stats', {})
            o_counter_df = o_counter_stats.get('o_counter_dataframe', [])
            
            message += "\n**O-Counter Overview:**\n"
            if not o_counter_df.empty:
                total_scenes = len(o_counter_df)
                scenes_with_o_counter = len(o_counter_df[o_counter_df['o_counter'] > 0])
                message += f"- Total Scenes: {total_scenes}\n"
                message += f"- Scenes with O-Counter: {scenes_with_o_counter}\n"
            
            # Correlation Statistics
            corr_stats = stats.get('correlation_stats', {})
            cup_letter_stats = corr_stats.get('cup_letter_o_stats', [])
            
            if cup_letter_stats:
                message += "\n**Cup Size O-Counter Correlation:**\n"
                for stat in sorted(cup_letter_stats, key=lambda x: x['avg_o_count'], reverse=True)[:3]:
                    message += f"- Cup {stat['cup_letter']}: Avg O-Count {stat['avg_o_count']:.2f} (n={stat['performer_count']})\n"
            
            # Ratio Statistics
            ratio_stats = stats.get('ratio_stats', {})
            ratio_data = ratio_stats.get('ratio_stats', [])
            
            if ratio_data:
                message += "\n**Ratio Statistics:**\n"
                for stat in ratio_data[:3]:
                    message += (f"- Cup {stat['cup_letter']}: "
                                f"BMI Ratio {stat.get('avg_cup_to_bmi', 0):.3f}, "
                                f"Height Ratio {stat.get('avg_cup_to_height', 0):.4f}\n")
            
            return message
        
        except Exception as e:
            logger.error(f"Error formatting statistics: {e}")
            return "Error generating statistics."
    
    def send_statistics(self):
        """Send statistics to output (console and Discord)"""
        # Format statistics
        stats_message = self.format_statistics()
        
        # Print to console
        print("\n" + stats_message)
        
        # Send to Discord
        if self.webhook_url:
            self._send_to_discord(stats_message)
        else:
            logger.warning("No Discord webhook URL configured. Skipping Discord message.")
    
    def _send_to_discord(self, message):
        """Send message to Discord webhook"""
        try:
            payload = {
                "content": message,
                "username": "Stash Statistics Bot"
            }
            
            response = requests.post(
                self.webhook_url, 
                json=payload, 
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 204:
                logger.info("Successfully sent statistics to Discord")
            else:
                logger.error(f"Failed to send message to Discord. Status code: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error sending message to Discord: {e}")

def send_statistics(stats_module, config_path=None):
    """Helper function to send statistics"""
    discord_module = DiscordModule(stats_module=stats_module, config_path=config_path)
    discord_module.send_statistics()import logging
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
