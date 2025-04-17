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
            
            if not performer_recs:
                logger.warning("No performer recommendations found")
                return "No performer recommendations found."
            
            # Build recommendation message
            message = "🌟 Performer Recommendations 🌟\n\n"
            
            for rec in performer_recs:
                base_performer = rec['performer']
                similar = rec.get('similar_performers', [])
                
                message += f"**Empfohlener Performer**: {base_performer.get('name', 'Unknown')}\n"
                message += f"Measurements: {base_performer.get('measurements', 'N/A')}\n"
                message += f"O-Counter: {base_performer.get('o_count', 0)}\n"
                message += f"Cup-to-BMI Factor: {base_performer.get('cup_to_bmi', 'N/A')}\n"
                
                if similar:
                    message += "\nSimilar to:\n"
                    for sp in similar:
                        message += (f"- {sp['name']} (Cup Size: {sp['cup_size']}, "
                                   f"O-Count: {sp['o_count']}, "
                                   f"Similarity: {sp['similarity']:.2f})\n")
                
                message += "\n"
            
            return message
        
        except Exception as e:
            logger.error(f"Error formatting performer recommendations: {e}")
            return f"Error generating performer recommendations: {e}"
    
    def format_scene_recommendations(self):
        """Format scene recommendations for output"""
        try:
            # Get scene recommendations
            scene_recs = self.recommendation_module.recommend_scenes()
            
            if not scene_recs:
                logger.warning("No scene recommendations found")
                return "No scene recommendations found."
            
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
            return f"Error generating scene recommendations: {e}"
    
    def format_statistics(self):
        """Format statistics for output"""
        try:
            # Generate statistics
            stats = self.stats_module.generate_all_stats()
            
            # Build stats message
            message = "📊 Stash Statistiken 📊\n\n"
            
            # Cup Size Distribution
            cup_stats = stats.get('cup_size_stats', {})
            cup_counts = cup_stats.get('cup_size_counts', {})
            
            message += "**Cup-Größen Verteilung:**\n"
            if cup_counts:
                for cup, count in sorted(cup_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    message += f"- {cup}: {count} Performer\n"
            else:
                message += "Keine Cup-Größen Daten verfügbar\n"
            
            # O-Counter Statistics
            o_counter_stats = stats.get('o_counter_stats', {})
            o_counter_df = o_counter_stats.get('o_counter_dataframe', [])
            
            message += "\n**O-Counter Übersicht:**\n"
            if not isinstance(o_counter_df, list):
                total_scenes = len(o_counter_df)
                scenes_with_o_counter = len(o_counter_df[o_counter_df['o_counter'] > 0])
                message += f"- Gesamtszenen: {total_scenes}\n"
                message += f"- Szenen mit O-Counter: {scenes_with_o_counter}\n"
            
            # Ratio Statistics (Cup-to-BMI)
            ratio_stats = stats.get('ratio_stats', {})
            ratio_data = ratio_stats.get('ratio_stats', [])
            
            if ratio_data:
                message += "\n**Cup-to-BMI Verhältnis:**\n"
                for stat in sorted(ratio_data, key=lambda x: abs(x.get('avg_cup_to_bmi', 0)), reverse=True)[:5]:
                    message += (f"- Cup {stat['cup_letter']}: "
                               f"Cup-to-BMI = {stat.get('avg_cup_to_bmi', 0):.4f} "
                               f"(n = {stat.get('performer_count', 0)})\n")
            
            # Top O-Counter Performer
            top_o_counter = stats.get('top_o_counter_performers', [])
            if top_o_counter:
                message += "\n**Top O-Counter Performer:**\n"
                for i, performer in enumerate(top_o_counter, 1):
                    message += (f"{i}. **{performer['name']}**\n"
                                f"   O-Counter: {performer['o_counter']}\n"
                                f"   Szenenanzahl: {performer['scene_count']}\n"
                                f"   Measurements: {performer['measurements']}\n"
                                f"   Cup-to-BMI: {performer.get('cup_to_bmi', 'N/A'):.4f}\n\n")
            
            return message
        
        except Exception as e:
            logger.error(f"Error formatting statistics: {e}")
            return f"Fehler beim Generieren der Statistiken: {e}"
    
    def send_recommendations(self):
        """Send recommendations to output (console and Discord)"""
        logger.info("Preparing to send recommendations")
        
        # Format recommendations
        performer_recs = self.format_performer_recommendations()
        scene_recs = self.format_scene_recommendations()
        
        # Combine messages
        full_message = performer_recs + "\n" + scene_recs
        
        # Print to console
        print("\n" + full_message)
        
        # Send to Discord
        self._send_to_discord(full_message)
    
    def send_statistics(self):
        """Send statistics to output (console and Discord)"""
        logger.info("Preparing to send statistics")
        
        # Format statistics
        stats_message = self.format_statistics()
        
        # Print to console
        print("\n" + stats_message)
        
        # Send to Discord
        self._send_to_discord(stats_message)
    
    def _send_to_discord(self, message):
        """Send message to Discord webhook"""
        if not self.webhook_url:
            logger.error("No Discord webhook URL configured")
            return
        
        try:
            # Split message if too long (Discord has 2000 character limit)
            max_length = 1900  # Leave room for potential truncation message
            message_chunks = []
            
            # Split long messages into chunks
            while message:
                if len(message) <= max_length:
                    message_chunks.append(message)
                    break
                
                # Find a good split point
                split_point = message.rfind('\n', 0, max_length)
                if split_point == -1:
                    split_point = max_length
                
                message_chunks.append(message[:split_point])
                message = message[split_point:].lstrip()
            
            # Send each chunk
            for i, chunk in enumerate(message_chunks, 1):
                # Add chunk number if multiple chunks
                if len(message_chunks) > 1:
                    chunk = f"Part {i}/{len(message_chunks)}\n" + chunk
                
                response = requests.post(
                    self.webhook_url, 
                    json={"content": chunk},
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 204:
                    logger.info(f"Successfully sent chunk {i} to Discord")
                else:
                    logger.error(f"Failed to send chunk {i} to Discord. Status code: {response.status_code}, Response: {response.text}")
        
        except Exception as e:
            logger.error(f"Error sending message to Discord: {e}")

def send_recommendations(recommendation_module, config_path=None):
    """Helper function to send recommendations"""
    discord_module = DiscordModule(recommendation_module=recommendation_module, config_path=config_path)
    discord_module.send_recommendations()

def send_statistics(stats_module, config_path=None):
    """Helper function to send statistics"""
    discord_module = DiscordModule(stats_module=stats_module, config_path=config_path)
    discord_module.send_statistics()
