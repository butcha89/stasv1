#!/usr/bin/env python3
import argparse
import configparser
import os
import sys
import logging
import json

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stash_tools.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    from stash_api.stash_client import StashClient
    from modules.statistics import StatisticsModule
    from modules.recommendations import RecommendationModule
    from modules.discord import send_recommendations, send_statistics
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

def main():
    # Ensure a default configuration exists
    default_config_path = os.path.join(project_root, 'config', 'configuration.ini')
    
    # Default configuration creation
    def create_default_config():
        config = configparser.ConfigParser()
        config['stash'] = {
            'host': 'localhost',
            'port': '9999',
            'api_key': 'your_api_key_here'
        }
        config['discord'] = {
            'webhook_url': 'your_discord_webhook_url_here',
            'enable_stats_posting': 'false',
            'enable_performer_recommendations': 'false',
            'enable_scene_recommendations': 'false'
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(default_config_path), exist_ok=True)
        
        with open(default_config_path, 'w') as configfile:
            config.write(configfile)
        
        logger.warning(f"Created default configuration at {default_config_path}")
    
    # Create default config if it doesn't exist
    if not os.path.exists(default_config_path):
        create_default_config()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Stash API Tools')
    parser.add_argument('--config', 
                        help='Path to configuration file', 
                        default=default_config_path)
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Generate statistics')
    stats_parser.add_argument('--output', 
                              help='Output file for statistics', 
                              default='stash_stats.json')
    
    # Recommendations command
    rec_parser = subparsers.add_parser('recommend', help='Generate recommendations')
    rec_parser.add_argument('--type', 
                            choices=['performers', 'scenes', 'all'], 
                            default='all', 
                            help='Type of recommendations')
    
    # Discord recommendations command
    discord_parser = subparsers.add_parser('discord', help='Send to Discord')
    discord_parser.add_argument('--type', 
                                choices=['recommendations', 'stats', 'all'], 
                                default='all', 
                                help='Type of Discord message')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, default to stats
    if not args.command:
        args.command = 'stats'
    
    try:
        # Initialize client and modules
        client = StashClient(args.config)
        stats_module = StatisticsModule(client)
        rec_module = RecommendationModule(client, stats_module)
    except Exception as e:
        logger.error(f"Failed to initialize modules: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'stats':
            logger.info("Generating statistics...")
            stats = stats_module.generate_all_stats()
            
            # Save to JSON
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Statistics saved to {args.output}")
            
            # Print some key stats
            cup_stats = stats.get('cup_size_stats', {})
            cup_counts = cup_stats.get('cup_size_counts', {})
            
            logger.info("\nCup Size Distribution:")
            for cup, count in sorted(cup_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {cup}: {count}")
        
        elif args.command == 'recommend':
            logger.info("Generating recommendations...")
            if args.type in ['performers', 'all']:
                performer_recs = rec_module.recommend_performers()
                logger.info(f"Found {len(performer_recs)} performer recommendations")
            
            if args.type in ['scenes', 'all']:
                scene_recs = rec_module.recommend_scenes()
                logger.info("Scene recommendations generated")
        
        elif args.command == 'discord':
            logger.info("Sending information to Discord...")
            if args.type in ['recommendations', 'all']:
                logger.info("Sending recommendations...")
                send_recommendations(rec_module, args.config)
            
            if args.type in ['stats', 'all']:
                logger.info("Sending statistics...")
                send_statistics(stats_module, args.config)
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
