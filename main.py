#!/usr/bin/env python3
import argparse
import configparser
import os
import sys
from stash_api import StashClient
from modules.statistics import StatisticsModule
from modules.recommendations import RecommendationModule
from modules.dashboard import DashboardModule
from modules.updater import UpdaterModule
from modules.discord import DiscordModule

def main():
    parser = argparse.ArgumentParser(description='Stash API Tools')
    parser.add_argument('--config', help='Path to configuration file', 
                        default='config/configuration.ini')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Generate statistics')
    stats_parser.add_argument('--output', help='Output file for statistics', default=None)
    
    # Recommendations command
    rec_parser = subparsers.add_parser('recommend', help='Generate recommendations')
    rec_parser.add_argument('--type', choices=['performers', 'scenes', 'all'], 
                           default='all', help='Type of recommendations')
    
    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Run the dashboard')
    dash_parser.add_argument('--port', type=int, default=8050, help='Port for the dashboard')
    dash_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # Updater command
    update_parser = subparsers.add_parser('update', help='Update performer data')
    update_parser.add_argument('--type', choices=['cup-sizes', 'ratios', 'all'], 
                             default='all', help='Type of updates to perform')
    
    # Discord command
    discord_parser = subparsers.add_parser('discord', help='Post updates to Discord')
    discord_parser.add_argument('--type', choices=['stats', 'performers', 'scenes', 'all'], 
                              default='all', help='Type of updates to post')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration file...")
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        # Create default config
        config = configparser.ConfigParser()
        config['stash'] = {
            'api_key': 'your_api_key_here',
            'host': 'localhost',
            'port': '9999'
        }
        config['discord'] = {
            'webhook_url': 'your_discord_webhook_url_here',
            'enable_stats_posting': 'true',
            'enable_performer_recommendations': 'true',
            'enable_scene_recommendations': 'true'
        }
        config['statistics'] = {
            'enable': 'true',
            'cup_size_stats': 'true',
            'o_counter_stats': 'true',
            'ratio_stats': 'true'
        }
        config['recommendations'] = {
            'enable_performer_recommendations': 'true',
            'enable_scene_recommendations': 'true',
            'min_similarity_score': '0.7',
            'max_recommendations': '10'
        }
        config['updater'] = {
            'auto_tag_performers': 'true',
            'update_cup_sizes': 'true',
            'update_ratios': 'true'
        }
        
        with open(args.config, 'w') as configfile:
            config.write(configfile)
            
        print(f"Default configuration created at {args.config}")
        print("Please edit the configuration file with your settings and run again.")
        sys.exit(0)
    
    # Initialize client
    client = StashClient(args.config)
    
    # Initialize modules
    stats_module = StatisticsModule(client)
    rec_module = RecommendationModule(client, stats_module)
    update_module = UpdaterModule(client, stats_module)
    discord_module = DiscordModule(stats_module, rec_module, args.config)
    
    # Execute command
    if args.command == 'stats':
        print("Generating statistics...")
        stats = stats_module.generate_all_stats()
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to {args.output}")
        else:
            # Print some summary statistics
            cup_stats = stats['cup_size_stats']
            cup_counts = cup_stats['cup_size_counts']
            
            print("\nCup Size Distribution:")
            for cup, count in sorted(cup_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cup}: {count}")
            
            corr_stats = stats['correlation_stats']
            if corr_stats:
                cup_letter_stats = corr_stats.get('cup_letter_o_stats', [])
                
                if cup_letter_stats:
                    print("\nTop Cup Letters by O-Count:")
                    for stat in sorted(cup_letter_stats, key=lambda x: x['avg_o_count'], reverse=True)[:5]:
                        print(f"  {stat['cup_letter']}: {stat['avg_o_count']:.2f} (n={stat['performer_count']})")
    
    elif args.command == 'recommend':
        if args.type in ['performers', 'all']:
            print("Generating performer recommendations...")
            performer_recs = rec_module.recommend_performers()
            
            print(f"\nFound {len(performer_recs)} performers with recommendations")
            
            for i, rec in enumerate(performer_recs[:5]):
                performer = rec['performer']
                similar = rec['similar_performers'][:3]
                
                print(f"\n{i+1}. {performer['name']} ({performer['cup_size']})")
                print(f"   O-Count: {performer['o_count']}")
                print("   Similar performers:")
                
                for sp in similar:
                    print(f"   - {sp['name']} ({sp['cup_size']}) - Similarity: {sp['similarity']:.2f}")
        
        if args.type in ['scenes', 'all']:
            print("\nGenerating scene recommendations...")
            scene_recs = rec_module.recommend_scenes()
            
            for key, scenes in scene_recs.items():
                print(f"\n{key.replace('_', ' ').title()}:")
                print(f"Found {len(scenes)} recommendations")
                
                for i, scene in enumerate(scenes[:3]):
                    print(f"\n{i+1}. {scene['title']}")
                    print(f"   Similarity: {scene['similarity']:.2f}")
                    print(f"   Performers: {', '.join(scene['performers'])}")
    
    elif args.command == 'dashboard':
        print(f"Starting dashboard on port {args.port}...")
        dashboard = DashboardModule(stats_module, rec_module)
        dashboard.run_server(debug=args.debug, port=args.port)
    
    elif args.command == 'update':
        print("Running updater...")
        
        if args.type in ['cup-sizes', 'all']:
            print("\nUpdating cup sizes...")
            cup_results = update_module.update_cup_sizes()
            
            if cup_results['success']:
                print(f"Updated {cup_results['count']} performers with cup size tags")
            else:
                print(f"Cup size update failed: {cup_results['message']}")
        
        if args.type in ['ratios', 'all']:
            print("\nUpdating ratio information...")
            ratio_results = update_module.update_ratios()
            
            if ratio_results['success']:
                print(f"Updated {ratio_results['count']} performers with ratio information")
            else:
                print(f"Ratio update failed: {ratio_results['message']}")
    
    elif args.command == 'discord':
        print("Posting to Discord...")
        
        if args.type in ['stats', 'all']:
            print("\nPosting statistics...")
            stats_result = discord_module.post_statistics()
            
            if stats_result['success']:
                print("Statistics posted successfully")
            else:
                print(f"Failed to post statistics: {stats_result.get('message', stats_result)}")
        
        if args.type in ['performers', 'all']:
            print("\nPosting performer recommendations...")
            performer_result = discord_module.post_performer_recommendations()
            
            if performer_result['success']:
                print("Performer recommendations posted successfully")
            else:
                print(f"Failed to post performer recommendations: {performer_result.get('message', performer_result)}")
        
        if args.type in ['scenes', 'all']:
            print("\nPosting scene recommendations...")
            scene_result = discord_module.post_scene_recommendations()
            
            if scene_result['success']:
                print("Scene recommendations posted successfully")
            else:
                print(f"Failed to post scene recommendations: {scene_result.get('message', scene_result)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
