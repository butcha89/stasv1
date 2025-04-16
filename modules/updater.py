import re
import pandas as pd
import sys
import os

# Füge das übergeordnete Verzeichnis zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stash_api import StashClient
from modules.statistics import StatisticsModule

class UpdaterModule:
    def __init__(self, stash_client=None, stats_module=None):
        """Initialize the updater module"""
        self.stash_client = stash_client or StashClient()
        self.stats_module = stats_module or StatisticsModule(self.stash_client)
        
    def update_cup_sizes(self):
        """Update cup sizes in EU format for performers"""
        # Get statistics data
        stats = self.stats_module.get_cup_size_stats()
        cup_df = stats['cup_size_dataframe']
        
        if cup_df.empty:
            return {'success': False, 'message': 'No cup size data available'}
            
        # Get performers that need updating
        updates = []
        
        for _, row in cup_df.iterrows():
            performer_id = row['id']
            band_size = row['band_size']
            cup_letter = row['cup_letter']
            
            if not band_size or not cup_letter:
                continue
                
            # Format cup size in EU format
            eu_cup_size = f"{int(band_size)}{cup_letter}"
            
            # Check if performer has a tag with the cup size
            performer = next((p for p in self.stats_module.performers_data if p['id'] == performer_id), None)
            
            if not performer:
                continue
                
            # Get current tags
            current_tags = performer.get('tags', [])
            current_tag_ids = [tag['id'] for tag in current_tags]
            
            # Check if there's already a cup size tag
            has_cup_size_tag = any(re.match(r'\d{2,3}[A-K]', tag['name']) for tag in current_tags)
            
            if not has_cup_size_tag:
                # Create a new tag for the cup size if it doesn't exist
                tag_query = """
                query($filter: FindFilterType) {
                  findTags(filter: $filter) {
                    tags {
                      id
                      name
                    }
                  }
                }
                """
                
                variables = {
                    "filter": {
                        "q": eu_cup_size
                    }
                }
                
                result = self.stash_client.call_graphql(tag_query, variables)
                tags = result.get('data', {}).get('findTags', {}).get('tags', [])
                
                tag_id = None
                
                if tags:
                    # Use existing tag
                    tag_id = tags[0]['id']
                else:
                    # Create new tag
                    create_tag_query = """
                    mutation($input: TagCreateInput!) {
                      tagCreate(input: $input) {
                        id
                      }
                    }
                    """
                    
                    variables = {
                        "input": {
                            "name": eu_cup_size
                        }
                    }
                    
                    result = self.stash_client.call_graphql(create_tag_query, variables)
                    tag_id = result.get('data', {}).get('tagCreate', {}).get('id')
                
                if tag_id:
                    # Add tag to performer
                    current_tag_ids.append(tag_id)
                    
                    update_query = """
                    mutation($input: PerformerUpdateInput!) {
                      performerUpdate(input: $input) {
                        id
                      }
                    }
                    """
                    
                    variables = {
                        "input": {
                            "id": performer_id,
                            "tag_ids": current_tag_ids
                        }
                    }
                    
                    result = self.stash_client.call_graphql(update_query, variables)
                    
                    if result.get('data', {}).get('performerUpdate', {}).get('id'):
                        updates.append({
                            'id': performer_id,
                            'name': performer.get('name', ''),
                            'cup_size': eu_cup_size
                        })
        
        return {
            'success': True,
            'updates': updates,
            'count': len(updates)
        }
    
    def update_ratios(self):
        """Update performers with ratio tags (cup-to-bmi, cup-to-height, cup-to-weight)"""
        # Get ratio statistics
        ratio_stats = self.stats_module.get_ratio_stats()
        ratio_df = ratio_stats.get('ratio_dataframe')
        
        if ratio_df.empty:
            return {'success': False, 'message': 'No ratio data available'}
            
        # Get performers that need updating
        updates = []
        
        for _, row in ratio_df.iterrows():
            performer_id = row['id']
            
            # Skip rows with missing data
            if pd.isna(row.get('cup_to_bmi')) and pd.isna(row.get('cup_to_height')) and pd.isna(row.get('cup_to_weight')):
                continue
                
            # Format ratio values
            ratios = {}
            
            if not pd.isna(row.get('cup_to_bmi')):
                ratios['cup_to_bmi'] = f"cup-to-bmi:{row['cup_to_bmi']:.3f}"
                
            if not pd.isna(row.get('cup_to_height')):
                ratios['cup_to_height'] = f"cup-to-height:{row['cup_to_height']:.5f}"
                
            if not pd.isna(row.get('cup_to_weight')):
                ratios['cup_to_weight'] = f"cup-to-weight:{row['cup_to_weight']:.4f}"
            
            if not ratios:
                continue
                
            # Get performer details
            performer = next((p for p in self.stats_module.performers_data if p['id'] == performer_id), None)
            
            if not performer:
                continue
                
            # Update performer details field with ratio information
            details = performer.get('details', '')
            
            # Check if details already has ratio information
            has_ratio_info = any(f"{key}:" in details for key in ratios.keys())
            
            if not has_ratio_info:
                # Add ratio information to details
                ratio_text = "\n\nRatios:\n" + "\n".join(ratios.values())
                
                if details:
                    new_details = details + ratio_text
                else:
                    new_details = ratio_text
                
                # Update performer
                update_query = """
                mutation($input: PerformerUpdateInput!) {
                  performerUpdate(input: $input) {
                    id
                  }
                }
                """
                
                variables = {
                    "input": {
                        "id": performer_id,
                        "details": new_details
                    }
                }
                
                result = self.stash_client.call_graphql(update_query, variables)
                
                if result.get('data', {}).get('performerUpdate', {}).get('id'):
                    updates.append({
                        'id': performer_id,
                        'name': performer.get('name', ''),
                        'ratios': list(ratios.values())
                    })
        
        return {
            'success': True,
            'updates': updates,
            'count': len(updates)
        }
    
    def run_all_updates(self):
        """Run all updater functions"""
        cup_size_results = self.update_cup_sizes()
        ratio_results = self.update_ratios()
        
        return {
            'cup_size_updates': cup_size_results,
            'ratio_updates': ratio_results
        }
