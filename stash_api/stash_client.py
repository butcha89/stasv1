import requests
import json
import configparser
import os
import logging

logger = logging.getLogger(__name__)

class StashClient:
    def __init__(self, config_path=None):
        """Initialize the StashClient with configuration"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'config', 'configuration.ini')
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.host = config.get('stash', 'host', fallback='localhost')
        self.port = config.get('stash', 'port', fallback='9999')
        self.api_key = config.get('stash', 'api_key', fallback='')
        self.url = f"http://{self.host}:{self.port}/graphql"
        
        self.headers = {
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
            "DNT": "1"
        }
        
        if self.api_key:
            self.headers["ApiKey"] = self.api_key
    
    def call_graphql(self, query, variables=None):
        """Make a GraphQL request to the Stash API"""
        json_data = {'query': query}
        if variables:
            json_data['variables'] = variables
        
        try:
            response = requests.post(self.url, headers=self.headers, json=json_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Log any GraphQL errors
                if 'errors' in result:
                    logger.error(f"GraphQL Errors: {result['errors']}")
                    return None
                
                return result.get('data')
            else:
                logger.error(f"Query failed with status code {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"GraphQL request error: {e}")
            return None
    
    def get_performers(self, filter_criteria=None):
        """Get performers with optional filtering and pagination"""
        query = """
        query findPerformers($filter: FindFilterType) {
          findPerformers(filter: $filter) {
            count
            performers {
              id
              name
              gender
              url
              twitter
              instagram
              birthdate
              ethnicity
              country
              eye_color
              height_cm
              measurements
              fake_tits
              career_length
              tattoos
              piercings
              favorite
              image_path
              scene_count
              stash_ids {
                endpoint
                stash_id
              }
              rating100
              details
              death_date
              hair_color
              weight
              tags {
                id
                name
              }
              o_counter
            }
          }
        }
        """
        
        # Default filter to return all performers without pagination limit
        default_filter = {
            "per_page": 10000,  # Large number to fetch all performers
            "page": 1
        }
        
        # Merge or replace default filter with user-provided filter
        if filter_criteria:
            # Merge dictionaries, giving priority to user-provided filter
            merged_filter = {**default_filter, **filter_criteria}
        else:
            merged_filter = default_filter
        
        variables = {"filter": merged_filter}
        
        result = self.call_graphql(query, variables)
        
        if result:
            performers = result.get('findPerformers', {}).get('performers', [])
            total_count = result.get('findPerformers', {}).get('count', 0)
            
            logger.info(f"Retrieved {len(performers)} performers out of {total_count} total")
            
            return performers
        
        return []
    
    def get_scenes(self, filter_criteria=None):
        """Get scenes with optional filtering and pagination"""
        query = """
        query findScenes($filter: FindFilterType) {
          findScenes(filter: $filter) {
            count
            scenes {
              id
              title
              details
              url
              date
              rating100
              o_counter
              organized
              interactive
              interactive_speed
              created_at
              updated_at
              files {
                path
                size
                duration
                video_codec
                audio_codec
                width
                height
                frame_rate
                bit_rate
              }
              paths {
                screenshot
                preview
                stream
                webp
                vtt
                sprite
                funscript
              }
              scene_markers {
                id
                title
                seconds
                tags {
                  id
                  name
                }
              }
              galleries {
                id
                title
              }
              studio {
                id
                name
              }
              movies {
                movie {
                  id
                  name
                }
                scene_index
              }
              tags {
                id
                name
              }
              performers {
                id
                name
                gender
                favorite
                o_counter
              }
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        
        # Default filter to return all scenes without pagination limit
        default_filter = {
            "per_page": 10000,  # Large number to fetch all scenes
            "page": 1
        }
        
        # Merge or replace default filter with user-provided filter
        if filter_criteria:
            # Merge dictionaries, giving priority to user-provided filter
            merged_filter = {**default_filter, **filter_criteria}
        else:
            merged_filter = default_filter
        
        variables = {"filter": merged_filter}
        
        result = self.call_graphql(query, variables)
        
        if result:
            scenes = result.get('findScenes', {}).get('scenes', [])
            total_count = result.get('findScenes', {}).get('count', 0)
            
            logger.info(f"Retrieved {len(scenes)} scenes out of {total_count} total")
            
            return scenes
        
        return []
