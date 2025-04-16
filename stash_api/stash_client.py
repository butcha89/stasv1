import requests
import json
import configparser
import os

class StashClient:
    def __init__(self, config_path=None):
        """Initialize the StashClient with configuration"""
        if config_path is None:
            # Default to the config in the project directory
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'config', 'configuration.ini')
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.host = config.get('stash', 'host')
        self.port = config.get('stash', 'port')
        self.api_key = config.get('stash', 'api_key')
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
            
        response = requests.post(self.url, headers=self.headers, json=json_data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed with status code {response.status_code}: {response.text}")
    
    def get_performers(self, filter_criteria=None):
        """Get performers with optional filtering"""
        query = """
        query($filter: FindFilterType) {
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
            }
          }
        }
        """
        
        variables = {}
        if filter_criteria:
            variables["filter"] = filter_criteria
            
        result = self.call_graphql(query, variables)
        return result.get('data', {}).get('findPerformers', {}).get('performers', [])
    
    def get_scenes(self, filter_criteria=None):
        """Get scenes with optional filtering"""
        query = """
        query($filter: SceneFilterType) {
          findScenes(filter: $filter) {
            count
            scenes {
              id
              title
              details
              url
              date
              rating
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
                chapters_vtt
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
              }
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        
        variables = {}
        if filter_criteria:
            variables["filter"] = filter_criteria
            
        result = self.call_graphql(query, variables)
        return result.get('data', {}).get('findScenes', {}).get('scenes', [])
    
    def update_performer(self, performer_id, performer_data):
        """Update a performer with the given data"""
        query = """
        mutation($input: PerformerUpdateInput!) {
          performerUpdate(input: $input) {
            id
          }
        }
        """
        
        performer_data['id'] = performer_id
        variables = {
            "input": performer_data
        }
        
        return self.call_graphql(query, variables)
