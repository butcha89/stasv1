import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from modules.statistics import StatisticsModule
from modules.recommendations import RecommendationModule

class DashboardModule:
    def __init__(self, stats_module=None, recommendation_module=None):
        """Initialize the dashboard module"""
        self.stats_module = stats_module or StatisticsModule()
        self.recommendation_module = recommendation_module or RecommendationModule(
            stats_module=self.stats_module
        )
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        
    def setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Stash Analytics Dashboard"),
            
            dcc.Tabs([
                dcc.Tab(label="Statistics", children=[
                    html.Div([
                        html.H2("Cup Size Statistics"),
                        dcc.Graph(id='cup-size-distribution'),
                        
                        html.H2("O-Counter Statistics"),
                        dcc.Graph(id='o-counter-by-cup'),
                        
                        html.H2("Ratio Statistics"),
                        dcc.Graph(id='ratio-stats')
                    ])
                ]),
                
                dcc.Tab(label="Performer Recommendations", children=[
                    html.Div([
                        html.H2("Similar Performers"),
                        html.Div(id='performer-recommendations')
                    ])
                ]),
                
                dcc.Tab(label="Scene Recommendations", children=[
                    html.Div([
                        html.H2("Recommended Scenes with Favorite Performers"),
                        html.Div(id='favorite-scene-recommendations'),
                        
                        html.H2("Recommended Scenes with Non-Favorite Performers"),
                        html.Div(id='non-favorite-scene-recommendations'),
                        
                        html.H2("Recommended Scenes with Recommended Performers"),
                        html.Div(id='recommended-performer-scene-recommendations')
                    ])
                ])
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=60*60*1000,  # refresh every hour
                n_intervals=0
            )
        ])
        
        self.setup_callbacks()
        
    def setup_callbacks(self):
        """Set up the dashboard callbacks"""
        @self.app.callback(
            [dash.dependencies.Output('cup-size-distribution', 'figure'),
             dash.dependencies.Output('o-counter-by-cup', 'figure'),
             dash.dependencies.Output('ratio-stats', 'figure')],
            [dash.dependencies.Input('interval-component', 'n_intervals')]
        )
        def update_statistics(_):
            # Get statistics
            stats = self.stats_module.generate_all_stats()
            
            # Cup size distribution
            cup_stats = stats['cup_size_stats']
            cup_counts = cup_stats['cup_size_counts']
            
            cup_fig = go.Figure()
            if cup_counts:
                # Sort by cup size
                sorted_cups = sorted(cup_counts.items(), 
                                    key=lambda x: (int(x[0][:-1]), x[0][-1]))
                
                cups, counts = zip(*sorted_cups)
                
                cup_fig = px.bar(x=cups, y=counts, 
                                title="Cup Size Distribution",
                                labels={'x': 'Cup Size', 'y': 'Count'})
            
            # O-counter by cup
            corr_stats = stats['correlation_stats']
            o_fig = go.Figure()
            
            if corr_stats:
                cup_letter_stats = corr_stats.get('cup_letter_o_stats', [])
                if cup_letter_stats:
                    df = pd.DataFrame(cup_letter_stats)
                    o_fig = px.bar(df, x='cup_letter', y='avg_o_count',
                                  title="Average O-Counter by Cup Size",
                                  labels={'cup_letter': 'Cup Letter', 
                                         'avg_o_count': 'Average O-Counter'})
                    
                    # Add performer count as text
                    o_fig.update_traces(text=df['performer_count'].apply(lambda x: f"n={x}"),
                                      textposition='outside')
            
            # Ratio statistics
            ratio_stats = stats['ratio_stats']
            ratio_fig = go.Figure()
            
            if ratio_stats:
                ratio_data = ratio_stats.get('ratio_stats', [])
                if ratio_data:
                    df = pd.DataFrame(ratio_data)
                    
                    # Create a figure with multiple traces
                    ratio_fig = go.Figure()
                    
                    if 'avg_cup_to_bmi' in df.columns:
                        ratio_fig.add_trace(go.Bar(
                            x=df['cup_letter'],
                            y=df['avg_cup_to_bmi'],
                            name='Cup to BMI'
                        ))
                    
                    if 'avg_cup_to_height' in df.columns:
                        ratio_fig.add_trace(go.Bar(
                            x=df['cup_letter'],
                            y=df['avg_cup_to_height'],
                            name='Cup to Height'
                        ))
                    
                    if 'avg_cup_to_weight' in df.columns:
                        ratio_fig.add_trace(go.Bar(
                            x=df['cup_letter'],
                            y=df['avg_cup_to_weight'],
                            name='Cup to Weight'
                        ))
                    
                    ratio_fig.update_layout(
                        title="Average Ratios by Cup Size",
                        xaxis_title="Cup Letter",
                        yaxis_title="Ratio Value",
                        barmode='group'
                    )
            
            return cup_fig, o_fig, ratio_fig
        
        @self.app.callback(
            dash.dependencies.Output('performer-recommendations', 'children'),
            [dash.dependencies.Input('interval-component', 'n_intervals')]
        )
        def update_performer_recommendations(_):
            recommendations = self.recommendation_module.recommend_performers()
            
            if not recommendations:
                return html.Div("No recommendations available")
                
            recommendation_divs = []
            
            for rec in recommendations[:10]:  # Limit to top 10
                performer = rec['performer']
                similar_performers = rec['similar_performers']
                
                performer_div = html.Div([
                    html.H3(f"{performer['name']} ({performer['cup_size']})"),
                    html.P(f"O-Count: {performer['o_count']}, " +
                          f"Height: {performer['height_cm']}cm, " +
                          f"Weight: {performer['weight']}kg, " +
                          f"BMI: {performer['bmi']:.1f}"),
                    
                    html.H4("Similar Performers:"),
                    html.Ul([
                        html.Li([
                            html.Strong(f"{sp['name']} ({sp['cup_size']})"),
                            html.Span(f" - Similarity: {sp['similarity']:.2f}, O-Count: {sp['o_count']}")
                        ]) for sp in similar_performers
                    ])
                ], className="recommendation-card")
                
                recommendation_divs.append(performer_div)
            
            return html.Div(recommendation_divs)
        
        @self.app.callback(
            [dash.dependencies.Output('favorite-scene-recommendations', 'children'),
             dash.dependencies.Output('non-favorite-scene-recommendations', 'children'),
             dash.dependencies.Output('recommended-performer-scene-recommendations', 'children')],
            [dash.dependencies.Input('interval-component', 'n_intervals')]
        )
        def update_scene_recommendations(_):
            scene_recs = self.recommendation_module.recommend_scenes()
            
            outputs = []
            
            for key in ['favorite_performer_scenes', 'non_favorite_performer_scenes', 
                       'recommended_performer_scenes']:
                recommendations = scene_recs.get(key, [])
                
                if not recommendations:
                    outputs.append(html.Div("No recommendations available"))
                    continue
                    
                scene_divs = []
                
                for rec in recommendations:
                    scene_div = html.Div([
                        html.H3(rec['title']),
                        html.P(f"Similarity: {rec['similarity']:.2f}"),
                        
                        html.H4("Performers:"),
                        html.Ul([html.Li(p) for p in rec['performers']]),
                        
                        html.H4("Tags:"),
                        html.Div(", ".join(rec['tags']), style={'maxHeight': '100px', 'overflow': 'auto'})
                    ], className="scene-card")
                    
                    scene_divs.append(scene_div)
                
                outputs.append(html.Div(scene_divs))
            
            return outputs
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)
