#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Dashboard for Social Media Sentiment Analysis
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.sentiment_analyzer import SentimentAnalyzer

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer(use_gpu=False)

# Sample data (replace with your actual data loading logic)
def load_sample_data():
    try:
        # Try to load real data
        data_files = []
        for root, dirs, files in os.walk('../data/raw'):
            for file in files:
                if file.endswith('.csv') and ('tweet' in file.lower() or 'reddit' in file.lower()):
                    data_files.append(os.path.join(root, file))
        
        if data_files:
            # Load and combine all data files
            dfs = []
            for file in data_files:
                df = pd.read_csv(file)
                # Add source column if not present
                if 'source' not in df.columns:
                    if 'tweet' in file.lower():
                        df['source'] = 'twitter'
                    elif 'reddit' in file.lower():
                        df['source'] = 'reddit'
                dfs.append(df)
            
            df = pd.concat(dfs, ignore_index=True)
            
            # Generate sample sentiment if not already present
            if 'consensus_sentiment' not in df.columns:
                text_col = next((col for col in df.columns if col in ['text', 'title', 'content', 'body']), None)
                if text_col:
                    # Sample a subset for demo purposes
                    sample_size = min(100, len(df))
                    df_sample = df.sample(sample_size)
                    sentiment_results = analyzer.analyze_dataframe(df_sample, text_col)
                    df = sentiment_results
            
            return df
        else:
            raise FileNotFoundError("No data files found")
    
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Using generated sample data instead")
        
        # Generate sample data
        np.random.seed(42)
        sources = ['twitter', 'reddit', 'youtube']
        topics = ['climate', 'politics', 'technology', 'entertainment', 'sports']
        sentiments = ['positive', 'neutral', 'negative']
        sentiment_probabilities = [0.4, 0.3, 0.3]
        
        # Generate dates over past 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = [start_date + timedelta(days=x) for x in range(31)]
        
        # Create sample data
        sample_size = 1000
        data = {
            'created_at': np.random.choice(dates, sample_size),
            'source': np.random.choice(sources, sample_size, p=[0.5, 0.3, 0.2]),
            'topic': np.random.choice(topics, sample_size),
            'consensus_sentiment': np.random.choice(sentiments, sample_size, p=sentiment_probabilities),
            'consensus_score': np.random.normal(0, 0.5, sample_size),
            'followers_count': np.random.exponential(1000, sample_size),
            'engagement': np.random.exponential(50, sample_size)
        }
        
        # Ensure sentiment and score match
        for i, sentiment in enumerate(data['consensus_sentiment']):
            if sentiment == 'positive':
                data['consensus_score'][i] = abs(data['consensus_score'][i])
            elif sentiment == 'negative':
                data['consensus_score'][i] = -abs(data['consensus_score'][i])
            else:
                data['consensus_score'][i] = data['consensus_score'][i] * 0.2
        
        df = pd.DataFrame(data)
        
        # Convert datetime objects to pandas datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Sort by date
        df = df.sort_values('created_at')
        
        return df

# Load data
df = load_sample_data()

# Initialize Dash app
app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                title="Social Media Sentiment Dashboard")

# Create layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Social Media Sentiment Analysis Dashboard", 
                style={"margin-bottom": "0px", "color": "white"}),
        html.H5("Real-time sentiment tracking across platforms", 
                style={"margin-top": "0px", "color": "white"})
    ], style={"text-align": "center", "padding": "1rem", "background-color": "#2c3e50"}),
    
    # Filters and Controls
    html.Div([
        html.Div([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id="date-picker",
                min_date_allowed=df['created_at'].min().date(),
                max_date_allowed=df['created_at'].max().date(),
                start_date=df['created_at'].min().date(),
                end_date=df['created_at'].max().date()
            )
        ], className="four columns"),
        
        html.Div([
            html.Label("Platform:"),
            dcc.Dropdown(
                id="platform-dropdown",
                options=[{"label": "All Platforms", "value": "all"}] + 
                         [{"label": src.capitalize(), "value": src} for src in df['source'].unique()],
                value="all",
                multi=False
            )
        ], className="four columns"),
        
        html.Div([
            html.Label("Topic:") if 'topic' in df.columns else html.Label("Filter:"),
            dcc.Dropdown(
                id="topic-dropdown",
                options=[{"label": "All Topics", "value": "all"}] + 
                         [{"label": topic.capitalize(), "value": topic} 
                          for topic in df['topic'].unique()] if 'topic' in df.columns 
                         else [{"label": "All", "value": "all"}],
                value="all",
                multi=False
            )
        ], className="four columns")
    ], className="row", style={"padding": "1rem"}),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.Div([
                html.H4("Total Posts"),
                html.H2(id="total-posts", children="0")
            ], className="kpi-card")
        ], className="four columns"),
        
        html.Div([
            html.Div([
                html.H4("Average Sentiment"),
                html.H2(id="avg-sentiment", children="0")
            ], className="kpi-card")
        ], className="four columns"),
        
        html.Div([
            html.Div([
                html.H4("Sentiment Distribution"),
                html.Div(id="sentiment-distribution", children=[
                    html.Span("Positive: 0%", style={"color": "green", "margin-right": "10px"}),
                    html.Span("Neutral: 0%", style={"color": "gray", "margin-right": "10px"}),
                    html.Span("Negative: 0%", style={"color": "red"})
                ])
            ], className="kpi-card")
        ], className="four columns")
    ], className="row", style={"padding": "1rem"}),
    
    # Charts
    html.Div([
        # Sentiment over time chart
        html.Div([
            html.H4("Sentiment Trends Over Time"),
            dcc.Graph(id="sentiment-time-chart")
        ], className="six columns"),
        
        # Platform comparison chart
        html.Div([
            html.H4("Sentiment by Platform"),
            dcc.Graph(id="platform-sentiment-chart")
        ], className="six columns")
    ], className="row", style={"padding": "1rem"}),
    
    html.Div([
        # Topic sentiment chart
        html.Div([
            html.H4("Sentiment by Topic" if 'topic' in df.columns else "Sentiment Distribution"),
            dcc.Graph(id="topic-sentiment-chart")
        ], className="six columns"),
        
        # Engagement vs Sentiment
        html.Div([
            html.H4("Engagement vs. Sentiment"),
            dcc.Graph(id="engagement-sentiment-chart")
        ], className="six columns")
    ], className="row", style={"padding": "1rem"}),
    
    # Data Table
    html.Div([
        html.H4("Recent Posts"),
        dash_table.DataTable(
            id="posts-table",
            columns=[
                {"name": "Date", "id": "created_at"},
                {"name": "Platform", "id": "source"},
                {"name": "Topic" if 'topic' in df.columns else "Content", "id": "topic" if 'topic' in df.columns else "text"},
                {"name": "Sentiment", "id": "consensus_sentiment"},
                {"name": "Score", "id": "consensus_score"}
            ],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "whiteSpace": "normal",
                "height": "auto"
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{consensus_sentiment} = 'positive'"},
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "color": "green"
                },
                {
                    "if": {"filter_query": "{consensus_sentiment} = 'negative'"},
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "color": "red"
                }
            ],
            style_header={
                "backgroundColor": "#2c3e50",
                "color": "white",
                "fontWeight": "bold"
            }
        )
    ], style={"padding": "1rem"}),
    
    # Footer
    html.Div([
        html.P("© 2025 Social Media Sentiment Analysis Project | Last updated: " + 
               datetime.now().strftime("%Y-%m-%d"))
    ], style={"text-align": "center", "padding": "1rem", "background-color": "#f8f9fa"})
    
], style={"font-family": "Arial, sans-serif"})

# Define callbacks
@app.callback(
    [
        Output("total-posts", "children"),
        Output("avg-sentiment", "children"),
        Output("sentiment-distribution", "children"),
        Output("sentiment-time-chart", "figure"),
        Output("platform-sentiment-chart", "figure"),
        Output("topic-sentiment-chart", "figure"),
        Output("engagement-sentiment-chart", "figure"),
        Output("posts-table", "data")
    ],
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("platform-dropdown", "value"),
        Input("topic-dropdown", "value")
    ]
)
def update_dashboard(start_date, end_date, platform, topic):
    # Filter data by date
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['created_at'] >= start_date) & 
                              (filtered_df['created_at'] <= end_date)]
    
    # Filter by platform
    if platform != "all":
        filtered_df = filtered_df[filtered_df['source'] == platform]
    
    # Filter by topic if available
    if 'topic' in filtered_df.columns and topic != "all":
        filtered_df = filtered_df[filtered_df['topic'] == topic]
    
    # Calculate KPIs
    total_posts = len(filtered_df)
    
    avg_sentiment = filtered_df['consensus_score'].mean()
    avg_sentiment_text = f"{avg_sentiment:.2f}"
    
    # Count sentiments
    sentiment_counts = filtered_df['consensus_sentiment'].value_counts(normalize=True) * 100
    positive_pct = sentiment_counts.get('positive', 0)
    neutral_pct = sentiment_counts.get('neutral', 0)
    negative_pct = sentiment_counts.get('negative', 0)
    
    sentiment_distribution = [
        html.Span(f"Positive: {positive_pct:.1f}%", style={"color": "green", "margin-right": "10px"}),
        html.Span(f"Neutral: {neutral_pct:.1f}%", style={"color": "gray", "margin-right": "10px"}),
        html.Span(f"Negative: {negative_pct:.1f}%", style={"color": "red"})
    ]
    
    # Sentiment over time chart
    filtered_df['date'] = filtered_df['created_at'].dt.date
    sentiment_by_day = filtered_df.groupby('date')['consensus_score'].mean().reset_index()
    
    sentiment_time_chart = px.line(
        sentiment_by_day, 
        x='date', 
        y='consensus_score',
        title="Average Sentiment Score Over Time",
        labels={"date": "Date", "consensus_score": "Sentiment Score"},
        color_discrete_sequence=["#2c3e50"]
    )
    sentiment_time_chart.add_hline(y=0, line_dash="dash", line_color="gray")
    sentiment_time_chart.update_layout(
        plot_bgcolor="white",
        yaxis_range=[-1, 1]
    )
    
    # Platform sentiment chart
    platform_sentiment = filtered_df.groupby('source')['consensus_score'].mean().reset_index()
    
    platform_sentiment_chart = px.bar(
        platform_sentiment,
        x='source',
        y='consensus_score',
        title="Average Sentiment by Platform",
        labels={"source": "Platform", "consensus_score": "Sentiment Score"},
        color='consensus_score',
        color_continuous_scale=["red", "yellow", "green"],
        range_color=[-1, 1]
    )
    platform_sentiment_chart.add_hline(y=0, line_dash="dash", line_color="gray")
    platform_sentiment_chart.update_layout(plot_bgcolor="white")
    
    # Topic sentiment chart
    if 'topic' in filtered_df.columns:
        topic_sentiment = filtered_df.groupby('topic')['consensus_score'].mean().reset_index()
        
        topic_chart = px.bar(
            topic_sentiment,
            x='topic',
            y='consensus_score',
            title="Average Sentiment by Topic",
            labels={"topic": "Topic", "consensus_score": "Sentiment Score"},
            color='consensus_score',
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[-1, 1]
        )
        topic_chart.add_hline(y=0, line_dash="dash", line_color="gray")
        topic_chart.update_layout(plot_bgcolor="white")
    else:
        # If no topic column, show sentiment distribution pie chart
        topic_chart = px.pie(
            filtered_df,
            names='consensus_sentiment',
            title="Sentiment Distribution",
            color='consensus_sentiment',
            color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        )
        topic_chart.update_layout(plot_bgcolor="white")
    
    # Engagement vs Sentiment chart
    engagement_col = next((col for col in filtered_df.columns 
                          if col in ['engagement', 'retweet_count', 'favorite_count', 'score']), None)
    
    if engagement_col:
        engagement_chart = px.scatter(
            filtered_df,
            x='consensus_score',
            y=engagement_col,
            color='source',
            title=f"Engagement vs. Sentiment",
            labels={"consensus_score": "Sentiment Score", engagement_col: "Engagement"},
            opacity=0.7
        )
        engagement_chart.add_vline(x=0, line_dash="dash", line_color="gray")
        engagement_chart.update_layout(plot_bgcolor="white")
    else:
        # Create dummy scatter if no engagement column
        engagement_chart = px.scatter(
            filtered_df,
            x='consensus_score',
            y=[1] * len(filtered_df),
            color='source',
            title="Sentiment Distribution",
            labels={"consensus_score": "Sentiment Score", "y": "Count"},
            opacity=0.7
        )
        engagement_chart.add_vline(x=0, line_dash="dash", line_color="gray")
        engagement_chart.update_layout(plot_bgcolor="white")
    
    # Table data
    table_df = filtered_df.sort_values('created_at', ascending=False).head(10)
    
    # Format dates for the table
    table_df['created_at'] = table_df['created_at'].dt.strftime('%Y-%m-%d')
    
    # Format scores for the table
    table_df['consensus_score'] = table_df['consensus_score'].round(2)
    
    # Select and prepare columns for table
    if 'topic' in table_df.columns:
        table_data = table_df[['created_at', 'source', 'topic', 'consensus_sentiment', 'consensus_score']].to_dict('records')
    else:
        text_col = next((col for col in table_df.columns 
                        if col in ['text', 'title', 'content', 'body']), 'consensus_sentiment')
        # Rename the text column for display
        table_df = table_df.rename(columns={text_col: 'text'})
        table_data = table_df[['created_at', 'source', 'text', 'consensus_sentiment', 'consensus_score']].to_dict('records')
    
    return total_posts, avg_sentiment_text, sentiment_distribution, sentiment_time_chart, \
           platform_sentiment_chart, topic_chart, engagement_chart, table_data

# Add CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .kpi-card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 20px;
                text-align: center;
                margin-bottom: 15px;
                height: 120px;
            }
            .kpi-card h4 {
                margin-top: 0;
                color: #2c3e50;
                font-size: 1rem;
            }
            .kpi-card h2 {
                margin-bottom: 0;
                color: #2c3e50;
                font-size: 2rem;
            }
            body {
                background-color: #f5f5f5;
                margin: 0;
                padding: 0;
            }
            .row {
                margin-left: 0;
                margin-right: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 