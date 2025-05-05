# Social Media Sentiment Analysis

## Overview
This project analyzes sentiment patterns across various social media platforms during major global events. By applying natural language processing techniques, the analysis reveals how public sentiment evolves over time and differs across platforms and demographics.

## Objectives
- Track sentiment trends across multiple social media platforms
- Identify key opinion leaders and influential users
- Map sentiment geographic distribution
- Detect emerging topics and sentiment shifts
- Compare sentiment across different demographic groups

## Methodology
1. **Data Collection**
   - Twitter/X API for tweets
   - Reddit API for subreddit comments
   - YouTube API for video comments
   - Instagram scraping for public post comments

2. **Preprocessing**
   - Text cleaning and normalization
   - Emoji interpretation
   - Language detection and translation
   - Duplicate and bot content filtering

3. **Sentiment Analysis**
   - VADER for rule-based sentiment scoring
   - RoBERTa for contextual sentiment analysis
   - Custom fine-tuned models for platform-specific language patterns

4. **Topic Modeling**
   - LDA (Latent Dirichlet Allocation)
   - BERTopic for contextual topic extraction
   - Temporal topic evolution tracking

5. **Visualization**
   - Interactive dashboards using Plotly and Dash
   - Geographic sentiment mapping
   - Topic-sentiment relationship networks
   - Temporal trend analysis

## Key Findings
- Sentiment varies significantly by platform for identical events
- Regional sentiment clusters emerge during polarizing global events
- Sentiment shifts predict public opinion changes by 3-5 days
- Platform-specific language patterns require customized sentiment models

## Future Work
- Real-time sentiment monitoring system
- Cross-platform user identity resolution
- Multimodal sentiment analysis (text + images)
- Causality analysis between events and sentiment shifts

## Technologies Used
- Python (pandas, numpy)
- Natural Language Processing (NLTK, spaCy, Transformers)
- Machine Learning (scikit-learn, PyTorch)
- Data Visualization (Plotly, Dash, Matplotlib)
- Database (MongoDB, PostgreSQL)

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for analysis
- `models/`: Trained sentiment and topic models
- `visualizations/`: Generated charts and dashboards

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Configure API keys in `config.py`
3. Run data collection scripts: `python scripts/collect_data.py`
4. Execute analysis notebooks in `notebooks/` directory

## License
This project is licensed under the MIT License - see the LICENSE file for details. 