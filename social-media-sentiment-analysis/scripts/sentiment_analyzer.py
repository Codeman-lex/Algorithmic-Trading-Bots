#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment Analysis Module for Social Media Data
"""

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import re
import emoji
import os
import pickle
from datetime import datetime

# Download NLTK data if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

class SentimentAnalyzer:
    """
    A class for performing sentiment analysis on social media text data using
    both rule-based (VADER) and transformer-based (RoBERTa) approaches.
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the sentiment analyzer with multiple models
        
        Parameters:
        -----------
        use_gpu : bool
            Whether to use GPU acceleration if available
        """
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize RoBERTa
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.roberta_model.to(self.device)
        
        # Labels for RoBERTa model
        self.roberta_labels = {0: "negative", 1: "neutral", 2: "positive"}
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis
        
        Parameters:
        -----------
        text : str
            Input text to preprocess
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
        
        # Replace user mentions
        text = re.sub(r'@\w+', ' USER ', text)
        
        # Replace emojis with text representation
        text = emoji.demojize(text)
        text = re.sub(r':[a-z_]+:', ' \g<0> ', text)
        
        # Replace hashtags
        text = re.sub(r'#(\w+)', r' \1 ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            VADER sentiment scores
        """
        preprocessed_text = self.preprocess_text(text)
        return self.vader.polarity_scores(preprocessed_text)
    
    def analyze_roberta(self, text, batch_size=32):
        """
        Analyze sentiment using RoBERTa
        
        Parameters:
        -----------
        text : str or list
            Text(s) to analyze
        batch_size : int
            Batch size for processing multiple texts
            
        Returns:
        --------
        dict or list of dicts
            RoBERTa sentiment predictions
        """
        # Handle single text or list of texts
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        preprocessed_texts = [self.preprocess_text(t) for t in texts]
        
        results = []
        
        # Process in batches
        for i in range(0, len(preprocessed_texts), batch_size):
            batch_texts = preprocessed_texts[i:i+batch_size]
            
            # Tokenize
            encoded_inputs = self.roberta_tokenizer(batch_texts, padding=True, truncation=True, 
                                                 max_length=128, return_tensors="pt")
            encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.roberta_model(**encoded_inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert predictions to dictionaries
            for pred in predictions:
                pred_dict = {
                    "roberta_negative": pred[0].item(),
                    "roberta_neutral": pred[1].item(),
                    "roberta_positive": pred[2].item(),
                    "roberta_sentiment": self.roberta_labels[torch.argmax(pred).item()],
                    "roberta_score": (pred[2].item() - pred[0].item())  # -1 to 1 scale
                }
                results.append(pred_dict)
        
        return results[0] if is_single else results
    
    def analyze_text(self, text):
        """
        Analyze text using multiple sentiment models
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Combined sentiment analysis results
        """
        vader_results = self.analyze_vader(text)
        roberta_results = self.analyze_roberta(text)
        
        # Combine results
        results = {
            "text": text,
            "processed_text": self.preprocess_text(text),
            "vader_negative": vader_results["neg"],
            "vader_neutral": vader_results["neu"],
            "vader_positive": vader_results["pos"],
            "vader_compound": vader_results["compound"],
            **roberta_results,
            "consensus_score": (vader_results["compound"] + roberta_results["roberta_score"]) / 2
        }
        
        # Determine consensus sentiment label
        if results["consensus_score"] >= 0.05:
            results["consensus_sentiment"] = "positive"
        elif results["consensus_score"] <= -0.05:
            results["consensus_sentiment"] = "negative"
        else:
            results["consensus_sentiment"] = "neutral"
            
        return results
    
    def analyze_dataframe(self, df, text_column, output_prefix=None, batch_size=32):
        """
        Analyze sentiment for all texts in a dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing text data
        text_column : str
            Column name with text to analyze
        output_prefix : str, optional
            Prefix for output file
        batch_size : int
            Batch size for processing
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment results
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
            
        texts = df[text_column].fillna("").tolist()
        print(f"Analyzing sentiment for {len(texts)} texts...")
        
        # Get VADER results for all texts
        vader_results = [self.analyze_vader(text) for text in tqdm(texts, desc="VADER Analysis")]
        
        # Get RoBERTa results in batches
        roberta_results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa Analysis"):
            batch_texts = texts[i:i+batch_size]
            batch_results = self.analyze_roberta(batch_texts, batch_size=batch_size)
            roberta_results.extend(batch_results)
        
        # Prepare new dataframe with all results
        results_df = pd.DataFrame({
            "processed_text": [self.preprocess_text(text) for text in texts],
            "vader_negative": [r["neg"] for r in vader_results],
            "vader_neutral": [r["neu"] for r in vader_results],
            "vader_positive": [r["pos"] for r in vader_results],
            "vader_compound": [r["compound"] for r in vader_results],
            "roberta_negative": [r["roberta_negative"] for r in roberta_results],
            "roberta_neutral": [r["roberta_neutral"] for r in roberta_results],
            "roberta_positive": [r["roberta_positive"] for r in roberta_results],
            "roberta_sentiment": [r["roberta_sentiment"] for r in roberta_results],
            "roberta_score": [r["roberta_score"] for r in roberta_results]
        })
        
        # Calculate consensus scores
        results_df["consensus_score"] = (results_df["vader_compound"] + results_df["roberta_score"]) / 2
        
        # Determine consensus sentiment labels
        conditions = [
            (results_df["consensus_score"] >= 0.05),
            (results_df["consensus_score"] <= -0.05)
        ]
        choices = ["positive", "negative"]
        results_df["consensus_sentiment"] = np.select(conditions, choices, default="neutral")
        
        # Merge with original dataframe
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Save results if output_prefix is provided
        if output_prefix:
            output_dir = os.path.join("../results")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(output_dir, f"{output_prefix}_sentiment_{timestamp}.csv")
            final_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return final_df


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer(use_gpu=True)
    
    # Single text analysis
    text = "I'm really excited about the new climate initiatives! Hope they make a difference. #ClimateAction"
    result = analyzer.analyze_text(text)
    print("\nSample Text Analysis:")
    print(f"Text: {text}")
    print(f"VADER Score: {result['vader_compound']:.3f}")
    print(f"RoBERTa Score: {result['roberta_score']:.3f}")
    print(f"Consensus: {result['consensus_sentiment']} ({result['consensus_score']:.3f})")
    
    # Dataframe analysis
    try:
        sample_df = pd.read_csv("../data/raw/sample_tweets.csv")
        if len(sample_df) > 0:
            print("\nAnalyzing sample dataframe...")
            results_df = analyzer.analyze_dataframe(sample_df, "text", output_prefix="sample")
            print(f"Analysis complete for {len(results_df)} records")
    except Exception as e:
        print(f"Could not process sample data: {e}") 