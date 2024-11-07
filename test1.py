import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from difflib import SequenceMatcher
import re
from datetime import datetime
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import hashlib
import random

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide"
)

# Constants from environment variables
SERP_API_KEY = os.getenv("SERP_API_KEY")
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.1"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MIN_MATCH_LENGTH = int(os.getenv("MIN_MATCH_LENGTH", "40"))

# Initialize session states
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'history' not in st.session_state:
    st.session_state.history = []

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")

download_nltk_data()

class Cache:
    @staticmethod
    def get_cache_key(text: str) -> str:
        """Generate a cache key for the given text."""
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def get_cached_result(text: str) -> Optional[Dict[str, Any]]:
        """Get cached result for the given text."""
        cache_key = Cache.get_cache_key(text)
        return st.session_state.cache.get(cache_key)

    @staticmethod
    def set_cached_result(text: str, result: Dict[str, Any]):
        """Cache the result for the given text."""
        cache_key = Cache.get_cache_key(text)
        st.session_state.cache[cache_key] = result

    @staticmethod
    def clear_cache():
        """Clear the cache"""
        st.session_state.cache = {}

class WebScraper:
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]

    def __init__(self):
        self.session = requests.Session()

    def get_webpage_content(self, url: str) -> str:
        """Extract text content from a webpage."""
        try:
            headers = {
                'User-Agent': random.choice(self.USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
                element.decompose()
            
            # Get main content
            text = soup.get_text()
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
        except Exception as e:
            st.warning(f"Error fetching content from {url}: {str(e)}")
            return ""

class PlagiarismDetector:
    def __init__(self):
        self.sentence_model = self.initialize_model()
        self.web_scraper = WebScraper()

    @st.cache_resource
    def initialize_model(_self):
        """Initialize the MPNet model."""
        try:
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            st.session_state.model_loaded = True
            return model
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            raise e

    def search_web(self, text: str) -> List[Dict[str, Any]]:
        """Search for potential matches using SERP API."""
        try:
            if not SERP_API_KEY:
                st.error("SERP API key not configured.")
                return []

            # Split text into segments
            segments = sent_tokenize(text)
            results = []
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, segment in enumerate(segments[:3]):  # Limit to first 3 segments
                status_text.text(f"Searching segment {i+1} of {min(len(segments), 3)}...")
                
                params = {
                    "q": f'"{segment}"',  # Exact match search
                    "api_key": SERP_API_KEY,
                    "engine": "google",
                    "num": MAX_SEARCH_RESULTS,
                    "gl": "us",  # Set region to US for broader results
                }
                
                search = GoogleSearch(params)
                search_results = search.get_dict()
                
                if "organic_results" in search_results:
                    results.extend([
                        {
                            "title": result.get("title", ""),
                            "link": result.get("link", ""),
                            "snippet": result.get("snippet", "")
                        }
                        for result in search_results["organic_results"]
                    ])
                
                progress_bar.progress((i + 1) / min(len(segments), 3))
            
            progress_bar.empty()
            status_text.empty()
            
            return self._deduplicate_results(results)
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('link')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results

    def check_plagiarism(self, text: str) -> Dict[str, Any]:
        """Check text for plagiarism."""
        try:
            # Check cache first
            cache_result = Cache.get_cached_result(text)
            if cache_result:
                return cache_result

            results = {
                'original_text': text,
                'matches': [],
                'sources': [],
                'similarity_scores': [],
                'overall_similarity': 0.0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Search for potential matches
            search_results = self.search_web(text)
            
            if search_results:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, result in enumerate(search_results):
                    status_text.text(f"Analyzing source {i+1} of {len(search_results)}...")
                    
                    content = self.web_scraper.get_webpage_content(result['link'])
                    if not content:
                        continue
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(text, content)
                    
                    if similarity > MIN_SIMILARITY_THRESHOLD:
                        # Find matching segments
                        matches = self.find_matching_segments(text, content)
                        
                        if matches:
                            results['matches'].extend(matches)
                            results['sources'].append({
                                'url': result['link'],
                                'title': result['title'],
                                'similarity': similarity,
                                'match_count': len(matches)
                            })
                            results['similarity_scores'].append(similarity)
                    
                    progress_bar.progress((i + 1) / len(search_results))
                
                progress_bar.empty()
                status_text.empty()
            
            if results['similarity_scores']:
                results['overall_similarity'] = max(results['similarity_scores'])
            
            # Cache results
            Cache.set_cached_result(text, results)
            
            return results
            
        except Exception as e:
            st.error(f"Error checking plagiarism: {str(e)}")
            return {
                'original_text': text,
                'matches': [],
                'sources': [],
                'similarity_scores': [],
                'overall_similarity': 0.0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using MPNet embeddings."""
        try:
            # Generate embeddings
            embedding1 = self.sentence_model.encode(text1, convert_to_tensor=True)
            embedding2 = self.sentence_model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            st.warning(f"Error calculating similarity: {str(e)}")
            return 0.0

    def find_matching_segments(self, text1: str, text2: str) -> List[Dict[str, Any]]:
        """Find matching text segments."""
        matches = []
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        for i, sent1 in enumerate(sentences1):
            embeddings1 = self.sentence_model.encode([sent1])[0]
            
            for sent2 in sentences2:
                embeddings2 = self.sentence_model.encode([sent2])[0]
                similarity = util.pytorch_cos_sim(
                    torch.tensor(embeddings1).unsqueeze(0),
                    torch.tensor(embeddings2).unsqueeze(0)
                ).item()
                
                if similarity > MIN_SIMILARITY_THRESHOLD:
                    matches.append({
                        'original': sent1,
                        'match': sent2,
                        'similarity': similarity
                    })
                    break  # Move to next original sentence after finding a match
        
        return matches

class PlagiarismUI:
    def __init__(self):
        if not st.session_state.model_loaded:
            with st.spinner("Loading MPNet model... This may take a few minutes..."):
                self.detector = PlagiarismDetector()
        else:
            self.detector = PlagiarismDetector()

    def render(self):
        """Render the main UI."""
        st.title("🔍 MPNet Plagiarism Detector")
        st.caption("Powered by MPNet and SERP API")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            text = st.text_area(
                "Enter text to check for plagiarism:",
                height=200,
                help="Paste or type your text here",
                placeholder="Enter your text here..."
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 4, 1])
            with col_btn1:
                check_button = st.button("Check", type="primary")
            with col_btn2:
                if text:
                    st.write(f"Text length: {len(text)} characters")
            with col_btn3:
                if st.button("Clear"):
                    text = ""
                    st.rerun()
        
        with col2:
            st.markdown("### Recent Checks")
            if st.session_state.history:
                for item in reversed(st.session_state.history[-5:]):
                    with st.expander(f"{item['timestamp']} ({item['similarity']:.1f}% match)"):
                        st.write(f"Sources found: {item['sources']}")
            else:
                st.info("No recent checks")
        
        if check_button and text:
            if not text.strip():
                st.warning("Please enter some text to check.")
                return
                
            with st.spinner("Analyzing text for potential matches..."):
                self.display_results(text)

    def render_sidebar(self):
        """Render the sidebar content."""
        with st.sidebar:
            st.header("About")
            st.markdown("""
            ### Advanced Plagiarism Detector
            
            This tool uses state-of-the-art technology:
            - MPNet for semantic analysis
            - SERP API for web search
            - Neural similarity matching
            
            #### Features:
            - Semantic similarity detection
            - Web content analysis
            - Source attribution
            - Match highlighting
            
            #### How it works:
            1. Text analysis
            2. Web search
            3. Content comparison
            4. Similarity scoring
            """)
            
            if st.button("Clear History"):
                st.session_state.history = []
                Cache.clear_cache()
                st.rerun()

    def display_results(self, text: str):
        """Display plagiarism check results."""
        results = self.detector.check_plagiarism(text)
        
        # Update history
        st.session_state.history.append({
            'timestamp': results['timestamp'],
            'similarity': results['overall_similarity'] * 100,
            'sources': len(results['sources'])
        })
        
        # Display overall similarity
        st.header("Plagiarism Analysis")
        similarity_percentage = results['overall_similarity'] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Maximum Similarity", f"{similarity_percentage:.1f}%")
        with col2:
            if similarity_percentage >= 80:
                st.error("❌ High similarity detected!")
            elif similarity_percentage >= 40:
                st.warning("⚠️ Moderate similarity detected")
            else:
                st.success("✅ Low similarity detected")
        
        # Display sources
        if results['sources']:
            st.header("Sources Found")
            for source in sorted(results['sources'], key=lambda x: x['similarity'], reverse=True):
                with st.expander(
                    f"Source: {source['title']} ({source['similarity']*100:.1f}% similar)"
                ):
                    st.write(f"URL: {source['url']}")
                    st.write(f"Number of matches: {source['match_count']}")

        # Display matching segments
        if results['matches']:
            st.header("Matching Segments")
            df = pd.DataFrame(results['matches'])
            df = df.rename(columns={
                'original': 'Original Text',
                'match': 'Matched Text',
                'similarity': 'Similarity'
            })
            df['Similarity'] = df['Similarity'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True
            )

        # Display highlighted text
        if results['matches']:
            st.header("Original Text with Highlights")
            highlighted_text = self.highlight_matches(
                results['original_text'],
                results['matches']
            )
            st.markdown(
                f'<div style="border:1px solid #ccc; padding:10px; '
                f'border-radius:5px;">{highlighted_text}</div>',
                unsafe_allow_html=True
            )

        # Download report button
        if results['sources'] or results['matches']:
            report = self.generate_report(results)
            st.download_button(
                "Download Detailed Report",
                report,
                file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    def highlight_matches(self, text: str, matches: List[Dict[str, Any]]) -> str:
        """Highlight matching segments in the original text."""
        # Sort matches by length (longest first) to avoid nested highlights
        sorted_matches = sorted(matches, key=lambda x: len(x['original']), reverse=True)
        
        # Replace matching segments with highlighted versions
        highlighted_text = text
        for match in sorted_matches:
            original = match['original']
            color = self._get_highlight_color(match['similarity'])
            highlight = f'<span style="background-color: {color}">{original}</span>'
            highlighted_text = highlighted_text.replace(original, highlight)
        
        return highlighted_text

    def _get_highlight_color(self, similarity: float) -> str:
        """Get highlight color based on similarity score."""
        if similarity >= 0.8:
            return "#ffcccb"  # Red
        elif similarity >= 0.6:
            return "#ffddcc"  # Orange
        else:
            return "#ffffcc"  # Yellow

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed plagiarism report."""
        report = [
            "Plagiarism Detection Report",
            "=========================",
            f"Date: {results['timestamp']}",
            f"Overall Similarity: {results['overall_similarity']*100:.1f}%",
            "\nSources Found",
            "-------------"
        ]
        
        for source in sorted(results['sources'], key=lambda x: x['similarity'], reverse=True):
            report.extend([
                f"\nURL: {source['url']}",
                f"Title: {source['title']}",
                f"Similarity: {source['similarity']*100:.1f}%",
                f"Matches: {source['match_count']}"
            ])
        
        if results['matches']:
            report.extend([
                "\nMatching Segments",
                "----------------"
            ])
            
            for match in results['matches']:
                report.extend([
                    f"\nOriginal: {match['original']}",
                    f"Matched: {match['match']}",
                    f"Similarity: {match['similarity']*100:.1f}%"
                ])
        
        return "\n".join(report)

def main():
    try:
        # Custom CSS for better styling
        st.markdown("""
            <style>
            .stProgress > div > div > div > div {
                background-color: #1f77b4;
            }
            .stAlert > div {
                padding-top: 10px;
                padding-bottom: 10px;
            }
            .st-emotion-cache-16idsys p {
                font-size: 16px;
            }
            span.highlight {
                padding: 1px 4px;
                border-radius: 3px;
            }
            </style>
            """, unsafe_allow_html=True)

        # Check for API key
        if not SERP_API_KEY:
            st.error("""
                ⚠️ SERP API key not configured. Please add your API key to the .env file:
                SERP_API_KEY=your_api_key_here
            """)
            st.stop()

        # Initialize and render UI
        ui = PlagiarismUI()
        ui.render()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()