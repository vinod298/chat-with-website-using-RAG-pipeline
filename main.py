import os
import logging
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse

import google.generativeai as genai

from website_db import WebsiteDataExtractor, WebDataVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WebsiteSearchApp:
    def __init__(self):
        """
        Initialize the Flask application with configuration
        """
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configure Gemini API
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise
        
        # Initialize class variables
        self.url: str = ""
        self.collection_name: str = ""
        self.website_data: Dict[str, Any] = {}
        self.processed_text: List[str] = []
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """
        Define application routes
        """
        self.app.route('/')(self.index)
        self.app.route('/extract', methods=['POST'])(self.extract_data)
        self.app.route('/query', methods=['POST'])(self.process_query)
    
    def index(self):
        """
        Render the main index page
        """
        return render_template("index.html")
    
    def extract_website(self, url: str) -> Dict[str, Any]:
        """
        Extract website data
        
        Args:
            url (str): Website URL to extract
        
        Returns:
            Dict: Extracted website data
        """
        try:
            extractor = WebsiteDataExtractor(
                max_depth=2,
                max_pages=10,
                delay=1.0
            )
            
            # Extract website data
            website_data = extractor.extract_website_data(url)
            
            # Create a collection name based on domain
            domain = urlparse(url).netloc.replace('.', '_')
            self.collection_name = f"website_{domain}"
            
            return website_data
        
        except Exception as e:
            logger.error(f"Website extraction failed: {e}")
            raise
    
    def vectorize_data(self, 
                       website_data: Dict[str, Any], 
                       query: str, 
                       n_results: int = 50) -> List[str]:
        """
        Vectorize website data and perform similarity search
        
        Args:
            website_data (Dict): Extracted website data
            query (str): Search query
            n_results (int): Number of search results
        
        Returns:
            List[str]: Search results
        """
        try:
            vectorizer = WebDataVectorizer()
            
            # Check if data is already processed
            if not self.processed_text:
                self.processed_text = vectorizer.vectorize_and_store(
                    website_data, 
                    self.collection_name
                )
                logger.info(f"Vectorized {len(self.processed_text)} texts")
            
            # Perform similarity search
            collection = vectorizer.chroma_client.get_collection(
                name=self.collection_name
            )
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = [
                f"{i+1}. {doc}" 
                for i, doc in enumerate(results['documents'][0])
            ]
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            raise
    
    def generate_llm_response(self, 
                               query: str, 
                               data: List[str]) -> str:
        """
        Generate LLM response using Gemini
        
        Args:
            query (str): User query
            data (List[str]): Retrieved search results
        
        Returns:
            str: LLM generated response
        """
        try:
            system_template = f"""
            You are an expert in summarizing website contents.
            Website URL: {self.url}
            User Query: {query}
            Related Documents: {' '.join(data)}
            
            Provide a concise and informative summary addressing the user's query.
            """
            
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(system_template)
            
            return response.text
        
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            raise
    
    def extract_data(self):
        """
        Handle website data extraction endpoint
        """
        try:
            self.url = request.json.get('url')
            
            if not self.url:
                return jsonify({'error': 'URL is required'}), 400
            
            logger.info(f"Extracting data from URL: {self.url}")
            self.website_data = self.extract_website(self.url)
            
            return jsonify({'status': 'ok'})
        
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def process_query(self):
        """
        Handle query processing endpoint
        """
        try:
            query = request.json.get('query')
            
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            logger.info(f"Processing query: {query}")
            
            # Vectorize and search
            search_results = self.vectorize_data(
                self.website_data, 
                query, 
                n_results=50
            )
            
            # Generate LLM response
            response = self.generate_llm_response(query, search_results)
            
            return jsonify({'response': response})
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def run(self, debug: bool = True):
        """
        Run the Flask application
        
        Args:
            debug (bool): Enable debug mode
        """
        self.app.run(debug=debug)

def create_app():
    """
    Application factory function
    """
    app_instance = WebsiteSearchApp()
    return app_instance.app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)