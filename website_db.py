import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import Dict, List, Set, Union
import chromadb
from sentence_transformers import SentenceTransformer

class WebsiteDataExtractor:
    def __init__(self, max_depth: int = 2, max_pages: int = 10, delay: float = 1.0):
        """
        Initialize the web scraping extractor
        
        Args:
            max_depth (int): Maximum depth of link traversal
            max_pages (int): Maximum number of pages to scrape
            delay (float): Delay between requests to avoid overwhelming servers
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.extracted_data: Dict[str, Dict] = {}
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """
        Check if the URL is valid and within the same domain
        
        Args:
            url (str): URL to validate
            base_domain (str): Base domain to compare against
        
        Returns:
            bool: Whether the URL is valid and within the same domain
        """
        try:
            parsed_url = urlparse(url)
            return (
                parsed_url.scheme in ['http', 'https'] and
                parsed_url.netloc == base_domain and
                url not in self.visited_urls
            )
        except Exception:
            return False
    
    def extract_page_data(self, url: str) -> Dict:
        """
        Extract data from a single webpage
        
        Args:
            url (str): URL of the webpage to scrape
        
        Returns:
            dict: Extracted webpage data
        """
        try:
            # Send a GET request to the website
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract metadata
            metadata = {}
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name', tag.get('property', 'unnamed'))
                content = tag.get('content', '')
                if name and content:
                    metadata[name] = content
            
            # Extract title
            metadata['title'] = soup.title.string if soup.title else 'No Title'
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text_content = {
                'paragraphs': [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)],
                'headings': {
                    'h1': [h.get_text(strip=True) for h in soup.find_all('h1')],
                    'h2': [h.get_text(strip=True) for h in soup.find_all('h2')],
                    'h3': [h.get_text(strip=True) for h in soup.find_all('h3')]
                }
            }
            
            # Extract links
            links = [
                urljoin(url, link.get('href', ''))
                for link in soup.find_all('a')
                if link.get('href')
            ]
            
            # Extract images
            images = [
                {
                    'src': urljoin(url, img.get('src', '')),
                    'alt': img.get('alt', '')
                } 
                for img in soup.find_all('img')
            ]
            
            return {
                'url': url,
                'metadata': metadata,
                'text_content': text_content,
                'links': links,
                'images': images
            }
        
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def recursive_crawl(self, start_url: str, current_depth: int = 0) -> None:
        """
        Recursively crawl and extract data from links
        
        Args:
            start_url (str): Starting URL to crawl
            current_depth (int): Current depth of crawling
        """
        # Check depth and page limits
        if (current_depth > self.max_depth or 
            len(self.visited_urls) >= self.max_pages):
            return
        
        # Parse base domain
        base_domain = urlparse(start_url).netloc
        
        # Skip if already visited
        if start_url in self.visited_urls:
            return
        
        # Mark as visited
        self.visited_urls.add(start_url)
        
        # Extract page data
        page_data = self.extract_page_data(start_url)
        if not page_data:
            return
        
        # Store extracted data
        self.extracted_data[start_url] = page_data
        
        # Recursive link extraction
        for link in page_data['links']:
            # Validate and explore links
            if (self.is_valid_url(link, base_domain) and 
                link not in self.visited_urls):
                # Add delay to be respectful of server resources
                time.sleep(self.delay)
                
                # Recursively crawl
                self.recursive_crawl(link, current_depth + 1)
    
    def extract_website_data(self, start_url: str) -> Dict:
        """
        Main method to start website data extraction
        
        Args:
            start_url (str): URL to start crawling from
        
        Returns:
            dict: Comprehensive extracted website data
        """
        # Reset data before starting
        self.visited_urls.clear()
        self.extracted_data.clear()
        
        # Start recursive crawling
        self.recursive_crawl(start_url)
        
        return {
            'total_pages_scraped': len(self.extracted_data),
            'extracted_data': self.extracted_data
        }

class WebDataVectorizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vectorization pipeline
        
        Args:
            model_name (str): Sentence transformer model for embedding generation
        """
        # Load pre-trained sentence transformer model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    
    def preprocess_text(self, text: Union[str, List[str]]) -> List[str]:
        """
        Preprocess text for vectorization
        
        Args:
            text (str or List[str]): Text to preprocess
        
        Returns:
            List[str]: Cleaned and processed text
        """
        if isinstance(text, str):
            text = [text]
        
        # Basic text cleaning
        cleaned_text = [
            ' '.join(t.lower().split())  # Normalize whitespace
            for t in text
            if t and len(t) > 10  # Filter out very short texts
        ]
        
        return cleaned_text
    
    def extract_vectorizable_content(self, extracted_data: Dict) -> List[str]:
        """
        Extract and organize vectorizable content from scraped website data
        
        Args:
            extracted_data (Dict): Extracted website data
        
        Returns:
            List[str]: Vectorizable content texts
        """
        vectorizable_content = []
        
        # Iterate through extracted pages
        for url, page_data in extracted_data.get('extracted_data', {}).items():
            # Extract metadata text
            for key, value in page_data.get('metadata', {}).items():
                if isinstance(value, str):
                    vectorizable_content.append(f"{key}: {value}")
            
            # Extract paragraphs
            paragraphs = page_data.get('text_content', {}).get('paragraphs', [])
            vectorizable_content.extend(paragraphs)
            
            # Extract headings
            for heading_level, headings in page_data.get('text_content', {}).get('headings', {}).items():
                vectorizable_content.extend(headings)
            
            # Extract link texts
            links = page_data.get('links', [])
            vectorizable_content.extend(links)
        
        return vectorizable_content
    
    def vectorize_and_store(self, extracted_data: Dict, collection_name: str):
        """
        Vectorize website data and store in ChromaDB
        
        Args:
            extracted_data (Dict): Extracted website data
            collection_name (str): Name of ChromaDB collection
        
        Returns:
            List[str]: Processed texts
        """
        # Extract vectorizable content
        vectorizable_content = self.extract_vectorizable_content(extracted_data)
        
        # Preprocess texts
        processed_texts = self.preprocess_text(vectorizable_content)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(processed_texts)
        
        # Create or get collection
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # Add documents to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=processed_texts,
            ids=[f"id_{i}" for i in range(len(processed_texts))]
        )
        
        return processed_texts

def main():
    # Infinite loop for interactive search
    while True:
        # Get website URL to crawl
        start_url = input("Enter a website URL to crawl (or 'quit' to exit): ").strip()
        
        if start_url.lower() == 'quit':
            break
        
        try:
            # Create extractor and vectorizer
            extractor = WebsiteDataExtractor(
                max_depth=2,  # How deep to go into links
                max_pages=10,  # Maximum number of pages to scrape
                delay=1.0     # Delay between requests
            )
            vectorizer = WebDataVectorizer()
            
            # Extract website data
            website_data = extractor.extract_website_data(start_url)
            
            # Create a collection name based on domain
            domain = urlparse(start_url).netloc.replace('.', '_')
            collection_name = f"website_{domain}"
            
            # Vectorize and store data
            processed_texts = vectorizer.vectorize_and_store(website_data, collection_name)
            print(f"Vectorized {len(processed_texts)} texts from {start_url}")
            
            # Search loop
            while True:
                # Get search query
                query = input("Enter search query (or 'back' to choose another website): ").strip()
                
                if query.lower() == 'back':
                    break
                
                # Perform similarity search
                collection = vectorizer.chroma_client.get_collection(name=collection_name)
                results = collection.query(
                    query_texts=[query],
                    n_results=n
                )
                
                # Print results
                print("\nSearch Results:")
                for i, doc in enumerate(results['documents'][0], 1):
                    print(f"{i}. {doc}")
        
        except Exception as e:
            print(f"An error occurred: {e}")
def search_in_database(url,query,n):
    # Infinite loop for interactive search
        # Get website URL to crawl
    
    try:
        # Create extractor and vectorizer
        extractor = WebsiteDataExtractor(
            max_depth=2,  # How deep to go into links
            max_pages=10,  # Maximum number of pages to scrape
            delay=1.0     # Delay between requests
        )
        
        # Extract website data
        website_data = extractor.extract_website_data(url)
        
        # Create a collection name based on domain
        domain = urlparse(url).netloc.replace('.', '_')
        collection_name = f"website_{domain}"
        
        vectorizer = WebDataVectorizer()
        # Vectorize and store data
        processed_texts = vectorizer.vectorize_and_store(website_data, collection_name)
        print(f"Vectorized {len(processed_texts)} texts from {url}")
        
        # Search loop
        
            # Get search query
            
            # Perform similarity search
        collection = vectorizer.chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=50
        )
        all_results=[]
        # Print results
        print("\nSearch Results:")
        for i, doc in enumerate(results['documents'][0], 1):
            print(f"{i}. {doc}\n")
            all_results.append(f"{i}. {doc}\n")
        return all_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# if __name__ == '__main__':
#     main()