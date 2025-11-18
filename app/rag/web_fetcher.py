"""
Web fetching utility for medical knowledge sources.
Implements the fetch_website tool from the paper.
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import time


class MedicalWebFetcher:
    """Fetches and processes medical information from authorized websites."""

    # Authorized medical sources per expert
    MAYO_CLINIC_DOMAINS = ["mayoclinic.org", "www.mayoclinic.org"]
    WEBMD_DOMAINS = ["webmd.com", "www.webmd.com"]

    MAX_CONTENT_LENGTH = 2000  # As per paper
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Medical Research Bot)'
        })

    def fetch_website(self, url: str, source_type: str) -> Dict[str, str]:
        """
        Fetch and process web content from authorized medical sources.

        Args:
            url: The URL to fetch
            source_type: Either 'mayo' or 'webmd'

        Returns:
            Dictionary with 'content' and 'url' keys
        """
        # Validate source
        if source_type == 'mayo':
            allowed_domains = self.MAYO_CLINIC_DOMAINS
        elif source_type == 'webmd':
            allowed_domains = self.WEBMD_DOMAINS
        else:
            return {"content": "", "url": "", "error": "Invalid source type"}

        # Check if URL is from allowed domain
        if not any(domain in url.lower() for domain in allowed_domains):
            return {
                "content": "",
                "url": url,
                "error": f"URL not from authorized {source_type} domain"
            }

        try:
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Limit content length (as per paper)
            if len(text) > self.MAX_CONTENT_LENGTH:
                text = text[:self.MAX_CONTENT_LENGTH] + "..."

            return {
                "content": text,
                "url": url,
                "length": len(text)
            }

        except requests.RequestException as e:
            return {
                "content": "",
                "url": url,
                "error": f"Failed to fetch: {str(e)}"
            }

    def search_and_fetch(self, query: str, source_type: str, num_results: int = 1) -> list:
        """
        Search for medical information and fetch top results.

        Note: This is a simplified version. In production, you would use
        a proper search API or pre-indexed content.

        Args:
            query: Search query (e.g., "aspirin side effects")
            source_type: Either 'mayo' or 'webmd'
            num_results: Number of results to fetch

        Returns:
            List of fetched content dictionaries
        """
        # Construct search URLs (simplified approach)
        if source_type == 'mayo':
            # Mayo Clinic search pattern
            base_url = "https://www.mayoclinic.org"
            search_url = f"{base_url}/search/search-results?q={query.replace(' ', '+')}"
        elif source_type == 'webmd':
            # WebMD search pattern
            base_url = "https://www.webmd.com"
            search_url = f"{base_url}/search/search_results_default.aspx?query={query.replace(' ', '+')}"
        else:
            return []

        results = []

        # Note: In a real implementation, you would:
        # 1. Use the search API to get result URLs
        # 2. Fetch each result URL
        # 3. Process and return the content

        # For now, return a placeholder indicating the tool is ready
        # Users should provide direct URLs to medical content
        results.append({
            "content": f"Search tool ready for {source_type}. Please provide direct URLs to medical articles.",
            "url": search_url,
            "note": "Direct URL fetching is recommended for accurate content retrieval"
        })

        return results


# Global instance
_fetcher = None

def get_fetcher() -> MedicalWebFetcher:
    """Get or create the global web fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = MedicalWebFetcher()
    return _fetcher
