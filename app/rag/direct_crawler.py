"""
Direct web crawler for Mayo Clinic and WebMD.
Replaces DuckDuckGo to avoid rate limiting.

This crawler:
1. Uses site's native search functionality
2. Parses search result pages directly
3. Extracts article URLs
4. Fetches content from those URLs
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from langchain_core.documents import Document
from app.rag.web_fetcher import MedicalWebFetcher
import time
import re
from urllib.parse import urljoin, quote_plus


class DirectMedicalCrawler:
    """
    Direct crawler for Mayo Clinic and WebMD.
    No third-party search engines required!
    """

    def __init__(self, timeout: int = 10):
        """
        Initialize direct crawler.

        Args:
            timeout: Timeout for web requests in seconds
        """
        self.timeout = timeout
        self.web_fetcher = MedicalWebFetcher()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def search_mayo_clinic(self, query: str, k: int = 3) -> List[Document]:
        """
        Search Mayo Clinic directly and fetch articles.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of Documents with content
        """
        try:
            # Mayo Clinic search URL
            search_url = f"https://www.mayoclinic.org/search/search-results?q={quote_plus(query)}"

            print(f"  🔍 Searching Mayo Clinic: {search_url[:80]}...")

            # Fetch search results page
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()

            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract article URLs from search results
            article_urls = []

            # Look for all links on the search results page
            for link in soup.find_all('a', href=True):
                url = link['href']

                # Make absolute URL
                if url.startswith('/'):
                    url = f"https://www.mayoclinic.org{url}"

                # Filter for main content pages only
                if 'mayoclinic.org' in url and url not in article_urls:
                    # Skip unwanted URL patterns
                    skip_patterns = [
                        'search', 'javascript:', '#', 'connect.', 'mcpress.',
                        'newsnetwork.', 'philanthropies.', 'jobs.',
                        'store.', 'giving.', 'login', 'signup',
                        '/about-', '/healthy-lifestyle/recipes',
                        '.pdf', '.jpg', '.png'
                    ]

                    # Only include disease/condition/symptom/treatment pages
                    include_patterns = [
                        '/diseases-conditions/',
                        '/symptoms/',
                        '/tests-procedures/',
                        '/drugs-supplements/',
                        '/healthy-lifestyle/'
                    ]

                    should_skip = any(pattern in url.lower() for pattern in skip_patterns)
                    should_include = any(pattern in url.lower() for pattern in include_patterns)

                    if should_include and not should_skip:
                        article_urls.append(url)
                        if len(article_urls) >= k * 2:
                            break

            print(f"  ✓ Found {len(article_urls)} Mayo Clinic URLs")

            # Fetch content from each URL
            documents = []
            for i, url in enumerate(article_urls[:k]):
                try:
                    fetched = self.web_fetcher.fetch_website(url, 'mayo')

                    if fetched.get('error'):
                        print(f"  ⚠️  Skipped {url[:60]}... (error)")
                        continue

                    content = fetched['content']
                    if not content or len(content) < 100:
                        continue

                    # Extract title from URL or content
                    title = url.split('/')[-1].replace('-', ' ').title()

                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': 'online_crawler',
                            'url': url,
                            'title': title,
                            'expert': 'Mayo Clinic',
                            'query': query,
                            'rank': i + 1
                        }
                    )

                    documents.append(doc)
                    print(f"  ✓ Fetched: {title[:60]}...")

                    # Small delay to be polite
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  ✗ Error fetching {url[:60]}: {e}")
                    continue

            print(f"✓ Successfully fetched {len(documents)} Mayo Clinic documents\n")
            return documents

        except Exception as e:
            print(f"✗ Mayo Clinic search error: {e}")
            return []

    def search_webmd(self, query: str, k: int = 3) -> List[Document]:
        """
        Search WebMD directly and fetch articles.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of Documents with content
        """
        try:
            # WebMD updated search URL format
            search_url = f"https://www.webmd.com/search?query={quote_plus(query)}"

            print(f"  🔍 Searching WebMD: {search_url[:80]}...")

            # Fetch search results page
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()

            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract article URLs from search results
            article_urls = []

            # Look for all links on the search results page
            for link in soup.find_all('a', href=True):
                url = link['href']

                # Make absolute URL
                if url.startswith('/'):
                    url = f"https://www.webmd.com{url}"

                # Filter for main content pages only
                if 'webmd.com' in url and url not in article_urls:
                    # Skip unwanted URL patterns
                    skip_patterns = [
                        'search', 'javascript:', '#', 'login', 'signup',
                        'directory', 'newsletter', 'subscribe',
                        'apps.', 'img.', 'ad.', 'privacy',
                        '.pdf', '.jpg', '.png'
                    ]

                    # Only include medical content pages
                    include_patterns = [
                        '/a-to-z-guides/',
                        '/diet/',
                        '/drugs/',
                        '/vitamins/',
                        '/fitness-exercise/',
                        '/mental-health/',
                        '/pain-management/',
                        '/cancer/',
                        '/diabetes/',
                        '/heart/',
                        '/lung/',
                        '/digestive-disorders/',
                        '/skin-problems-and-treatments/'
                    ]

                    should_skip = any(pattern in url.lower() for pattern in skip_patterns)
                    should_include = any(pattern in url.lower() for pattern in include_patterns)

                    if should_include and not should_skip:
                        article_urls.append(url)
                        if len(article_urls) >= k * 2:
                            break

            print(f"  ✓ Found {len(article_urls)} WebMD URLs")

            # Fetch content from each URL
            documents = []
            for i, url in enumerate(article_urls[:k]):
                try:
                    fetched = self.web_fetcher.fetch_website(url, 'webmd')

                    if fetched.get('error'):
                        print(f"  ⚠️  Skipped {url[:60]}... (error)")
                        continue

                    content = fetched['content']
                    if not content or len(content) < 100:
                        continue

                    # Extract title from URL or content
                    title = url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()

                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': 'online_crawler',
                            'url': url,
                            'title': title,
                            'expert': 'WebMD',
                            'query': query,
                            'rank': i + 1
                        }
                    )

                    documents.append(doc)
                    print(f"  ✓ Fetched: {title[:60]}...")

                    # Small delay to be polite
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  ✗ Error fetching {url[:60]}: {e}")
                    continue

            print(f"✓ Successfully fetched {len(documents)} WebMD documents\n")
            return documents

        except Exception as e:
            print(f"✗ WebMD search error: {e}")
            return []

    def search_for_expert(self, query: str, expert: str, k: int = 3) -> List[Document]:
        """
        Search for a specific expert's knowledge source.

        Args:
            query: Search query
            expert: "A" for Mayo Clinic, "B" for WebMD
            k: Number of results to return

        Returns:
            List of Documents from direct crawling
        """
        if expert == "A":
            return self.search_mayo_clinic(query, k)
        elif expert == "B":
            return self.search_webmd(query, k)
        else:
            raise ValueError(f"Unknown expert: {expert}")


# Global crawler instance
_crawler = None


def get_direct_crawler(timeout: int = 10) -> DirectMedicalCrawler:
    """
    Get or create global direct crawler instance.

    Args:
        timeout: Timeout for web requests

    Returns:
        DirectMedicalCrawler instance
    """
    global _crawler
    if _crawler is None:
        _crawler = DirectMedicalCrawler(timeout=timeout)
    return _crawler
