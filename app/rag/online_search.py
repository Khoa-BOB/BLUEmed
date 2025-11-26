"""
Online search capability for medical information from Mayo Clinic and WebMD.

Uses direct web crawling instead of DuckDuckGo to avoid rate limiting.
Crawls Mayo Clinic and WebMD search results directly.
"""
from typing import List
from langchain_core.documents import Document
from app.rag.direct_crawler import get_direct_crawler


class OnlineMedicalSearch:
    """
    Performs online searches for medical information from authorized sources.

    Uses direct web crawling of Mayo Clinic and WebMD search pages.
    No third-party search engines required - no rate limiting!
    """

    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Initialize online search with direct crawler.

        Args:
            max_results: Maximum number of search results to fetch
            timeout: Timeout for web requests in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.crawler = get_direct_crawler(timeout=timeout)

    def search_mayo_clinic(self, query: str, k: int = 3) -> List[Document]:
        """
        Search Mayo Clinic website directly for medical information.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of Documents with fetched content
        """
        return self.crawler.search_mayo_clinic(query, k)

    def search_webmd(self, query: str, k: int = 3) -> List[Document]:
        """
        Search WebMD website directly for medical information.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of Documents with fetched content
        """
        return self.crawler.search_webmd(query, k)

    def search_for_expert(self, query: str, expert: str, k: int = 3) -> List[Document]:
        """
        Search online for a specific expert's knowledge source.

        Args:
            query: Search query
            expert: "A" for Mayo Clinic, "B" for WebMD
            k: Number of results to return

        Returns:
            List of Documents from online search
        """
        if expert == "A":
            return self.search_mayo_clinic(query, k)
        elif expert == "B":
            return self.search_webmd(query, k)
        else:
            raise ValueError(f"Unknown expert: {expert}")


# Global online search instance
_online_search = None


def get_online_search(
    max_results: int = 5,
    timeout: int = 10
) -> OnlineMedicalSearch:
    """
    Get or create global online search instance.

    Args:
        max_results: Maximum number of search results per query
        timeout: Timeout for web requests

    Returns:
        OnlineMedicalSearch instance
    """
    global _online_search
    if _online_search is None:
        _online_search = OnlineMedicalSearch(
            max_results=max_results,
            timeout=timeout
        )
    return _online_search
