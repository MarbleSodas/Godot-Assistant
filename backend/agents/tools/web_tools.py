"""
Web search and documentation tools for the planning agent.

Provides tools for searching documentation and fetching web content.
"""

import re
from typing import Dict
import httpx
from bs4 import BeautifulSoup

from strands import tool


@tool
async def search_documentation(query: str, source: str = "general") -> Dict[str, any]:
    """
    Search for documentation on a specific topic.

    This tool performs a web search to find relevant documentation.
    Use this when you need to find information about libraries, frameworks,
    or programming concepts.

    Args:
        query: Search query (e.g., "Python async/await tutorial")
        source: Documentation source to search (default: "general")
                Options: "general", "godot", "python", "fastapi"

    Returns:
        Dictionary with status and search results
    """
    try:
        # Map sources to specific documentation sites
        source_urls = {
            "godot": "https://docs.godotengine.org",
            "python": "https://docs.python.org",
            "fastapi": "https://fastapi.tiangolo.com",
            "strands": "https://strandsagents.com"
        }

        # Build search query based on source
        if source != "general" and source in source_urls:
            search_query = f"site:{source_urls[source]} {query}"
            result_text = f"Searching {source} documentation for: {query}\n\n"
        else:
            search_query = f"{query} documentation"
            result_text = f"Searching for: {query}\n\n"

        # Note: In a production environment, you would integrate with a real search API
        # (Google Custom Search API, DuckDuckGo API, etc.)
        # For now, we'll provide guidance on where to find information

        result_text += "To find this documentation:\n"
        result_text += f"1. Search for: '{search_query}'\n"

        if source in source_urls:
            result_text += f"2. Visit: {source_urls[source]}\n"

        result_text += "\nRecommended search engines:\n"
        result_text += "- Google: https://www.google.com\n"
        result_text += "- DuckDuckGo: https://duckduckgo.com\n"
        result_text += "- DevDocs: https://devdocs.io\n"

        result_text += "\nNote: For production use, integrate a search API like:\n"
        result_text += "- Google Custom Search API\n"
        result_text += "- Bing Search API\n"
        result_text += "- SerpAPI\n"

        return {
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error searching documentation: {str(e)}"}]
        }


@tool
async def fetch_webpage(url: str, extract_text: bool = True) -> Dict[str, any]:
    """
    Fetch and extract content from a webpage.

    Use this tool to retrieve information from web pages, documentation sites,
    or API references.

    Args:
        url: URL of the webpage to fetch
        extract_text: If True, extract and clean text content (default: True)
                     If False, return raw HTML

    Returns:
        Dictionary with status and webpage content
    """
    try:
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return {
                "status": "error",
                "content": [{"text": "Invalid URL: must start with http:// or https://"}]
            }

        # Fetch webpage
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        content_type = response.headers.get('content-type', '')

        # Check if it's HTML
        if 'text/html' not in content_type:
            return {
                "status": "success",
                "content": [{
                    "text": f"URL: {url}\n\n{response.text[:1000]}..."
                }]
            }

        # Extract text if requested
        if extract_text:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Limit length
            if len(text) > 5000:
                text = text[:5000] + "\n\n... (content truncated)"

            result_text = f"URL: {url}\n"
            result_text += f"Title: {soup.title.string if soup.title else 'No title'}\n\n"
            result_text += text

        else:
            # Return raw HTML (truncated)
            result_text = f"URL: {url}\n\n"
            html = response.text
            if len(html) > 5000:
                html = html[:5000] + "\n\n... (HTML truncated)"
            result_text += html

        return {
            "status": "success",
            "content": [{"text": result_text}]
        }

    except httpx.HTTPStatusError as e:
        return {
            "status": "error",
            "content": [{"text": f"HTTP error {e.response.status_code}: {url}"}]
        }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "content": [{"text": f"Timeout while fetching: {url}"}]
        }
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error fetching webpage: {str(e)}"}]
        }


@tool
async def get_godot_api_reference(class_name: str) -> Dict[str, any]:
    """
    Get API reference for a Godot class.

    Fetches the official Godot API documentation for a specific class.

    Args:
        class_name: Name of the Godot class (e.g., "Node2D", "CharacterBody3D")

    Returns:
        Dictionary with status and API reference information
    """
    try:
        # Construct Godot docs URL
        base_url = "https://docs.godotengine.org/en/stable/classes"
        url = f"{base_url}/class_{class_name.lower()}.html"

        # Fetch the page using fetch_webpage tool
        result = await fetch_webpage(url, extract_text=True)

        if result["status"] == "success":
            content = result["content"][0]["text"]
            # Add context about the class
            formatted = f"Godot API Reference: {class_name}\n"
            formatted += "=" * 50 + "\n\n"
            formatted += content

            return {
                "status": "success",
                "content": [{"text": formatted}]
            }
        else:
            return {
                "status": "error",
                "content": [{
                    "text": f"Could not find documentation for class: {class_name}\n"
                            f"Please check the class name spelling."
                }]
            }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error fetching Godot API reference: {str(e)}"}]
        }
