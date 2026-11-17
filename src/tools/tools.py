
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from rich import print
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re

load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and relevant information.
    Returns the title, URL, and snippet of the top 5 results.
    """

    try:
        response = tavily.search(
            query=query,
            max_results=5
        )

        results = response.get("results", [])

        if not results:
            return "No relevant search results found."

        out = []

        for index, r in enumerate(results, start=1):

            out.append(
                f"""
Result {index}

Title: {r.get('title', 'N/A')}

URL: {r.get('url', 'N/A')}

Snippet: {r.get('content', 'N/A')[:300]}
"""
            )

        return "\n---\n".join(out)

    except Exception as error:
        return f"Web search failed: {str(error)}"


@tool
def scrape_url(url: str) -> str:

    """
        Scrape and extract clean readable content from a URL.
        Uses multiple extraction strategies for better reliablility.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referrer": "https://www.google.com"
    }

    try:
        # Fetch Page
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        html = response.text

        # ------------------------------------
        # Strategy 1 -> trafilatura (Best for articles and blogs)
        #------------------------------------

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
        )

        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return cleaned[:5000]

        # ------------------------------------
        # Strategy 2 -> readability
        #------------------------------------
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, 'html.parser')

        for tag in soup([
            "script", 
            "style", 
            "header", 
            "footer", 
            "nav", 
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return cleaned[:5000]

        #------------------------------------
        # Strategy 3 -> fallback full page extraction
        #------------------------------------

        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup([
            "script", 
            "style", 
            "header", 
            "footer", 
            "nav", 
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        cleaned = re.sub(r'\s+', ' ', text)

        if cleaned:
            return cleaned[:5000]

        return "Failed to extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return "Request timed out while trying to fetch the URL."
    
    except requests.exceptions.HTTPError as e:
        return f"Http error occurred: {str(e)}"

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
