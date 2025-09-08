from fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# إنشاء السيرفر
mcp = FastMCP("my-mcp-server")

@mcp.tool()
def tavily_search(query: str, max_results: int = 5):
    """Search for products using Tavily API"""
    if not TAVILY_API_KEY:
        return {"error": "Missing TAVILY_API_KEY in environment."}

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    data = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "num_results": max_results
    }

    try:
        res = requests.post(url, json=data, headers=headers, timeout=20)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def scrape_page(url: str):
    """Scrape a web page and return text preview"""
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text_content = "\n".join(paragraphs[:10])

        return {
            "url": url,
            "content_preview": text_content[:500]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # تشغيل السيرفر على HTTP بدلاً من STDIO
    mcp.run(transport="http", host="127.0.0.1", port=8080)
