import os
import json
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import agentops
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List
from tavily import TavilyClient
from scrapegraph_py import Client
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SGAI_API_KEY = os.getenv("SGAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["SGAI_API_KEY"] = SGAI_API_KEY
os.environ["AGENTOPS_API_KEY"] = AGENTOPS_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

agentops.init(
    api_key=AGENTOPS_API_KEY,
    skip_auto_end_session=True,
    default_tags=['crewai']
)

output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)

basic_llm = LLM(model="gemini/gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)

about_company = "three is a company that provides AI solutions to help websites refine their search and recommendation systems."
company_context = StringKnowledgeSource(content=about_company)

search_client = TavilyClient(api_key=TAVILY_API_KEY)
scrape_client = Client()

no_keywords = 10

class SuggestedSearchQueries(BaseModel):
    queries: List[str] = Field(..., title="Suggested search queries to be passed to the search engine",
                               min_items=1, max_items=no_keywords)

search_queries_recommendation_agent = Agent(
    role="Search Queries Recommendation Agent",
    goal="To provide a list of suggested search queries to be passed to the search engine.",
    backstory="Helps generate varied search queries based on context provided.",
    llm=basic_llm,
    verbose=True,
)

search_queries_recommendation_task = Task(
    description="\n".join([
        "three is looking to buy {product_name} at the best prices (value for a price strategy)",
        "The company targets any of these websites: {websites_list}",
        "The company wants to reach all available products on the internet to be compared later.",
        "The stores must sell the product in {country_name}",
        "Generate at maximum {no_keywords} queries.",
        "The search keywords must be in {language} language.",
        "Search keywords must contain specific brands, types or technologies. Avoid general keywords.",
        "The search query must reach an ecommerce webpage for product, and not a blog or listing page."
    ]),
    expected_output="A JSON object containing a list of suggested search queries.",
    output_json=SuggestedSearchQueries,
    output_file=os.path.join(output_dir, "step_1_suggested_search_queries.json"),
    agent=search_queries_recommendation_agent
)

class SignleSearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    search_query: str

class AllSearchResults(BaseModel):
    results: List[SignleSearchResult]

@tool
def search_engine_tool(query: str):
    return search_client.search(query)

search_engine_agent = Agent(
    role="Search Engine Agent",
    goal="To search for products based on the suggested search query.",
    backstory="Searches for products based on suggested queries.",
    llm=basic_llm,
    verbose=True,
    tools=[search_engine_tool]
)

search_engine_task = Task(
    description="\n".join([
        "Search for products based on the suggested search queries.",
        "Collect results from multiple search queries.",
        "Ignore suspicious links or non-ecommerce product pages.",
        "Ignore search results with confidence score less than ({score_th}).",
        "The search results will be used to compare prices from different websites."
    ]),
    expected_output="A JSON object containing the search results.",
    output_json=AllSearchResults,
    output_file=os.path.join(output_dir, "step_2_search_results.json"),
    agent=search_engine_agent
)

class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str

class SingleExtractedProduct(BaseModel):
    page_url: str
    product_title: str
    product_image_url: str
    product_url: str
    product_current_price: float
    product_original_price: float = None
    product_discount_percentage: float = None
    product_specs: List[ProductSpec]
    agent_recommendation_rank: int
    agent_recommendation_notes: List[str]

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

@tool
def web_scraping_tool(page_url: str):
    details = scrape_client.smartscraper(
        website_url=page_url,
        user_prompt="Extract ```json\n" + SingleExtractedProduct.schema_json() + "```\n From the web page"
    )
    return {"page_url": page_url, "details": details}

scraping_agent = Agent(
    role="Web scraping agent",
    goal="To extract details from any website.",
    backstory="Scrapes ecommerce websites for product details.",
    llm=basic_llm,
    tools=[web_scraping_tool],
    verbose=True,
)

scraping_task = Task(
    description="\n".join([
        "Extract product details from ecommerce store pages.",
        "Collect results from multiple page URLs.",
        "Collect the best {top_recommendations_no} products from the search results."
    ]),
    expected_output="A JSON object containing products details.",
    output_json=AllExtractedProducts,
    output_file=os.path.join(output_dir, "step_3_search_results.json"),
    agent=scraping_agent
)

procurement_report_author_agent = Agent(
    role="Procurement Report Author Agent",
    goal="Generate a professional HTML page for the procurement report.",
    backstory="Generates a final procurement report with findings.",
    llm=basic_llm,
    verbose=True,
)

procurement_report_author_task = Task(
    description="\n".join([
        "Generate a professional HTML page for the procurement report.",
        "Use Bootstrap CSS framework for a better UI.",
        "Include: Executive Summary, Introduction, Methodology, Findings, Analysis, Recommendations, Conclusion, Appendices."
    ]),
    expected_output="A professional HTML page for the procurement report.",
    output_file=os.path.join(output_dir, "step_4_procurement_report.html"),
    agent=procurement_report_author_agent,
)

three_crew = Crew(
    agents=[
        search_queries_recommendation_agent,
        search_engine_agent,
        scraping_agent,
        procurement_report_author_agent,
    ],
    tasks=[
        search_queries_recommendation_task,
        search_engine_task,
        scraping_task,
        procurement_report_author_task,
    ],
    process=Process.sequential,
    knowledge_sources=[company_context]
)

if __name__ == "__main__":
    crew_results = three_crew.kickoff(
        inputs={
            "product_name": "coffee machine for the office",
            "websites_list": ["www.amazon.eg", "www.jumia.com.eg", "www.noon.com/egypt-en"],
            "country_name": "Egypt",
            "no_keywords": 10,
            "language": "English",
            "score_th": 0.10,
            "top_recommendations_no": 10
        }
    )
    print("Crew finished! Results saved in:", output_dir)
