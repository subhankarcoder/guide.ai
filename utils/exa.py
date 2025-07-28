# utils/exa.py
import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from exa_py import Exa
    exa_client = Exa(os.getenv("EXA_API_KEY")) if os.getenv("EXA_API_KEY") else None
except ImportError:
    logger.warning("exa_py not installed. Install with: pip install exa_py")
    exa_client = None

async def search_web(query: str, k: int = 5):
    """Enhanced web search with comprehensive error handling and answer retrieval"""
    if not exa_client:
        logger.warning("Exa client not available - returning empty results")
        return [], None
    
    loop = asyncio.get_running_loop()
    
    def _blocking_search():
        try:
            # Get search results
            search_response = exa_client.search_and_contents(
                query=query, 
                num_results=k, 
                text=True,
                highlights=True
            )
            
            results = []
            for result in search_response.results:
                results.append({
                    "url": result.url,
                    "title": result.title or "Unknown Title",
                    "text": (result.text or "")[:1200],  # Limit text length
                    "highlights": getattr(result, 'highlights', [])
                })

            # Get a direct answer
            answer_response = exa_client.search(query, use_autoprompt=True, type="answer")
            answer = answer_response.answer if hasattr(answer_response, 'answer') else None

            return results, answer
            
        except Exception as e:
            logger.error(f"Exa search error: {e}")
            return [], None
    
    return await loop.run_in_executor(None, _blocking_search)