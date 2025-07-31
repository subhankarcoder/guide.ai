# utils/exa.py - Fixed version
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
    """Enhanced web search with fixed API parameters"""
    if not exa_client:
        logger.warning("Exa client not available - returning empty results")
        return [], None
    
    loop = asyncio.get_running_loop()
    
    def _blocking_search():
        try:
            logger.info(f"Exa search for: '{query}' with {k} results")
            
            # Fixed search call - removed invalid parameters
            search_response = exa_client.search_and_contents(
                query=query, 
                num_results=k, 
                text=True,
                highlights=True,
                # Removed type parameter that was causing 400 error
            )
            
            results = []
            for result in search_response.results:
                results.append({
                    "url": result.url,
                    "title": result.title or "Unknown Title",
                    "text": (result.text or "")[:1200],  # Limit text length
                    "highlights": getattr(result, 'highlights', [])
                })

            logger.info(f"Exa search successful: {len(results)} results")
            
            # Try to get answer using a separate call if needed
            answer = None
            try:
                # Use basic search for answer if available
                answer_response = exa_client.search(
                    query=query,
                    num_results=1,
                    use_autoprompt=True
                )
                if hasattr(answer_response, 'results') and answer_response.results:
                    # Extract answer from first result
                    first_result = answer_response.results[0]
                    answer = getattr(first_result, 'text', None) or getattr(first_result, 'summary', None)
            except Exception as answer_error:
                logger.warning(f"Could not get Exa answer: {answer_error}")
                answer = None

            return results, answer
            
        except Exception as e:
            logger.error(f"Exa search error: {e}")
            return [], None
    
    return await loop.run_in_executor(None, _blocking_search)