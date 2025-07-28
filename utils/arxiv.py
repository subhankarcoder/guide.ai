# utils/arxiv.py
import arxiv
import asyncio
import textwrap
import logging

logger = logging.getLogger(__name__)

async def search(query: str, k: int = 5):
    """Enhanced ArXiv search with better error handling"""
    loop = asyncio.get_running_loop()
    
    def _blocking_search():
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=k,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in client.results(search):
                # Clean and validate the result
                title = result.title.strip() if result.title else "Unknown Title"
                authors = [author.name for author in result.authors] if result.authors else ["Unknown Author"]
                summary = textwrap.shorten(result.summary.strip(), width=1200, placeholder="...") if result.summary else "No summary available"
                pdf_url = result.pdf_url if hasattr(result, 'pdf_url') else ""
                
                results.append({
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "pdf_link": pdf_url
                })
                
            return results
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    return await loop.run_in_executor(None, _blocking_search)
