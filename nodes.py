# nodes.py
import logging
import json
import asyncio
import traceback
import time
from typing import List, Dict, Any
from pocketflow import AsyncNode, AsyncParallelBatchNode, AsyncFlow

from utils import llm, arxiv, exa, prompts

# Configure logging with UTF-8 support for Windows
import sys
if sys.platform.startswith('win'):
    # Fix Windows console encoding issues
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research_agent.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Production metrics tracking
class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.node_times = {}
        self.error_count = 0
        self.success_count = 0
    
    def track_node_start(self, node_name: str):
        self.node_times[node_name] = time.time()
    
    def track_node_end(self, node_name: str, success: bool = True):
        if node_name in self.node_times:
            duration = time.time() - self.node_times[node_name]
            logger.info(f"Node {node_name} completed in {duration:.2f}s - {'Success' if success else 'Failed'}")
            if success:
                self.success_count += 1
            else:
                self.error_count += 1

# Global metrics instance
metrics = MetricsTracker()

# ----------  Enhanced Utility Functions  ----------
def push_stream_update(shared: Dict, message: str):
    """
    Push real-time updates to both the streams array and SSE queue
    """
    try:
        # Remove or replace emoji characters if they cause issues
        clean_message = message.encode('ascii', 'ignore').decode('ascii')
        if not clean_message.strip():
            clean_message = message  # Keep original if completely filtered out
        
        # Add to streams array for final response
        shared.setdefault("streams", []).append(clean_message)
        
        # Push to real-time SSE queue if available
        sse_queue = shared.get("_sse_queue")
        if sse_queue and not sse_queue.full():
            try:
                sse_queue.put_nowait(clean_message)
            except asyncio.QueueFull:
                logger.warning("SSE queue is full, dropping message")
        
        logger.debug(f"Pushed stream update: {clean_message[:50]}...")
        
    except Exception as e:
        logger.error(f"Error pushing stream update: {str(e)}")


async def call_arxiv_safe(query: str, k: int = 5) -> List[Dict]:
    """Enhanced ArXiv search with comprehensive error handling"""
    try:
        logger.info(f"Searching ArXiv for: '{query}' (limit: {k})")
        result = await arxiv.search(query, k)
        
        if not result:
            logger.warning("ArXiv search returned no results")
            return []
            
        # Validate and clean results
        clean_results = []
        for paper in result:
            if paper.get("title") and paper.get("summary"):
                clean_results.append({
                    "title": paper.get("title", "").strip(),
                    "authors": paper.get("authors", []),
                    "summary": paper.get("summary", "").strip()[:1200],  # Limit summary length
                    "pdf_link": paper.get("pdf_link", "")
                })
        
        logger.info(f"ArXiv search returned {len(clean_results)} valid papers")
        return clean_results
        
    except Exception as e:
        logger.error(f"ArXiv search failed: {str(e)}\n{traceback.format_exc()}")
        return []

async def call_exa_safe(query: str, k: int = 5) -> List[Dict]:
    """Enhanced Exa search with comprehensive error handling"""
    try:
        logger.info(f"Searching Exa for: '{query}' (limit: {k})")
        result = await exa.search_web(query, k)
        
        if not result:
            logger.warning("Exa search returned no results")
            return []
            
        logger.info(f"Exa search returned {len(result)} results")
        return result
        
    except Exception as e:
        logger.error(f"Exa search failed: {str(e)}\n{traceback.format_exc()}")
        return []

# ----------  Production-Ready Async Nodes  ----------
class TopicWeightNode(AsyncParallelBatchNode):
    """Extract topics and weights from research papers with robust error handling"""
    
    async def prep_async(self, shared):
        papers = shared.get("papers", [])
        logger.info(f"TopicWeightNode processing {len(papers)} papers")
        metrics.track_node_start("TopicWeightNode")
        
        if not papers:
            logger.warning("No papers provided to TopicWeightNode")
            return []
            
        return papers

    async def exec_async(self, paper):
        paper_title = paper.get('title', 'Unknown')[:50]
        logger.info(f"Analyzing topics for: {paper_title}...")
        
        try:
            summary = paper.get("summary", "").strip()
            if not summary:
                logger.warning(f"Paper '{paper_title}' has no summary, using title")
                summary = paper.get("title", "No content available")
            
            # Use the enhanced prompt
            prompt = prompts.TOPIC_WEIGHT_PROMPT.format(summary=summary)
            
            # Make LLM call with retries
            raw_response = await llm.chat([{"role": "user", "content": prompt}])
            
            if not raw_response:
                logger.error(f"Empty LLM response for paper: {paper_title}")
                return prompts.FALLBACK_TOPICS
            
            # Parse JSON with reasoning token handling
            topics = llm.safe_json_parse(raw_response, fallback_value=[])
            
            # Validate structure
            if not isinstance(topics, list):
                logger.warning(f"Expected list but got {type(topics)}, using fallback")
                topics = prompts.FALLBACK_TOPICS
            
            # Validate each topic
            valid_topics = []
            for topic in topics:
                if isinstance(topic, dict) and "topic" in topic and "weight" in topic:
                    valid_topics.append({
                        "topic": str(topic.get("topic", "Unknown")),
                        "weight": str(topic.get("weight", "Medium")),
                        "subtopics": topic.get("subtopics", [])
                    })
            
            if not valid_topics:
                logger.warning(f"No valid topics extracted for paper: {paper_title}")
                return prompts.FALLBACK_TOPICS
            
            logger.info(f"Successfully extracted {len(valid_topics)} topics from: {paper_title}")
            return valid_topics
            
        except Exception as e:
            logger.error(f"Error analyzing topics for '{paper_title}': {str(e)}\n{traceback.format_exc()}")
            return prompts.FALLBACK_TOPICS

    async def post_async(self, shared, prep_res, exec_res_list):
        all_topics = []
        
        for topic_list in exec_res_list:
            if isinstance(topic_list, list):
                all_topics.extend(topic_list)
            elif isinstance(topic_list, dict):
                all_topics.append(topic_list)
        
        # Remove duplicates and limit count
        unique_topics = []
        seen_topics = set()
        for topic in all_topics:
            topic_name = topic.get("topic", "").lower()
            if topic_name and topic_name not in seen_topics:
                unique_topics.append(topic)
                seen_topics.add(topic_name)
        
        shared["topics"] = unique_topics[:20]  # Limit to 20 topics
        
        logger.info(f"Total unique topics collected: {len(shared['topics'])}")
        metrics.track_node_end("TopicWeightNode", success=len(shared['topics']) > 0)
        
        # Update streaming progress
        push_stream_update(shared, f"TOPICS: Analyzed topics from {len(prep_res)} papers")
        
        return "default"

class GapAnalysisNode(AsyncParallelBatchNode):
    """Identify research gaps with comprehensive error handling"""
    
    async def prep_async(self, shared):
        papers = shared.get("papers", [])
        logger.info(f"GapAnalysisNode processing {len(papers)} papers")
        metrics.track_node_start("GapAnalysisNode")
        
        if not papers:
            logger.warning("No papers provided to GapAnalysisNode")
            return []
            
        return papers
    
    async def exec_async(self, paper):
        paper_title = paper.get('title', 'Unknown')[:50]
        logger.info(f"Analyzing gaps for: {paper_title}...")
        
        try:
            title = paper.get("title", "Unknown Title")
            summary = paper.get("summary", "").strip()
            
            if not summary:
                logger.warning(f"Paper '{paper_title}' has no summary for gap analysis")
                return f"Limited gap analysis for '{title}' - insufficient content"
            
            prompt = prompts.GAP_PROMPT.format(title=title, summary=summary)
            raw_response = await llm.chat([{"role": "user", "content": prompt}])
            
            if not raw_response:
                logger.error(f"Empty LLM response for gap analysis: {paper_title}")
                return f"Gap analysis unavailable for '{title}'"
            
            # Clean and validate gaps
            gaps = llm._strip_think(raw_response).strip()
            
            if not gaps:
                return f"No specific gaps identified in '{title}'"
            
            logger.info(f"Successfully identified gaps for: {paper_title}")
            return gaps
            
        except Exception as e:
            logger.error(f"Error in gap analysis for '{paper_title}': {str(e)}\n{traceback.format_exc()}")
            return f"Gap analysis failed for '{paper.get('title', 'Unknown')}': {str(e)}"
    
    async def post_async(self, shared, prep_res, exec_res_list):
        # Filter and clean gaps
        valid_gaps = []
        for gap in exec_res_list:
            if gap and isinstance(gap, str) and gap.strip():
                valid_gaps.append(gap.strip())
        
        shared["gaps"] = valid_gaps[:15]  # Limit to 15 gaps
        
        logger.info(f"Total research gaps collected: {len(shared['gaps'])}")
        metrics.track_node_end("GapAnalysisNode", success=len(shared['gaps']) > 0)
        
        # Update streaming progress
        push_stream_update(shared, f"GAPS: Identified research gaps from {len(prep_res)} papers")
        
        return "default"

class DirectionSynthesizerNode(AsyncNode):
    """Synthesize research directions with advanced error handling"""
    
    async def prep_async(self, shared):
        topics = shared.get("topics", [])
        gaps = shared.get("gaps", [])
        
        logger.info(f"DirectionSynthesizerNode processing {len(topics)} topics and {len(gaps)} gaps")
        metrics.track_node_start("DirectionSynthesizerNode")
        
        return {"topics": topics, "gaps": gaps}
    
    async def exec_async(self, data):
        logger.info("Synthesizing research directions...")
        
        try:
            topics = data.get("topics", [])
            gaps = data.get("gaps", [])
            
            if not topics and not gaps:
                logger.warning("No topics or gaps available for direction synthesis")
                return prompts.FALLBACK_DIRECTIONS
            
            # Prepare context for the prompt
            topics_text = json.dumps(topics, indent=2) if topics else "No specific topics identified"
            gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "No specific gaps identified"
            
            prompt = prompts.DIRECTION_PROMPT.format(topics=topics_text, gaps=gaps_text)
            raw_response = await llm.chat([{"role": "user", "content": prompt}])
            
            if not raw_response:
                logger.error("Empty LLM response for direction synthesis")
                return prompts.FALLBACK_DIRECTIONS
            
            # Parse and validate directions
            directions = llm.safe_json_parse(raw_response, fallback_value=[])
            
            if not isinstance(directions, list):
                logger.warning(f"Expected list but got {type(directions)} for directions")
                directions = prompts.FALLBACK_DIRECTIONS
            
            # Validate each direction
            valid_directions = []
            for direction in directions:
                if isinstance(direction, dict) and all(key in direction for key in ["question", "methods", "impact"]):
                    valid_directions.append({
                        "question": str(direction.get("question", "")).strip(),
                        "methods": str(direction.get("methods", "")).strip(),
                        "impact": str(direction.get("impact", "")).strip()
                    })
            
            if not valid_directions:
                logger.warning("No valid directions generated, using fallback")
                return prompts.FALLBACK_DIRECTIONS
            
            logger.info(f"Successfully generated {len(valid_directions)} research directions")
            return valid_directions[:5]  # Limit to 5 directions
            
        except Exception as e:
            logger.error(f"Error in direction synthesis: {str(e)}\n{traceback.format_exc()}")
            return prompts.FALLBACK_DIRECTIONS
    
    async def post_async(self, shared, prep_res, exec_res):
        directions = exec_res if exec_res else []
        shared["directions"] = directions
        
        logger.info(f"Final research directions stored: {len(directions)}")
        metrics.track_node_end("DirectionSynthesizerNode", success=len(directions) > 0)
        
        # Update streaming progress
        push_stream_update(shared, f"DIRECTIONS: Generated {len(directions)} research directions")
        
        return "default"

class ResearchAgentNode(AsyncNode):
    """Enhanced research agent with real paper fetching and streaming updates"""

    async def prep_async(self, shared):
        logger.info("Initializing Enhanced ResearchAgent...")

        # Initialize shared data here, as in the old pattern
        shared["streams"] = []
        shared["papers"] = []
        shared["search_queries"] = []

        # Add the initial stream message directly to the shared state
        shared["streams"].append("STARTED: Starting research analysis...")

        logger.info("ResearchAgent initialized successfully")
        
        # Return the query from shared state to be passed to exec_async
        return shared.get("query", "")

    async def exec_async(self, query: str):
        logger.info(f"ResearchAgent processing query: '{query}'")
        metrics.track_node_start("ResearchAgentNode")

        # Local lists to hold results, which will be returned for post_async
        local_streams = []
        papers = []
        
        try:
            local_streams.append(f"QUERY: Analyzing query: '{query}'")

            # Generate search queries
            search_queries = self._generate_search_queries(query)
            local_streams.append(f"SEARCH: Generated {len(search_queries)} search queries")

            # Search for papers
            all_papers = []
            for search_query in search_queries:
                local_streams.append(f"ARXIV: Searching for: {search_query}")
                
                # Search ArXiv
                arxiv_papers = await call_arxiv_safe(search_query, k=3)
                if arxiv_papers:
                    all_papers.extend(arxiv_papers)
                    local_streams.append(f"FOUND: {len(arxiv_papers)} papers from ArXiv")

                await asyncio.sleep(0.5) # Avoid rate limiting

            # Remove duplicates and limit total papers
            unique_papers = []
            seen_titles = set()
            for paper in all_papers:
                title_key = paper.get("title", "").lower().strip()
                if title_key and title_key not in seen_titles and len(unique_papers) < 10:
                    unique_papers.append(paper)
                    seen_titles.add(title_key)
            
            papers = unique_papers
            logger.info(f"Collected {len(papers)} unique papers")
            local_streams.append(f"COLLECTED: {len(papers)} unique research papers")
            
            # Prepare results to be returned
            return {
                "papers": papers,
                "streams": local_streams,
                "search_queries": search_queries,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in ResearchAgent execution: {str(e)}\n{traceback.format_exc()}")
            local_streams.append(f"ERROR: Error in research: {str(e)}")
            
            # Return fallback papers on error
            return {
                "papers": self._get_fallback_papers(query),
                "streams": local_streams,
                "search_queries": search_queries if 'search_queries' in locals() else [],
                "success": False
            }

    def _generate_search_queries(self, query: str) -> List[str]:
        """Generate relevant search queries from the main query"""
        base_queries = [query]
        
        if "machine learning" in query.lower():
            base_queries.extend([
                f"{query} deep learning",
                f"{query} neural networks",
                f"{query} AI applications"
            ])
        elif "quantum" in query.lower():
            base_queries.extend([
                f"{query} quantum computing",
                f"{query} quantum algorithms"
            ])
        else:
            base_queries.extend([
                f"{query} recent advances",
                f"{query} survey"
            ])
        
        return base_queries[:4]

    def _get_fallback_papers(self, query: str) -> List[Dict]:
        """Provide fallback papers when search fails"""
        return [
            {
                "title": f"Research Overview: {query}",
                "authors": ["Research Team"],
                "summary": f"This paper provides a comprehensive overview of {query}, discussing current methodologies, challenges, and future research directions in the field.",
                "pdf_link": "https://example.com/paper1.pdf"
            },
            {
                "title": f"Recent Advances in {query}",
                "authors": ["Academic Consortium"],
                "summary": f"Recent developments in {query} have shown promising results. This work reviews the latest techniques and identifies key areas for future investigation.",
                "pdf_link": "https://example.com/paper2.pdf"
            }
        ]

    async def post_async(self, shared, prep_res, exec_res):
        # Now we update the shared state with the results from exec_async
        if exec_res and isinstance(exec_res, dict):
            papers = exec_res.get("papers", [])
            streams = exec_res.get("streams", [])
            search_queries = exec_res.get("search_queries", [])
            success = exec_res.get("success", False)

            shared["papers"] = papers
            shared["search_queries"] = search_queries
            shared["streams"].extend(streams) # Append new streams from execution

            logger.info(f"ResearchAgent stored {len(papers)} papers")
            metrics.track_node_end("ResearchAgentNode", success=success)

            shared["streams"].append(f"READY: Ready to analyze {len(papers)} papers")
        else:
            logger.warning("No valid results from exec_async")
            metrics.track_node_end("ResearchAgentNode", success=False)
            shared["streams"].append("ERROR: Research agent failed to produce results.")

        return "analysis"
