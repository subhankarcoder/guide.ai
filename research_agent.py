# improved_research_agent.py
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ToolType(Enum):
    ARXIV_SEARCH = "ArXivSearch"
    EXA_SEARCH = "ExaSearch"
    TOPIC_ANALYZER = "TopicWeightAnalyzer"
    GAP_IDENTIFIER = "GapIdentifier"
    DIRECTION_SYNTHESIZER = "DirectionSynthesizer"

@dataclass
class ReActStep:
    step_type: str  # "Thought", "Action", "Observation"
    content: str
    tool_used: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_output: Optional[Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class ResearchGuideAgent:
    """
    Enhanced Research Guide Agent implementing ReAct framework
    """
    
    def __init__(self, tools_manager):
        self.tools = tools_manager
        self.context = {}
        self.react_steps = []
        self.max_iterations = 15
        
        # Tool descriptions for ReAct prompting
        self.tool_descriptions = {
            ToolType.ARXIV_SEARCH: {
                "description": "Searches any topic in ArXiv database through its API and returns summary and metadata about the papers.",
                "input_format": '{"search": "topic name"}',
                "when_to_use": "When you need academic papers on a specific research topic"
            },
            ToolType.EXA_SEARCH: {
                "description": "Searches anything in the web and retrieves text and its sources. Used for real-time insights about any topic.",
                "input_format": '{"query": "search query", "text": true}',
                "when_to_use": "When you need current web information, discussions, or real-time insights"
            },
            ToolType.TOPIC_ANALYZER: {
                "description": "For any given research paper summary it can extract their topics, weights and sub-topics using LLM",
                "input_format": '{"summary": "paper summary text"}',
                "when_to_use": "When you have paper summaries and need to extract key topics and their importance"
            },
            ToolType.GAP_IDENTIFIER: {
                "description": "For any given research paper summary it identifies the gaps in it with help of LLM",
                "input_format": '{"title": "paper title", "summary": "paper summary"}',
                "when_to_use": "When you need to identify research gaps in specific papers"
            },
            ToolType.DIRECTION_SYNTHESIZER: {
                "description": "For given list of topics and gaps it returns a direction of research containing the core research question, proposed methods and relevance",
                "input_format": '{"topics": ["topic1", "topic2"], "gaps": ["gap1", "gap2"]}',
                "when_to_use": "When you have collected topics and gaps and need to synthesize research directions"
            }
        }

    async def process_query(self, query: str, stream_callback=None) -> Dict[str, Any]:
        """
        Process research query using ReAct framework with streaming
        """
        self.context = {
            "original_query": query,
            "papers_found": [],
            "topics_extracted": [],
            "gaps_identified": [],
            "research_directions": [],
            "web_insights": []
        }
        self.react_steps = []
        
        # Initial goal setting
        await self._add_thought(
            f"I need to provide comprehensive research guidance for the query: '{query}'. "
            f"My goal is to analyze relevant research papers, identify key topics and gaps, "
            f"and synthesize actionable research directions with proper sources.",
            stream_callback
        )
        
        # ReAct loop
        iteration = 0
        while iteration < self.max_iterations and not self._is_goal_achieved():
            iteration += 1
            
            # Reasoning step
            next_action = await self._reason_next_action(stream_callback)
            if not next_action:
                break
                
            # Action step
            action_result = await self._execute_action(next_action, stream_callback)
            
            # Observation step
            await self._observe_result(action_result, stream_callback)
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Final synthesis
        final_answer = await self._synthesize_final_answer(stream_callback)
        
        return {
            "query": query,
            "react_steps": [step.__dict__ for step in self.react_steps],
            "final_answer": final_answer,
            "context": self.context,
            "processing_time": time.time() - self.react_steps[0].timestamp if self.react_steps else 0
        }

    async def _add_thought(self, thought: str, stream_callback=None):
        """Add a reasoning step and stream it"""
        step = ReActStep("Thought", thought)
        self.react_steps.append(step)
        
        if stream_callback:
            await stream_callback({
                "type": "reasoning",
                "step": "Thought",
                "content": thought,
                "timestamp": step.timestamp
            })

    async def _reason_next_action(self, stream_callback=None) -> Optional[Dict[str, Any]]:
        """Determine the next action based on current context"""
        
        # Check what we have and what we need
        papers_count = len(self.context["papers_found"])
        topics_count = len(self.context["topics_extracted"])
        gaps_count = len(self.context["gaps_identified"])
        directions_count = len(self.context["research_directions"])
        web_insights_count = len(self.context["web_insights"])
        
        # Reasoning logic
        if papers_count == 0:
            reasoning = "I haven't found any research papers yet. I should start by searching ArXiv for academic papers related to the query."
            action = {
                "tool": ToolType.ARXIV_SEARCH,
                "input": {"search": self.context["original_query"]},
                "rationale": "Need to gather academic papers as foundation for research analysis"
            }
        elif web_insights_count == 0:
            reasoning = f"I have {papers_count} papers from ArXiv. Now I should search the web for current discussions and real-time insights about this topic."
            action = {
                "tool": ToolType.EXA_SEARCH,
                "input": {"query": self.context["original_query"], "text": True},
                "rationale": "Need current web insights to complement academic papers"
            }
        elif topics_count < papers_count:
            reasoning = f"I have {papers_count} papers but only extracted topics from {topics_count}. I need to analyze more papers to extract their key topics and weights."
            # Find unanalyzed papers
            unanalyzed_papers = [p for p in self.context["papers_found"] if not p.get("topics_analyzed")]
            if unanalyzed_papers:
                paper = unanalyzed_papers[0]
                action = {
                    "tool": ToolType.TOPIC_ANALYZER,
                    "input": {"summary": paper["summary"]},
                    "rationale": f"Extracting topics from paper: '{paper['title'][:50]}...'"
                }
            else:
                action = None
        elif gaps_count < papers_count:
            reasoning = f"I have analyzed topics for {topics_count} papers but only identified gaps for {gaps_count}. I need to identify research gaps in more papers."
            # Find papers without gap analysis
            ungapped_papers = [p for p in self.context["papers_found"] if not p.get("gaps_analyzed")]
            if ungapped_papers:
                paper = ungapped_papers[0]
                action = {
                    "tool": ToolType.GAP_IDENTIFIER,
                    "input": {"title": paper["title"], "summary": paper["summary"]},
                    "rationale": f"Identifying gaps in paper: '{paper['title'][:50]}...'"
                }
            else:
                action = None
        elif directions_count == 0 and topics_count > 0 and gaps_count > 0:
            reasoning = f"I have extracted {topics_count} topic sets and {gaps_count} gap analyses. Now I can synthesize research directions."
            action = {
                "tool": ToolType.DIRECTION_SYNTHESIZER,
                "input": {
                    "topics": [topic for topic_set in self.context["topics_extracted"] for topic in topic_set],
                    "gaps": self.context["gaps_identified"]
                },
                "rationale": "Synthesizing final research directions from collected topics and gaps"
            }
        else:
            reasoning = "I have gathered sufficient information to provide comprehensive research guidance."
            action = None
        
        await self._add_thought(reasoning, stream_callback)
        return action

    async def _execute_action(self, action: Dict[str, Any], stream_callback=None) -> Any:
        """Execute the determined action"""
        tool_type = action["tool"]
        tool_input = action["input"]
        rationale = action["rationale"]
        
        # Create action step
        action_step = ReActStep(
            "Action", 
            f"Using {tool_type.value}: {rationale}",
            tool_used=tool_type.value,
            tool_input=tool_input
        )
        self.react_steps.append(action_step)
        
        if stream_callback:
            await stream_callback({
                "type": "action",
                "step": "Action",
                "tool": tool_type.value,
                "input": tool_input,
                "rationale": rationale,
                "timestamp": action_step.timestamp
            })
        
        # Execute the tool
        try:
            if tool_type == ToolType.ARXIV_SEARCH:
                result = await self.tools.arxiv_search(tool_input["search"])
            elif tool_type == ToolType.EXA_SEARCH:
                result = await self.tools.exa_search(tool_input["query"])
            elif tool_type == ToolType.TOPIC_ANALYZER:
                result = await self.tools.analyze_topics(tool_input["summary"])
            elif tool_type == ToolType.GAP_IDENTIFIER:
                result = await self.tools.identify_gaps(tool_input["title"], tool_input["summary"])
            elif tool_type == ToolType.DIRECTION_SYNTHESIZER:
                result = await self.tools.synthesize_directions(tool_input["topics"], tool_input["gaps"])
            else:
                result = {"error": f"Unknown tool: {tool_type}"}
            
            action_step.tool_output = result
            return result
            
        except Exception as e:
            error_result = {"error": str(e)}
            action_step.tool_output = error_result
            return error_result

    async def _observe_result(self, result: Any, stream_callback=None):
        """Observe and contextualize the action result"""
        
        if isinstance(result, dict) and "error" in result:
            observation = f"Error occurred: {result['error']}. I'll need to adapt my approach."
        else:
            # Process result based on last action
            last_action = next((step for step in reversed(self.react_steps) if step.step_type == "Action"), None)
            
            if last_action and last_action.tool_used == ToolType.ARXIV_SEARCH.value:
                paper_count = len(result) if isinstance(result, list) else 0
                observation = f"Found {paper_count} research papers from ArXiv. Adding them to my knowledge base for analysis."
                self.context["papers_found"].extend(result if isinstance(result, list) else [])
                
            elif last_action and last_action.tool_used == ToolType.EXA_SEARCH.value:
                insight_count = len(result) if isinstance(result, list) else 0
                observation = f"Gathered {insight_count} web insights about the topic. This provides current context beyond academic papers."
                self.context["web_insights"].extend(result if isinstance(result, list) else [])
                
            elif last_action and last_action.tool_used == ToolType.TOPIC_ANALYZER.value:
                topic_count = len(result) if isinstance(result, list) else 0
                observation = f"Extracted {topic_count} topics with their weights and subtopics. This helps understand the research landscape."
                self.context["topics_extracted"].append(result if isinstance(result, list) else [])
                # Mark paper as analyzed
                if self.context["papers_found"]:
                    for paper in self.context["papers_found"]:
                        if not paper.get("topics_analyzed"):
                            paper["topics_analyzed"] = True
                            break
                            
            elif last_action and last_action.tool_used == ToolType.GAP_IDENTIFIER.value:
                observation = f"Identified research gaps in the paper. These gaps reveal opportunities for future research."
                if isinstance(result, list):
                    self.context["gaps_identified"].extend(result)
                else:
                    self.context["gaps_identified"].append(result)
                # Mark paper as gap-analyzed
                if self.context["papers_found"]:
                    for paper in self.context["papers_found"]:
                        if not paper.get("gaps_analyzed"):
                            paper["gaps_analyzed"] = True
                            break
                            
            elif last_action and last_action.tool_used == ToolType.DIRECTION_SYNTHESIZER.value:
                direction_count = len(result) if isinstance(result, list) else 0
                observation = f"Synthesized {direction_count} concrete research directions with methods and expected impact."
                self.context["research_directions"] = result if isinstance(result, list) else []
                
            else:
                observation = f"Processed result: {str(result)[:100]}..."

        # Create observation step
        observation_step = ReActStep("Observation", observation)
        self.react_steps.append(observation_step)
        
        if stream_callback:
            await stream_callback({
                "type": "observation",
                "step": "Observation", 
                "content": observation,
                "timestamp": observation_step.timestamp
            })

    def _is_goal_achieved(self) -> bool:
        """Check if we have achieved our research guidance goal"""
        return (
            len(self.context["papers_found"]) >= 3 and
            len(self.context["topics_extracted"]) >= 2 and
            len(self.context["gaps_identified"]) >= 2 and
            len(self.context["research_directions"]) >= 1
        )

    async def _synthesize_final_answer(self, stream_callback=None) -> Dict[str, Any]:
        """Synthesize the final comprehensive research guidance"""
        
        await self._add_thought(
            "Now I'll synthesize all the information I've gathered into comprehensive research guidance.",
            stream_callback
        )
        
        # Compile sources
        sources = []
        for paper in self.context["papers_found"]:
            sources.append({
                "type": "academic_paper",
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "link": paper.get("pdf_link", ""),
                "source": "ArXiv"
            })
        
        for insight in self.context["web_insights"]:
            sources.append({
                "type": "web_resource",
                "title": insight.get("title", ""),
                "url": insight.get("url", ""),
                "source": "Web Search"
            })
        
        # Flatten topics
        all_topics = []
        for topic_set in self.context["topics_extracted"]:
            if isinstance(topic_set, list):
                all_topics.extend(topic_set)
        
        final_answer = {
            "research_query": self.context["original_query"],
            "research_directions": self.context["research_directions"],
            "key_topics": all_topics[:10] if all_topics else [],
            "research_gaps": self.context["gaps_identified"][:10],
            "sources": sources,
            "summary": f"Based on analysis of {len(self.context['papers_found'])} research papers and {len(self.context['web_insights'])} web sources, I've identified {len(self.context['research_directions'])} actionable research directions.",
            "methodology": "Used ReAct framework to systematically search academic papers, analyze web insights, extract topics, identify gaps, and synthesize research directions."
        }
        
        if stream_callback:
            await stream_callback({
                "type": "final_answer",
                "content": final_answer,
                "timestamp": time.time()
            })
        
        return final_answer


# Enhanced Tools Manager that works with existing utilities
class ToolsManager:
    """
    Manages all research tools with proper error handling
    """
    
    def __init__(self, arxiv_util, exa_util, llm_util):
        self.arxiv = arxiv_util
        self.exa = exa_util  
        self.llm = llm_util

    async def arxiv_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search ArXiv papers"""
        try:
            results = await self.arxiv.search(query, k)
            return results if results else []
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    async def exa_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search web using Exa"""
        try:
            results = await self.exa.search_web(query, k)
            return results if results else []
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return []

    async def analyze_topics(self, summary: str) -> List[Dict]:
        """Extract topics from paper summary"""
        try:
            from utils.prompts import TOPIC_WEIGHT_PROMPT
            prompt = TOPIC_WEIGHT_PROMPT.format(summary=summary)
            response = await self.llm.chat([{"role": "user", "content": prompt}])
            return self.llm.safe_json_parse(response, fallback_value=[])
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return []

    async def identify_gaps(self, title: str, summary: str) -> List[str]:
        """Identify research gaps"""
        try:
            from utils.prompts import GAP_PROMPT
            prompt = GAP_PROMPT.format(title=title, summary=summary)
            response = await self.llm.chat([{"role": "user", "content": prompt}])
            
            # Split response into individual gaps
            gaps = [gap.strip() for gap in response.split('\n') if gap.strip() and not gap.strip().startswith('RESEARCH GAPS')]
            return gaps[:5]  # Limit to 5 gaps
        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
            return []

    async def synthesize_directions(self, topics: List[Dict], gaps: List[str]) -> List[Dict]:
        """Synthesize research directions"""
        try:
            from utils.prompts import DIRECTION_PROMPT
            topics_text = json.dumps(topics[:10], indent=2)  # Limit topics
            gaps_text = "\n".join(f"- {gap}" for gap in gaps[:10])  # Limit gaps
            
            prompt = DIRECTION_PROMPT.format(topics=topics_text, gaps=gaps_text)
            response = await self.llm.chat([{"role": "user", "content": prompt}])
            return self.llm.safe_json_parse(response, fallback_value=[])
        except Exception as e:
            logger.error(f"Direction synthesis failed: {e}")
            return []