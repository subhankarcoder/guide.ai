# research_agent.py - Enhanced version with AI answer
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
        self.max_iterations = 12  # Reduced to prevent infinite loops
        
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
        start_time = time.time()
        
        self.context = {
            "original_query": query,
            "papers_found": [],
            "topics_extracted": [],
            "gaps_identified": [],
            "research_directions": [],
            "web_insights": [],
            "arxiv_completed": False,
            "exa_completed": False,
            "analysis_completed": False
        }
        self.react_steps = []
        
        # Initial goal setting
        await self._add_thought(
            f"I need to provide comprehensive research guidance for the query: '{query}'. "
            f"My goal is to: 1) Search ArXiv for academic papers, 2) Search web for current insights, "
            f"3) Analyze topics and gaps from papers, 4) Synthesize actionable research directions.",
            stream_callback
        )
        
        # ReAct loop with better termination conditions
        iteration = 0
        consecutive_failures = 0
        
        while iteration < self.max_iterations and not self._is_goal_achieved():
            iteration += 1
            
            # Reasoning step
            next_action = await self._reason_next_action(stream_callback)
            if not next_action:
                await self._add_thought(
                    "I have gathered sufficient information to provide research guidance. "
                    "Moving to final synthesis.",
                    stream_callback
                )
                break
                
            # Action step
            action_result = await self._execute_action(next_action, stream_callback)
            
            # Check for consecutive failures
            if isinstance(action_result, dict) and action_result.get("error"):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    await self._add_thought(
                        "Multiple tool failures detected. Proceeding with available data to provide best possible guidance.",
                        stream_callback
                    )
                    break
            else:
                consecutive_failures = 0
            
            # Observation step
            await self._observe_result(action_result, stream_callback)
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Generate AI answer
        ai_answer = await self._generate_ai_answer(stream_callback)
        
        # Final synthesis
        final_answer = await self._synthesize_final_answer(ai_answer, stream_callback)
        
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "react_steps": [step.__dict__ for step in self.react_steps],
            "final_answer": final_answer,
            "context": self.context,
            "processing_time": processing_time
        }

    async def _add_thought(self, thought: str, stream_callback=None):
        """Add a reasoning step and stream it"""
        step = ReActStep("Thought", thought)
        self.react_steps.append(step)
        logger.info(f"Thought: {thought[:100]}...")
        
        if stream_callback:
            await stream_callback({
                "type": "reasoning",
                "step": "Thought",
                "content": thought,
                "timestamp": step.timestamp
            })

    async def _reason_next_action(self, stream_callback=None) -> Optional[Dict[str, Any]]:
        """Enhanced reasoning with better flow control"""
        
        papers_count = len(self.context["papers_found"])
        topics_count = len(self.context["topics_extracted"])
        gaps_count = len(self.context["gaps_identified"])
        directions_count = len(self.context["research_directions"])
        web_insights_count = len(self.context["web_insights"])
        
        # Step 1: Get ArXiv papers first
        if not self.context["arxiv_completed"]:
            reasoning = "I need to start by searching ArXiv for academic papers related to the query. This will provide the foundation for research analysis."
            action = {
                "tool": ToolType.ARXIV_SEARCH,
                "input": {"search": self.context["original_query"]},
                "rationale": "Gathering academic papers as foundation for research analysis"
            }
            
        # Step 2: Get web insights
        elif not self.context["exa_completed"]:
            reasoning = f"I have searched ArXiv and found {papers_count} papers. Now I need to search the web for current discussions and real-time insights about this topic."
            action = {
                "tool": ToolType.EXA_SEARCH,
                "input": {"query": self.context["original_query"], "text": True},
                "rationale": "Gathering current web insights to complement academic papers"
            }
            
        # Step 3: Analyze topics from papers
        elif topics_count == 0 and papers_count > 0:
            reasoning = f"I have {papers_count} papers. Now I need to analyze them to extract key topics and their weights."
            paper = self.context["papers_found"][0]  # Analyze first paper
            action = {
                "tool": ToolType.TOPIC_ANALYZER,
                "input": {"summary": paper["summary"]},
                "rationale": f"Extracting topics from: '{paper['title'][:50]}...'"
            }
            
        # Step 4: Identify gaps from papers
        elif gaps_count == 0 and papers_count > 0:
            reasoning = f"I have analyzed some topics. Now I need to identify research gaps from the papers."
            paper = self.context["papers_found"][0]  # Analyze first paper for gaps
            action = {
                "tool": ToolType.GAP_IDENTIFIER,
                "input": {"title": paper["title"], "summary": paper["summary"]},
                "rationale": f"Identifying gaps in: '{paper['title'][:50]}...'"
            }
            
        # Step 5: Synthesize research directions
        elif directions_count == 0 and topics_count > 0 and gaps_count > 0:
            reasoning = f"I have extracted topics and identified gaps. Now I can synthesize concrete research directions."
            # Flatten topics for synthesis
            all_topics = []
            for topic_set in self.context["topics_extracted"]:
                if isinstance(topic_set, list):
                    all_topics.extend(topic_set)
                    
            action = {
                "tool": ToolType.DIRECTION_SYNTHESIZER,
                "input": {
                    "topics": all_topics[:10],  # Limit topics
                    "gaps": self.context["gaps_identified"][:10]  # Limit gaps
                },
                "rationale": "Synthesizing final research directions from collected data"
            }
            
        # Analysis complete
        else:
            reasoning = "I have completed the analysis with sufficient information to provide comprehensive research guidance."
            self.context["analysis_completed"] = True
            action = None
        
        await self._add_thought(reasoning, stream_callback)
        return action

    async def _execute_action(self, action: Dict[str, Any], stream_callback=None) -> Any:
        """Execute the determined action with enhanced error handling"""
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
        logger.info(f"Action: {tool_type.value} - {rationale}")
        
        if stream_callback:
            await stream_callback({
                "type": "action",
                "step": "Action",
                "tool": tool_type.value,
                "input": tool_input,
                "rationale": rationale,
                "timestamp": action_step.timestamp
            })
        
        # Execute the tool with error handling
        try:
            if tool_type == ToolType.ARXIV_SEARCH:
                result = await self.tools.arxiv_search(tool_input["search"])
                self.context["arxiv_completed"] = True
                
            elif tool_type == ToolType.EXA_SEARCH:
                result = await self.tools.exa_search(tool_input["query"])
                self.context["exa_completed"] = True
                
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
            logger.error(f"Tool execution failed: {e}")
            error_result = {"error": str(e)}
            action_step.tool_output = error_result
            return error_result

    async def _observe_result(self, result: Any, stream_callback=None):
        """Enhanced observation with better result processing"""
        
        if isinstance(result, dict) and "error" in result:
            observation = f"Tool execution failed: {result['error']}. I'll continue with available data."
        else:
            # Process result based on last action
            last_action = next((step for step in reversed(self.react_steps) if step.step_type == "Action"), None)
            
            if last_action and last_action.tool_used == ToolType.ARXIV_SEARCH.value:
                paper_count = len(result) if isinstance(result, list) else 0
                observation = f"Successfully found {paper_count} research papers from ArXiv. These will form the foundation of my analysis."
                if isinstance(result, list):
                    self.context["papers_found"].extend(result)
                
            elif last_action and last_action.tool_used == ToolType.EXA_SEARCH.value:
                if isinstance(result, tuple) and len(result) == 2:
                    web_results, answer = result
                    insight_count = len(web_results) if web_results else 0
                    observation = f"Gathered {insight_count} web insights. This provides current context beyond academic papers."
                    if web_results:
                        self.context["web_insights"].extend(web_results)
                else:
                    observation = "Web search completed with limited results."
                
            elif last_action and last_action.tool_used == ToolType.TOPIC_ANALYZER.value:
                topic_count = len(result) if isinstance(result, list) else 0
                observation = f"Extracted {topic_count} topics with their importance weights. This helps map the research landscape."
                if isinstance(result, list):
                    self.context["topics_extracted"].append(result)
                    
            elif last_action and last_action.tool_used == ToolType.GAP_IDENTIFIER.value:
                if isinstance(result, list):
                    gap_count = len(result)
                    self.context["gaps_identified"].extend(result)
                else:
                    gap_count = 1
                    self.context["gaps_identified"].append(result)
                observation = f"Identified {gap_count} research gaps. These represent opportunities for future investigation."
                    
            elif last_action and last_action.tool_used == ToolType.DIRECTION_SYNTHESIZER.value:
                direction_count = len(result) if isinstance(result, list) else 0
                observation = f"Synthesized {direction_count} concrete research directions with methods and expected impact."
                if isinstance(result, list):
                    self.context["research_directions"] = result
                    
            else:
                observation = f"Processed result successfully."

        # Create observation step
        observation_step = ReActStep("Observation", observation)
        self.react_steps.append(observation_step)
        logger.info(f"Observation: {observation}")
        
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
            self.context["analysis_completed"] or
            (len(self.context["papers_found"]) >= 1 and
             len(self.context["topics_extracted"]) >= 1 and
             len(self.context["gaps_identified"]) >= 1 and
             len(self.context["research_directions"]) >= 1)
        )

    async def _generate_ai_answer(self, stream_callback=None) -> str:
        """Generate comprehensive AI answer in markdown format"""
        
        await self._add_thought(
            "Now I'll synthesize all findings into a comprehensive research guide essay.",
            stream_callback
        )
        
        # Prepare data for essay generation
        papers_count = len(self.context["papers_found"])
        web_insights_count = len(self.context["web_insights"])
        
        # Flatten topics
        all_topics = []
        for topic_set in self.context["topics_extracted"]:
            if isinstance(topic_set, list):
                all_topics.extend(topic_set)
        
        # Create comprehensive essay
        essay_parts = []
        
        # Introduction
        essay_parts.append(f"# Research Guide: {self.context['original_query']}")
        essay_parts.append("")
        essay_parts.append(f"Based on my comprehensive analysis of {papers_count} academic papers and {web_insights_count} web sources, I'll provide you with a detailed research guide for **{self.context['original_query']}**.")
        essay_parts.append("")
        
        # Current Landscape
        essay_parts.append("## Current Research Landscape")
        essay_parts.append("")
        
        if all_topics:
            essay_parts.append("### Key Research Topics")
            for i, topic in enumerate(all_topics[:8], 1):
                if isinstance(topic, dict):
                    topic_name = topic.get('topic', 'Unknown')
                    weight = topic.get('weight', 'Medium')
                    subtopics = topic.get('subtopics', [])
                    
                    essay_parts.append(f"**{i}. {topic_name}** (*{weight} Priority*)")
                    if subtopics:
                        essay_parts.append(f"   - Subtopics: {', '.join(subtopics[:3])}")
                    essay_parts.append("")
        
        # Research Gaps
        if self.context["gaps_identified"]:
            essay_parts.append("## Identified Research Gaps")
            essay_parts.append("")
            essay_parts.append("Through my analysis, I've identified several key areas where current research is lacking:")
            essay_parts.append("")
            
            for i, gap in enumerate(self.context["gaps_identified"][:5], 1):
                gap_text = gap if isinstance(gap, str) else str(gap)
                essay_parts.append(f"**{i}.** {gap_text}")
                essay_parts.append("")
        
        # Research Directions
        if self.context["research_directions"]:
            essay_parts.append("## Recommended Research Directions")
            essay_parts.append("")
            essay_parts.append("Based on the identified gaps and current trends, here are my recommended research directions:")
            essay_parts.append("")
            
            for i, direction in enumerate(self.context["research_directions"], 1):
                if isinstance(direction, dict):
                    question = direction.get('question', 'Research question not specified')
                    methods = direction.get('methods', 'Methods not specified')
                    impact = direction.get('impact', 'Impact not specified')
                    
                    essay_parts.append(f"### {i}. {question}")
                    essay_parts.append(f"**Proposed Methods:** {methods}")
                    essay_parts.append(f"**Expected Impact:** {impact}")
                    essay_parts.append("")
        
        # Practical Recommendations
        essay_parts.append("## Practical Next Steps")
        essay_parts.append("")
        essay_parts.append("To advance research in this field, I recommend:")
        essay_parts.append("")
        essay_parts.append("1. **Literature Review**: Start with the academic papers I've analyzed to understand the current state of research")
        essay_parts.append("2. **Gap Analysis**: Focus on the identified research gaps as potential areas for contribution")
        essay_parts.append("3. **Methodology Development**: Consider the proposed methods in my research directions")
        essay_parts.append("4. **Collaboration**: Look for opportunities to collaborate with researchers working on complementary aspects")
        essay_parts.append("")
        
        # Conclusion
        essay_parts.append("## Conclusion")
        essay_parts.append("")
        essay_parts.append(f"The field of {self.context['original_query']} presents numerous opportunities for impactful research. By focusing on the identified gaps and following the recommended directions, researchers can make significant contributions to advancing knowledge in this area.")
        essay_parts.append("")
        
        return "\n".join(essay_parts)

    async def _synthesize_final_answer(self, ai_answer: str, stream_callback=None) -> Dict[str, Any]:
        """Synthesize the final comprehensive research guidance"""
        
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
            "ai_answer": ai_answer,  # The comprehensive markdown essay
            "research_directions": self.context["research_directions"],
            "key_topics": all_topics[:10] if all_topics else [],
            "research_gaps": self.context["gaps_identified"][:10],
            "sources": sources,
            "summary": f"Analyzed {len(self.context['papers_found'])} academic papers and {len(self.context['web_insights'])} web sources to identify {len(self.context['research_directions'])} actionable research directions.",
            "methodology": "Used ReAct framework to systematically search academic papers, analyze web insights, extract topics, identify gaps, and synthesize research directions."
        }
        
        if stream_callback:
            await stream_callback({
                "type": "final_answer",
                "content": final_answer,
                "timestamp": time.time()
            })
        
        return final_answer


# Enhanced Tools Manager with better error handling
class ToolsManager:
    """
    Manages all research tools with proper error handling
    """
    
    def __init__(self, arxiv_util, exa_util, llm_util):
        self.arxiv = arxiv_util
        self.exa = exa_util  
        self.llm = llm_util

    async def arxiv_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search ArXiv papers with timeout"""
        try:
            logger.info(f"ArXiv search: {query}")
            results = await asyncio.wait_for(self.arxiv.search(query, k), timeout=60.0)
            return results if results else []
        except asyncio.TimeoutError:
            logger.error("ArXiv search timeout")
            return []
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    async def exa_search(self, query: str, k: int = 5) -> tuple:
        """Search web using Exa with timeout"""
        try:
            logger.info(f"Exa search: {query}")
            results = await asyncio.wait_for(self.exa.search_web(query, k), timeout=60.0)
            return results if results else ([], None)
        except asyncio.TimeoutError:
            logger.error("Exa search timeout")
            return [], None
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return [], None

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