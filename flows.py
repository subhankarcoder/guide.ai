# flows.py
import logging
from pocketflow import AsyncFlow
from nodes import ResearchAgentNode, TopicWeightNode, GapAnalysisNode, DirectionSynthesizerNode

logger = logging.getLogger(__name__)

def create_research_flow():
    """
    Create the main research analysis flow with proper error handling
    """
    logger.info("Creating research analysis flow...")
    
    # Initialize nodes
    agent_node = ResearchAgentNode()
    topic_node = TopicWeightNode()
    gap_node = GapAnalysisNode() 
    direction_node = DirectionSynthesizerNode()
    
    # Connect nodes in sequence
    agent_node - "analysis" >> topic_node
    topic_node >> gap_node >> direction_node
    
    # Create the async flow
    research_flow = AsyncFlow(start=agent_node)
    
    logger.info("Research flow created successfully")
    return research_flow

# Export the main flow
ResearchFlow = create_research_flow()
