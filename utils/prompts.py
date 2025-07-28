# utils/prompts.py

SYSTEM_REACT = """You are an AI research-guide agent.
Tools available: {tool_desc}

IMPORTANT: Always respond with either:
1. Thought: <your reasoning>
2. Action: <tool_name>[JSON]
OR when finished:
1. Answer: <final answer with citations>

Keep responses concise and focused."""

TOPIC_WEIGHT_PROMPT = """
Analyze the research paper summary below and extract topics with their importance weights.

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON, no other text
- No reasoning tokens or explanations
- Use this exact format

REQUIRED JSON FORMAT:
[
  {{
    "topic": "specific research topic name",
    "weight": "High|Medium|Low", 
    "subtopics": ["subtopic1", "subtopic2"]
  }}
]

PAPER SUMMARY:
{summary}

JSON OUTPUT:"""

GAP_PROMPT = """
Analyze the research paper and identify specific research gaps.

PAPER TITLE: {title}

PAPER SUMMARY:
{summary}

INSTRUCTIONS:
- Identify 3-5 specific research gaps
- Focus on technical limitations, missing methods, or unexplored areas
- Be concrete and actionable
- Return as bullet points

RESEARCH GAPS:"""

DIRECTION_PROMPT = """
Based on the identified topics and research gaps, propose concrete research directions.

TOPICS IDENTIFIED:
{topics}

RESEARCH GAPS:
{gaps}

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON, no other text
- No reasoning tokens or explanations
- Propose 3-5 specific research directions

REQUIRED JSON FORMAT:
[
  {{
    "question": "Specific research question",
    "methods": "Proposed methodology or approach", 
    "impact": "Expected impact and significance"
  }}
]

JSON OUTPUT:"""

ESSAY_PROMPT = """
You are a research assistant. Based on the provided research analysis, write a comprehensive, human-like essay in markdown format.

The essay should summarize the findings, including the key topics, research gaps, and future directions.

**IMPORTANT**:
- The essay should be well-structured with a clear introduction, body, and conclusion.
- Use markdown for formatting (e.g., headings, bold text, lists).
- The tone should be professional and informative.
- Do not include any JSON or code in the output.

**RESEARCH ANALYSIS:**

**Key Topics:**
{topics}

**Research Gaps:**
{gaps}

**Future Research Directions:**
{directions}

**ESSAY:**
"""


# Fallback prompts for when APIs fail
FALLBACK_TOPICS = [
    {
        "topic": "Machine Learning Applications",
        "weight": "High",
        "subtopics": ["Neural Networks", "Deep Learning", "Algorithm Optimization"]
    }
]

FALLBACK_DIRECTIONS = [
    {
        "question": "How can we improve the efficiency of current machine learning models?",
        "methods": "Develop novel optimization algorithms and architectural improvements",
        "impact": "Reduced computational costs and improved accessibility of ML models"
    }
]