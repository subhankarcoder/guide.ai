# utils/llm.py
import os
import asyncio
import json
import re
import logging
import functools
from typing import AsyncGenerator, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# Import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)

# Configuration
MODEL = "gemini-1.5-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
if GEMINI_AVAILABLE and API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        client = genai.GenerativeModel(MODEL)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        client = None
else:
    client = None
    if not GEMINI_AVAILABLE:
        logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")
    if not API_KEY:
        logger.warning("GEMINI_API_KEY not found - LLM calls will use fallback responses")

# Regex for reasoning token extraction
RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from text"""
    if not text:
        return ""
    return RE_THINK.sub("", text).strip()

def _ensure_text(response) -> str:
    """Ensure Gemini response is converted to string"""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        return response.candidates[0].content.parts[0].text
    return str(response)

async def chat(
    messages: List[Dict[str, str]],
    tools: List[Dict] | None = None,
    stream: bool = False,
    max_retries: int = 3,
) -> str:
    """
    Async chat completion using Google Gemini
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: Not used with Gemini (for compatibility)
        stream: Whether to stream response
        max_retries: Number of retry attempts
    
    Returns:
        String response from Gemini
    """
    if not client:
        logger.error("Gemini client not available")
        return json.dumps({
            "error": "Gemini client not initialized", 
            "fallback": True,
            "message": "Please set GEMINI_API_KEY environment variable"
        })
    
    # Convert messages to Gemini format
    if len(messages) == 1 and messages[0].get("role") == "user":
        prompt = messages[0]["content"]
    else:
        # Combine multiple messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n\n".join(prompt_parts)
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Gemini API call attempt {attempt + 1}/{max_retries}")
            
            if stream:
                # Streaming response
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: client.generate_content(prompt, stream=True)
                )
                
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        full_response += chunk.text
                
                logger.debug("Gemini streaming call successful")
                return full_response
            else:
                # Non-streaming response
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.generate_content(prompt)
                )
                
                result = _ensure_text(response)
                logger.debug("Gemini API call successful")
                return result
                
        except Exception as e:
            logger.error(f"Gemini API call attempt {attempt + 1} failed: {str(e)}")
            
            if attempt == max_retries - 1:
                # Last attempt failed, return fallback
                logger.error("All Gemini API call attempts failed")
                return json.dumps({
                    "error": f"Gemini API failed after {max_retries} attempts: {str(e)}",
                    "fallback": True
                })
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
    
    # This shouldn't be reached, but just in case
    return json.dumps({"error": "Unexpected error in chat function", "fallback": True})

async def stream_llm(
    messages: List[Dict[str, str]], 
    tools: List[Dict] | None = None
) -> AsyncGenerator[str, None]:
    """
    Stream LLM responses with error handling
    
    Args:
        messages: List of message dicts
        tools: Not used with Gemini
    
    Yields:
        String chunks from the response
    """
    if not client:
        logger.error("Cannot stream - Gemini client not initialized")
        yield json.dumps({"error": "No API key provided"})
        return
    
    try:
        # Convert messages to prompt
        if len(messages) == 1 and messages[0].get("role") == "user":
            prompt = messages[0]["content"]
        else:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt = "\n\n".join(prompt_parts)
        
        # Generate streaming response
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.generate_content(prompt, stream=True)
        )
        
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.error(f"Gemini streaming failed: {str(e)}")
        yield json.dumps({"error": f"Streaming failed: {str(e)}"})

def safe_json_parse(text: str, fallback_value=None):
    """
    Safely parse JSON with reasoning token handling
    
    Args:
        text: Raw text that may contain JSON
        fallback_value: Value to return if parsing fails
    
    Returns:
        Parsed JSON object or fallback value
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for JSON parsing")
        return fallback_value or {}
    
    # Remove reasoning tokens
    cleaned = _strip_think(text)
    
    # Try to extract JSON from code blocks
    json_match = re.search(r'``````', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1).strip()
    
    try:
        parsed = json.loads(cleaned)
        logger.debug("Successfully parsed JSON")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}. Cleaned content: {cleaned[:200]}...")
        
        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            parsed = json.loads(fixed)
            logger.debug("Successfully parsed JSON after fixing trailing commas")
            return parsed
        except:
            pass
        
        return fallback_value or {}
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return fallback_value or {}
