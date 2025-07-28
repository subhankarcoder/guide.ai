# updated_main.py
import asyncio
import json
import uvicorn
import os
import logging
import traceback
import time
import sys
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import the new ReAct agent and existing utilities
from research_agent import ResearchGuideAgent, ToolsManager
from utils import llm, arxiv, exa

# Fix Windows console encoding
if sys.platform.startswith('win'):
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        else:
            os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception as e:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research_agent.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app with metadata
app = FastAPI(
    title="AI Research Guide Agent - ReAct Framework",
    description="Advanced AI research agent using ReAct framework for intelligent research guidance",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the enhanced research agent
tools_manager = ToolsManager(arxiv, exa, llm)
research_agent = ResearchGuideAgent(tools_manager)

# Global metrics
request_count = 0
active_requests = 0
start_time = time.time()

@app.middleware("http")
async def track_requests(request, call_next):
    global request_count, active_requests
    request_count += 1
    active_requests += 1
    
    start_time_req = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time_req
    
    active_requests -= 1
    
    logger.info(f"Request {request_count} completed in {process_time:.2f}s")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.post("/research")
async def run_research_analysis(query: str, background_tasks: BackgroundTasks):
    """
    Run comprehensive research analysis using ReAct framework with real-time streaming
    
    Args:
        query: Research query to analyze
        
    Returns:
        Server-Sent Events stream with ReAct steps and final research guidance
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter is required and cannot be empty")
    
    query = query.strip()
    logger.info(f"Starting ReAct research analysis for query: '{query}'")
    
    # Create shared context for streaming
    stream_queue = asyncio.Queue(maxsize=100)
    request_id = f"req_{int(time.time())}_{request_count}"
    
    async def stream_callback(data: dict):
        """Callback to stream ReAct steps in real-time"""
        try:
            if not stream_queue.full():
                await stream_queue.put(data)
        except Exception as e:
            logger.error(f"Error in stream callback: {e}")

    async def run_research_agent():
        """Run the research agent in background"""
        try:
            logger.info("Starting ReAct research agent...")
            result = await research_agent.process_query(query, stream_callback)
            
            # Signal completion with final result
            await stream_queue.put({
                "type": "completion",
                "result": result,
                "timestamp": time.time()
            })
            
            logger.info("ReAct research agent completed successfully")
        except Exception as e:
            logger.error(f"Research agent failed: {str(e)}\n{traceback.format_exc()}")
            await stream_queue.put({
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            })
        finally:
            await stream_queue.put({"type": "stream_end"})

    async def generate_react_stream() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events stream with ReAct steps"""
        
        # Start the research agent
        agent_task = asyncio.create_task(run_research_agent())
        
        try:
            # Send initial status
            initial_data = {
                'type': 'initialization',
                'message': f'Starting ReAct research analysis for: "{query}"',
                'query': query,
                'request_id': request_id,
                'timestamp': time.time()
            }
            yield f"data: {json.dumps(initial_data, ensure_ascii=False)}\n\n"
            
            # Stream ReAct steps in real-time
            final_result = None
            while True:
                try:
                    # Wait for next step with timeout
                    step_data = await asyncio.wait_for(stream_queue.get(), timeout=30.0)
                    
                    if step_data.get("type") == "stream_end":
                        break
                    elif step_data.get("type") == "completion":
                        final_result = step_data.get("result")
                        break
                    elif step_data.get("type") == "error":
                        error_data = {
                            "type": "error",
                            "status": "failed",
                            "query": query,
                            "error": step_data.get("error"),
                            "request_id": request_id,
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                        return
                    else:
                        # Stream the ReAct step
                        yield f"data: {json.dumps(step_data, ensure_ascii=False)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    heartbeat = {
                        'type': 'heartbeat',
                        'message': 'Research analysis in progress...',
                        'timestamp': time.time()
                    }
                    yield f"data: {json.dumps(heartbeat, ensure_ascii=False)}\n\n"
                    continue
                except Exception as e:
                    logger.error(f"Error in ReAct stream: {str(e)}")
                    break
            
            # Send final comprehensive results
            if final_result:
                processing_time = final_result.get("processing_time", 0)
                react_steps = final_result.get("react_steps", [])
                final_answer = final_result.get("final_answer", {})
                
                completion_data = {
                    "type": "final_results",
                    "status": "completed",
                    "query": query,
                    "request_id": request_id,
                    "processing_time": round(processing_time, 2),
                    "react_steps_count": len(react_steps),
                    "research_guidance": final_answer,
                    "metrics": {
                        "papers_analyzed": len(final_answer.get("sources", [])),
                        "research_directions": len(final_answer.get("research_directions", [])),
                        "key_topics": len(final_answer.get("key_topics", [])),
                        "research_gaps": len(final_answer.get("research_gaps", [])),
                        "react_iterations": len([s for s in react_steps if s.get("step_type") == "Thought"])
                    },
                    "summary": final_answer.get("summary", "Research analysis completed"),
                    "timestamp": time.time()
                }
                
                yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
                logger.info(f"ReAct research analysis completed for query: '{query}' in {processing_time:.2f}s")
            
        except Exception as e:
            error_details = {
                "type": "error",
                "status": "failed",
                "query": query,
                "error": str(e),
                "request_id": request_id,
                "timestamp": time.time()
            }
            logger.error(f"ReAct research analysis failed: {str(e)}\n{traceback.format_exc()}")
            yield f"data: {json.dumps(error_details, ensure_ascii=False)}\n\n"
        
        finally:
            # Clean up background task
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass
            
            # Send stream end marker
            yield f"data: {json.dumps({'type': 'stream_complete', 'timestamp': time.time()}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_react_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with ReAct agent status"""
    gemini_status = "available" if os.getenv("GEMINI_API_KEY") else "missing_api_key"
    exa_status = "available" if os.getenv("EXA_API_KEY") else "missing_api_key"
    
    return {
        "status": "healthy",
        "version": "3.0.0",
        "framework": "ReAct (Reasoning, Acting, Observing)",
        "timestamp": time.time(),
        "services": {
            "gemini_llm": gemini_status,
            "exa_search": exa_status,
            "arxiv_search": "available",
            "react_agent": "initialized"
        },
        "capabilities": [
            "Dynamic tool selection",
            "Real-time reasoning streaming", 
            "Research gap identification",
            "Topic weight analysis",
            "Research direction synthesis"
        ],
        "metrics": {
            "total_requests": request_count,
            "active_requests": active_requests,
            "uptime_seconds": time.time() - start_time
        }
    }

@app.get("/tools")
async def list_available_tools():
    """List all available research tools and their descriptions"""
    tools_info = {}
    for tool_type, info in research_agent.tool_descriptions.items():
        tools_info[tool_type.value] = {
            "description": info["description"],
            "input_format": info["input_format"],
            "when_to_use": info["when_to_use"]
        }
    
    return {
        "available_tools": tools_info,
        "total_tools": len(tools_info),
        "framework": "ReAct - Reasoning, Acting, Observing"
    }

@app.get("/metrics")
async def get_detailed_metrics():
    """Get detailed system and ReAct framework metrics"""
    return {
        "system": {
            "total_requests": request_count,
            "active_requests": active_requests,
            "uptime_seconds": time.time() - start_time
        },
        "react_framework": {
            "max_iterations": research_agent.max_iterations,
            "tool_count": len(research_agent.tool_descriptions),
            "supported_steps": ["Thought", "Action", "Observation"]
        },
        "tools_status": {
            "arxiv_search": "operational",
            "exa_search": "operational" if os.getenv("EXA_API_KEY") else "missing_key",
            "llm_analysis": "operational" if os.getenv("GEMINI_API_KEY") else "missing_key"
        }
    }

@app.get("/")
async def enhanced_root():
    """Enhanced API information with ReAct framework details"""
    return {
        "message": "AI Research Guide Agent - ReAct Framework v3.0",
        "version": "3.0.0", 
        "framework": "ReAct (Reasoning, Acting, Observing)",
        "description": "Intelligent research agent that dynamically reasons about research queries, selects appropriate tools, and provides comprehensive research guidance with proper citations.",
        "key_features": [
            "ReAct framework for intelligent reasoning",
            "Dynamic tool selection based on context",
            "Real-time streaming of reasoning steps",
            "Comprehensive research paper analysis",
            "Research gap identification",
            "Actionable research direction synthesis"
        ],
        "endpoints": {
            "research": {
                "method": "POST",
                "path": "/research?query=<your_research_query>",
                "description": "Analyze research query using ReAct framework with real-time streaming",
                "response_type": "Server-Sent Events (SSE)",
                "stream_types": ["initialization", "reasoning", "action", "observation", "final_results"]
            },
            "health": {
                "method": "GET",
                "path": "/health", 
                "description": "Check system health and ReAct agent status"
            },
            "tools": {
                "method": "GET",
                "path": "/tools",
                "description": "List available research tools and their usage"
            },
            "metrics": {
                "method": "GET", 
                "path": "/metrics",
                "description": "Get detailed system and ReAct framework metrics"
            }
        },
        "example_usage": {
            "curl": "curl -X POST 'http://localhost:8000/research?query=quantum+machine+learning+applications' -H 'Accept: text/event-stream'",
            "javascript": "const eventSource = new EventSource('/research?query=quantum+machine+learning'); eventSource.onmessage = (event) => { const data = JSON.parse(event.data); console.log(data.step + ':', data.content); };",
            "python": "import requests; response = requests.post('http://localhost:8000/research?query=your+query', stream=True); for line in response.iter_lines(): print(line.decode())"
        },
        "react_workflow": {
            "1": "Thought: Agent reasons about the query and current context",
            "2": "Action: Agent selects and executes appropriate research tools", 
            "3": "Observation: Agent analyzes results and updates context",
            "4": "Repeat: Continue until research goal is achieved",
            "5": "Answer: Synthesize comprehensive research guidance"
        }
    }

async def run_enhanced_cli(query: str):
    """Enhanced CLI mode with ReAct framework visualization"""
    print(f"\n{'='*80}")
    print(f"AI RESEARCH GUIDE AGENT - ReAct FRAMEWORK v3.0")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Framework: ReAct (Reasoning, Acting, Observing)")
    print(f"{'='*80}\n")
    
    # Stream callback for CLI
    step_count = 0
    async def cli_stream_callback(data: dict):
        nonlocal step_count
        step_count += 1
        
        step_type = data.get("type", "unknown")
        
        if step_type == "reasoning":
            print(f"\n🤔 THOUGHT #{step_count}:")
            print(f"   {data.get('content', '')}")
            
        elif step_type == "action":
            print(f"\n🔧 ACTION #{step_count}:")
            print(f"   Tool: {data.get('tool', '')}")
            print(f"   Rationale: {data.get('rationale', '')}")
            print(f"   Input: {json.dumps(data.get('input', {}), indent=6)}")
            
        elif step_type == "observation":
            print(f"\n👁️  OBSERVATION #{step_count}:")
            print(f"   {data.get('content', '')}")
            
        elif step_type == "final_answer":
            print(f"\n{'='*80}")
            print(f"🎯 FINAL RESEARCH GUIDANCE")
            print(f"{'='*80}")
            
            final_data = data.get("content", {})
            directions = final_data.get("research_directions", [])
            topics = final_data.get("key_topics", [])
            gaps = final_data.get("research_gaps", [])
            sources = final_data.get("sources", [])
            
            print(f"\n📋 SUMMARY: {final_data.get('summary', '')}")
            
            if directions:
                print(f"\n🎯 RESEARCH DIRECTIONS ({len(directions)}):")
                for i, direction in enumerate(directions, 1):
                    print(f"\n   {i}. {direction.get('question', 'Unknown question')}")
                    print(f"      🔬 Methods: {direction.get('methods', 'Not specified')}")
                    print(f"      💫 Impact: {direction.get('impact', 'Not specified')}")
            
            if topics:
                print(f"\n📊 KEY TOPICS ({len(topics)}):")
                for topic in topics[:8]:
                    if isinstance(topic, dict):
                        weight_emoji = {"High": "🔥", "Medium": "⚡", "Low": "💡"}.get(topic.get("weight", ""), "📌")
                        print(f"   {weight_emoji} {topic.get('topic', 'Unknown')} ({topic.get('weight', 'Unknown')} priority)")
            
            if gaps:
                print(f"\n🔍 RESEARCH GAPS ({len(gaps)}):")
                for i, gap in enumerate(gaps[:5], 1):
                    gap_text = gap if isinstance(gap, str) else str(gap)
                    print(f"   {i}. {gap_text[:100]}{'...' if len(gap_text) > 100 else ''}")
            
            if sources:
                print(f"\n📚 SOURCES ({len(sources)}):")
                for i, source in enumerate(sources[:5], 1):
                    if source.get("type") == "academic_paper":
                        print(f"   {i}. 📄 {source.get('title', 'Unknown title')}")
                        authors = source.get('authors', [])
                        if authors:
                            print(f"      👥 Authors: {', '.join(authors[:3])}")
                        if source.get('link'):
                            print(f"      🔗 {source.get('link')}")
                    else:
                        print(f"   {i}. 🌐 {source.get('title', 'Unknown title')}")
                        if source.get('url'):
                            print(f"      🔗 {source.get('url')}")
    
    try:
        # Run the research agent
        result = await research_agent.process_query(query, cli_stream_callback)
        
        processing_time = result.get("processing_time", 0)
        react_steps = result.get("react_steps", [])
        
        print(f"\n{'='*80}")
        print(f"✅ ANALYSIS COMPLETED")
        print(f"{'='*80}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"ReAct Steps: {len(react_steps)} total")
        thoughts = len([s for s in react_steps if s.get("step_type") == "Thought"])
        actions = len([s for s in react_steps if s.get("step_type") == "Action"])
        observations = len([s for s in react_steps if s.get("step_type") == "Observation"])
        print(f"   - Thoughts: {thoughts}")
        print(f"   - Actions: {actions}")
        print(f"   - Observations: {observations}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Check environment setup
    gemini_key = os.getenv("GEMINI_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")
    
    missing_keys = []
    if not gemini_key:
        missing_keys.append("GEMINI_API_KEY")
    if not exa_key:
        missing_keys.append("EXA_API_KEY")
    
    if missing_keys:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_keys)}")
        print("Set them with: export VARIABLE_NAME='your_key_here'")
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Enhanced CLI mode
        if len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
        else:
            query = input("🔍 Enter your research query: ")
        
        asyncio.run(run_enhanced_cli(query))
    else:
        # Web server mode
        print("🚀 Starting AI Research Guide Agent - ReAct Framework v3.0")
        print("="*80)
        print("🌐 Server: http://localhost:8000")
        print("📖 API Docs: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("🛠️  Tools Info: http://localhost:8000/tools")
        print("📊 Metrics: http://localhost:8000/metrics")
        print("="*80)
        print("🧠 Framework: ReAct (Reasoning, Acting, Observing)")
        print("✨ Features: Dynamic tool selection, Real-time reasoning streaming")
        print("="*80)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )