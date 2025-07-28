# main.py
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
from flows import ResearchFlow
from nodes import metrics

# Fix Windows console encoding
if sys.platform.startswith('win'):
    try:
        import codecs
        # Only apply fix if buffer attribute exists
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        else:
            # Set environment variable as fallback
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception as e:
        # If encoding fix fails, just continue
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
    title="AI Research Guide Agent",
    description="Production-ready AI agent for research analysis and direction synthesis",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global request tracking
request_count = 0
active_requests = 0

@app.middleware("http")
async def track_requests(request, call_next):
    global request_count, active_requests
    request_count += 1
    active_requests += 1
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    active_requests -= 1
    
    logger.info(f"Request {request_count} completed in {process_time:.2f}s")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.post("/research")
async def run_research(query: str, background_tasks: BackgroundTasks):
    """
    Run comprehensive research analysis with real-time streaming
    
    Args:
        query: Research query to analyze
        
    Returns:
        Server-Sent Events stream with real-time progress and final results
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter is required and cannot be empty")
    
    query = query.strip()
    logger.info(f"Starting research analysis for query: '{query}'")
    
    # Create shared context with SSE queue
    shared = {
        "query": query,
        "request_id": f"req_{int(time.time())}_{request_count}",
        "start_time": time.time(),
        "_sse_queue": asyncio.Queue(maxsize=100)  # Real-time channel
    }
    
    async def run_flow():
        """Run the research flow in background"""
        try:
            logger.info("Starting research flow execution...")
            await ResearchFlow.run_async(shared)
            logger.info("Research flow completed successfully")
        except Exception as e:
            logger.error(f"Research flow failed: {str(e)}\n{traceback.format_exc()}")
            # Push error to queue
            try:
                await shared["_sse_queue"].put(f"ERROR: Research analysis failed: {str(e)}")
            except:
                pass
        finally:
            # Signal completion
            try:
                await shared["_sse_queue"].put("__FLOW_DONE__")
            except:
                pass

    async def generate_sse_stream() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events stream with comprehensive error handling"""
        
        # Start the flow in background
        flow_task = asyncio.create_task(run_flow())
        
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting research analysis...', 'query': query}, ensure_ascii=False)}\n\n"
            
            # Stream real-time updates
            while True:
                try:
                    # Wait for next message with timeout
                    message = await asyncio.wait_for(shared["_sse_queue"].get(), timeout=30.0)
                    
                    if message == "__FLOW_DONE__":
                        # Flow completed, send final results
                        break
                    
                    # Send progress update
                    progress_data = {
                        'type': 'progress',
                        'message': message.strip(),
                        'timestamp': time.time()
                    }
                    yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()}, ensure_ascii=False)}\n\n"
                    continue
                except Exception as e:
                    logger.error(f"Error in SSE stream: {str(e)}")
                    break
            
            # Prepare comprehensive final results
            papers = shared.get("papers", [])
            topics = shared.get("topics", [])
            gaps = shared.get("gaps", [])
            directions = shared.get("directions", [])
            processing_time = time.time() - shared["start_time"]
            
            final_results = {
                "type": "final_results",
                "status": "completed",
                "query": query,
                "request_id": shared["request_id"],
                "processing_time": round(processing_time, 2),
                "metrics": {
                    "papers_analyzed": len(papers),
                    "topics_identified": len(topics),
                    "gaps_found": len(gaps),
                    "directions_generated": len(directions),
                    "success_rate": metrics.success_count / max(metrics.success_count + metrics.error_count, 1)
                },
                "results": {
                    "research_directions": directions,
                    "key_topics": topics[:10],  # Top 10 topics
                    "research_gaps": gaps[:10],  # Top 10 gaps
                    "sources": [
                        {
                            "title": paper.get("title", ""),
                            "authors": paper.get("authors", []),
                            "link": paper.get("pdf_link", "")
                        }
                        for paper in papers[:5]  # Top 5 sources
                    ]
                },
                "summary": f"Analyzed {len(papers)} research papers and generated {len(directions)} actionable research directions in {processing_time:.1f} seconds."
            }
            
            # Stream final results
            yield f"data: {json.dumps(final_results, ensure_ascii=False)}\n\n"
            logger.info(f"Research analysis completed successfully for query: '{query}' in {processing_time:.2f}s")
            
        except Exception as e:
            error_details = {
                "type": "error",
                "status": "failed", 
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__,
                "request_id": shared.get("request_id", "unknown"),
                "message": "An error occurred during research analysis. Please try again."
            }
            
            logger.error(f"Research analysis failed for query '{query}': {str(e)}\n{traceback.format_exc()}")
            yield f"data: {json.dumps(error_details, ensure_ascii=False)}\n\n"
        
        finally:
            # Clean up background task
            if not flow_task.done():
                flow_task.cancel()
                try:
                    await flow_task
                except asyncio.CancelledError:
                    pass
            
            # Send completion marker
            yield f"data: {json.dumps({'type': 'stream_end'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    gemini_status = "available" if os.getenv("GEMINI_API_KEY") else "missing_api_key"
    exa_status = "available" if os.getenv("EXA_API_KEY") else "missing_api_key"
    
    return {
        "status": "healthy",
        "version": "2.1.0",
        "timestamp": time.time(),
        "services": {
            "gemini_llm": gemini_status,
            "exa_search": exa_status
        },
        "metrics": {
            "total_requests": request_count,
            "active_requests": active_requests,
            "uptime_seconds": time.time() - metrics.start_time
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed system metrics"""
    return {
        "requests": {
            "total": request_count,
            "active": active_requests
        },
        "processing": {
            "successful_nodes": metrics.success_count,
            "failed_nodes": metrics.error_count,
            "node_timings": metrics.node_times
        },
        "uptime": time.time() - metrics.start_time
    }

@app.get("/")
async def root():
    """API information and usage guide"""
    return {
        "message": "AI Research Guide Agent - Production Ready v2.1",
        "version": "2.1.0",
        "description": "Advanced AI agent for research analysis and direction synthesis",
        "endpoints": {
            "research": {
                "method": "POST",
                "path": "/research?query=<your_research_query>",
                "description": "Analyze research query and get actionable directions",
                "response_type": "Server-Sent Events (SSE)"
            },
            "health": {
                "method": "GET", 
                "path": "/health",
                "description": "Check system health and API availability"
            },
            "metrics": {
                "method": "GET",
                "path": "/metrics", 
                "description": "Get detailed system performance metrics"
            }
        },
        "example_usage": {
            "curl": "curl -X POST 'http://localhost:8000/research?query=quantum+machine+learning' -H 'Accept: text/event-stream'",
            "javascript": "const eventSource = new EventSource('/research?query=quantum+machine+learning'); eventSource.onmessage = (event) => console.log(JSON.parse(event.data));"
        }
    }

async def run_cli(query: str):
    """Enhanced CLI mode for testing and development"""
    logger.info(f"CLI Mode - Processing query: '{query}'")
    
    shared = {
        "query": query, 
        "start_time": time.time(),
        "_sse_queue": asyncio.Queue(maxsize=100)
    }
    flow = ResearchFlow
    
    try:
        print(f"\n{'='*60}")
        print(f"AI RESEARCH GUIDE AGENT - CLI MODE")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Run the analysis
        await flow.run_async(shared)
        
        # Display results
        papers = shared.get("papers", [])
        topics = shared.get("topics", [])
        gaps = shared.get("gaps", [])
        directions = shared.get("directions", [])
        processing_time = time.time() - shared["start_time"]
        
        print(f"ANALYSIS COMPLETED IN {processing_time:.2f}s")
        print(f"{'='*60}")
        print(f"SUMMARY:")
        print(f"   -  Papers Analyzed: {len(papers)}")
        print(f"   -  Topics Identified: {len(topics)}")
        print(f"   -  Research Gaps: {len(gaps)}")
        print(f"   -  Research Directions: {len(directions)}")
        print(f"{'='*60}")
        
        if directions:
            print(f"\nRESEARCH DIRECTIONS:")
            for i, direction in enumerate(directions, 1):
                print(f"\n{i}. {direction.get('question', 'Unknown question')}")
                print(f"   Methods: {direction.get('methods', 'Not specified')}")
                print(f"   Impact: {direction.get('impact', 'Not specified')}")
        
        if topics:
            print(f"\nKEY TOPICS:")
            for topic in topics[:5]:
                print(f"   -  {topic.get('topic', 'Unknown')} ({topic.get('weight', 'Unknown')} priority)")
        
        if papers:
            print(f"\nSOURCES:")
            for paper in papers[:3]:
                print(f"   -  {paper.get('title', 'Unknown title')}")
                print(f"     Authors: {', '.join(paper.get('authors', []))}")
                print(f"     Link: {paper.get('pdf_link', 'No link')}")
        
        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        
    except Exception as e:
        logger.error(f"CLI error: {str(e)}\n{traceback.format_exc()}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Check environment setup
    gemini_key = os.getenv("GEMINI_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")
    
    if not gemini_key:
        print("Warning: GEMINI_API_KEY not set. Set with: export GEMINI_API_KEY='your_key_here'")
    
    if not exa_key:
        print("Warning: EXA_API_KEY not set. Set with: export EXA_API_KEY='your_key_here'")
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI mode
        query = input("Enter your research query: ") if len(sys.argv) <= 2 else " ".join(sys.argv[2:])
        asyncio.run(run_cli(query))
    else:
        # Web server mode
        print("Starting AI Research Guide Agent - Production Server v2.1")
        print("="*60)
        print("Server: http://localhost:8000")
        print("API Docs: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/health")
        print("Metrics: http://localhost:8000/metrics")
        print("="*60)
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            access_log=True
        )
