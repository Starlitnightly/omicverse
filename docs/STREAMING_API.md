# OmicVerse Streaming API Guide

## Overview

The OmicVerse agent provides real-time streaming capabilities, allowing you to receive incremental updates as code is generated, executed, and results are produced. This enables responsive user interfaces and better visibility into the agent's reasoning process.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Stream Event Types](#stream-event-types)
3. [Usage Examples](#usage-examples)
4. [Type Hints](#type-hints)
5. [Error Handling](#error-handling)
6. [Provider Support](#provider-support)
7. [Performance Considerations](#performance-considerations)
8. [Best Practices](#best-practices)

---

## Quick Start

### Basic Streaming Example

```python
import asyncio
from omicverse.utils.smart_agent import OmicVerseAgent
import scanpy as sc

async def stream_analysis():
    # Initialize agent
    agent = OmicVerseAgent(
        model="gpt-4o",
        temperature=0.1
    )

    # Load data
    adata = sc.datasets.pbmc3k()

    # Stream query execution
    async for event in agent.stream_async(
        "Preprocess and cluster my single-cell data",
        adata=adata
    ):
        # Handle each event
        event_type = event.get("event_type")
        data = event.get("data")

        if event_type == "llm_chunk":
            print(data["chunk"], end="", flush=True)
        elif event_type == "code":
            print(f"\n\nGenerated code:\n{data['code']}")
        elif event_type == "result":
            print(f"\n\nExecution completed: {data['success']}")

# Run the async function
asyncio.run(stream_analysis())
```

---

## Stream Event Types

The streaming API emits a sequence of events as the agent processes your query. Each event is a dictionary with `event_type` and `data` fields.

### 1. Skill Match Event

**When:** After relevant skills are identified for the query

**Structure:**
```python
{
    "event_type": "skill_match",
    "data": {
        "matched_skills": List[str],        # List of skill names
        "skill_scores": Dict[str, float],   # Relevance scores (if available)
        "method": str                       # "llm" or "algorithmic"
    }
}
```

**Example:**
```python
{
    "event_type": "skill_match",
    "data": {
        "matched_skills": [
            "single-preprocessing",
            "single-clustering"
        ],
        "skill_scores": {
            "single-preprocessing": 0.95,
            "single-clustering": 0.88
        },
        "method": "llm"
    }
}
```

**Usage:**
```python
if event["event_type"] == "skill_match":
    skills = event["data"]["matched_skills"]
    print(f"Using skills: {', '.join(skills)}")
```

---

### 2. LLM Chunk Event

**When:** As the LLM generates each token/chunk of the response

**Structure:**
```python
{
    "event_type": "llm_chunk",
    "data": {
        "chunk": str,              # Current text chunk
        "cumulative_text": str     # All text so far
    }
}
```

**Example:**
```python
{
    "event_type": "llm_chunk",
    "data": {
        "chunk": "import ",
        "cumulative_text": "import "
    }
}
```

**Usage:**
```python
if event["event_type"] == "llm_chunk":
    # Real-time display of LLM output
    print(event["data"]["chunk"], end="", flush=True)
```

---

### 3. Code Event

**When:** After code is extracted and validated from LLM response

**Structure:**
```python
{
    "event_type": "code",
    "data": {
        "code": str,          # Extracted Python code
        "validated": bool,    # Whether AST validation passed
        "method": str         # "fenced" or "inline"
    }
}
```

**Example:**
```python
{
    "event_type": "code",
    "data": {
        "code": "import omicverse as ov\n\nadata = ov.pp.qc(adata)\nadata = ov.pp.normalize(adata)",
        "validated": True,
        "method": "fenced"
    }
}
```

**Usage:**
```python
if event["event_type"] == "code":
    code = event["data"]["code"]
    print(f"\n\nExecuting code:\n{code}\n")

    # Optional: Display with syntax highlighting
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import TerminalFormatter

    highlighted = highlight(code, PythonLexer(), TerminalFormatter())
    print(highlighted)
```

---

### 4. Execution Event

**When:** During code execution (optional, if enabled)

**Structure:**
```python
{
    "event_type": "execution",
    "data": {
        "status": str,        # "started" or "progress"
        "message": str,       # Status message
        "line_number": int    # Current line being executed (if available)
    }
}
```

**Example:**
```python
{
    "event_type": "execution",
    "data": {
        "status": "progress",
        "message": "Normalizing data...",
        "line_number": 3
    }
}
```

---

### 5. Result Event

**When:** After code execution completes (success or failure)

**Structure:**
```python
{
    "event_type": "result",
    "data": {
        "success": bool,              # Whether execution succeeded
        "result": Dict[str, Any],     # Execution result (if success)
        "error": str,                 # Error message (if failure)
        "execution_time": float       # Time in seconds
    }
}
```

**Example (Success):**
```python
{
    "event_type": "result",
    "data": {
        "success": True,
        "result": {
            "adata": AnnData_object,
            "namespace": {...}
        },
        "execution_time": 2.5
    }
}
```

**Example (Failure):**
```python
{
    "event_type": "result",
    "data": {
        "success": False,
        "error": "NameError: name 'undefined_variable' is not defined",
        "execution_time": 0.1
    }
}
```

**Usage:**
```python
if event["event_type"] == "result":
    if event["data"]["success"]:
        print(f"✓ Success in {event['data']['execution_time']:.2f}s")
        adata = event["data"]["result"]["adata"]
    else:
        print(f"✗ Error: {event['data']['error']}")
```

---

### 6. Reflection Event

**When:** When the agent reflects on an error and generates corrected code

**Structure:**
```python
{
    "event_type": "reflection",
    "data": {
        "iteration": int,              # Reflection iteration number
        "original_error": str,         # Original error message
        "reflection_prompt": str,      # Prompt sent to LLM
        "corrected_code": str          # New code attempt
    }
}
```

**Example:**
```python
{
    "event_type": "reflection",
    "data": {
        "iteration": 1,
        "original_error": "NameError: name 'ov' is not defined",
        "reflection_prompt": "The code failed with: NameError...",
        "corrected_code": "import omicverse as ov\n..."
    }
}
```

**Usage:**
```python
if event["event_type"] == "reflection":
    iteration = event["data"]["iteration"]
    print(f"\nReflecting on error (attempt {iteration})...")
    print(f"Error: {event['data']['original_error']}")
```

---

### 7. Review Event

**When:** When the agent reviews the result for quality

**Structure:**
```python
{
    "event_type": "review",
    "data": {
        "satisfactory": bool,         # Whether result is satisfactory
        "issues": List[str],          # List of identified issues
        "suggestions": List[str],     # Suggestions for improvement
        "confidence": float           # Confidence score (0-1)
    }
}
```

**Example:**
```python
{
    "event_type": "review",
    "data": {
        "satisfactory": True,
        "issues": [],
        "suggestions": ["Consider adding PCA for visualization"],
        "confidence": 0.92
    }
}
```

---

### 8. Usage Event

**When:** At the end of streaming, reporting token usage

**Structure:**
```python
{
    "event_type": "usage",
    "data": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "model": str
    }
}
```

**Example:**
```python
{
    "event_type": "usage",
    "data": {
        "prompt_tokens": 1234,
        "completion_tokens": 567,
        "total_tokens": 1801,
        "model": "openai/gpt-4o-2024-11-20"
    }
}
```

**Usage:**
```python
if event["event_type"] == "usage":
    usage = event["data"]
    print(f"\nTokens used: {usage['total_tokens']} "
          f"(prompt: {usage['prompt_tokens']}, "
          f"completion: {usage['completion_tokens']})")
```

---

### 9. Error Event

**When:** When an unrecoverable error occurs

**Structure:**
```python
{
    "event_type": "error",
    "data": {
        "error_type": str,        # Error class name
        "message": str,           # Error message
        "traceback": str,         # Full traceback
        "recoverable": bool       # Whether error is recoverable
    }
}
```

**Example:**
```python
{
    "event_type": "error",
    "data": {
        "error_type": "RateLimitError",
        "message": "Rate limit exceeded",
        "traceback": "Traceback (most recent call last)...",
        "recoverable": True
    }
}
```

---

## Usage Examples

### Example 1: Simple Progress Display

```python
import asyncio
from omicverse.utils.smart_agent import OmicVerseAgent

async def simple_stream():
    agent = OmicVerseAgent(model="gpt-4o")

    async for event in agent.stream_async("Cluster my cells", adata):
        if event["event_type"] == "llm_chunk":
            print(event["data"]["chunk"], end="", flush=True)
        elif event["event_type"] == "result":
            if event["data"]["success"]:
                print("\n✓ Completed successfully")
            else:
                print(f"\n✗ Error: {event['data']['error']}")

asyncio.run(simple_stream())
```

---

### Example 2: Detailed Event Handling

```python
import asyncio
from typing import Dict, Any

class StreamHandler:
    def __init__(self):
        self.events = []
        self.current_code = None

    async def handle_event(self, event: Dict[str, Any]):
        """Handle each streaming event"""
        self.events.append(event)

        event_type = event["event_type"]
        data = event["data"]

        if event_type == "skill_match":
            self._handle_skill_match(data)
        elif event_type == "llm_chunk":
            self._handle_llm_chunk(data)
        elif event_type == "code":
            self._handle_code(data)
        elif event_type == "result":
            self._handle_result(data)
        elif event_type == "reflection":
            self._handle_reflection(data)
        elif event_type == "usage":
            self._handle_usage(data)
        elif event_type == "error":
            self._handle_error(data)

    def _handle_skill_match(self, data):
        skills = data.get("matched_skills", [])
        print(f"\n📚 Using skills: {', '.join(skills)}")

    def _handle_llm_chunk(self, data):
        print(data["chunk"], end="", flush=True)

    def _handle_code(self, data):
        self.current_code = data["code"]
        print(f"\n\n💻 Generated code ({len(data['code'])} chars)")
        print("─" * 50)
        print(data["code"])
        print("─" * 50)

    def _handle_result(self, data):
        if data["success"]:
            time = data.get("execution_time", 0)
            print(f"\n✓ Execution completed in {time:.2f}s")
        else:
            print(f"\n✗ Execution failed: {data['error']}")

    def _handle_reflection(self, data):
        iteration = data["iteration"]
        print(f"\n🔄 Reflecting on error (attempt {iteration})...")

    def _handle_usage(self, data):
        print(f"\n📊 Tokens: {data['total_tokens']} total "
              f"({data['prompt_tokens']} prompt + "
              f"{data['completion_tokens']} completion)")

    def _handle_error(self, data):
        print(f"\n❌ Error: {data['message']}")
        if data.get("recoverable"):
            print("   (Attempting recovery...)")

async def main():
    agent = OmicVerseAgent(model="gpt-4o")
    handler = StreamHandler()

    async for event in agent.stream_async("Analyze my data", adata):
        await handler.handle_event(event)

    print(f"\nTotal events: {len(handler.events)}")

asyncio.run(main())
```

---

### Example 3: Web UI Integration (FastAPI + WebSockets)

```python
from fastapi import FastAPI, WebSocket
from omicverse.utils.smart_agent import OmicVerseAgent
import json

app = FastAPI()
agent = OmicVerseAgent(model="gpt-4o")

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Receive query from client
    data = await websocket.receive_json()
    query = data["query"]
    adata = load_user_data(data["adata_id"])

    try:
        # Stream events to client
        async for event in agent.stream_async(query, adata):
            # Send event as JSON
            await websocket.send_json(event)

    except Exception as e:
        await websocket.send_json({
            "event_type": "error",
            "data": {"message": str(e)}
        })
    finally:
        await websocket.close()
```

**Client-side (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => {
    ws.send(JSON.stringify({
        query: "Cluster my cells",
        adata_id: "user123_pbmc3k"
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch(data.event_type) {
        case 'llm_chunk':
            document.getElementById('output').innerText += data.data.chunk;
            break;
        case 'code':
            document.getElementById('code').innerText = data.data.code;
            break;
        case 'result':
            if (data.data.success) {
                showSuccess();
            } else {
                showError(data.data.error);
            }
            break;
    }
};
```

---

### Example 4: CLI Progress Bar

```python
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from omicverse.utils.smart_agent import OmicVerseAgent

async def stream_with_progress():
    console = Console()
    agent = OmicVerseAgent(model="gpt-4o")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Create tasks for each stage
        skill_task = progress.add_task("Matching skills...", total=None)
        code_task = progress.add_task("Generating code...", total=None)
        exec_task = progress.add_task("Executing code...", total=None)

        async for event in agent.stream_async("Cluster my data", adata):
            event_type = event["event_type"]

            if event_type == "skill_match":
                progress.update(skill_task, completed=True)
                skills = event["data"]["matched_skills"]
                console.print(f"✓ Matched {len(skills)} skills")

            elif event_type == "code":
                progress.update(code_task, completed=True)
                code_lines = event["data"]["code"].count('\n') + 1
                console.print(f"✓ Generated {code_lines} lines of code")

            elif event_type == "result":
                progress.update(exec_task, completed=True)
                if event["data"]["success"]:
                    console.print("✓ Execution completed", style="green")
                else:
                    console.print("✗ Execution failed", style="red")

asyncio.run(stream_with_progress())
```

---

### Example 5: Jupyter Notebook Display

```python
from IPython.display import display, HTML, Code, clear_output
import asyncio
from omicverse.utils.smart_agent import OmicVerseAgent

async def stream_in_notebook():
    agent = OmicVerseAgent(model="gpt-4o")

    # Create output containers
    status_html = HTML("<p>Initializing...</p>")
    code_display = None
    display(status_html)

    async for event in agent.stream_async("Cluster my cells", adata):
        event_type = event["event_type"]

        if event_type == "skill_match":
            skills = event["data"]["matched_skills"]
            status_html.data = f"<p>🎯 Using skills: {', '.join(skills)}</p>"

        elif event_type == "llm_chunk":
            # Update status
            status_html.data = "<p>💭 Generating code...</p>"

        elif event_type == "code":
            # Display code with syntax highlighting
            code = event["data"]["code"]
            clear_output(wait=True)
            display(status_html)
            display(Code(code, language='python'))

        elif event_type == "result":
            if event["data"]["success"]:
                status_html.data = "<p style='color: green'>✓ Completed successfully!</p>"
            else:
                error = event["data"]["error"]
                status_html.data = f"<p style='color: red'>✗ Error: {error}</p>"

# Run in Jupyter
await stream_in_notebook()
```

---

## Type Hints

For proper type checking, use these type hints:

```python
from typing import Dict, Any, AsyncIterator
from omicverse.utils.smart_agent import OmicVerseAgent
from anndata import AnnData

# Event type
Event = Dict[str, Any]

# Stream function signature
async def process_stream(
    agent: OmicVerseAgent,
    query: str,
    adata: AnnData
) -> None:
    event: Event
    async for event in agent.stream_async(query, adata):
        event_type: str = event["event_type"]
        data: Dict[str, Any] = event["data"]

        # Handle events
        ...

# Event handler type
from typing import Callable, Awaitable

EventHandler = Callable[[Event], Awaitable[None]]

async def handle_event(event: Event) -> None:
    """Type-safe event handler"""
    if event["event_type"] == "result":
        result_data: Dict[str, Any] = event["data"]
        success: bool = result_data["success"]
        ...
```

---

## Error Handling

### Handling Stream Errors

```python
import asyncio
from omicverse.utils.smart_agent import OmicVerseAgent

async def robust_stream():
    agent = OmicVerseAgent(model="gpt-4o")

    try:
        async for event in agent.stream_async("Analyze my data", adata):
            try:
                # Process event
                handle_event(event)
            except Exception as e:
                print(f"Error handling event: {e}")
                # Continue streaming despite event handling error
                continue

    except asyncio.CancelledError:
        print("Stream cancelled by user")
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        print("Stream ended")

asyncio.run(robust_stream())
```

### Timeout Handling

```python
import asyncio

async def stream_with_timeout():
    agent = OmicVerseAgent(model="gpt-4o")

    try:
        async with asyncio.timeout(30):  # 30 second timeout
            async for event in agent.stream_async("Query", adata):
                handle_event(event)
    except asyncio.TimeoutError:
        print("Stream timed out after 30 seconds")
```

### Retry on Error

```python
async def stream_with_retry(max_retries=3):
    agent = OmicVerseAgent(model="gpt-4o")

    for attempt in range(max_retries):
        try:
            async for event in agent.stream_async("Query", adata):
                if event["event_type"] == "error":
                    if not event["data"].get("recoverable"):
                        raise Exception(event["data"]["message"])
                handle_event(event)

            # Success, exit retry loop
            break

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed")
                raise
```

---

## Provider Support

All providers support streaming:

| Provider | Streaming Support | Notes |
|----------|------------------|-------|
| OpenAI | ✅ Full | Native SDK streaming |
| Anthropic | ✅ Full | Context manager streaming |
| Google Gemini | ✅ Full | Async generator streaming |
| DeepSeek | ✅ Full | OpenAI-compatible |
| Moonshot | ✅ Full | OpenAI-compatible |
| xAI (Grok) | ✅ Full | OpenAI-compatible |
| DashScope | ✅ Full | Sync-to-async bridge |
| Zhipu AI | ✅ Full | OpenAI-compatible |

### Provider-Specific Considerations

**GPT-5 Models (Responses API):**
```python
# GPT-5 falls back to non-streaming for Responses API
agent = OmicVerseAgent(model="gpt-5-high")

async for event in agent.stream_async("Query", adata):
    # Events will still be emitted, but LLM response
    # will arrive as one chunk instead of streaming
    pass
```

**Anthropic Claude:**
```python
# Claude streaming uses text_stream for clean output
agent = OmicVerseAgent(model="claude-4-5-sonnet")

async for event in agent.stream_async("Query", adata):
    # Chunks are clean text without metadata
    if event["event_type"] == "llm_chunk":
        print(event["data"]["chunk"], end="")
```

---

## Performance Considerations

### Memory Usage

Streaming reduces memory footprint:

```python
# Non-streaming: Full response in memory
response = agent.query("Query", adata)  # Large memory spike

# Streaming: Process chunks incrementally
async for event in agent.stream_async("Query", adata):
    # Process and discard chunks
    if event["event_type"] == "llm_chunk":
        process_chunk(event["data"]["chunk"])
    # Minimal memory usage
```

### Latency

First token latency (time to first chunk):

| Provider | Typical Latency | Notes |
|----------|----------------|-------|
| OpenAI GPT-4o | 200-500ms | Very fast |
| Claude Sonnet | 300-700ms | Fast |
| Gemini Flash | 150-400ms | Fastest |
| GPT-5 | 1-3s | Slower (reasoning) |

### Throughput

Tokens per second during streaming:

- **GPT-4o**: ~50-80 tokens/sec
- **Claude Sonnet**: ~40-70 tokens/sec
- **Gemini Flash**: ~60-100 tokens/sec
- **GPT-5**: ~20-40 tokens/sec (reasoning mode)

### Concurrent Streams

```python
import asyncio

async def concurrent_streams():
    agent = OmicVerseAgent(model="gpt-4o")

    # Run multiple streams concurrently
    tasks = [
        agent.stream_async("Query 1", adata1),
        agent.stream_async("Query 2", adata2),
        agent.stream_async("Query 3", adata3)
    ]

    # Process all streams concurrently
    async for stream in asyncio.as_completed(tasks):
        async for event in await stream:
            handle_event(event)
```

---

## Best Practices

### 1. Always Handle All Event Types

```python
async for event in agent.stream_async(query, adata):
    event_type = event["event_type"]

    if event_type == "llm_chunk":
        # Handle chunks
        pass
    elif event_type == "code":
        # Handle code
        pass
    elif event_type == "result":
        # Handle result
        pass
    elif event_type == "error":
        # Handle error
        pass
    else:
        # Log unexpected event types
        logger.warning(f"Unexpected event: {event_type}")
```

### 2. Flush Output for Real-Time Display

```python
# Good: Real-time display
print(chunk, end="", flush=True)

# Bad: Buffered output
print(chunk, end="")  # May not display immediately
```

### 3. Clean Up Resources

```python
async def safe_stream():
    agent = OmicVerseAgent(model="gpt-4o")

    try:
        async for event in agent.stream_async(query, adata):
            handle_event(event)
    finally:
        # Clean up resources
        await agent.cleanup()  # If available
```

### 4. Separate Event Handling Logic

```python
# Good: Separate concerns
class EventProcessor:
    async def process(self, event):
        handler = getattr(self, f"handle_{event['event_type']}", None)
        if handler:
            await handler(event["data"])

    async def handle_code(self, data):
        ...

    async def handle_result(self, data):
        ...

# Bad: Large if-elif chain
async for event in stream:
    if event["event_type"] == "code":
        # 50 lines of code
        ...
    elif event["event_type"] == "result":
        # 50 lines of code
        ...
```

### 5. Provide User Feedback

```python
# Show progress to user
async for event in agent.stream_async(query, adata):
    if event["event_type"] == "skill_match":
        print("🎯 Analyzing your query...")
    elif event["event_type"] == "llm_chunk":
        print("💭 Generating code...")
    elif event["event_type"] == "code":
        print("✅ Code ready")
    elif event["event_type"] == "result":
        print("🎉 Analysis complete!")
```

### 6. Log Events for Debugging

```python
import logging

logger = logging.getLogger(__name__)

async for event in agent.stream_async(query, adata):
    # Log all events for debugging
    logger.debug(f"Event: {event['event_type']}", extra=event)

    # Handle events
    ...
```

---

## Troubleshooting

### Issue: No Events Received

```python
# Check: Are you awaiting the async generator?
# Wrong:
for event in agent.stream_async(query, adata):  # Missing async
    ...

# Correct:
async for event in agent.stream_async(query, adata):
    ...
```

### Issue: Events Out of Order

Events are emitted in order, but async processing may display them out of order:

```python
# Use asyncio.Queue to preserve order
import asyncio

async def ordered_processing():
    queue = asyncio.Queue()

    async def producer():
        async for event in agent.stream_async(query, adata):
            await queue.put(event)
        await queue.put(None)  # Sentinel

    async def consumer():
        while True:
            event = await queue.get()
            if event is None:
                break
            handle_event(event)  # Guaranteed in-order

    await asyncio.gather(producer(), consumer())
```

### Issue: Stream Hangs

```python
# Add timeout
import asyncio

async with asyncio.timeout(60):
    async for event in agent.stream_async(query, adata):
        ...
```

---

## API Reference

### OmicVerseAgent.stream_async

```python
async def stream_async(
    self,
    query: str,
    adata: AnnData,
    max_reflections: int = 3,
    enable_review: bool = True,
    skill_matching_method: str = "llm",
    **kwargs
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream agent execution with real-time updates.

    Args:
        query: Natural language query
        adata: AnnData object to analyze
        max_reflections: Maximum reflection iterations on error
        enable_review: Whether to review results
        skill_matching_method: "llm" or "algorithmic"
        **kwargs: Additional parameters

    Yields:
        Dict[str, Any]: Event dictionaries with keys:
            - event_type: str
            - data: Dict[str, Any]

    Raises:
        ValueError: Invalid input
        LLMAPIError: LLM API errors
        ExecutionError: Code execution errors
    """
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-08
**Author:** OmicVerse Development Team
