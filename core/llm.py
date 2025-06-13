# -*- coding: utf-8 -*-
"""llm.py ─ UltimateReVALAgent v2 with all improvements

Key Improvements:
• Docker sandboxing for create_and_test_tool
• Real simple_verifier implementation
• ScratchPad memory management with async cleanup
• Multi-model token counting
• Hierarchical message trimming
• Rich logging with color-safe output
• Adaptive tool timeouts
• Persistent _fc_supported caching
• Improved JSON parsing

Dependencies: tiktoken, pydantic, python-dotenv, openai>=1.0.0, rich, docker
"""
from __future__ import annotations

###############################################################################
# 0.  Imports & global helpers
###############################################################################
import asyncio
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Tuple

import tiktoken
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError, create_model
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# Try to import docker, but make it optional
try:
    import docker
    # Check if Docker daemon is actually running and accessible
    try:
        client = docker.from_env()
        client.ping()
        DOCKER_AVAILABLE = True
    except Exception:
        DOCKER_AVAILABLE = False
except ImportError:
    DOCKER_AVAILABLE = False

###############################################################################
# 1.  Enhanced Tool decorator & ScratchPad with memory management
###############################################################################
@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., Awaitable[Any]]
    expected_runtime: Optional[float] = None  # New field for adaptive timeout

def tool(desc: str = "", expected_runtime: Optional[float] = None) -> Callable[[Callable], Callable]:
    """Enhanced decorator that turns a function/method into a registered tool."""

    def decorator(func: Callable):
        sig = inspect.signature(func)
        fields: Dict[str, tuple] = {}
        for pname, param in sig.parameters.items():
            if pname == "self" or pname.startswith("_"):
                continue
            annot = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else Any
            )
            default = (
                ... if param.default is inspect.Parameter.empty else param.default
            )
            fields[pname] = (annot, Field(default=default))
        ArgsModel = create_model(f"{func.__name__.title()}Args", **fields)  # type: ignore

        async def _wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

        # Combine description and docstring for richer tool documentation
        combined_desc = desc
        if func.__doc__:
            if desc:
                combined_desc = f"{desc}\n\n{func.__doc__}"
            else:
                combined_desc = func.__doc__
                
        func.__tool_spec__ = ToolSpec(  # type: ignore[attr-defined]
            func.__name__, 
            combined_desc or "", 
            ArgsModel, 
            _wrapper,
            expected_runtime
        )
        return func

    return decorator


class ScratchPad(dict):
    """Enhanced in-memory TTL-aware key-value store with automatic cleanup."""

    def __init__(self):
        super().__init__()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._items_count = 0  # For metrics

    def store(self, value: Any, ttl: Optional[int] = None) -> str:
        key = f"sp_{uuid.uuid4().hex[:8]}"
        expiry = time.time() + ttl if ttl else None
        super().__setitem__(key, (value, expiry))
        self._items_count = len(self)
        return key

    def load(self, key: str):
        if key not in self:
            raise KeyError(key)
        val, exp = self[key]
        if exp and exp < time.time():
            del self[key]
            self._items_count = len(self)
            raise KeyError(f"{key} expired")
        return val

    async def cleanup_expired(self):
        """Remove expired items from the scratchpad."""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, expiry) in self.items():
            if expiry and expiry < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self[key]
        
        self._items_count = len(self)
        return len(expired_keys)

    def start_cleanup_task(self, interval_seconds: int = 300):
        """Start a background task to clean up expired items periodically."""
        async def _cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    removed = await self.cleanup_expired()
                    if removed > 0:
                        logging.debug(f"ScratchPad cleanup: removed {removed} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"ScratchPad cleanup error: {e}")

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(_cleanup_loop())

    def stop_cleanup_task(self):
        """Stop the cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    @property
    def items_count(self) -> int:
        """Get current number of items (for metrics)."""
        return self._items_count

###############################################################################
# 2.  Enhanced logging with Rich
###############################################################################
def setup_rich_logging(debug: bool = True, log_file: Optional[str] = None) -> logging.Logger:
    """Set up enhanced logging with Rich console output."""
    console = Console(
        force_terminal=True if sys.stdout.isatty() else None,
        force_jupyter=False,
        force_interactive=False
    )
    
    # Create logger
    logger = logging.getLogger("UltimateReVAL")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    
    # Rich console handler
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=debug,
        show_time=True,
        show_path=debug
    )
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

###############################################################################
# 3.  JSON parsing utilities
###############################################################################
def strip_json_markdown(text: str) -> Optional[str]:
    """Extract JSON from markdown code blocks more robustly."""
    # Try multiple patterns for JSON blocks
    patterns = [
        r"```json\s*\n([\s\S]*?)\n```",
        r"```JSON\s*\n([\s\S]*?)\n```",
        r"```\s*\n(\{[\s\S]*?\})\s*\n```",
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"  # Direct JSON object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            json_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
            try:
                # Validate it's proper JSON
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue
    
    return None

###############################################################################
# 4.  Docker sandbox utilities
###############################################################################
class DockerSandbox:
    """Docker-based sandbox for safe code execution."""
    
    def __init__(self):
        self.client = docker.from_env() if DOCKER_AVAILABLE else None
        self.image = "python:3.11-slim"
        
    async def execute_code(
        self, 
        code: str, 
        timeout: int = 10, 
        memory_limit: str = "128m"
    ) -> Tuple[bool, str]:
        """Execute Python code in a Docker container."""
        if not DOCKER_AVAILABLE:
            return False, "Docker not available - install docker package"
        
        # Create temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name
        
        try:
            # Run in container
            container = self.client.containers.run(
                self.image,
                f"python /code/script.py",
                volumes={
                    os.path.dirname(code_file): {'bind': '/code', 'mode': 'ro'}
                },
                working_dir="/tmp",
                mem_limit=memory_limit,
                network_mode="none",
                read_only=True,
                remove=True,
                detach=True,
                command=["/bin/sh", "-c", f"timeout {timeout} python /code/{os.path.basename(code_file)}"]
            )
            
            # Wait for completion
            result = container.wait(timeout=timeout + 2)
            output = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            success = result.get('StatusCode', 1) == 0
            return success, output
            
        except Exception as e:
            return False, f"Sandbox error: {str(e)}"
        finally:
            # Cleanup
            if os.path.exists(code_file):
                os.unlink(code_file)

###############################################################################
# 5.  Cache management for _fc_supported
###############################################################################
class FunctionCallingCache:
    """Persistent cache for function calling support status."""
    
    def __init__(self, cache_file: str = ".fc_cache.json"):
        self.cache_file = Path(cache_file)
        self._cache: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
    
    def _save(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception:
            pass
    
    def get(self, model: str) -> Optional[bool]:
        """Get cached function calling support status."""
        if model in self._cache:
            entry = self._cache[model]
            # Check if entry is still valid (24h TTL)
            if datetime.fromisoformat(entry['timestamp']) > datetime.now() - timedelta(days=1):
                return entry['supported']
        return None
    
    def set(self, model: str, supported: bool):
        """Cache function calling support status."""
        self._cache[model] = {
            'supported': supported,
            'timestamp': datetime.now().isoformat()
        }
        self._save()

###############################################################################
# 6.  UltimateReVALAgent – Enhanced core engine
###############################################################################
class UltimateReVALAgent:
    """Enhanced ReVAL agent with all improvements."""

    STRICT_JSON_ERROR = (
        "⛔️ Format invalide. Réponds UNIQUEMENT par un bloc ```json``` contenant `tool_call`."
    )

    def __init__(
        self,
        model: str = "deepseek/deepseek-r1-0528:free",
        *,
        tool_support: bool = False,
        temperature: float = 0.2,
        max_model_tokens: int = 16_000,
        max_response_tokens: int = 2_048,
        persona_prompt: str | None = None,
        debug: bool = True,
        debug_log_file: str | None = None,
        encoding_override: Optional[str] = None,  # New parameter
        verifier_model: str = "gpt-3.5-turbo",   # For simple_verifier
        register_default_tools: bool = True,     # Control default tool registration
    ) -> None:
        # ── ENV & LLM client ────────────────────────────────────────────────
        load_dotenv(".env.local")
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key env var")
            
        # Check if using OpenRouter
        is_openrouter = "openrouter" in os.getenv("LLM_BASE_URL", "").lower() or \
                      any(provider in model for provider in ["deepseek", "anthropic", "mistral", "google"])
        
        if is_openrouter:
            self.is_openrouter = True
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                timeout=45,
                default_headers={
                    "HTTP-Referer": os.getenv("HTTP_REFERER", "https://synapsereval.local"),
                    "X-Title": os.getenv("X_TITLE", "SynapseREval")
                }
            )
        else:
            self.is_openrouter = False
            self.client = AsyncOpenAI(api_key=api_key, timeout=45)

        # ── core settings ───────────────────────────────────────────────────
        self.model = model
        self.verifier_model = verifier_model
        self.tool_support_flag = tool_support
        self.temperature = temperature
        self.max_model_tokens = max_model_tokens
        self.max_response_tokens = max_response_tokens
        
        # Enhanced token encoding
        if encoding_override:
            self._enc = tiktoken.get_encoding(encoding_override)
        else:
            try:
                self._enc = tiktoken.encoding_for_model(model.split("/")[-1])
            except KeyError:
                self._enc = tiktoken.get_encoding("cl100k_base")
        
        self.debug = debug

        # ── runtime state ───────────────────────────────────────────────────
        self.scratch = ScratchPad()
        self.scratch.start_cleanup_task()  # Start automatic cleanup
        self.tools: Dict[str, ToolSpec] = {}
        self.fc_cache = FunctionCallingCache()
        self._fc_supported: Optional[bool] = self.fc_cache.get(model)
        self.docker_sandbox = DockerSandbox() if DOCKER_AVAILABLE else None

        # ── logging ─────────────────────────────────────────────────────────
        self.log = setup_rich_logging(debug, debug_log_file)
        self.log.debug(f"Initializing UltimateReVALAgent with model: {model}")
        self.log.debug(f"OpenRouter mode: {is_openrouter}")
        self.log.debug(f"Docker available: {DOCKER_AVAILABLE}")

        # ── register built‑in tools ─────────────────────────────────────────────
        if register_default_tools:
            self._debug_step("Registering built-in tools")
            self.register_tool(self.update_goal_state)
            self.register_tool(self.save_to_scratchpad)
            self.register_tool(self.load_from_scratchpad)
            self.register_tool(self.self_reflect_and_replan)
            self.register_tool(self.complexity_estimator)
            self.register_tool(self.simple_verifier)
            self.register_tool(self.create_and_test_tool)
            self.register_tool(self.get_scratchpad_metrics)
        else:
            self._debug_step("Skipping built-in tools registration")

        # ── system prompt ───────────────────────────────────────────────────────
        self._debug_step("Setting up system prompt")
        self.persona_prompt = (
            persona_prompt or "You are ReVAL, an elite autonomous agent known for rigorous reasoning and brutal honesty."
        )
        self._refresh_system_prompt()

    def _debug_step(self, message: str, reval_step: bool = False):
        """Log a debug step if debug mode is enabled."""
        if self.debug:
            # Replace emojis with text fallbacks for Windows
            import os
            if os.name == 'nt':
                message = message.replace("🧠", "[BRAIN]")
                message = message.replace("✅", "[OK]")
                message = message.replace("🔧", "[TOOL]")
                message = message.replace("❌", "[ERROR]")
                message = message.replace("⚠️", "[WARNING]")
                message = message.replace("📦", "[PACKAGE]")
                
            if reval_step:
                self.log.debug(f"[bold magenta]ReVAL STEP:[/bold magenta] {message}", extra={"markup": True})
            else:
                self.log.debug(f"[blue]STEP:[/blue] {message}", extra={"markup": True})
    
    def _debug_dump_conversation(self, msgs: List[Dict], prefix: str = ""):
        """Dump the current conversation state to the log file."""
        if not self.debug:
            return
            
        self.log.debug(f"{prefix}Conversation state:")
        for i, msg in enumerate(msgs):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content or "") > 100:
                content = content[:97] + "..."
            self.log.debug(f"  [{i}] {role}: {content}")
            
            # Log tool calls if present
            if "tool_calls" in msg and msg["tool_calls"] is not None:
                for tc in msg["tool_calls"]:
                    if tc.get("type") == "function":
                        fn = tc.get("function", {})
                        self.log.debug(f"      Tool call: {fn.get('name')}({fn.get('arguments', '')})")
                        
            # Log tool responses if present
            if role == "tool":
                tool_name = msg.get("name", "unknown_tool")
                self.log.debug(f"      Tool response from: {tool_name}")
                
        self.log.debug(f"{prefix}End of conversation state.")

    # ======================================================================
    # Tool registration & helpers
    # ======================================================================
    def register_tool(self, func: Callable):
        """Register a callable that has been decorated with @tool."""
        if not hasattr(func, "__tool_spec__"):
            self._debug_step(f"Auto-decorating function {func.__name__}")
            func = tool()(func)  # auto‑decorate if not already
            
        spec: ToolSpec = func.__tool_spec__  # type: ignore[attr-defined]
            
        # If we receive a *bound* method, ensure the handler is the method itself
        if inspect.ismethod(func):
            # The handler should be the method itself, not a wrapper around it
            spec.handler = func
            
        self._debug_step(f"Registering tool: {spec.name}")
        self.tools[spec.name] = spec
        # Only refresh system prompt if persona_prompt is already set
        if hasattr(self, "persona_prompt"):
            self._refresh_system_prompt()

    def register_tools_from_instance(self, obj: Any):
        """Register all callable attributes of an object as tools."""
        for name in dir(obj):
            if name.startswith("_"):
                continue
            attr = getattr(obj, name)
            if callable(attr):
                try:
                    # Check if the attribute already has a tool_spec
                    if hasattr(attr, "__tool_spec__"):
                        tool_spec = attr.__tool_spec__
                        if all(hasattr(tool_spec, field) for field in ["name", "description", "args_schema", "handler"]):
                            # Create a new wrapper that properly binds the method to the instance
                            original_handler = tool_spec.handler
                            
                            async def bound_handler(*args, _orig=original_handler, **kwargs):
                                return await _orig(obj, *args, **kwargs)
                            
                            # Create a new ToolSpec with the bound handler
                            bound_spec = ToolSpec(
                                name=tool_spec.name,
                                description=tool_spec.description,
                                args_schema=tool_spec.args_schema,
                                handler=bound_handler,
                                expected_runtime=getattr(tool_spec, 'expected_runtime', None)
                            )
                            self.tools[bound_spec.name] = bound_spec
                            if hasattr(self, "persona_prompt"):
                                self._refresh_system_prompt()
                            continue
                        
                    # For methods, create a wrapper
                    if inspect.ismethod(attr):
                        method = attr
                        
                        if asyncio.iscoroutinefunction(method):
                            async def async_wrapper(*args, _m=method, **kwargs):
                                return await _m(*args, **kwargs)
                            wrapper = async_wrapper
                        else:
                            def sync_wrapper(*args, _m=method, **kwargs):
                                return _m(*args, **kwargs)
                            wrapper = sync_wrapper
                            
                        wrapper.__name__ = method.__name__
                        wrapper.__doc__ = method.__doc__
                        self.register_tool(wrapper)
                    else:
                        # For non-methods, bind to instance
                        method = attr
                        
                        if asyncio.iscoroutinefunction(method):
                            async def async_bound_wrapper(*args, _m=method, **kwargs):
                                return await _m(obj, *args, **kwargs)
                            wrapper = async_bound_wrapper
                        else:
                            def sync_bound_wrapper(*args, _m=method, **kwargs):
                                return _m(obj, *args, **kwargs)
                            wrapper = sync_bound_wrapper
                            
                        wrapper.__name__ = method.__name__
                        wrapper.__doc__ = method.__doc__
                        self.register_tool(wrapper)
                except Exception as e:
                    self.log.warning(f"Failed to register tool {name}: {e}")

    # ----------------------------------------------------------------------
    # Built‑in meta tools
    # ----------------------------------------------------------------------
    @tool("Update or query long‑term goal‑state store.")
    async def update_goal_state(
        self,
        original_request: Optional[str] = None,
        plan: Optional[List[str]] = None,
        completed_step: Optional[str] = None,
        finding_key: Optional[str] = None,
        finding_value: Optional[Any] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        gs = self.scratch.setdefault(
            "goal_state", {"plan": [], "done": [], "findings": {}, "conf": None}
        )
        if original_request:
            gs["request"] = original_request
        if plan is not None:
            gs["plan"] = plan
        if completed_step:
            gs["done"].append(completed_step)
        if finding_key and finding_value is not None:
            gs["findings"][finding_key] = finding_value
        if confidence is not None:
            gs["conf"] = confidence
        return gs

    @tool("Persist data into scratchpad; returns key.")
    async def save_to_scratchpad(self, value: Any, ttl_s: int | None = None) -> str:
        return self.scratch.store(value, ttl_s)

    @tool("Retrieve value from scratchpad by key.")
    async def load_from_scratchpad(self, key: str):
        try:
            return self.scratch.load(key)
        except KeyError as err:
            return {"error": str(err)}

    @tool("Get scratchpad metrics including item count.")
    async def get_scratchpad_metrics(self) -> Dict[str, Any]:
        """Return current scratchpad metrics."""
        return {
            "items_count": self.scratch.items_count,
            "memory_usage_estimate": sys.getsizeof(self.scratch),
        }

    @tool("Self‑reflect: critique current plan and propose new one.")
    async def self_reflect_and_replan(self, critique: str, new_plan: List[str]):
        return {"meta": "reflect", "critique": critique, "plan": new_plan}

    # ── ReVAL specific tools ───────────────────────────────────────────────
    @tool("Estimate problem complexity (0‑1) for dynamic budgeting.")
    async def complexity_estimator(self, prompt: str) -> float:
        score = min(len(prompt.split()) / 4000, 1.0)
        return score

    @tool("Verify answer correctness using a secondary model.", expected_runtime=3.0)
    async def simple_verifier(self, answer: str, question: str) -> Dict[str, Any]:
        """Real implementation of simple_verifier using a small model."""
        checklist_prompt = f"""
You are a verification assistant. Evaluate if the answer correctly addresses the question.

Question: {question}
Answer: {answer}

Checklist:
1. Does the answer directly address the question?
2. Is the answer factually plausible?
3. Is the answer complete (not missing key parts)?
4. Is the answer internally consistent?

Respond with JSON:
{{"is_correct": true/false, "explanation": "brief reason", "confidence": 0.0-1.0}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.verifier_model,
                messages=[
                    {"role": "system", "content": "You are a precise verification assistant."},
                    {"role": "user", "content": checklist_prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content or "{}")
            return {
                "verified": result.get("is_correct", False),
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", 0.5)
            }
        except Exception as e:
            self.log.error(f"Verifier error: {e}")
            return {
                "verified": True,  # Default to true on error
                "explanation": f"Verification error: {str(e)}",
                "confidence": 0.0
            }

    # ── Dynamic toolsmith with Docker sandbox ──────────────────────────────
    @tool("Create a new Python tool, test it in sandbox, and register if safe.", expected_runtime=15.0)
    async def create_and_test_tool(
        self,
        tool_name: str,
        description: str,
        python_code: str,
        test_code: str,
    ) -> str:
        """Create and test a new tool in a Docker sandbox."""
        if not self.docker_sandbox:
            return "Error: Docker sandbox not available. Install docker package."
        
        # Combine tool code and test code
        full_code = f"""
{python_code}

# Test section
if __name__ == "__main__":
    candidate = {tool_name}
    {test_code}
    print("All tests passed!")
"""
        
        # Execute in sandbox
        success, output = await self.docker_sandbox.execute_code(
            full_code,
            timeout=10,
            memory_limit="128m"
        )
        
        if not success:
            return f"Tool testing failed:\n{output}"
        
        # If tests pass, register the tool (but still in a safe namespace)
        try:
            ns: Dict[str, Any] = {}
            exec(python_code, {"__builtins__": {}}, ns)
            
            if tool_name not in ns:
                return "Error: function not defined in code."
            
            fn = ns[tool_name]
            # Wrap the function to ensure it's async
            if not asyncio.iscoroutinefunction(fn):
                async def async_fn(*args, **kwargs):
                    return fn(*args, **kwargs)
                fn = async_fn
            
            # Create a bound method
            setattr(self, tool_name, fn.__get__(self))
            self.register_tool(getattr(self, tool_name))
            
            # Update description
            if tool_name in self.tools:
                self.tools[tool_name] = ToolSpec(
                    name=tool_name,
                    description=description,
                    args_schema=self.tools[tool_name].args_schema,
                    handler=self.tools[tool_name].handler,
                    expected_runtime=None
                )
            
            return f"Tool '{tool_name}' created and registered successfully.\nTest output:\n{output}"
        except Exception as e:
            return f"Error registering tool: {str(e)}"

    # ======================================================================
    # System prompt
    # ======================================================================
    def _refresh_system_prompt(self):
        tool_lines = []
        for spec in self.tools.values():
            params = ", ".join(
                spec.args_schema.model_json_schema().get("properties", {}).keys()
            )
            tool_lines.append(f"- `{spec.name}({params})`: {spec.description}")
        tools_doc = "\n".join(tool_lines)
        usage_doc = (
            "Tools are available via *native function‑calling* (if supported) **or** via a JSON fall‑back.\n\n"
            "When using the fall‑back, reply **only** with one JSON block:\n"
            "```json\n{\"tool_call\": {\"name\": <tool_name>, \"arguments\": {…}}}\n```"
        )
        self.system_prompt = f"{self.persona_prompt}\n\n### Tools\n{tools_doc}\n\n### Usage\n{usage_doc}"

    # ======================================================================
    # LLM invocation helpers
    # ======================================================================
    async def _call_llm(self, messages: List[Dict]) -> Dict:
        """Call the LLM with graceful degradation between modes."""
        self._debug_step("[bold blue]🧠 Calling LLM[/bold blue]", reval_step=False)
        
        want_native = self.tool_support_flag or (self._fc_supported is not False)
        fc_schema = None
        if want_native:
            fc_schema = [
                {
                    "type": "function",
                    "function": {
                        "name": s.name,
                        "description": s.description,
                        "parameters": s.args_schema.model_json_schema(),
                    },
                }
                for s in self.tools.values()
            ]
            self._debug_step(f"[cyan]Using native function calling with {len(fc_schema)} tools[/cyan]")
        else:
            self._debug_step("[yellow]Using JSON fallback for tool calling[/yellow]")
            
        params = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_response_tokens,
            tools=fc_schema if want_native else None,
            tool_choice="auto" if want_native else None,
        )
        
        # Add OpenRouter specific parameters
        if hasattr(self, "is_openrouter") and self.is_openrouter:
            self._debug_step("Adding OpenRouter specific parameters")
            headers = {
                "HTTP-Referer": os.getenv("HTTP_REFERER", "https://synapsereval.local"),
                "X-Title": os.getenv("X_TITLE", "SynapseREval")
            }
            params["extra_headers"] = headers
            params["extra_body"] = {
                "transforms": ["middle-out"],
            }
        
        for attempt in range(3):
            try:
                self._debug_step(f"LLM request attempt {attempt+1}/3")
                resp = await self.client.chat.completions.create(**params)
                
                if getattr(resp, "error", None):
                    error_code = resp.error.get("code", "unknown")
                    error_msg = resp.error.get("message", "Unknown error")
                    self._debug_step(f"[red]❌ LLM backend error {error_code}: {error_msg}[/red]")
                    raise RuntimeError(f"LLM backend error {error_code}: {error_msg}")
                
                if not getattr(resp, "choices", None):
                    self._debug_step("[red]❌ LLM returned no choices[/red]")
                    raise RuntimeError("LLM returned no choices")
                
                # Update function calling support status
                if self._fc_supported != want_native:
                    self._fc_supported = want_native
                    self.fc_cache.set(self.model, want_native)
                
                self._debug_step("[green]✅ LLM request successful[/green]")
                return resp.choices[0].message.model_dump()
                
            except OpenAIError as err:
                self._debug_step(f"[red]❌ LLM error: {err}[/red]")
                if want_native and (
                    getattr(err, "status_code", None) == 404 or "No endpoints" in str(err)
                ):
                    self._debug_step("[yellow]⚠️ Function calling not supported, downgrading[/yellow]")
                    self._fc_supported = False
                    self.fc_cache.set(self.model, False)
                    want_native = False
                    params["tools"] = None
                    params["tool_choice"] = None
                    continue
                if isinstance(err, (RateLimitError, APIConnectionError)):
                    wait_time = 2 ** attempt
                    self._debug_step(f"[yellow]⚠️ Rate limit, retrying in {wait_time}s[/yellow]")
                    await asyncio.sleep(wait_time)
                    continue
                if attempt < 2:
                    wait_time = 2 ** attempt
                    self._debug_step(f"[yellow]⚠️ Error, retrying in {wait_time}s[/yellow]")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as ex:
                self._debug_step(f"[red]❌ Unexpected error: {ex}[/red]")
                if attempt < 2:
                    wait_time = 2 ** attempt
                    self._debug_step(f"[yellow]⚠️ Retrying in {wait_time}s[/yellow]")
                    await asyncio.sleep(wait_time)
                    continue
                raise
        
        self._debug_step("[red bold]❌ LLM FAILED AFTER ALL RETRIES[/red bold]")
        raise RuntimeError("LLM failed after retries")

    # ======================================================================
    # Tool execution helper with adaptive timeout
    # ======================================================================
    async def _execute_tool(self, name: str, args: Dict[str, Any]):
        self._debug_step(f"[cyan bold]🔧 Executing tool: {name}[/cyan bold]")
        if name not in self.tools:
            self._debug_step(f"[red]❌ Unknown tool: {name}[/red]")
            return {"error": f"Unknown tool {name}"}
        
        spec = self.tools[name]
        try:
            validated = spec.args_schema(**args)
            self._debug_step(f"Arguments validated for {name}")
            
            # Adaptive timeout based on expected_runtime
            timeout = 25  # default
            if spec.expected_runtime:
                timeout = max(25, spec.expected_runtime * 1.5)
                self._debug_step(f"Using adaptive timeout: {timeout}s for {name}")
            
            result = await asyncio.wait_for(
                spec.handler(**validated.dict()),
                timeout=timeout
            )
            self._debug_step(f"[green]✅ Tool {name} executed successfully[/green]")
            
            # Large payload off‑loading
            if (
                isinstance(result, (dict, list, str))
                and len(json.dumps(result, default=str)) > 4000
            ):
                key = self.scratch.store(result, ttl=300)
                self._debug_step(f"[yellow]📦 Large result stored: {key}[/yellow]")
                return {"scratchpad_key": key, "info": f"{name} output stored (large payload)"}
            return result
            
        except asyncio.TimeoutError:
            self._debug_step(f"[red]❌ Tool {name} timed out[/red]")
            return {"error": f"Tool {name} timed out"}
        except ValidationError as ve:
            self._debug_step(f"[red]❌ Validation error in {name}: {ve}[/red]")
            return {"error": str(ve)}
        except Exception as exc:
            self._debug_step(f"[red]❌ Exception in {name}: {exc}[/red]")
            if self.debug:
                self.log.exception(f"Tool execution error in {name}")
            return {"error": str(exc)}

    # ======================================================================
    # Parsing helpers
    # ======================================================================
    def _extract_tool_calls(self, resp: Dict) -> Optional[List[Dict]]:
        """Extract tool calls using improved parsing."""
        # Native function‑calling
        if resp.get("tool_calls"):
            self._debug_step(f"[cyan]Found {len(resp['tool_calls'])} native tool calls[/cyan]")
            return resp["tool_calls"]
        
        # Enhanced JSON parsing
        content = resp.get("content")
        if not content:
            return None
        
        json_str = strip_json_markdown(content)
        if not json_str:
            return None
        
        try:
            blob = json.loads(json_str)
            if "tool_call" in blob and {"name", "arguments"} <= set(blob["tool_call"].keys()):
                tc = blob["tool_call"]
                tool_id = f"tc_{uuid.uuid4().hex[:6]}"
                self._debug_step(f"[cyan]Extracted JSON tool call: {tc['name']}[/cyan]")
                return [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                ]
        except Exception as e:
            self._debug_step(f"[red]Error parsing JSON: {e}[/red]")
        return None

    def _tool_messages(self, calls: List[Dict], results: List[Any]):
        msgs = []
        for call, res in zip(calls, results):
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": json.dumps(res, default=str)[:12_000],
                }
            )
        return msgs

    # ======================================================================
    # Utility with hierarchical trimming
    # ======================================================================
    def _tokens(self, txt: str | None) -> int:
        return len(self._enc.encode(txt or ""))

    def _trim_hierarchical(self, msgs: List[Dict]) -> List[Dict]:
        """Hierarchical message trimming preserving important messages."""
        budget = self.max_model_tokens - self.max_response_tokens
        current_tokens = sum(self._tokens(m.get("content")) for m in msgs)
        
        if current_tokens <= budget:
            return msgs
        
        # Never remove: system prompt (0), first user message (1), last few messages
        protected_indices = {0, 1, len(msgs) - 1}
        if len(msgs) > 3:
            protected_indices.add(len(msgs) - 2)
        
        # Build priority list (tool messages have lowest priority)
        priorities = []
        for i, msg in enumerate(msgs):
            if i in protected_indices:
                priority = 0  # Highest priority
            elif msg.get("role") == "tool":
                priority = 3  # Lowest priority
            elif msg.get("role") == "user":
                priority = 2  # Medium priority
            else:
                priority = 1  # Assistant messages
            priorities.append((priority, i))
        
        # Sort by priority (highest priority first) and age (older first within same priority)
        priorities.sort(key=lambda x: (x[0], x[1]))
        
        # Remove messages until within budget
        removed_indices = set()
        for priority, idx in priorities:
            if idx in protected_indices:
                continue
            
            # Simulate removal
            test_msgs = [m for i, m in enumerate(msgs) if i not in removed_indices and i != idx]
            test_tokens = sum(self._tokens(m.get("content")) for m in test_msgs)
            
            if test_tokens <= budget:
                removed_indices.add(idx)
                current_tokens = test_tokens
                if current_tokens <= budget:
                    break
        
        # Return filtered messages
        return [m for i, m in enumerate(msgs) if i not in removed_indices]

    # ======================================================================
    # Chat loop (ReVAL)
    # ======================================================================
    async def chat(self, user_prompt: str, history: Optional[List[Dict]] = None) -> str:
        self._debug_step(f"Starting chat with prompt: {user_prompt[:50]}...", reval_step=True)
        history = history or []
        msgs: List[Dict] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_prompt},
        ]
        await self.update_goal_state(original_request=user_prompt)
        self._debug_step("Goal state updated with original request")
        
        # Dump initial conversation state
        self._debug_dump_conversation(msgs, prefix="INITIAL ")
        max_cycles = 30
        
        for cycle in range(max_cycles):
            self._debug_step(f"[magenta bold]ReVAL CYCLE {cycle+1}/{max_cycles} STARTED[/magenta bold]", reval_step=True)
            
            # REASON phase
            self._debug_step("[yellow]REASON PHASE: Preparing to call LLM[/yellow]", reval_step=True)
            msgs = self._trim_hierarchical(msgs)  # Use hierarchical trimming
            self._debug_step(f"Messages trimmed to {len(msgs)} messages")
            
            assistant = await self._call_llm(msgs)
            msgs.append(assistant)
            self._debug_step("Assistant response received and added to messages")
            
            # Dump conversation after assistant response
            self._debug_dump_conversation([assistant], prefix=f"CYCLE {cycle+1} ASSISTANT ")

            # VERIFY phase - Confidence gating
            self._debug_step("[yellow]VERIFY PHASE: Checking confidence[/yellow]", reval_step=True)
            conf_match = re.search(r"CONF\s*=\s*([0-9.]+)", assistant.get("content", ""))
            if conf_match:
                conf_val = float(conf_match.group(1))
                self._debug_step(f"Confidence value detected: {conf_val}")
                await self.update_goal_state(confidence=conf_val)
                if conf_val < 0.7:
                    self._debug_step(f"[red]Low confidence {conf_val} detected[/red]", reval_step=True)
                    
                    # ADAPT phase
                    self._debug_step("[yellow]ADAPT PHASE: Initiating self-reflection[/yellow]", reval_step=True)
                    reflection = await self.self_reflect_and_replan(
                        critique="Low confidence", new_plan=["Retry with deeper reasoning"]
                    )
                    msgs.append({"role": "assistant", "content": json.dumps(reflection)})
                    self._debug_step("Self-reflection added to messages")
                    continue

            # Tool phase
            tool_calls = self._extract_tool_calls(assistant)
            if not tool_calls:
                self._debug_step("[green]COMPLETION: No tool calls, returning final response[/green]", reval_step=True)
                return assistant.get("content", "")

            self._debug_step(f"Extracted {len(tool_calls)} tool calls")
            
            # Execute tools with adaptive timeout
            results = []
            for call in tool_calls:
                result = await self._execute_tool(
                    call["function"]["name"],
                    json.loads(call["function"]["arguments"]),
                )
                results.append(result)
            
            self._debug_step(f"All tool calls executed, got {len(results)} results")

            # VERIFY phase - verification pass
            self._debug_step("[yellow]VERIFY PHASE: Running verification[/yellow]", reval_step=True)
            if results and "simple_verifier" not in [tc["function"]["name"] for tc in tool_calls]:
                # Don't verify the verifier itself
                verification = await self.simple_verifier(
                    answer=str(results[0]), 
                    question=user_prompt
                )
                if not verification.get("verified", True):
                    # ADAPT phase
                    self._debug_step(f"[red]Verification failed: {verification.get('explanation', '')}[/red]", reval_step=True)
                    self._debug_step("[yellow]ADAPT PHASE: Initiating self-reflection[/yellow]", reval_step=True)
                    reflection = await self.self_reflect_and_replan(
                        critique=f"Verifier failed: {verification.get('explanation', '')}", 
                        new_plan=["Revise answer based on verification feedback"]
                    )
                    msgs.append({"role": "assistant", "content": json.dumps(reflection)})
                    continue
                self._debug_step("[green]Verification passed[/green]", reval_step=True)

            # LOOP phase
            self._debug_step("[yellow]LOOP PHASE: Processing tool results[/yellow]", reval_step=True)
            tool_messages = self._tool_messages(tool_calls, results)
            msgs.extend(tool_messages)
            self._debug_step(f"Added {len(tool_messages)} tool messages to conversation")
            
            # Dump conversation after tool responses
            self._debug_dump_conversation(tool_messages, prefix=f"CYCLE {cycle+1} TOOLS ")
            self._debug_step(f"[magenta bold]ReVAL CYCLE {cycle+1} COMPLETED[/magenta bold]", reval_step=True)
            
        self._debug_step("[red]REACHED MAX REASONING CYCLES[/red]", reval_step=True)
        return "⚠️ Reached max reasoning cycles."

    def __del__(self):
        """Cleanup when agent is destroyed."""
        if hasattr(self, 'scratch'):
            self.scratch.stop_cleanup_task()

