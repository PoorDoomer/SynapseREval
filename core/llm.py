# -*- coding: utf-8 -*-
"""llm.py ─ merged v1


───────────────────────────────────────────────────────────────────────────────
Key Features
────────────
• **ReVAL loop** (Reason‑Verify‑Adapt‑Loop) with confidence gating & self‑reflection.
• Dual **tool‑calling** strategy: native OpenAI function‑calling *or* JSON fallback.
• **ScratchPad** with TTL + automatic large‑payload off‑loading.
• Built‑in **meta‑tools** (goal‑state store, complexity estimator, verifier, etc.).
• Automatic **toolsmith** (create & test new tools on‑the‑fly).
• Robust **guard‑rails**: strict JSON schema, timeout sandbox, adaptive token trim.
• Drop‑in compatibility with any project already using `tools.py` helpers.

Dependencies: tiktoken, pydantic, python‑dotenv, openai>=1.0.0
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
import sys
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

import tiktoken
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError, create_model

# ANSI color codes for terminal output
class Colors:
    if os.name == 'nt':  # Windows
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        BOLD = ""
        UNDERLINE = ""
        END = ""
    else:
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"
        END = "\033[0m"
###############################################################################
# 1.  Primitive helpers – Tool decorator & ScratchPad
###############################################################################
@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., Awaitable[Any]]

def tool(desc: str = "") -> Callable[[Callable], Callable]:
    """Decorator that turns a function/method into a registered *tool*."""

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
            # If we have both a description and docstring, combine them
            if desc:
                combined_desc = f"{desc}\n\n{func.__doc__}"
            else:
                combined_desc = func.__doc__
                
        func.__tool_spec__ = ToolSpec(  # type: ignore[attr-defined]
            func.__name__, combined_desc or "", ArgsModel, _wrapper
        )
        return func

    return decorator


class ScratchPad(dict):
    """In‑memory TTL‑aware key‑value store (auto‑cleans on access)."""

    def store(self, value: Any, ttl: Optional[int] = None) -> str:
        key = f"sp_{uuid.uuid4().hex[:8]}"
        expiry = time.time() + ttl if ttl else None
        super().__setitem__(key, (value, expiry))
        return key

    def load(self, key: str):
        if key not in self:
            raise KeyError(key)
        val, exp = self[key]
        if exp and exp < time.time():
            del self[key]
            raise KeyError(f"{key} expired")
        return val

###############################################################################
# 2.  UltimateReVALAgent – core engine
###############################################################################
class UltimateReVALAgent:
    """Merged agent implementing ReVAL + Ultimate features."""

    # Regex to capture JSON blocks in fallback mode
    _JSON_RE = re.compile(r"```json\s*([\s\S]*?)```", re.I)

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
    ) -> None:
        # ── ENV & LLM client ────────────────────────────────────────────────
        load_dotenv(".env.local")
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key env var")
            
        # Check if using OpenRouter based on model name or explicit env var
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
        self.tool_support_flag = tool_support
        self.temperature = temperature
        self.max_model_tokens = max_model_tokens
        self.max_response_tokens = max_response_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")
        self.debug = debug

        # ── runtime state ───────────────────────────────────────────────────
        self.scratch = ScratchPad()
        self.tools: Dict[str, ToolSpec] = {}
        self._fc_supported: Optional[bool] = None  # unknown until first attempt

        # ── logging ─────────────────────────────────────────────────────────
        self.log = logging.getLogger("UltimateReVAL")
        
        # Setup enhanced debugging if enabled
        if debug:
            self.log.setLevel(logging.DEBUG)
            
            # Console handler with formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_format)
            self.log.addHandler(console_handler)
            
            # File handler if log file is specified
            if debug_log_file:
                file_handler = logging.FileHandler(debug_log_file)
                file_handler.setLevel(logging.DEBUG)
                file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_format)
                self.log.addHandler(file_handler)
                
            self.log.debug(f"Initializing UltimateReVALAgent with model: {model}")
            self.log.debug(f"OpenRouter mode: {is_openrouter}")
        else:
            self.log.setLevel(logging.INFO)
            
        if debug and not self.log.handlers:
            logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        # ── register built‑in tools ─────────────────────────────────────────────
        self._debug_step("Registering built-in tools")
        self.register_tool(self.update_goal_state)
        self.register_tool(self.save_to_scratchpad)
        self.register_tool(self.load_from_scratchpad)
        self.register_tool(self.self_reflect_and_replan)
        self.register_tool(self.complexity_estimator)
        self.register_tool(self.simple_verifier)
        self.register_tool(self.create_and_test_tool)

        # ── system prompt ───────────────────────────────────────────────────────
        self._debug_step("Setting up system prompt")
        self.persona_prompt = (
            persona_prompt or "You are ReVAL, an elite autonomous agent known for rigorous reasoning and brutal honesty."
        )
        self._refresh_system_prompt()

    def _debug_step(self, message: str, reval_step: bool = False):
        """Log a debug step if debug mode is enabled."""
        if self.debug:
            # Remove problematic Unicode characters for Windows
            clean_message = message
            if os.name == 'nt':  # Windows
                # Replace problematic Unicode characters
                clean_message = (clean_message
                            .replace('🧠', '[BRAIN]')
                            .replace('🔧', '[TOOL]')
                            .replace('✅', '[OK]')
                            .replace('❌', '[ERROR]')
                            .replace('⚠️', '[WARNING]')
                            .replace('📦', '[PACKAGE]'))
            
            if reval_step:
                self.log.debug(f"{Colors.MAGENTA}{Colors.BOLD}ReVAL STEP: {clean_message}{Colors.END}")
            else:
                self.log.debug(f"{Colors.BLUE}STEP: {clean_message}{Colors.END}")
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
            
        # If we receive a *bound* method, remember the instance so we can
        # re-inject it when the tool is executed.
        if inspect.ismethod(func):
            owning_instance = func.__self__

            async def _bound_handler(*args, _h=func.__tool_spec__.handler, **kwargs):
                # Always pass the original `self` first
                return await _h(owning_instance, *args, **kwargs)

            func.__tool_spec__.handler = _bound_handler      # 🔑
            
        spec: ToolSpec = func.__tool_spec__  # type: ignore[attr-defined]
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
                    # Check if the attribute already has a tool_spec (from @tool decorator)
                    if hasattr(attr, "__tool_spec__"):
                        # If using a different tool decorator with compatible ToolSpec,
                        # we need to handle the method binding properly
                        tool_spec = attr.__tool_spec__
                        if all(hasattr(tool_spec, field) for field in ["name", "description", "args_schema", "handler"]):
                            # Create a new wrapper that properly binds the method to the instance
                            original_handler = tool_spec.handler
                            
                            # Create a new handler that binds the method to the instance
                            async def bound_handler(*args, _orig=original_handler, **kwargs):
                                # ALWAYS pass the owning instance (obj) as first arg
                                return await _orig(obj, *args, **kwargs)
                            
                            # Create a new ToolSpec with the bound handler
                            bound_spec = ToolSpec(
                                name=tool_spec.name,
                                description=tool_spec.description,
                                args_schema=tool_spec.args_schema,
                                handler=bound_handler
                            )
                            self.tools[bound_spec.name] = bound_spec
                            if hasattr(self, "persona_prompt"):
                                self._refresh_system_prompt()
                            continue
                        
                    # For methods, we need to create a wrapper function that can have attributes
                    if inspect.ismethod(attr):
                        # Create a wrapper function that calls the method
                        method = attr
                        
                        # Create a proper wrapper that preserves async/sync behavior
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
                        # For non-methods, we need to create a bound method
                        method = attr
                        
                        # Create a proper wrapper that preserves async/sync behavior and binds to instance
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

    @tool("Self‑reflect: critique current plan and propose new one.")
    async def self_reflect_and_replan(self, critique: str, new_plan: List[str]):
        return {"meta": "reflect", "critique": critique, "plan": new_plan}

    # ── ReVAL specific tools ───────────────────────────────────────────────
    @tool("Estimate problem complexity (0‑1) for dynamic budgeting.")
    async def complexity_estimator(self, prompt: str) -> float:
        score = min(len(prompt.split()) / 4000, 1.0)
        return score

    @tool("Simple self‑verifier. Returns True if answer likely correct (stub).")
    async def simple_verifier(self, answer: str, question: str) -> bool:
        # Placeholder implementation; always returns True.
        return True

    # ── Dynamic toolsmith ──────────────────────────────────────────────────
    @tool("Create a new Python tool, test it, and register it if tests pass.")
    async def create_and_test_tool(
        self,
        tool_name: str,
        description: str,
        python_code: str,
        test_code: str,
    ) -> str:
        ns: Dict[str, Any] = {}
        exec(python_code, globals(), ns)
        if tool_name not in ns:
            return "Error: function not defined."
        fn = ns[tool_name]
        exec(test_code, globals(), {"candidate": fn})
        setattr(self, tool_name, fn.__get__(self))
        self.register_tool(getattr(self, tool_name))
        self.tools[tool_name].description = description  # type: ignore
        return f"Tool '{tool_name}' created and registered."

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
            "Tools are available via *native function‑calling* (if supported) **or** via a JSON fall‑back.\n\n"  # noqa: E501
            "When using the fall‑back, reply **only** with one JSON block:\n"
            "```json\n{\"tool_call\": {\"name\": <tool_name>, \"arguments\": {…}}}\n```"
        )
        self.system_prompt = f"{self.persona_prompt}\n\n### Tools\n{tools_doc}\n\n### Usage\n{usage_doc}"

    # ======================================================================
    # LLM invocation helpers
    # ======================================================================
    async def _call_llm(self, messages: List[Dict]) -> Dict:
        """Call the LLM with graceful degradation between modes."""
        self._debug_step(f"{Colors.BOLD}{Colors.BLUE}🧠 Calling LLM with {len(messages)} messages{Colors.END}")
        if self.debug:
            self.log.debug(f"Last message: {messages[-1].get('content', '')[:100]}...")
            
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
            self._debug_step(f"{Colors.CYAN}Using native function calling with {len(fc_schema)} tools{Colors.END}")
        else:
            self._debug_step(f"{Colors.YELLOW}Using JSON fallback for tool calling{Colors.END}")
            
        params = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_response_tokens,
            tools=fc_schema if want_native else None,
            tool_choice="auto" if want_native else None,
        )
        
        # Add OpenRouter specific parameters if using OpenRouter
        if hasattr(self, "is_openrouter") and self.is_openrouter:
            self._debug_step("Adding OpenRouter specific parameters")
            # OpenRouter uses these headers
            headers = {
                "HTTP-Referer": os.getenv("HTTP_REFERER", "https://synapsereval.local"),
                "X-Title": os.getenv("X_TITLE", "SynapseREval")
            }
            params["extra_headers"] = headers
            params["extra_body"] = {
                "transforms": ["middle-out"],  # Recommended by OpenRouter
            }
        
        # Log the request parameters if in debug mode
        self._debug_llm_request(params)
            
        for attempt in range(3):
            try:
                self._debug_step(f"LLM request attempt {attempt+1}/3")
                resp = await self.client.chat.completions.create(**params)
                
                # Check for error in response
                if getattr(resp, "error", None):
                    error_code = resp.error.get("code", "unknown")
                    error_msg = resp.error.get("message", "Unknown error")
                    self._debug_step(f"{Colors.RED}❌ LLM backend error {error_code}: {error_msg}{Colors.END}")
                    raise RuntimeError(f"LLM backend error {error_code}: {error_msg}")
                
                # Check for missing choices
                if not getattr(resp, "choices", None):
                    self._debug_step(f"{Colors.RED}❌ LLM returned no choices{Colors.END}")
                    raise RuntimeError("LLM returned no choices")
                
                self._fc_supported = want_native
                self._debug_step(f"{Colors.GREEN}✅ LLM request successful{Colors.END}")
                
                # Log the response if in debug mode
                self._debug_llm_response(resp)
                
                return resp.choices[0].message.model_dump()
            except OpenAIError as err:
                self._debug_step(f"{Colors.RED}❌ LLM error: {err}{Colors.END}")
                if want_native and (
                    getattr(err, "status_code", None) == 404 or "No endpoints" in str(err)
                ):
                    # function‑calling not supported → downgrade once
                    self._debug_step(f"{Colors.YELLOW}⚠️ Function calling not supported, downgrading to JSON fallback{Colors.END}")
                    self._fc_supported = False
                    want_native = False
                    params["tools"] = None
                    params["tool_choice"] = None
                    continue
                if isinstance(err, (RateLimitError, APIConnectionError)):
                    wait_time = 2 ** attempt
                    self._debug_step(f"{Colors.YELLOW}⚠️ Rate limit or connection error, retrying in {wait_time}s{Colors.END}")
                    await asyncio.sleep(wait_time)
                    continue
                if attempt < 2:  # Try again for any other error, but only twice
                    wait_time = 2 ** attempt
                    self._debug_step(f"{Colors.YELLOW}⚠️ OpenAI error, retrying in {wait_time}s{Colors.END}")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as ex:
                self._debug_step(f"{Colors.RED}❌ Unexpected error: {ex}{Colors.END}")
                if attempt < 2:  # Try again for any other error, but only twice
                    wait_time = 2 ** attempt
                    self._debug_step(f"{Colors.YELLOW}⚠️ Unexpected error, retrying in {wait_time}s{Colors.END}")
                    await asyncio.sleep(wait_time)
                    continue
                raise
        self._debug_step(f"{Colors.RED}{Colors.BOLD}❌ LLM FAILED AFTER ALL RETRIES{Colors.END}")
        raise RuntimeError("LLM failed after retries")
        
    def _debug_llm_request(self, params: Dict):
        """Log the LLM request parameters if in debug mode."""
        if not self.debug:
            return
            
        # Create a safe copy of the parameters to log
        safe_params = params.copy()
        
        # Remove sensitive information
        if "extra_headers" in safe_params:
            safe_params["extra_headers"] = {k: v for k, v in safe_params["extra_headers"].items()}
            
        # Truncate messages for readability
        if "messages" in safe_params:
            safe_params["messages"] = [
                {
                    "role": m.get("role", "unknown"),
                    "content": (m.get("content", "")[:50] + "...") if m.get("content") else None,
                    "has_tool_calls": "tool_calls" in m,
                    "is_tool": "name" in m and m.get("role") == "tool",
                }
                for m in safe_params["messages"]
            ]
            
        # Truncate tool definitions for readability
        if "tools" in safe_params and safe_params["tools"]:
            safe_params["tools"] = [
                {
                    "type": t.get("type", "unknown"),
                    "function": {
                        "name": t.get("function", {}).get("name", "unknown"),
                        "description": (t.get("function", {}).get("description", "")[:50] + "..."),
                        "params_count": len(t.get("function", {}).get("parameters", {}).get("properties", {})),
                    }
                }
                for t in safe_params["tools"]
            ]
            
        self.log.debug(f"LLM REQUEST: {json.dumps(safe_params, default=str)}")
        
    def _debug_llm_response(self, resp):
        """Log the LLM response if in debug mode."""
        if not self.debug:
            return
            
        try:
            # Extract key information from the response
            resp_data = {
                "id": getattr(resp, "id", None),
                "model": getattr(resp, "model", None),
                "usage": getattr(resp, "usage", None),
                "choices": [],
                "error": getattr(resp, "error", None)
            }
            
            # Extract choice information if present
            choices = getattr(resp, "choices", [])
            for choice in choices:
                choice_data = {
                    "index": getattr(choice, "index", None),
                    "finish_reason": getattr(choice, "finish_reason", None),
                }
                
                # Extract message information
                message = getattr(choice, "message", None)
                if message:
                    msg_data = {
                        "role": getattr(message, "role", None),
                        "content": (getattr(message, "content", "")[:100] + "...") if getattr(message, "content", None) else None,
                    }
                    
                    # Extract tool calls if present
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        msg_data["tool_calls"] = []
                        for tc in tool_calls:
                            tc_data = {
                                "id": getattr(tc, "id", None),
                                "type": getattr(tc, "type", None),
                            }
                            
                            # Extract function information
                            function = getattr(tc, "function", None)
                            if function:
                                tc_data["function"] = {
                                    "name": getattr(function, "name", None),
                                    "arguments": (getattr(function, "arguments", "")[:50] + "...") if getattr(function, "arguments", None) else None,
                                }
                                
                            msg_data["tool_calls"].append(tc_data)
                            
                    choice_data["message"] = msg_data
                    
                resp_data["choices"].append(choice_data)
                
            self.log.debug(f"LLM RESPONSE: {json.dumps(resp_data, default=str)}")
        except Exception as e:
            self.log.debug(f"Error logging LLM response: {e}")
            self.log.debug(f"LLM RESPONSE RAW: {resp}")

    # ======================================================================
    # Tool execution helper (with large‑payload off‑loading)
    # ======================================================================
    async def _execute_tool(self, name: str, args: Dict[str, Any]):
        self._debug_step(f"{Colors.CYAN}{Colors.BOLD}🔧 Executing tool: {name}{Colors.END} with args: {args}")
        if name not in self.tools:
            self._debug_step(f"{Colors.RED}❌ Unknown tool: {name}{Colors.END}")
            return {"error": f"Unknown tool {name}"}
        spec = self.tools[name]
        try:
            validated = spec.args_schema(**args)
            self._debug_step(f"Arguments validated for {name}")
            result = await spec.handler(**validated.dict())
            self._debug_step(f"{Colors.GREEN}✅ Tool {name} executed successfully{Colors.END}")
            
            # Large payload? → off‑load to scratchpad
            if (
                isinstance(result, (dict, list, str))
                and len(json.dumps(result, default=str)) > 4000
            ):
                key = self.scratch.store(result, ttl=300)
                self._debug_step(f"{Colors.YELLOW}📦 Large result from {name} stored in scratchpad with key {key}{Colors.END}")
                return {"scratchpad_key": key, "info": f"{name} output stored (large payload)"}
            return result
        except ValidationError as ve:
            self._debug_step(f"{Colors.RED}❌ Validation error in {name}: {ve}{Colors.END}")
            return {"error": str(ve)}
        except Exception as exc:  # noqa: B902
            self._debug_step(f"{Colors.RED}❌ Exception in {name}: {exc}{Colors.END}")
            if self.debug:
                self.log.exception(f"Tool execution error in {name}")
            return {"error": str(exc)}

    # ======================================================================
    # Parsing helpers
    # ======================================================================
    def _extract_tool_calls(self, resp: Dict) -> Optional[List[Dict]]:
        """Return a list of tool‑call dicts or None (for both modes)."""
        # Native function‑calling
        if resp.get("tool_calls"):
            if self.debug:
                self.log.debug(f"{Colors.CYAN}Found native tool calls: {len(resp['tool_calls'])}{Colors.END}")
            return resp["tool_calls"]
        
        # JSON fall‑back parsing
        content = resp.get("content")
        if not content:
            if self.debug:
                self.log.debug(f"{Colors.RED}No content in response{Colors.END}")
            return None
            
        m = self._JSON_RE.search(content)
        if not m:
            if self.debug:
                self.log.debug(f"{Colors.RED}No JSON block found in content{Colors.END}")
            return None
            
        try:
            blob = json.loads(m.group(1))
            if "tool_call" in blob and {"name", "arguments"} <= set(blob["tool_call"].keys()):
                tc = blob["tool_call"]
                tool_id = f"tc_{uuid.uuid4().hex[:6]}"
                if self.debug:
                    self.log.debug(f"{Colors.CYAN}Extracted JSON tool call: {tc['name']} (ID: {tool_id}){Colors.END}")
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
            else:
                if self.debug:
                    self.log.debug(f"{Colors.RED}Invalid tool_call format in JSON: {blob}{Colors.END}")
        except Exception as e:
            if self.debug:
                self.log.debug(f"{Colors.RED}Error parsing JSON tool call: {str(e)}{Colors.END}")
            return None
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
    # Utility
    # ======================================================================
    def _tokens(self, txt: str | None) -> int:
        return len(self._enc.encode(txt or ""))

    def _trim(self, msgs: List[Dict]):
        budget = self.max_model_tokens - self.max_response_tokens
        while sum(self._tokens(m.get("content")) for m in msgs) > budget and len(msgs) > 3:
            msgs.pop(1)
        return msgs

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
        for cycle in range(max_cycles):  # max ReVAL cycles
            self._debug_step(f"{Colors.MAGENTA}{Colors.BOLD}ReVAL CYCLE {cycle+1}/{max_cycles} STARTED{Colors.END}", reval_step=True)
            
            # REASON phase
            self._debug_step(f"{Colors.YELLOW}REASON PHASE: Preparing to call LLM{Colors.END}", reval_step=True)
            msgs = self._trim(msgs)
            self._debug_step(f"Messages trimmed to {len(msgs)} messages")
            
            assistant = await self._call_llm(msgs)
            msgs.append(assistant)
            self._debug_step("Assistant response received and added to messages")
            
            # Dump conversation after assistant response
            self._debug_dump_conversation([assistant], prefix=f"CYCLE {cycle+1} ASSISTANT ")

            # VERIFY phase - Confidence gating (look for CONF=x.y in content)
            self._debug_step(f"{Colors.YELLOW}VERIFY PHASE: Checking confidence{Colors.END}", reval_step=True)
            conf_match = re.search(r"CONF\s*=\s*([0-9.]+)", assistant.get("content", ""))
            if conf_match:
                conf_val = float(conf_match.group(1))
                self._debug_step(f"Confidence value detected: {conf_val}")
                await self.update_goal_state(confidence=conf_val)
                if conf_val < 0.7:
                    self._debug_step(f"{Colors.RED}Low confidence {conf_val} detected{Colors.END}", reval_step=True)
                    
                    # ADAPT phase
                    self._debug_step(f"{Colors.YELLOW}ADAPT PHASE: Initiating self-reflection{Colors.END}", reval_step=True)
                    reflection = await self.self_reflect_and_replan(
                        critique="Low confidence", new_plan=["Retry with deeper reasoning"]
                    )
                    msgs.append({"role": "assistant", "content": json.dumps(reflection)})
                    self._debug_step("Self-reflection added to messages")
                    continue

            # Tool phase (part of REASON)
            tool_calls = self._extract_tool_calls(assistant)
            if not tool_calls:
                self._debug_step(f"{Colors.GREEN}COMPLETION: No tool calls, returning final response{Colors.END}", reval_step=True)
                return assistant.get("content", "")

            self._debug_step(f"Extracted {len(tool_calls)} tool calls")
            
            async def timed_exec(call):
                tool_name = call["function"]["name"]
                self._debug_step(f"Executing tool call: {tool_name}")
                try:
                    result = await asyncio.wait_for(
                        self._execute_tool(
                            tool_name,
                            json.loads(call["function"]["arguments"]),
                        ),
                        timeout=25,
                    )
                    self._debug_step(f"{Colors.GREEN}Tool {tool_name} execution completed{Colors.END}")
                    return result
                except asyncio.TimeoutError:
                    self._debug_step(f"{Colors.RED}Tool {tool_name} execution timed out{Colors.END}")
                    return {"error": "tool timeout"}

            results = await asyncio.gather(*[timed_exec(c) for c in tool_calls])
            self._debug_step(f"All tool calls executed, got {len(results)} results")

            # VERIFY phase - verification pass
            self._debug_step(f"{Colors.YELLOW}VERIFY PHASE: Checking results{Colors.END}", reval_step=True)
            if results:
                self._debug_step("Running verification on first result")
                ok = await self.simple_verifier(answer=str(results[0]), question=user_prompt)
                if not ok:
                    # ADAPT phase
                    self._debug_step(f"{Colors.RED}Verification failed{Colors.END}", reval_step=True)
                    self._debug_step(f"{Colors.YELLOW}ADAPT PHASE: Initiating self-reflection{Colors.END}", reval_step=True)
                    reflection = await self.self_reflect_and_replan(
                        critique="Verifier failed", new_plan=["Revise answer"]
                    )
                    msgs.append({"role": "assistant", "content": json.dumps(reflection)})
                    continue
                self._debug_step(f"{Colors.GREEN}Verification passed{Colors.END}", reval_step=True)

            # LOOP phase
            self._debug_step(f"{Colors.YELLOW}LOOP PHASE: Processing tool results{Colors.END}", reval_step=True)
            tool_messages = self._tool_messages(tool_calls, results)
            msgs.extend(tool_messages)
            self._debug_step(f"Added {len(tool_messages)} tool messages to conversation")
            
            # Dump conversation after tool responses
            self._debug_dump_conversation(tool_messages, prefix=f"CYCLE {cycle+1} TOOLS ")
            self._debug_step(f"{Colors.MAGENTA}{Colors.BOLD}ReVAL CYCLE {cycle+1} COMPLETED{Colors.END}", reval_step=True)
            
        self._debug_step(f"{Colors.RED}REACHED MAX REASONING CYCLES{Colors.END}", reval_step=True)
        return "⚠️ Reached max reasoning cycles."

