"""synapses.py

A *bio‑inspired* multi‑agent orchestration layer that treats each **Agent** as a
neuron and the message pathways as weighted synapses. This version introduces:

───────────────────────────────────────────────────────────────────────────────
✨ PER‑NEURON MEMORY      – local KV store & rolling transcript per neuron
✨ STDP PLASTICITY        – spike‑timing dependent synaptic updates
✨ MULTI‑MODAL SPIKES     – each spike can carry text / image / audio payloads
✨ ATTENTIONAL ROUTING    – softmax attention over outgoing synapses
✨ PERSISTENCE            – JSON (de)serialisation for long‑lived brains
✨ META‑SUPERVISOR        – a special neuron that rewires topology at runtime

Dependencies:  networkx, asyncio, numpy (for softmax) – all standard PyPI.
"""
from __future__ import annotations
import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from llm import UltimateReVALAgent
import networkx as nx
import numpy as np

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
log = logging.getLogger("SynapseNetwork")

###############################################################################
# Helpers
###############################################################################

NOW = lambda: time.time()  # simple wall‑clock helper
TAU = 0.2                  # STDP time constant (seconds)
MIN_W, MAX_W = 0.05, 5.0   # synaptic weight bounds


def softmax(vec: List[float]) -> List[float]:
    """Numerically stable softmax."""
    v = np.array(vec, dtype=float)
    v -= v.max()
    e = np.exp(v)
    return (e / e.sum()).tolist()


###############################################################################
# Core data structures
###############################################################################

@dataclass
class Spike:
    """Spike carrying payload + meta."""

    payload: Any  # can be str for text, bytes for image/audio, etc.
    modality: str = "text"  # text | image | audio | sensor
    weight: float = 1.0
    timestamp: float = field(default_factory=NOW)


@dataclass
class Synapse:
    """Directed connection with STDP‑capable plasticity."""

    src: str
    dst: str
    weight: float = 1.0
    plastic: bool = True
    last_pre_spike: float = field(default_factory=NOW)
    last_post_spike: float = field(default_factory=NOW)

    # --------------------------------------------------------------------- STDP
    def update_stdp(self, pre_t: float, post_t: float, lr: float):
        """Spike‑timing‑dependent plasticity.

        Δw = +lr·exp(-Δt/τ)  if pre precedes post (LTP)
             -lr·exp(+Δt/τ)  if post precedes pre (LTD)
        """
        if not self.plastic:
            return
        delta_t = post_t - pre_t
        dw: float
        if delta_t > 0:  # potentiation
            dw = lr * math.exp(-delta_t / TAU)
        else:           # depression
            dw = -lr * math.exp(delta_t / TAU)
        self.weight = float(min(max(self.weight + dw, MIN_W), MAX_W))
        self.last_pre_spike, self.last_post_spike = pre_t, post_t
        log.debug("STDP: %s→%s Δt=%.3f dw=%.4f w=%.3f", self.src, self.dst, delta_t, dw, self.weight)


@dataclass
class NeuronMemory:
    """Simple rolling memory per neuron (could be swapped with vectordb)."""

    capacity: int = 128  # number of recent entries to keep
    store: Deque[Tuple[float, Any]] = field(default_factory=deque)

    def add(self, item: Any):
        self.store.append((NOW(), item))
        if len(self.store) > self.capacity:
            self.store.popleft()

    def dump(self) -> List[Tuple[float, Any]]:
        return list(self.store)


@dataclass
class Neuron:
    """Agent wrapper acting as a neuron."""

    agent: Any  # must expose async chat(prompt, history, **kwargs)
    threshold: float = 1.0
    role: str = "processor"
    inbox: Deque[Spike] = field(default_factory=deque)
    memory: NeuronMemory = field(default_factory=NeuronMemory)
    last_fire: float = field(default_factory=NOW, init=False)

    # ------------------------------------------------------------------- Spikes
    async def maybe_fire(self, net_history: List[Dict[str, Any]]) -> Optional[Spike]:
        """If accumulated weight exceeds threshold → fire LLM."""
        total = sum(sp.weight for sp in self.inbox)
        if total < self.threshold:
            return None

        # Aggregate textual prompts only for now; skip other modalities in prompt.
        prompt_parts = [str(sp.payload) for sp in self.inbox if sp.modality == "text"]
        self.inbox.clear()

        try:
            response: str = await self.agent.chat("\n".join(prompt_parts), net_history)
        except Exception as ex:  # noqa: BLE001
            log.error("LLM error in %s: %s", self.role, ex)
            return None

        self.memory.add(response)
        self.last_fire = NOW()
        return Spike(payload=response, modality="text", weight=1.0, timestamp=self.last_fire)


###############################################################################
# Network
###############################################################################

class SynapseNetwork:
    """Bio‑inspired agentic network."""

    def __init__(self, learning_rate: float = 0.02):
        self.G = nx.DiGraph()
        self.neurons: Dict[str, Neuron] = {}
        self.history: List[Dict[str, Any]] = []  # global transcript
        self.lr = learning_rate

    # ----------------------------------------------------------------- Building
    def add_neuron(self, name: str, neuron: Neuron):
        self.G.add_node(name, role=neuron.role)
        self.neurons[name] = neuron
        log.info("Neuron added: %s (%s)", name, neuron.role)

    def connect(self, src: str, dst: str, *, weight: float = 1.0, plastic: bool = True):
        self.G.add_edge(src, dst, syn=Synapse(src, dst, weight, plastic))
        log.info("Synapse created: %s → %s  w=%.2f", src, dst, weight)

    # ------------------------------------------------------------- Persistence
    def save(self, path: str | Path):
        path = Path(path)
        data = {
            "nodes": {
                n: {
                    "role": self.neurons[n].role,
                    "threshold": self.neurons[n].threshold,
                    "memory": self.neurons[n].memory.dump(),
                }
                for n in self.G.nodes
            },
            "edges": [
                {
                    "src": u,
                    "dst": v,
                    "weight": d["syn"].weight,
                    "plastic": d["syn"].plastic,
                }
                for u, v, d in self.G.edges(data=True)
            ],
            "history": self.history,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        log.info("Network state saved → %s", path)

    @classmethod
    def load(cls, path: str | Path, agent_factory: Callable[[str, str], Any], lr: float = 0.02) -> "SynapseNetwork":
        """agent_factory(name, role) -> Agent instance"""
        path = Path(path)
        data = json.loads(path.read_text())
        net = cls(learning_rate=lr)
        for name, meta in data["nodes"].items():
            agent = agent_factory(name, meta["role"])
            neuron = Neuron(agent, threshold=meta["threshold"], role=meta["role"])
            for t, m in meta["memory"]:
                neuron.memory.add((t, m))
            net.add_neuron(name, neuron)
        for e in data["edges"]:
            net.connect(e["src"], e["dst"], weight=e["weight"], plastic=e["plastic"])
        net.history = data["history"]
        log.info("Network state loaded ← %s", path)
        return net

    # ---------------------------------------------------------------- Utilities
    def inject(self, dst: str, payload: Any, *, modality: str = "text", weight: float = 1.0):
        sp = Spike(payload=payload, modality=modality, weight=weight)
        self.neurons[dst].inbox.append(sp)
        log.debug("Injected spike into %s (%.2f)", dst, weight)

        # ------------------------------------------------------------------- Runner
    async def run(
        self,
        *,
        max_cycles: int = 6,
        entry_neuron: str | None = None,
        initial_prompt: str | None = None,
    ):
        """
        Execute the network for at most `max_cycles` synchronous steps.

        The traversal order is:

        - If the graph is a DAG  → use `networkx.topological_sort`
          (gives a stable, causally sensible order).
        - Otherwise             → fall back to plain node insertion order.
          This keeps feedback loops alive without raising
          `networkx.NetworkXUnfeasible`.
        """
        if entry_neuron and initial_prompt:
            self.inject(entry_neuron, initial_prompt, weight=1.0)

        for cycle in range(1, max_cycles + 1):
            log.info("────────── Cycle %d/%d ──────────", cycle, max_cycles)
            fired: Dict[str, Spike] = {}

            # ─────────────────────── choose a deterministic traversal order
            if nx.is_directed_acyclic_graph(self.G):
                node_order = list(nx.topological_sort(self.G))
            else:
                # graph has ≥1 cycles → just use node insertion order
                node_order = list(self.G)

            # ─────────────────────── neuron updates
            for node in node_order:
                neuron = self.neurons[node]
                sp_out = await neuron.maybe_fire(self.history)
                if sp_out is None:
                    continue
                fired[node] = sp_out
                log.info(
                    "⚡ %s fired → %.30s",
                    node,
                    str(sp_out.payload).replace("\n", " "),
                )

            if not fired:
                log.info("Stable state reached – no spikes fired.")
                break

            # ─────────────────────── STDP on edges that saw pre+post spikes
            for post, post_sp in fired.items():
                for pre in self.G.predecessors(post):
                    if pre in fired:
                        syn: Synapse = self.G.edges[pre, post]["syn"]
                        syn.update_stdp(
                            fired[pre].timestamp, post_sp.timestamp, self.lr
                        )

            # ─────────────────────── propagate with soft-attention routing
            for src, sp in fired.items():
                outs = list(self.G.out_edges(src, data=True))
                if not outs:
                    continue
                weights = [d["syn"].weight for *_ , d in outs]
                probs = softmax(weights)
                for (_u, v, d), p in zip(outs, probs):
                    if np.random.rand() < p:           # stochastic routing
                        w_eff = d["syn"].weight * sp.weight
                        self.neurons[v].inbox.append(
                            Spike(sp.payload, sp.modality, w_eff)
                        )
                        log.debug(
                            "Spike routed %s → %s  p=%.2f w=%.2f", src, v, p, w_eff
                        )

            # ─────────────────────── append to global transcript
            for n, sp in fired.items():
                self.history.append(
                    {
                        "cycle": cycle,
                        "neuron": n,
                        "payload": sp.payload,
                        "modality": sp.modality,
                        "timestamp": sp.timestamp,
                    }
                )

        return self.history

    ############################################################################
    # Dynamic topology tools – could be exposed to LLM agents as JSON‑tools
    ############################################################################

    async def tool_add_neuron(self, name: str, role: str, agent_factory: Callable[[str, str], Any], threshold: float = 1.0):
        if name in self.neurons:
            return f"Neuron '{name}' already exists."
        self.add_neuron(name, Neuron(agent_factory(name, role), threshold=threshold, role=role))
        return f"Neuron '{name}' added."

    async def tool_connect(self, src: str, dst: str, weight: float = 1.0, plastic: bool = True):
        if not (src in self.neurons and dst in self.neurons):
            return "Both src and dst neurons must exist."
        self.connect(src, dst, weight=weight, plastic=plastic)
        return f"Synapse {src}→{dst} created."

    async def tool_prune(self, src: str, dst: str):
        if not self.G.has_edge(src, dst):
            return "Edge not found."
        self.G.remove_edge(src, dst)
        return f"Synapse {src}→{dst} removed."


###############################################################################
# Demo (requires your own UltimateAgent implementation)
###############################################################################

if __name__ == "__main__":
    import importlib

    async def main():
        # Lazy import your agent implementation
        
        def agent_factory(name: str, role: str):
            prompt = f"You are {role} named {name}. Respond succinctly."
            return UltimateReVALAgent(
                model="deepseek/deepseek-chat-v3-0324:free",
                tool_support=True,
                temperature=0.2,
                max_model_tokens=16_000,
                max_response_tokens=2_048,
                persona_prompt=prompt,
                debug=True,
                debug_log_file="agent_log.txt",
            )

        net = SynapseNetwork(learning_rate=0.05)
        # Core neurons
        net.add_neuron("Planner", Neuron(agent_factory("Planner", "planner"), threshold=1.0, role="planner"))
        net.add_neuron("Executor", Neuron(agent_factory("Executor", "executor"), threshold=1.0, role="executor"))
        net.add_neuron("Finalizer", Neuron(agent_factory("Finalizer", "finalizer"), threshold=1.0, role="finalizer"))

        # Synapses
        net.connect("Planner", "Executor", weight=1.2)
        net.connect("Executor", "Finalizer", weight=1.0)
        net.connect("Finalizer", "Planner", weight=0.4)

        # Inject initial question
        await net.run(max_cycles=8, entry_neuron="Planner", initial_prompt="Résume la météo à Casablanca pour aujourd'hui.")

        # Persist
        net.save("brain.json")

        print("History size:", len(net.history))

    asyncio.run(main())
