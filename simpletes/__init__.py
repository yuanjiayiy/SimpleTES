"""
SimpleTES — LLM-driven code evolution.

Main components:
    - EngineConfig: Configuration for the evolution engine
    - SimpleTESEngine: Main evolution engine
    - Node, NodeDatabase: Program node data model and storage
    - Selector: Base class for inspiration selection policies

Exports are resolved lazily so that lightweight submodules (e.g.
``simpletes.construction``, imported by ``sitecustomize`` inside task-local
eval venvs) do not pull in litellm / rich / the engine.
"""
from importlib import import_module

_EXPORTS = {
    "EngineConfig": "simpletes.config",
    "SimpleTESEngine": "simpletes.engine",
    "GenerationTask": "simpletes.generator",
    "Evaluator": "simpletes.evaluator",
    "EvaluatorWorker": "simpletes.evaluator",
    "LLMBackend": "simpletes.llm",
    "LLMCallError": "simpletes.llm",
    "LLMClient": "simpletes.llm",
    "create_llm_client": "simpletes.llm",
    "Node": "simpletes.node",
    "NodeDatabase": "simpletes.node",
    "Status": "simpletes.node",
    "extract_code": "simpletes.node",
    "Selector": "simpletes.policies",
    "create_selector": "simpletes.policies",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'simpletes' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
