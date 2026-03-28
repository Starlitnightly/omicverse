"""Integration test harness for OmicVerse Agent and Jarvis layers.

This package provides reusable fake implementations and helpers that let
integration tests exercise agent and Jarvis runtime behavior without real
LLM providers or network dependencies.

Submodules
----------
fakes : Fake LLM, tool-runtime, and Jarvis presenter/adapter/router classes.
helpers : Factory functions for common data objects and agent wiring.
conftest : Pytest fixtures that expose the fakes as session-ready test doubles.
"""
