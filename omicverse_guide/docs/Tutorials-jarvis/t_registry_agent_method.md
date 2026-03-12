# OmicVerse Registry-Centered Agent Architecture

```text
OmicVerse registry-centered agent architecture
==============================================

[Source code layer]
    |
    |  Public API definitions in OmicVerse modules
    |  ------------------------------------------------------------
    |  A. Explicitly registered entry points
    |     - @register_function on functions
    |     - @register_function on classes
    |
    |  B. Latent capabilities hidden inside registered objects
    |     - public class methods
    |     - string-dispatch branches
    |       e.g. method='dynamo'
    |            backend='scvelo'
    |            method='bayesprism'
    |            method='regdiffusion'
    |
    |  C. Backend evidence inside branch bodies
    |     - import dynamo
    |     - import regdiffusion
    |     - import celltypist
    |     - from ... import Prism
    v

[Registry build layer: FunctionRegistry.register]
    |
    |  1. Build canonical runtime entry
    |     ------------------------------------------------------------
    |     fields:
    |     - function
    |     - full_name
    |     - short_name
    |     - module
    |     - aliases
    |     - category
    |     - description
    |     - examples
    |     - related
    |     - signature
    |     - parameters
    |     - docstring
    |     - prerequisites / requires / produces
    |     - source = runtime
    |     - virtual_entry = False
    |
    |  2. Derive method-level virtual entries from registered classes
    |     ------------------------------------------------------------
    |     e.g.
    |     - omicverse.single._velo.Velo.moments
    |     - omicverse.single._scenic.SCENIC.cal_grn
    |
    |  3. Derive branch-level virtual entries from selector branches
    |     ------------------------------------------------------------
    |     e.g.
    |     - ...Velo.moments[backend=dynamo]
    |     - ...Deconvolution.deconvolution[method=bayesprism]
    |     - ...SCENIC.cal_grn[method=regdiffusion]
    |
    |  4. Attach branch metadata
    |     ------------------------------------------------------------
    |     - source = runtime_derived_method / runtime_derived_branch
    |     - virtual_entry = True
    |     - parent_full_name
    |     - branch_parameter
    |     - branch_value
    |     - imports
    |
    |  5. Store all entries into _global_registry
    v

[_global_registry]
    |
    |  Searchable memory structure
    |  ------------------------------------------------------------
    |  contains:
    |  - canonical registered entries
    |  - virtual method entries
    |  - virtual branch entries
    |
    |  searchable over:
    |  - full_name
    |  - short_name
    |  - aliases
    |  - category
    |  - description
    |  - docstring
    |  - examples
    |  - related
    |  - imports
    v

[Hydration layer: ensure_registry_populated()]
    |
    |  Problem:
    |  OmicVerse uses lazy imports, so decorators do not fire until modules load.
    |
    |  Hydration procedure:
    |  ------------------------------------------------------------
    |  1. Read PHASE_WHITELIST from MCP overrides
    |  2. Convert whitelisted full_names -> module paths
    |  3. Try importlib.import_module(module)
    |  4. If package import fails for any reason:
    |     - attempt direct leaf-module loading
    |     - bypass fragile package __init__ chains
    |  5. Decorators execute
    |  6. _global_registry becomes populated
    v

[Export layer]
    |
    |  export_registry(filepath)
    |  ------------------------------------------------------------
    |  serializes unique entries from _global_registry
    |  output fields include:
    |  - full_name / short_name / module
    |  - aliases / category / description / examples
    |  - signature / docstring
    |  - source
    |  - virtual_entry
    |  - parent_full_name
    |  - branch_parameter
    |  - branch_value
    |  - imports
    v

[MCP manifest layer]
    |
    |  build_registry_manifest(...)
    |  ------------------------------------------------------------
    |  uses _global_registry as source of truth
    |  BUT filters out virtual_entry == True
    |
    |  rationale:
    |  - searchable branch variants should aid retrieval
    |  - they should not be exposed as independently executable tools
    v

[Agent initialization: OmicVerseAgent.__init__]
    |
    |  1. initialize skill registry
    |  2. hydrate / access registry context
    |  3. build system prompt
    |  4. create tool runtime + turn controller
    |
    |  system prompt contents
    |  ------------------------------------------------------------
    |  - OmicVerse task instructions
    |  - workflow rules
    |  - code quality rules
    |  - full registry summary text
    |  - skill catalog overview
    v

[Reasoning-time tool use]
    |
    |  Jarvis / Claw agentic loop
    |  ------------------------------------------------------------
    |  inspect_data()     -> understand current dataset state
    |  search_functions() -> query _global_registry
    |  search_skills()    -> retrieve workflow guidance
    |  execute_code()     -> final OmicVerse code action
    |  finish()           -> terminate turn
    |
    |  retrieval examples
    |  ------------------------------------------------------------
    |  query = "dynamo"
    |  -> Velo.moments[backend=dynamo]
    |  -> Velo.dynamics[backend=dynamo]
    |  -> Velo.cal_velocity[method=dynamo]
    |
    |  query = "bayesprism"
    |  -> Deconvolution.deconvolution[method=bayesprism]
    |
    |  query = "regdiffusion"
    |  -> SCENIC.cal_grn[method=regdiffusion]
    v

[Interface split]
    |
    |  Jarvis
    |  ------------------------------------------------------------
    |  - same agent
    |  - same registry
    |  - same skills
    |  - same search_functions / search_skills logic
    |  - execute_code really runs
    |
    |  Claw
    |  ------------------------------------------------------------
    |  - same agent
    |  - same registry
    |  - same skills
    |  - same search_functions / search_skills logic
    |  - execute_code is intercepted
    |  - final code is captured and returned without execution
    v

[Outcome]
    |
    |  Registry serves three roles simultaneously:
    |  ------------------------------------------------------------
    |  1. capability index for discovery
    |  2. prompt-grounding substrate for the agent
    |  3. exportable machine-readable API/capability snapshot
```

We implemented a registry-centered agent architecture in OmicVerse to make analytical capabilities machine-readable, retrievable, and executable through a unified interface. In this design, the runtime registry rather than repository-wide source search is treated as the primary representation of available functionality. Public OmicVerse capabilities are declared through `@register_function`, which records symbolic and semantic metadata including aliases, category labels, natural-language descriptions, examples, related entries, signatures, and state annotations such as prerequisites, required data structures, and expected outputs. Each decorated object yields a canonical runtime entry in `_global_registry`, establishing an explicit mapping between a public OmicVerse abstraction and an agent-consumable representation.

Decorator registration alone is insufficient for OmicVerse because a substantial fraction of biologically meaningful functionality is not exposed as standalone top-level functions. Many workflows are implemented as classes whose real analytical actions reside in public methods, and many backend-specific behaviors are selected through string-dispatch branches such as `method='dynamo'`, `backend='scvelo'`, `method='bayesprism'`, or `method='regdiffusion'`. To recover these hidden capabilities, the registry expands each registered object at runtime using abstract syntax tree analysis. Registered classes are converted into method-level virtual entries, and methods or functions containing selector branches are further expanded into branch-specific virtual entries. These derived entries preserve the semantic context of the parent callable while adding branch-specific metadata, including the selector parameter, selector value, and branch-local import evidence. This enables the registry to represent capability variants such as `Velo.cal_velocity[method=dynamo]`, `Deconvolution.deconvolution[method=bayesprism]`, and `SCENIC.cal_grn[method=regdiffusion]` even though these variants are not independent top-level APIs.

The resulting `_global_registry` is therefore a searchable capability graph rather than a simple name map. Entries can be matched through full names, short names, aliases, categories, descriptions, docstrings, examples, related-function annotations, and backend import tokens. This is particularly important for natural-language agent use, because users often refer to a backend package or workflow name rather than the exact OmicVerse wrapper symbol. By preserving import evidence and selector values as searchable fields, the registry can resolve backend-oriented requests to the appropriate OmicVerse wrapper branch. This design substantially improves retrieval fidelity for workflows whose scientific identity is carried by an internal backend choice rather than by a unique top-level function name.

Because OmicVerse uses lazy imports, registry completeness cannot be assumed at interpreter startup. We therefore use an explicit hydration procedure, `ensure_registry_populated()`, which loads a curated whitelist of modules before retrieval or agent reasoning begins. The hydration process first attempts standard module import and, when package initialization fails, falls back to direct leaf-module loading in order to bypass fragile package `__init__` chains. This behavior is necessary because some package-level imports transitively activate heavy or environment-sensitive dependencies. The fallback loader allows decorators to execute and registry entries to be constructed even when the broader package import graph is not fully stable. In the current implementation this was necessary to guarantee stable registration of public class-based workflows such as `SCENIC`, thereby allowing downstream branch expansion to expose `regdiffusion`, `grnboost2`, and `genie3` as retrievable variants.

The same runtime registry is used for two related but distinct downstream products. First, it is exported as a machine-readable JSON snapshot through `export_registry(...)`, preserving both canonical entries and derived virtual entries together with metadata such as `source`, `virtual_entry`, `parent_full_name`, `branch_parameter`, `branch_value`, and `imports`. This enriched export is intended for debugging, interface inspection, and external integration. Second, the registry is converted into an MCP manifest for tool execution. In this second path, virtual entries are deliberately filtered out. The rationale is that branch-derived entries are retrieval-oriented abstractions rather than independently executable tools. Separating expressive capability representation from the executable manifest allows OmicVerse to retain high discovery recall without presenting synthetic branch variants as standalone tool calls.

`OmicVerseAgent` uses this registry as both initialization context and runtime retrieval substrate. During initialization, the agent constructs a system prompt containing workflow rules, code-quality constraints, a registry-derived summary of available OmicVerse functions, and an overview of curated skills. During interactive reasoning, however, capability grounding is not left to the initial prompt alone. The agent can call `search_functions()` to query `_global_registry`, `search_skills()` to retrieve workflow guidance, `inspect_data()` to characterize the current dataset, and `execute_code()` to perform or emit the final OmicVerse analysis code. In this way, the registry acts as a renewable local memory that can be revisited during planning rather than as a static prompt appendix. Queries containing backend names such as `dynamo`, `bayesprism`, or `regdiffusion` can therefore retrieve the corresponding OmicVerse wrappers through the same mechanism used for ordinary function discovery.

This registry-centered reasoning path is shared between Jarvis and Claw. Jarvis uses the full agentic loop to inspect data, search the registry, retrieve skills, and execute code against a live object. Claw reuses the same OmicVerse agent, the same runtime registry, the same skill system, and the same tool-level retrieval logic, but intercepts the terminal `execute_code()` action so that the final Python program is captured rather than run. The divergence between Jarvis and Claw is therefore confined to execution semantics, not capability discovery or planning. As a consequence, any improvement in registry expressiveness directly benefits both interfaces. Once a workflow object or backend-specific branch is represented in `_global_registry`, it becomes simultaneously available for interactive execution in Jarvis and code-only synthesis in Claw.

This design addresses a common failure mode in bioinformatics agents, namely that function discovery lags behind the real operational surface of the package. A decorator-only registry misses hidden branch-specific methods, whereas unconstrained source search confounds internal implementation detail with user-facing capability. By combining explicit registration, runtime method and branch expansion, robust hydration, and retrieval-oriented search semantics, OmicVerse maintains a registry that more faithfully represents its analytical affordances. The registry thus functions simultaneously as a capability index, a prompt-grounding substrate, and an exportable interface description for downstream agent systems.
