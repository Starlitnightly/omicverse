"""
FM Router — LLM-based routing for single-cell foundation model tasks.
=====================================================================

Takes a natural language query and returns:

- Inferred task (embed / integrate / annotate / spatial / perturb / drug_response)
- Selected best-fit model from :mod:`omicverse.fm.registry`
- Resolved parameters (or clarifying questions)
- Executable tool-call plan
"""

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from .registry import (
    GeneIDScheme,
    ModelSpec,
    SkillReadyStatus,
    TaskType,
    get_registry,
)


# ===========================================================================
# Constants
# ===========================================================================

VALID_TASKS = [t.value for t in TaskType]

VALID_FM_TOOLS = [
    "fm_profile_data",
    "fm_preprocess_validate",
    "fm_run",
    "fm_interpret_results",
    "fm_list_models",
    "fm_describe_model",
    "fm_select_model",
]

ROUTER_SYSTEM_PROMPT = """You are an expert scFM (single-cell foundation model) router.
Your job is to analyze user queries about single-cell data analysis and determine:
1. Which task they want to perform (embed, integrate, annotate, spatial, perturb, drug_response)
2. Which model best fits their needs from the available registry
3. What parameters are needed for execution
4. If any clarification is required

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanation text outside JSON.

## Available Tasks
- embed: Generate cell embeddings using a foundation model
- integrate: Batch integration / correction using foundation model embeddings
- annotate: Cell type annotation (may require fine-tuning depending on model)
- spatial: Spatial transcriptomics analysis (requires spatial coordinates)
- perturb: Perturbation prediction / analysis
- drug_response: Drug response prediction

## Output Format
Return a JSON object with this exact structure:
{
  "intent": {
    "task": "<task_name>",
    "confidence": <0.0-1.0>,
    "constraints": {}
  },
  "inputs": {
    "query": "<original_query>",
    "adata_path": "<path_if_provided>"
  },
  "data_profile": <null_or_profile_object>,
  "selection": {
    "recommended": {"name": "<model_name>", "rationale": "<why>"},
    "fallbacks": [{"name": "<model_name>", "rationale": "<why>"}]
  },
  "resolved_params": {
    "output_path": "<path_or_null>",
    "batch_key": "<key_or_null>",
    "label_key": "<key_or_null>"
  },
  "plan": [
    {"tool": "<tool_name>", "args": {}}
  ],
  "questions": [
    {"field": "<param_name>", "question": "<clarification_question>", "options": []}
  ],
  "warnings": []
}

## CRITICAL: Model Selection Rules

**Match the user's specific requirements to each model's unique differentiator and "Use when" guidance.**
Do NOT default to any single model. Each model has a distinct strength — select based on what the user actually needs.

### Disambiguation Table (confusable models)
| User mentions...                          | Select         | NOT              |
|-------------------------------------------|---------------|------------------|
| multi-omics, CITE-seq, RNA+ATAC+Protein   | scmulan       | scgpt            |
| spatial transcriptomics, niche, Visium    | nicheformer   | scgpt            |
| ATAC-seq only, chromatin accessibility    | atacformer    | scgpt/scmulan    |
| denoising, ambient RNA, protein-coding    | scprint       | scgpt            |
| unsupervised clustering, label-free       | aidocell      | scgpt            |
| cell-cell communication, multicellular    | pulsar        | scgpt            |
| fast inference, high throughput, million+  | cellplm       | scgpt            |
| next-token, autoregressive, generative    | tgpt          | scgpt/geneformer |
| MLP architecture, largest scale           | cellfm        | scgpt            |
| compact 200-dim, lightweight              | scbert        | scgpt            |
| ontology, hierarchical cell types         | sccello       | scgpt            |
| plant, polyploidy, Arabidopsis            | scplantllm    | scgpt/uce        |
| text+cell alignment, NL cell queries      | langcell      | scgpt            |
| LLM fine-tuning, cells-as-text            | cell2sentence | scgpt            |
| gene-level (not cell), no GPU, API-based  | genept        | scgpt/geneformer |
| chat-based, conversational annotation     | chatcell      | scgpt            |
| Ensembl IDs, network biology, CPU-only    | geneformer    | scgpt            |
| cross-species, zebrafish/frog/pig/macaque | uce           | scgpt/geneformer |
| prior knowledge, gene regulatory networks | genecompass   | scgpt            |
| perturbation prediction (gene KO/KD)      | tabula        | scfoundation/scgpt |
| general RNA embed/integrate, no special needs | scgpt     | -                |

### Selection Priority
1. **Unique requirement match**: If the query mentions a specific capability listed in a model's "Use when" field, select that model — even if it is ⚠️ partial-spec.
2. **Modality/species match**: ATAC-only → atacformer. Plant → scplantllm. Multi-omics → scmulan. Non-standard species → uce.
3. **Task-specific match**: Zero-shot annotation → sccello or chatcell. Perturbation → tabula. Spatial → nicheformer.
4. **General fallback**: Only select scgpt or geneformer when no specific differentiating requirement is present.

### Rules
1. Always select models from the provided model cards
2. If uncertain about parameters (like batch_key), add a question
3. If data profile shows incompatibility, select alternative model or add warning
4. Generate a complete execution plan with tool calls
5. Set confidence based on how clear the user's intent is
6. Skill-ready status (✅ vs ⚠️) is about adapter documentation, NOT model quality — do not prefer ✅ models over ⚠️ models based on status alone
"""


# ===========================================================================
# Pydantic Data Models
# ===========================================================================

class RouterIntent(BaseModel):
    """Inferred task intent from user query."""
    task: str = Field(..., description="The inferred task type")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"Invalid task: {v}. Must be one of {VALID_TASKS}")
        return v


class RouterInputs(BaseModel):
    """Input information from user query."""
    query: str = Field(...)
    adata_path: Optional[str] = Field(default=None)


class ModelSelection(BaseModel):
    """Model selection with rationale."""
    name: str = Field(...)
    rationale: str = Field(default="")


class RouterSelection(BaseModel):
    """Model selection output."""
    recommended: ModelSelection = Field(...)
    fallbacks: list[ModelSelection] = Field(default_factory=list)


class ResolvedParams(BaseModel):
    """Resolved parameters for execution."""
    output_path: Optional[str] = Field(default=None)
    batch_key: Optional[str] = Field(default=None)
    label_key: Optional[str] = Field(default=None)


class ToolCall(BaseModel):
    """A single tool call in the execution plan."""
    tool: str = Field(...)
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, v: str) -> str:
        if v not in VALID_FM_TOOLS:
            raise ValueError(f"Invalid tool: {v}. Must be one of {VALID_FM_TOOLS}")
        return v


class Question(BaseModel):
    """A clarifying question for the user."""
    field: str = Field(...)
    question: str = Field(...)
    options: list[str] = Field(default_factory=list)


class RouterOutput(BaseModel):
    """Complete router output."""
    intent: RouterIntent
    inputs: RouterInputs
    data_profile: Optional[dict[str, Any]] = Field(default=None)
    selection: RouterSelection
    resolved_params: ResolvedParams = Field(default_factory=ResolvedParams)
    plan: list[ToolCall] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ===========================================================================
# Validation
# ===========================================================================

def validate_router_output(
    output_dict: dict[str, Any],
) -> tuple[bool, list[str], Optional[RouterOutput]]:
    """Validate router output against schema and registry.

    Returns
    -------
    tuple[bool, list[str], RouterOutput | None]
        ``(is_valid, error_messages, parsed_output)``
    """
    errors = []
    registry = get_registry()

    try:
        parsed = RouterOutput.model_validate(output_dict)
    except Exception as exc:
        errors.append(f"Schema validation error: {exc}")
        return False, errors, None

    recommended_model = parsed.selection.recommended.name.lower()
    if registry.get(recommended_model) is None:
        available = [m.name for m in registry.list_models()]
        errors.append(f"Model '{recommended_model}' not found in registry. Available: {available[:10]}")

    for fallback in parsed.selection.fallbacks:
        if registry.get(fallback.name.lower()) is None:
            errors.append(f"Fallback model '{fallback.name}' not found in registry")

    for tool_call in parsed.plan:
        if tool_call.tool not in VALID_FM_TOOLS:
            errors.append(f"Invalid tool '{tool_call.tool}' in plan. Valid: {VALID_FM_TOOLS}")

    if errors:
        return False, errors, parsed
    return True, [], parsed


# ===========================================================================
# Prompt Builder
# ===========================================================================

def build_model_cards(
    skill_ready_only: bool = False,
    max_vram_gb: Optional[int] = None,
    prefer_zero_shot: bool = True,
) -> str:
    """Build formatted model cards for LLM prompt.

    Parameters
    ----------
    skill_ready_only : bool
        Only include skill-ready models.
    max_vram_gb : int, optional
        Filter by max VRAM constraint.
    prefer_zero_shot : bool
        Highlight zero-shot capable models.

    Returns
    -------
    str
        Formatted string of model cards.
    """
    registry = get_registry()
    models = registry.list_models(skill_ready_only=skill_ready_only)

    if max_vram_gb:
        models = [m for m in models if m.hardware.min_vram_gb <= max_vram_gb]

    cards = []
    for spec in models:
        status_icon = "✅" if spec.skill_ready == SkillReadyStatus.READY else "⚠️"
        zero_shot_note = " [zero-shot]" if spec.zero_shot_embedding else ""

        card = f"""### {status_icon} {spec.name}{zero_shot_note}
- **Version**: {spec.version}
- **Tasks**: {', '.join(t.value for t in spec.tasks)}
- **Species**: {', '.join(spec.species)}
- **Gene IDs**: {spec.gene_id_scheme.value}
- **VRAM**: {spec.hardware.min_vram_gb}GB min
- **CPU fallback**: {"Yes" if spec.hardware.cpu_fallback else "No"}
- **Differentiator**: {spec.differentiator or "General-purpose"}
- **Use when**: {spec.prefer_when or "No specific preference"}
"""
        cards.append(card)

    return "\n".join(cards)


def build_router_prompt(
    query: str,
    data_profile: Optional[dict[str, Any]] = None,
    model_cards: str = "",
    prefer_zero_shot: bool = True,
    max_vram_gb: Optional[int] = None,
    skill_ready_only: bool = False,
    allow_partial: bool = True,
    allow_reference: bool = False,
) -> str:
    """Build the user prompt for the router LLM.

    Parameters
    ----------
    query : str
        User's natural language query.
    data_profile : dict, optional
        Data profile from ``fm.profile_data``.
    model_cards : str
        Formatted model cards string.
    prefer_zero_shot : bool
        Prefer zero-shot capable models.
    max_vram_gb : int, optional
        Max VRAM constraint.
    skill_ready_only : bool
        Only select skill-ready models.
    allow_partial : bool
        Allow partial-spec models.
    allow_reference : bool
        Allow reference-only models.

    Returns
    -------
    str
    """
    prompt_parts = [f"## User Query\n{query}"]

    if data_profile:
        profile_str = json.dumps(data_profile, indent=2, default=str)
        prompt_parts.append(f"## Data Profile\n```json\n{profile_str}\n```")

    if model_cards:
        prompt_parts.append(f"## Available Models\n{model_cards}")

    constraints = []
    if prefer_zero_shot:
        constraints.append("Zero-shot capability is available — but only prefer it if the user has no labeled reference data")
    if max_vram_gb:
        constraints.append(f"Max VRAM available: {max_vram_gb}GB")
    if skill_ready_only:
        constraints.append("Only select fully skill-ready (✅) models")
    elif not allow_partial:
        constraints.append("Avoid partial-spec (⚠️) models")
    if not allow_reference:
        constraints.append("Do not select reference-only models")

    if constraints:
        prompt_parts.append("## Constraints\n- " + "\n- ".join(constraints))

    prompt_parts.append("## Instructions\nAnalyze the query and provide your response in JSON format only.")

    return "\n\n".join(prompt_parts)


# ===========================================================================
# LLM Call Helper
# ===========================================================================

async def call_router_llm(
    context: dict[str, Any],
    prompt: str,
    system_prompt: str = ROUTER_SYSTEM_PROMPT,
    max_retries: int = 1,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Call the router LLM via ``context["_call_agent"]``.

    Parameters
    ----------
    context : dict
        Must contain ``_call_agent`` async callback.
    prompt : str
        User prompt to send.
    system_prompt : str
        System prompt for the LLM.
    max_retries : int
        Number of retries on validation failure.

    Returns
    -------
    tuple[bool, dict, list[str]]
        ``(success, result_dict, errors)``
    """
    _call_agent = context.get("_call_agent")
    if _call_agent is None:
        return False, {}, ["_call_agent not available in context"]

    model = None
    caller_models = context.get("caller_models")
    if caller_models and len(caller_models) > 0:
        model = caller_models[0]

    errors_accumulated: list[str] = []
    last_response = None

    for attempt in range(max_retries + 1):
        messages = [{"role": "user", "content": prompt}]

        if attempt > 0 and errors_accumulated:
            retry_prompt = (
                "Your previous response had validation errors:\n"
                + "\n".join(f"- {e}" for e in errors_accumulated)
                + "\n\nPlease fix these issues and return valid JSON only."
            )
            messages.append({"role": "user", "content": retry_prompt})

        try:
            result = await _call_agent(
                messages=messages,
                system_prompt=system_prompt,
                model=model,
            )
        except Exception as exc:
            errors_accumulated.append(f"LLM call failed: {exc}")
            continue

        if not result.get("success"):
            errors_accumulated.append(f"LLM call unsuccessful: {result.get('error', 'Unknown error')}")
            continue

        response_text = result.get("response", "")
        last_response = response_text

        parsed_json = _extract_json_from_response(response_text)
        if parsed_json is None:
            errors_accumulated.append(f"Failed to parse JSON from response: {response_text[:200]}...")
            continue

        is_valid, validation_errors, parsed_output = validate_router_output(parsed_json)

        if is_valid and parsed_output:
            return True, parsed_output.model_dump(), []

        errors_accumulated.extend(validation_errors)

    # All retries failed — return best effort
    if last_response:
        parsed_json = _extract_json_from_response(last_response)
        if parsed_json:
            return False, parsed_json, errors_accumulated

    return False, {}, errors_accumulated


def _extract_json_from_response(response: str) -> Optional[dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # JSON object boundaries
    start_idx = response.find("{")
    end_idx = response.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(response[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            pass

    return None


async def _reroute_on_incompatibility(
    context: dict[str, Any],
    original_result: dict[str, Any],
    incompatible_model: str,
    data_profile: dict[str, Any],
    model_cards: str,
    query: str,
) -> Optional[dict[str, Any]]:
    """Re-call LLM to select an alternative model after incompatibility.

    Parameters
    ----------
    context : dict
        Context with ``_call_agent``.
    original_result : dict
        Original router result.
    incompatible_model : str
        Name of the incompatible model.
    data_profile : dict
        Data profile showing compatibility.
    model_cards : str
        Available model cards.
    query : str
        Original user query.

    Returns
    -------
    dict or None
        New router result or ``None`` if reroute fails.
    """
    issues = (
        data_profile
        .get("model_compatibility", {})
        .get(incompatible_model, {})
        .get("issues", ["Unknown incompatibility"])
    )
    reroute_prompt = (
        f"## Reroute Request\n\n"
        f"The previously selected model '{incompatible_model}' is INCOMPATIBLE with the user's data.\n\n"
        f"### Incompatibility Issues:\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + f"\n\n### Original Query:\n{query}\n\n"
        f"### Data Profile:\n```json\n{json.dumps(data_profile, indent=2, default=str)}\n```\n\n"
        f"### Available Models (excluding {incompatible_model}):\n{model_cards}\n\n"
        f"Please select a DIFFERENT model that is compatible with this data. "
        f"Do NOT select '{incompatible_model}'.\n"
        f"Return valid JSON only with the same schema as before."
    )

    success, result, errors = await call_router_llm(context=context, prompt=reroute_prompt)
    if success:
        return result
    return None


# ===========================================================================
# Main Router
# ===========================================================================

async def route_query(
    query: str,
    context: dict[str, Any],
    adata_path: Optional[str] = None,
    data_profile: Optional[dict[str, Any]] = None,
    prefer_zero_shot: bool = True,
    max_vram_gb: Optional[int] = None,
    skill_ready_only: bool = False,
    allow_partial: bool = True,
    allow_reference: bool = False,
    output_path: Optional[str] = None,
    batch_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> dict[str, Any]:
    """Main router function — orchestrates profiling, LLM call, validation, rerouting.

    Parameters
    ----------
    query : str
        User's natural language query.
    context : dict
        Must contain ``_call_agent`` async callback.
    adata_path : str, optional
        Path to AnnData file.
    data_profile : dict, optional
        Pre-computed data profile (skips profiling if provided).
    prefer_zero_shot : bool
        Prefer zero-shot capable models.
    max_vram_gb : int, optional
        Maximum VRAM constraint.
    skill_ready_only : bool
        Only select skill-ready models.
    allow_partial : bool
        Allow partial-spec models.
    allow_reference : bool
        Allow reference-only models.
    output_path : str, optional
        Pre-specified output path.
    batch_key : str, optional
        Pre-specified batch key.
    label_key : str, optional
        Pre-specified label key.

    Returns
    -------
    dict
        RouterOutput dict with intent, selection, plan, questions, warnings.
    """
    registry = get_registry()

    if context.get("_call_agent") is None:
        return {
            "error": "Router requires _call_agent in context",
            "intent": {"task": "unknown", "confidence": 0.0, "constraints": {}},
            "inputs": {"query": query, "adata_path": adata_path},
            "data_profile": data_profile,
            "selection": {"recommended": {"name": "", "rationale": ""}, "fallbacks": []},
            "resolved_params": {"output_path": output_path, "batch_key": batch_key, "label_key": label_key},
            "plan": [],
            "questions": [],
            "warnings": ["Router cannot function without _call_agent callback"],
        }

    model_cards = build_model_cards(
        skill_ready_only=skill_ready_only,
        max_vram_gb=max_vram_gb,
        prefer_zero_shot=prefer_zero_shot,
    )

    prompt = build_router_prompt(
        query=query,
        data_profile=data_profile,
        model_cards=model_cards,
        prefer_zero_shot=prefer_zero_shot,
        max_vram_gb=max_vram_gb,
        skill_ready_only=skill_ready_only,
        allow_partial=allow_partial,
        allow_reference=allow_reference,
    )

    success, result, errors = await call_router_llm(context=context, prompt=prompt)

    if not success:
        return {
            "error": f"Router LLM failed: {'; '.join(errors)}",
            "intent": result.get("intent", {"task": "unknown", "confidence": 0.0, "constraints": {}}),
            "inputs": {"query": query, "adata_path": adata_path},
            "data_profile": data_profile,
            "selection": result.get("selection", {"recommended": {"name": "", "rationale": ""}, "fallbacks": []}),
            "resolved_params": {"output_path": output_path, "batch_key": batch_key, "label_key": label_key},
            "plan": result.get("plan", []),
            "questions": result.get("questions", []),
            "warnings": errors,
        }

    # Inject data_profile
    if data_profile and result.get("data_profile") is None:
        result["data_profile"] = data_profile

    # Override resolved_params with pre-specified values
    if result.get("resolved_params"):
        if output_path:
            result["resolved_params"]["output_path"] = output_path
        if batch_key:
            result["resolved_params"]["batch_key"] = batch_key
        if label_key:
            result["resolved_params"]["label_key"] = label_key

    # Check compatibility if we have data profile and a model selection
    if data_profile and result.get("selection", {}).get("recommended", {}).get("name"):
        model_name = result["selection"]["recommended"]["name"].lower()
        spec = registry.get(model_name)

        if spec:
            data_species = data_profile.get("species", "").replace(" (inferred)", "").lower()
            if data_species and data_species != "unknown":
                if not spec.supports_species(data_species):
                    result.setdefault("warnings", []).append(
                        f"Selected model '{model_name}' may not support species '{data_species}'"
                    )

            data_gene_scheme = data_profile.get("gene_scheme", "")
            if data_gene_scheme and data_gene_scheme != "unknown":
                model_scheme = spec.gene_id_scheme.value
                if data_gene_scheme != model_scheme:
                    result.setdefault("warnings", []).append(
                        f"Data uses {data_gene_scheme} gene IDs but model '{model_name}' expects {model_scheme}"
                    )

            model_compat = data_profile.get("model_compatibility", {}).get(model_name, {})
            if model_compat and not model_compat.get("compatible", True):
                issues = model_compat.get("issues", [])
                result.setdefault("warnings", []).extend(issues)

                reroute_result = await _reroute_on_incompatibility(
                    context=context,
                    original_result=result,
                    incompatible_model=model_name,
                    data_profile=data_profile,
                    model_cards=model_cards,
                    query=query,
                )

                if reroute_result:
                    new_model = reroute_result.get("selection", {}).get("recommended", {}).get("name", "").lower()
                    new_compat = data_profile.get("model_compatibility", {}).get(new_model, {})

                    if new_compat.get("compatible", True):
                        reroute_result.setdefault("warnings", []).extend(result.get("warnings", []))
                        reroute_result["warnings"].insert(
                            0, f"Rerouted from '{model_name}' to '{new_model}' due to incompatibility"
                        )
                        return reroute_result
                    else:
                        result.setdefault("questions", []).append({
                            "field": "model_name",
                            "question": (
                                f"Both '{model_name}' and '{new_model}' are incompatible with your data. "
                                "Please select a model manually."
                            ),
                            "options": [
                                m.name for m in registry.find_models()
                                if data_profile.get("model_compatibility", {}).get(m.name.lower(), {}).get("compatible", True)
                            ][:5],
                        })
                else:
                    if result.get("selection", {}).get("fallbacks"):
                        for fallback in result["selection"]["fallbacks"]:
                            fallback_name = fallback.get("name", "").lower()
                            fallback_compat = data_profile.get("model_compatibility", {}).get(fallback_name, {})
                            if fallback_compat.get("compatible", True):
                                result.setdefault("warnings", []).append(
                                    f"Recommended model '{model_name}' is incompatible. "
                                    f"Consider using fallback '{fallback_name}' instead."
                                )
                                break

    return result
