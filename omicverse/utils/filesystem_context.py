"""
FilesystemContextManager - Filesystem-based context engineering for OmicVerse agents.

This module provides a filesystem-based context management system that allows agents to:
- Write intermediate results and notes to a scratch pad
- Selectively retrieve relevant context using glob/grep patterns
- Compress and summarize old context to prevent overflow
- Share workspace between parent and sub-agents

Inspired by LangChain's context engineering principles:
- Write: Offload information to external storage early and often
- Select: Pull in only relevant context when needed
- Compress: Summarize using structured schema-driven approaches
- Isolate: Use sub-agent architecture to delegate and keep contexts separate

Reference: https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/
"""

import json
import os
import re
import uuid
import fnmatch
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextNote:
    """A single note stored in the context workspace.

    Attributes:
        key: Unique identifier for this note
        content: The note content (string or structured data)
        category: Category for organizing notes (e.g., 'intermediate_results', 'observations')
        timestamp: When the note was created
        metadata: Additional metadata (e.g., function name, step number)
        compressed: Whether this note has been compressed/summarized
    """
    key: str
    content: Union[str, Dict[str, Any]]
    category: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextNote':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExecutionPlan:
    """A structured execution plan that can be persisted and tracked.

    Attributes:
        steps: List of planned steps
        current_step: Index of current step being executed
        completed_steps: Indices of completed steps
        status: Overall plan status (pending, in_progress, completed, failed)
        created_at: When the plan was created
        updated_at: When the plan was last updated
    """
    steps: List[Dict[str, Any]]
    current_step: int = 0
    completed_steps: List[int] = field(default_factory=list)
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionPlan':
        return cls(**data)

    def mark_step_complete(self, step_index: int) -> None:
        """Mark a step as completed."""
        if step_index not in self.completed_steps:
            self.completed_steps.append(step_index)
        self.current_step = step_index + 1
        self.updated_at = datetime.now().isoformat()
        if len(self.completed_steps) == len(self.steps):
            self.status = "completed"
        else:
            self.status = "in_progress"


@dataclass
class ContextSearchResult:
    """Result from searching the context workspace.

    Attributes:
        key: Note key that matched
        category: Category of the matching note
        content_preview: Preview of matching content
        match_type: Type of match (glob, grep, semantic)
        relevance_score: How relevant this result is (0-1)
        file_path: Path to the file containing this note
    """
    key: str
    category: str
    content_preview: str
    match_type: str
    relevance_score: float = 1.0
    file_path: Optional[str] = None


class FilesystemContextManager:
    """Manages filesystem-based context for OmicVerse agents.

    This class implements the four pillars of context engineering:
    1. WRITE: Offload intermediate results to filesystem early and often
    2. SELECT: Use glob/grep to find relevant context
    3. COMPRESS: Summarize old context to reduce token usage
    4. ISOLATE: Session-based workspaces for sub-agent collaboration

    Example:
        >>> ctx = FilesystemContextManager()
        >>> ctx.write_note("pca_result", {"n_components": 50, "variance_ratio": 0.85}, "results")
        >>> ctx.write_note("observation", "High mitochondrial content in cluster 3", "observations")
        >>> results = ctx.search_context("pca*", match_type="glob")
        >>> relevant = ctx.get_relevant_context("dimensionality reduction")
    """

    DEFAULT_BASE_DIR = Path.home() / ".ovagent" / "context"

    # Categories for organizing notes
    CATEGORIES = {
        "notes": "General notes and observations",
        "results": "Intermediate computation results",
        "decisions": "Decision points and rationale",
        "snapshots": "Data state snapshots",
        "figures": "Generated figure paths",
        "errors": "Error logs and debugging info",
    }

    def __init__(
        self,
        session_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
        parent_session_id: Optional[str] = None,
        max_notes_per_category: int = 100,
        auto_compress_threshold: int = 50,
    ):
        """Initialize the filesystem context manager.

        Parameters
        ----------
        session_id : str, optional
            Unique session identifier. If not provided, a new one is generated.
        base_dir : Path, optional
            Base directory for context storage. Defaults to ~/.ovagent/context/
        parent_session_id : str, optional
            If this is a sub-agent, the parent's session ID for workspace sharing.
        max_notes_per_category : int
            Maximum notes to keep per category before auto-compression (default: 100)
        auto_compress_threshold : int
            Number of notes that triggers auto-compression (default: 50)
        """
        self.base_dir = Path(base_dir) if base_dir else self.DEFAULT_BASE_DIR
        self.session_id = session_id or self._generate_session_id()
        self.parent_session_id = parent_session_id
        self.max_notes_per_category = max_notes_per_category
        self.auto_compress_threshold = auto_compress_threshold

        # Initialize workspace
        self._workspace_dir = self._get_workspace_dir()
        self._ensure_workspace_structure()

        # In-memory cache for frequently accessed notes
        self._cache: Dict[str, ContextNote] = {}
        self._cache_dirty = False

        logger.info(f"FilesystemContextManager initialized: session={self.session_id}, workspace={self._workspace_dir}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:8]
        return f"session_{timestamp}_{unique_suffix}"

    def _get_workspace_dir(self) -> Path:
        """Get the workspace directory for this session."""
        if self.parent_session_id:
            # Sub-agent shares parent's workspace
            return self.base_dir / self.parent_session_id
        return self.base_dir / self.session_id

    def _ensure_workspace_structure(self) -> None:
        """Create the workspace directory structure if it doesn't exist."""
        # Create main directories
        directories = [
            self._workspace_dir,
            self._workspace_dir / "notes",
            self._workspace_dir / "results",
            self._workspace_dir / "snapshots",
            self._workspace_dir / "figures",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create shared directory if this is a root session
        if not self.parent_session_id:
            shared_dir = self.base_dir / "shared"
            shared_dir.mkdir(parents=True, exist_ok=True)

        # Initialize session metadata
        metadata_file = self._workspace_dir / "session_metadata.json"
        if not metadata_file.exists():
            metadata = {
                "session_id": self.session_id,
                "parent_session_id": self.parent_session_id,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
            }
            self._write_json(metadata_file, metadata)

    # =========================================================================
    # WRITE: Offload information to filesystem
    # =========================================================================

    def write_note(
        self,
        key: str,
        content: Union[str, Dict[str, Any]],
        category: str = "notes",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write a note to the context workspace.

        This is the primary method for offloading information from the context
        window to the filesystem. Use this liberally for:
        - Intermediate computation results
        - Observations about the data
        - Decision rationale
        - Step-by-step progress

        Parameters
        ----------
        key : str
            Unique identifier for this note. Will be used for retrieval.
        content : str or dict
            The note content. Can be free-form text or structured data.
        category : str
            Category for organizing notes (default: "notes").
            Options: notes, results, decisions, snapshots, figures, errors
        metadata : dict, optional
            Additional metadata to store with the note.

        Returns
        -------
        str
            The file path where the note was saved.

        Example
        -------
        >>> ctx.write_note(
        ...     "clustering_result",
        ...     {"n_clusters": 8, "resolution": 1.0, "modularity": 0.72},
        ...     category="results",
        ...     metadata={"function": "leiden", "step": 5}
        ... )
        """
        # Validate category
        if category not in self.CATEGORIES:
            logger.warning(f"Unknown category '{category}', using 'notes'")
            category = "notes"

        # Create the note
        note = ContextNote(
            key=key,
            content=content,
            category=category,
            metadata=metadata or {},
        )

        # Determine file path
        category_dir = self._workspace_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Use key as filename (sanitized)
        safe_key = self._sanitize_filename(key)
        file_path = category_dir / f"{safe_key}.json"

        # Write to file
        self._write_json(file_path, note.to_dict())

        # Update cache
        cache_key = f"{category}/{key}"
        self._cache[cache_key] = note

        # Check if auto-compression is needed
        self._check_auto_compress(category)

        logger.debug(f"Wrote note: {cache_key} -> {file_path}")
        return str(file_path)

    def write_plan(self, steps: List[Dict[str, Any]]) -> str:
        """Write an execution plan to the workspace.

        Parameters
        ----------
        steps : list of dict
            List of step definitions. Each step should have:
            - description: What this step does
            - status: pending, in_progress, completed, failed
            - optional: function, parameters, expected_output

        Returns
        -------
        str
            Path to the plan file.

        Example
        -------
        >>> ctx.write_plan([
        ...     {"description": "Load and QC data", "status": "pending"},
        ...     {"description": "Normalize and scale", "status": "pending"},
        ...     {"description": "Run PCA", "status": "pending"},
        ...     {"description": "Cluster cells", "status": "pending"},
        ... ])
        """
        plan = ExecutionPlan(steps=steps)
        plan_file = self._workspace_dir / "plan.json"
        self._write_json(plan_file, plan.to_dict())

        # Also write human-readable markdown version
        md_content = self._plan_to_markdown(plan)
        md_file = self._workspace_dir / "plan.md"
        md_file.write_text(md_content, encoding="utf-8")

        logger.info(f"Wrote execution plan with {len(steps)} steps")
        return str(plan_file)

    def update_plan_step(self, step_index: int, status: str, result: Optional[str] = None) -> None:
        """Update the status of a plan step.

        Parameters
        ----------
        step_index : int
            Index of the step to update (0-based)
        status : str
            New status: pending, in_progress, completed, failed
        result : str, optional
            Result or notes for this step
        """
        plan = self.load_plan()
        if plan is None:
            logger.warning("No plan exists to update")
            return

        if 0 <= step_index < len(plan.steps):
            plan.steps[step_index]["status"] = status
            if result:
                plan.steps[step_index]["result"] = result

            if status == "completed":
                plan.mark_step_complete(step_index)
            elif status == "in_progress":
                plan.current_step = step_index
                plan.status = "in_progress"
            elif status == "failed":
                plan.status = "failed"

            plan.updated_at = datetime.now().isoformat()

            # Save updated plan
            plan_file = self._workspace_dir / "plan.json"
            self._write_json(plan_file, plan.to_dict())

            # Update markdown
            md_content = self._plan_to_markdown(plan)
            md_file = self._workspace_dir / "plan.md"
            md_file.write_text(md_content, encoding="utf-8")

    def write_snapshot(
        self,
        snapshot_data: Dict[str, Any],
        step_number: Optional[int] = None,
        description: Optional[str] = None,
    ) -> str:
        """Write a data state snapshot.

        Parameters
        ----------
        snapshot_data : dict
            Dictionary containing data state information (shapes, keys, etc.)
        step_number : int, optional
            Step number for ordering snapshots
        description : str, optional
            Human-readable description of the snapshot

        Returns
        -------
        str
            Path to the snapshot file.
        """
        snapshots_dir = self._workspace_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Generate snapshot filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if step_number is not None:
            filename = f"step_{step_number:03d}_{timestamp}.json"
        else:
            filename = f"snapshot_{timestamp}.json"

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "step_number": step_number,
            "description": description,
            "data": snapshot_data,
        }

        file_path = snapshots_dir / filename
        self._write_json(file_path, snapshot)

        logger.debug(f"Wrote snapshot: {file_path}")
        return str(file_path)

    # =========================================================================
    # SELECT: Find relevant context using glob/grep
    # =========================================================================

    def read_note(self, key: str, category: str = "notes") -> Optional[ContextNote]:
        """Read a specific note by key.

        Parameters
        ----------
        key : str
            The note key to retrieve
        category : str
            Category where the note is stored

        Returns
        -------
        ContextNote or None
            The note if found, None otherwise.
        """
        # Check cache first
        cache_key = f"{category}/{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Read from filesystem
        safe_key = self._sanitize_filename(key)
        file_path = self._workspace_dir / category / f"{safe_key}.json"

        if file_path.exists():
            data = self._read_json(file_path)
            if data:
                note = ContextNote.from_dict(data)
                self._cache[cache_key] = note
                return note

        return None

    def load_plan(self) -> Optional[ExecutionPlan]:
        """Load the current execution plan.

        Returns
        -------
        ExecutionPlan or None
            The plan if it exists, None otherwise.
        """
        plan_file = self._workspace_dir / "plan.json"
        if plan_file.exists():
            data = self._read_json(plan_file)
            if data:
                return ExecutionPlan.from_dict(data)
        return None

    def search_context(
        self,
        pattern: str,
        match_type: str = "glob",
        categories: Optional[List[str]] = None,
        max_results: int = 20,
    ) -> List[ContextSearchResult]:
        """Search the context workspace using glob or grep patterns.

        This is the primary method for selective context retrieval. Use this
        to find relevant notes without loading everything into memory.

        Parameters
        ----------
        pattern : str
            Search pattern. For glob: "pca*", "cluster_*". For grep: regex pattern.
        match_type : str
            Type of search: "glob" (filename pattern) or "grep" (content search)
        categories : list of str, optional
            Categories to search in. Default: all categories.
        max_results : int
            Maximum number of results to return (default: 20)

        Returns
        -------
        list of ContextSearchResult
            Matching results with content previews.

        Example
        -------
        >>> results = ctx.search_context("cluster*", match_type="glob")
        >>> results = ctx.search_context("resolution.*1\\.0", match_type="grep")
        """
        results = []
        categories = categories or list(self.CATEGORIES.keys())

        for category in categories:
            category_dir = self._workspace_dir / category
            if not category_dir.exists():
                continue

            for file_path in category_dir.glob("*.json"):
                matched = False
                relevance = 1.0

                if match_type == "glob":
                    # Match against filename (key)
                    if fnmatch.fnmatch(file_path.stem, pattern):
                        matched = True

                elif match_type == "grep":
                    # Match against file content
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if re.search(pattern, content, re.IGNORECASE):
                            matched = True
                            # Calculate relevance based on match count
                            matches = len(re.findall(pattern, content, re.IGNORECASE))
                            relevance = min(1.0, matches / 10.0 + 0.5)
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")

                if matched:
                    # Load note for preview
                    try:
                        data = self._read_json(file_path)
                        if data:
                            content = data.get("content", "")
                            if isinstance(content, dict):
                                preview = json.dumps(content)[:200]
                            else:
                                preview = str(content)[:200]

                            results.append(ContextSearchResult(
                                key=data.get("key", file_path.stem),
                                category=category,
                                content_preview=preview,
                                match_type=match_type,
                                relevance_score=relevance,
                                file_path=str(file_path),
                            ))
                    except Exception as e:
                        logger.debug(f"Error processing {file_path}: {e}")

                if len(results) >= max_results:
                    break

            if len(results) >= max_results:
                break

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:max_results]

    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000,
        include_plan: bool = True,
        include_recent: int = 5,
    ) -> str:
        """Get context relevant to a query, formatted for LLM injection.

        This method combines multiple strategies:
        1. Load the current plan (if exists)
        2. Search for notes matching the query
        3. Include N most recent notes
        4. Format everything for LLM consumption

        Parameters
        ----------
        query : str
            The current task or query to find relevant context for
        max_tokens : int
            Approximate maximum tokens to return (rough estimate: 4 chars = 1 token)
        include_plan : bool
            Whether to include the execution plan
        include_recent : int
            Number of most recent notes to include

        Returns
        -------
        str
            Formatted context string ready for LLM injection.
        """
        sections = []
        current_chars = 0
        max_chars = max_tokens * 4  # Rough estimate

        # 1. Include plan if requested
        if include_plan:
            plan = self.load_plan()
            if plan:
                plan_md = self._plan_to_markdown(plan, brief=True)
                sections.append(f"## Current Execution Plan\n{plan_md}")
                current_chars += len(plan_md)

        # 2. Search for relevant notes
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        for keyword in keywords[:3]:  # Top 3 keywords
            if current_chars >= max_chars:
                break

            # Try both glob and grep
            results = self.search_context(f"*{keyword}*", match_type="glob", max_results=5)
            if not results:
                results = self.search_context(keyword, match_type="grep", max_results=5)

            for result in results:
                if current_chars >= max_chars:
                    break
                note = self.read_note(result.key, result.category)
                if note:
                    note_text = self._format_note_for_context(note)
                    if current_chars + len(note_text) < max_chars:
                        sections.append(note_text)
                        current_chars += len(note_text)

        # 3. Include recent notes
        recent_notes = self._get_recent_notes(include_recent)
        if recent_notes:
            recent_section = "## Recent Notes\n"
            for note in recent_notes:
                if current_chars >= max_chars:
                    break
                note_text = self._format_note_for_context(note, brief=True)
                if current_chars + len(note_text) < max_chars:
                    recent_section += note_text + "\n"
                    current_chars += len(note_text)
            if len(recent_section) > 20:
                sections.append(recent_section)

        if not sections:
            return ""

        return "\n\n".join(sections)

    def list_notes(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all notes in the workspace.

        Parameters
        ----------
        category : str, optional
            If provided, only list notes in this category.

        Returns
        -------
        list of dict
            List of note metadata (key, category, timestamp, preview).
        """
        notes = []
        categories = [category] if category else list(self.CATEGORIES.keys())

        for cat in categories:
            cat_dir = self._workspace_dir / cat
            if not cat_dir.exists():
                continue

            for file_path in cat_dir.glob("*.json"):
                try:
                    data = self._read_json(file_path)
                    if data:
                        content = data.get("content", "")
                        if isinstance(content, dict):
                            preview = json.dumps(content)[:100]
                        else:
                            preview = str(content)[:100]

                        notes.append({
                            "key": data.get("key", file_path.stem),
                            "category": cat,
                            "timestamp": data.get("timestamp", ""),
                            "preview": preview,
                            "compressed": data.get("compressed", False),
                        })
                except Exception as e:
                    logger.debug(f"Error reading {file_path}: {e}")

        # Sort by timestamp (most recent first)
        notes.sort(key=lambda n: n.get("timestamp", ""), reverse=True)
        return notes

    # =========================================================================
    # COMPRESS: Summarize old context
    # =========================================================================

    def compress_notes(
        self,
        category: str,
        keep_recent: int = 10,
        summarizer: Optional[callable] = None,
    ) -> str:
        """Compress old notes in a category into a summary.

        Parameters
        ----------
        category : str
            Category to compress
        keep_recent : int
            Number of recent notes to keep uncompressed
        summarizer : callable, optional
            Function to summarize notes. If not provided, uses simple concatenation.
            Should accept List[ContextNote] and return str.

        Returns
        -------
        str
            Path to the compressed summary file.
        """
        cat_dir = self._workspace_dir / category
        if not cat_dir.exists():
            return ""

        # Get all notes sorted by timestamp
        all_notes = []
        for file_path in cat_dir.glob("*.json"):
            if file_path.stem.startswith("_summary"):
                continue  # Skip existing summaries
            try:
                data = self._read_json(file_path)
                if data:
                    all_notes.append((file_path, ContextNote.from_dict(data)))
            except Exception:
                continue

        # Sort by timestamp
        all_notes.sort(key=lambda x: x[1].timestamp, reverse=True)

        # Keep recent, compress the rest
        to_keep = all_notes[:keep_recent]
        to_compress = all_notes[keep_recent:]

        if not to_compress:
            logger.debug(f"No notes to compress in {category}")
            return ""

        # Create summary
        notes_to_summarize = [n for _, n in to_compress]

        if summarizer:
            summary_content = summarizer(notes_to_summarize)
        else:
            # Default: simple concatenation with metadata
            summary_content = self._default_summarize(notes_to_summarize)

        # Write summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = cat_dir / f"_summary_{timestamp}.json"

        summary_note = ContextNote(
            key=f"summary_{timestamp}",
            content=summary_content,
            category=category,
            metadata={
                "type": "summary",
                "notes_count": len(to_compress),
                "date_range": {
                    "from": to_compress[-1][1].timestamp if to_compress else "",
                    "to": to_compress[0][1].timestamp if to_compress else "",
                }
            },
            compressed=True,
        )

        self._write_json(summary_file, summary_note.to_dict())

        # Delete compressed notes
        for file_path, _ in to_compress:
            try:
                file_path.unlink()
            except Exception as e:
                logger.debug(f"Error deleting {file_path}: {e}")

        logger.info(f"Compressed {len(to_compress)} notes in {category} -> {summary_file}")
        return str(summary_file)

    def get_session_summary(self) -> str:
        """Get a summary of the current session.

        Returns
        -------
        str
            Markdown-formatted session summary.
        """
        lines = [f"# Session Summary: {self.session_id}\n"]

        # Add plan status
        plan = self.load_plan()
        if plan:
            completed = len(plan.completed_steps)
            total = len(plan.steps)
            lines.append(f"## Plan Progress: {completed}/{total} steps completed")
            lines.append(f"Status: {plan.status}\n")

        # Count notes by category
        lines.append("## Notes by Category")
        for category in self.CATEGORIES:
            notes = self.list_notes(category)
            if notes:
                lines.append(f"- **{category}**: {len(notes)} notes")

        # Recent activity
        all_notes = self.list_notes()
        if all_notes:
            lines.append("\n## Recent Activity")
            for note in all_notes[:5]:
                lines.append(f"- [{note['category']}] {note['key']}: {note['preview'][:50]}...")

        return "\n".join(lines)

    # =========================================================================
    # ISOLATE: Sub-agent workspace sharing
    # =========================================================================

    def create_sub_agent_context(self, sub_agent_id: Optional[str] = None) -> 'FilesystemContextManager':
        """Create a context manager for a sub-agent that shares this workspace.

        Parameters
        ----------
        sub_agent_id : str, optional
            Identifier for the sub-agent. If not provided, one is generated.

        Returns
        -------
        FilesystemContextManager
            A new context manager that shares this session's workspace.
        """
        if sub_agent_id is None:
            sub_agent_id = f"sub_{uuid.uuid4().hex[:8]}"

        return FilesystemContextManager(
            session_id=sub_agent_id,
            base_dir=self.base_dir,
            parent_session_id=self.session_id,
            max_notes_per_category=self.max_notes_per_category,
            auto_compress_threshold=self.auto_compress_threshold,
        )

    def get_shared_workspace_path(self) -> Path:
        """Get the path to the shared workspace directory.

        Returns
        -------
        Path
            Path to the shared workspace.
        """
        return self.base_dir / "shared"

    def write_to_shared(self, key: str, content: Union[str, Dict[str, Any]]) -> str:
        """Write to the shared workspace (accessible across sessions).

        Parameters
        ----------
        key : str
            Key for the shared note
        content : str or dict
            Content to store

        Returns
        -------
        str
            Path to the shared note file.
        """
        shared_dir = self.get_shared_workspace_path()
        shared_dir.mkdir(parents=True, exist_ok=True)

        note = ContextNote(
            key=key,
            content=content,
            category="shared",
            metadata={"session_id": self.session_id},
        )

        safe_key = self._sanitize_filename(key)
        file_path = shared_dir / f"{safe_key}.json"
        self._write_json(file_path, note.to_dict())

        return str(file_path)

    def read_from_shared(self, key: str) -> Optional[ContextNote]:
        """Read from the shared workspace.

        Parameters
        ----------
        key : str
            Key to read

        Returns
        -------
        ContextNote or None
            The shared note if found.
        """
        safe_key = self._sanitize_filename(key)
        file_path = self.get_shared_workspace_path() / f"{safe_key}.json"

        if file_path.exists():
            data = self._read_json(file_path)
            if data:
                return ContextNote.from_dict(data)
        return None

    # =========================================================================
    # Utility methods
    # =========================================================================

    def cleanup_session(self, keep_summary: bool = True) -> None:
        """Clean up the current session workspace.

        Parameters
        ----------
        keep_summary : bool
            If True, create a summary before cleaning up.
        """
        if keep_summary:
            summary = self.get_session_summary()
            summary_file = self._workspace_dir / "final_summary.md"
            summary_file.write_text(summary, encoding="utf-8")

        # Clear cache
        self._cache.clear()

        logger.info(f"Session {self.session_id} cleaned up")

    def get_workspace_stats(self) -> Dict[str, Any]:
        """Get statistics about the workspace.

        Returns
        -------
        dict
            Workspace statistics.
        """
        stats = {
            "session_id": self.session_id,
            "workspace_dir": str(self._workspace_dir),
            "categories": {},
            "total_notes": 0,
            "total_size_bytes": 0,
        }

        for category in self.CATEGORIES:
            cat_dir = self._workspace_dir / category
            if cat_dir.exists():
                files = list(cat_dir.glob("*.json"))
                size = sum(f.stat().st_size for f in files)
                stats["categories"][category] = {
                    "count": len(files),
                    "size_bytes": size,
                }
                stats["total_notes"] += len(files)
                stats["total_size_bytes"] += size

        return stats

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to a file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON data from a file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Error reading JSON from {path}: {e}")
            return None

    def _sanitize_filename(self, key: str) -> str:
        """Sanitize a key for use as a filename."""
        # Replace unsafe characters
        safe = re.sub(r'[<>:"/\\|?*]', '_', key)
        # Limit length
        if len(safe) > 200:
            # Keep first 100, hash, last 50
            hash_part = hashlib.md5(key.encode()).hexdigest()[:8]
            safe = f"{safe[:100]}_{hash_part}_{safe[-50:]}"
        return safe

    def _check_auto_compress(self, category: str) -> None:
        """Check if auto-compression is needed for a category."""
        cat_dir = self._workspace_dir / category
        if not cat_dir.exists():
            return

        note_count = len(list(cat_dir.glob("*.json")))
        if note_count >= self.auto_compress_threshold:
            logger.info(f"Auto-compressing {category} ({note_count} notes)")
            self.compress_notes(category, keep_recent=10)

    def _default_summarize(self, notes: List[ContextNote]) -> str:
        """Default summarization: structured concatenation."""
        if not notes:
            return "No notes to summarize."

        summary_lines = [
            f"Summary of {len(notes)} notes:",
            f"Date range: {notes[-1].timestamp} to {notes[0].timestamp}",
            "",
            "Key points:",
        ]

        for note in notes:
            content = note.content
            if isinstance(content, dict):
                content_str = json.dumps(content)[:100]
            else:
                content_str = str(content)[:100]
            summary_lines.append(f"- [{note.key}]: {content_str}")

        return "\n".join(summary_lines)

    def _plan_to_markdown(self, plan: ExecutionPlan, brief: bool = False) -> str:
        """Convert plan to markdown format."""
        lines = []

        if not brief:
            lines.append(f"**Status**: {plan.status}")
            lines.append(f"**Progress**: {len(plan.completed_steps)}/{len(plan.steps)} steps")
            lines.append("")

        for i, step in enumerate(plan.steps):
            status_icon = "✅" if i in plan.completed_steps else ("🔄" if i == plan.current_step else "⬜")
            desc = step.get("description", f"Step {i+1}")
            lines.append(f"{status_icon} {i+1}. {desc}")

            if not brief and step.get("result"):
                lines.append(f"   → {step['result']}")

        return "\n".join(lines)

    def _format_note_for_context(self, note: ContextNote, brief: bool = False) -> str:
        """Format a note for LLM context injection."""
        if brief:
            content = note.content
            if isinstance(content, dict):
                content_str = json.dumps(content)[:150]
            else:
                content_str = str(content)[:150]
            return f"**{note.key}** ({note.category}): {content_str}"

        lines = [f"### {note.key}"]
        lines.append(f"*Category: {note.category} | Time: {note.timestamp}*")

        if isinstance(note.content, dict):
            lines.append("```json")
            lines.append(json.dumps(note.content, indent=2)[:500])
            lines.append("```")
        else:
            lines.append(str(note.content)[:500])

        return "\n".join(lines)

    def _get_recent_notes(self, n: int = 5) -> List[ContextNote]:
        """Get the N most recent notes across all categories."""
        all_notes = []

        for category in self.CATEGORIES:
            cat_dir = self._workspace_dir / category
            if not cat_dir.exists():
                continue

            for file_path in cat_dir.glob("*.json"):
                if file_path.stem.startswith("_summary"):
                    continue
                try:
                    data = self._read_json(file_path)
                    if data:
                        all_notes.append(ContextNote.from_dict(data))
                except Exception:
                    continue

        # Sort by timestamp
        all_notes.sort(key=lambda n: n.timestamp, reverse=True)
        return all_notes[:n]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for searching."""
        # Simple keyword extraction: lowercase, remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                     'very', 'just', 'also', 'now', 'here', 'there', 'when',
                     'where', 'why', 'how', 'all', 'each', 'every', 'both',
                     'few', 'more', 'most', 'other', 'some', 'such', 'no',
                     'run', 'execute', 'perform', 'data', 'adata', 'analysis'}

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return unique
