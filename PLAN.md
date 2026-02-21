# BioContext MCP Server 集成到 OmicVerse Agent 套件 — 实现计划

## 1. 目标

将 [BioContext.ai](https://biocontext.ai/) 的 MCP (Model Context Protocol) 生态系统集成到
OmicVerse Agent 中，使 Agent 在生成代码时能够 **实时查询外部生物医学数据库**
（STRING、UniProt、KEGG、Reactome、PanglaoDB、Open Targets 等 20+ 数据库），
并通过 `FilesystemContextManager` 缓存查询结果以避免重复请求。

### 核心价值
- **之前**：Agent 只能调用本地 OmicVerse 函数，无法访问在线数据库
- **之后**：Agent 生成的代码可以 `mcp_call("string_interaction_partners", {"identifiers": "TP53"})` 实时获取蛋白互作数据

---

## 2. 架构设计

```
用户 → ov.Agent(mcp_servers=["biocontext"])
                │
                ├─ MCPClientManager ← 连接 MCP 服务器, 发现工具
                │      │
                │      ├─ BioContext KB (remote HTTP)
                │      ├─ 自定义 MCP Server (stdio/HTTP)
                │      └─ OvIntelligence RAG MCP (本地)
                │
                ├─ _setup_agent()
                │      └─ 系统提示词 += MCP 工具描述
                │
                ├─ _build_sandbox_globals()
                │      └─ 注入 mcp_call() 函数
                │
                ├─ FilesystemContextManager
                │      └─ 缓存 MCP 查询结果 (Write/Select)
                │
                └─ Skill: biocontext-mcp
                       └─ 教 LLM 何时/如何使用 MCP 工具
```

---

## 3. 文件变更清单

### 3.1 新增文件 (4 个)

| 文件 | 大小估计 | 说明 |
|------|---------|------|
| `omicverse/utils/mcp_client.py` | ~400 行 | 通用 MCP 客户端管理器 |
| `omicverse/utils/biocontext_bridge.py` | ~250 行 | BioContext 预配置桥接层 |
| `.claude/skills/biocontext-mcp/SKILL.md` | ~200 行 | Agent Skill：教 LLM 使用 MCP |
| `tests/utils/test_mcp_client.py` | ~300 行 | MCP 客户端单元测试 |

### 3.2 修改文件 (5 个)

| 文件 | 修改范围 | 说明 |
|------|---------|------|
| `omicverse/utils/agent_config.py` | +20 行 | 新增 `MCPConfig` 数据类 |
| `omicverse/utils/smart_agent.py` | +120 行 (6 处) | MCP 集成到 Agent 管道 |
| `omicverse/agent/__init__.py` | +60 行 | 公开 API: `mcp_connect()`, `biocontext()` |
| `omicverse/utils/__init__.py` | +5 行 | 导出新模块 |
| `omicverse/utils/filesystem_context.py` | +1 行 (CATEGORIES) | 新增 `"mcp_cache"` 类别 |

---

## 4. 详细实现步骤

### Step 1: `omicverse/utils/mcp_client.py` — MCP 客户端管理器

**设计原则**: SDK-first, HTTP-fallback; 同步 API 封装异步底层

```python
# 核心数据类
@dataclass
class MCPToolParam:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False

@dataclass
class MCPTool:
    name: str
    description: str = ""
    parameters: List[MCPToolParam]
    server_name: str = ""

    @property
    def signature_text(self) -> str:
        """一行签名, 供 LLM 提示词使用"""

@dataclass
class MCPServerInfo:
    name: str
    url: Optional[str] = None          # HTTP 传输
    command: Optional[str] = None       # stdio 传输
    tools: List[MCPTool]
    transport: str = "http"             # "http" | "stdio"


# HTTP 轻量客户端 (无 SDK 依赖)
class _HTTPMCPClient:
    """JSON-RPC over HTTP, 兼容 Streamable HTTP + SSE 响应"""
    async def initialize() -> Dict
    async def list_tools() -> List[Dict]
    async def call_tool(name, arguments) -> Any


# 主管理器
class MCPClientManager:
    """管理一个或多个 MCP 服务器连接"""

    def connect(name, *, url=None, command=None, ...) -> MCPServerInfo
    def disconnect(name) -> None
    def list_tools(server_name=None) -> List[MCPTool]
    def call_tool(server_name, tool_name, arguments) -> Any
    def call(tool_name, arguments) -> Any   # 自动路由
    def tools_for_llm_prompt() -> str       # 供系统提示词注入
```

**关键决策**:
- HTTP 传输使用 `urllib.request`（无额外依赖），不引入 `requests`
- stdio 传输需要 `mcp` SDK，用 `ImportError` 优雅降级
- `_run_sync()` 处理 Jupyter 嵌套事件循环（线程桥接）
- 工具 schema 解析兼容 SDK 对象和 dict 两种格式

---

### Step 2: `omicverse/utils/biocontext_bridge.py` — BioContext 桥接层

**设计原则**: 零配置开箱即用；结果缓存到 FilesystemContextManager

```python
# 预配置常量
BIOCONTEXT_REMOTE_URL = "https://mcp.biocontext.ai/mcp/"
BIOCONTEXT_LOCAL_COMMAND = "uvx"
BIOCONTEXT_LOCAL_ARGS = ["biocontext_kb@latest"]

class BioContextBridge:
    """BioContext MCP 的高层封装，集成上下文缓存"""

    def __init__(
        self,
        mode: str = "remote",          # "remote" | "local" | "auto"
        context_manager: Optional[FilesystemContextManager] = None,
        cache_ttl: int = 3600,          # 缓存过期秒数
    )

    def connect(self) -> MCPServerInfo
    def query(self, tool_name, arguments, use_cache=True) -> Any
    def is_connected(self) -> bool
    def available_tools(self) -> List[MCPTool]

    # 便捷方法 (最常用的查询)
    def string_interactions(self, identifiers, species=9606) -> Dict
    def uniprot_lookup(self, accession) -> Dict
    def kegg_pathway(self, pathway_id) -> Dict
    def panglao_markers(self, cell_type) -> Dict
    def europepmc_search(self, query, limit=10) -> Dict
    def reactome_pathway(self, pathway_id) -> Dict
    def open_targets(self, target_id) -> Dict

    # 缓存管理
    def _cache_key(self, tool_name, arguments) -> str
    def _check_cache(self, cache_key) -> Optional[Any]
    def _write_cache(self, cache_key, result) -> None
```

**缓存策略**:
- 用 `FilesystemContextManager.write_note()` 写入 `"mcp_cache"` 类别
- `_cache_key` = `f"mcp_{tool_name}_{hashlib.md5(json.dumps(args)).hexdigest()[:12]}"`
- `_check_cache` 先 `search_context(cache_key, "glob")`，再检查 TTL
- 避免重复调用同一个 MCP 工具（如同一基因的 STRING 查询）

---

### Step 3: `omicverse/utils/agent_config.py` — 新增 MCPConfig

在现有四组配置之后添加第五组:

```python
@dataclass
class MCPConfig:
    """MCP server connection settings."""
    servers: List[Dict[str, Any]] = field(default_factory=list)
    # 每个 server: {"name": "biocontext", "url": "https://...", "enabled": True}
    enable_biocontext: bool = False     # 是否自动连接 BioContext
    biocontext_mode: str = "remote"     # "remote" | "local" | "auto"
    cache_ttl: int = 3600               # MCP 结果缓存秒数
    inject_tools_in_prompt: bool = True # 是否将工具描述注入系统提示词

@dataclass
class AgentConfig:
    llm: LLMConfig = ...
    reflection: ReflectionConfig = ...
    execution: ExecutionConfig = ...
    context: ContextConfig = ...
    mcp: MCPConfig = field(default_factory=MCPConfig)    # ← 新增
    verbose: bool = True
    ...
```

**向后兼容**: `from_flat_kwargs()` 新增 `mcp_servers`, `enable_biocontext` 参数映射。

---

### Step 4: `omicverse/utils/smart_agent.py` — 6 处集成点

#### 4a. `__init__()` — 初始化 MCP 组件 (~+30 行)

在 `_initialize_skill_registry()` 之后:

```python
# Initialize MCP client if configured
self._mcp_manager: Optional[MCPClientManager] = None
self._biocontext: Optional[BioContextBridge] = None

mcp_cfg = self._config.mcp if hasattr(self._config, 'mcp') else MCPConfig()
if mcp_cfg.enable_biocontext or mcp_cfg.servers:
    self._init_mcp(mcp_cfg)
```

新增方法:
```python
def _init_mcp(self, mcp_cfg: MCPConfig) -> None:
    """Initialize MCP connections."""
    from .mcp_client import MCPClientManager
    from .biocontext_bridge import BioContextBridge

    self._mcp_manager = MCPClientManager()

    # Auto-connect BioContext if enabled
    if mcp_cfg.enable_biocontext:
        self._biocontext = BioContextBridge(
            mode=mcp_cfg.biocontext_mode,
            context_manager=self._filesystem_context,
            cache_ttl=mcp_cfg.cache_ttl,
        )
        try:
            info = self._biocontext.connect()
            self._mcp_manager._servers["biocontext"] = ...  # 共享引用
            print(f"   🔗 BioContext MCP connected: {len(info.tools)} tools")
        except Exception as e:
            print(f"   ⚠️  BioContext connection failed: {e}")

    # Connect additional servers
    for srv in mcp_cfg.servers:
        try:
            self._mcp_manager.connect(**srv)
        except Exception as e:
            print(f"   ⚠️  MCP server '{srv.get('name')}' failed: {e}")
```

#### 4b. `_setup_agent()` — 系统提示词注入 MCP 工具描述 (~+15 行)

在 `instructions += self._build_filesystem_context_instructions()` 之后:

```python
# Add MCP tool descriptions if available
if self._mcp_manager and self._mcp_manager.connected_servers:
    mcp_cfg = getattr(self._config, 'mcp', None)
    if mcp_cfg is None or mcp_cfg.inject_tools_in_prompt:
        instructions += self._build_mcp_tools_instructions()
```

新增方法:
```python
def _build_mcp_tools_instructions(self) -> str:
    """Build MCP tools section for system prompt."""
    tools_text = self._mcp_manager.tools_for_llm_prompt()
    if not tools_text:
        return ""
    return f"""

## External Database Tools (MCP)

You have access to external biomedical databases via the Model Context Protocol.
To query these databases in your generated code, use:

```python
result = mcp_call("tool_name", {{"param": "value"}})
```

The `mcp_call` function is pre-loaded in the execution environment.
Results are automatically cached — repeated queries with the same parameters
return cached results without network calls.

**IMPORTANT**: Only call MCP tools when the user's request explicitly needs
external database information (e.g., protein interactions, pathway data,
gene markers). Do NOT call MCP tools for standard analysis operations.

{tools_text}
"""
```

#### 4c. `_build_sandbox_globals()` — 注入 `mcp_call()` 函数 (~+20 行)

在 `sandbox_globals["ov"] = omicverse` 之后:

```python
# Inject MCP tool caller if available
if self._mcp_manager and self._mcp_manager.connected_servers:
    def mcp_call(tool_name: str, arguments: dict = None) -> Any:
        """Call an MCP tool and return the result."""
        result = self._mcp_manager.call(tool_name, arguments or {})
        # Cache result via filesystem context
        if self._biocontext:
            self._biocontext._write_cache(
                self._biocontext._cache_key(tool_name, arguments or {}),
                result
            )
        return result

    sandbox_globals["mcp_call"] = mcp_call

    # Also inject convenience aliases for common BioContext tools
    if self._biocontext:
        sandbox_globals["biocontext"] = self._biocontext
```

#### 4d. `_run_skills_workflow()` — 注入 MCP 上下文 (~+10 行)

在 `priority2_prompt` 构建时，`{skill_guidance_section}` 之后添加:

```python
mcp_context_section = ""
if self._mcp_manager and self._mcp_manager.connected_servers:
    mcp_context_section = (
        "\nExternal Database Tools (MCP):\n"
        "Use `mcp_call(tool_name, args_dict)` to query external databases.\n"
        f"{self._mcp_manager.tools_for_llm_prompt()}\n"
    )

# 在 prompt 中插入
priority2_prompt = f'''...
{skill_guidance_section}
{mcp_context_section}
...'''
```

#### 4e. `Agent()` 工厂函数 — 新增 MCP 参数 (~+15 行)

```python
def Agent(
    model="gemini-2.5-flash",
    ...,
    # MCP parameters (新增)
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
    enable_biocontext: bool = False,
    biocontext_mode: str = "remote",
    ...
) -> OmicVerseAgent:
```

传递到 `AgentConfig.from_flat_kwargs()` 中构建 `MCPConfig`。

#### 4f. `__del__()` — 清理 MCP 连接 (~+5 行)

```python
if hasattr(self, '_mcp_manager') and self._mcp_manager:
    try:
        self._mcp_manager.disconnect_all()
    except:
        pass
```

---

### Step 5: `.claude/skills/biocontext-mcp/SKILL.md` — Agent Skill

教 LLM 何时/如何使用 BioContext MCP 工具:

```yaml
---
name: biocontext-mcp
title: BioContext External Database Queries via MCP
description: >
  Query external biomedical databases (STRING, UniProt, KEGG, Reactome,
  PanglaoDB, Open Targets, EuropePMC) in real-time using MCP tools.
  Use when analysis requires protein interactions, pathway data,
  cell type markers, or literature search.
---
```

Skill body 包含:
1. **When to Use** — 用户请求涉及外部数据库查询时
2. **Available Tools** — 主要工具列表和参数说明
3. **Code Patterns** — `mcp_call()` 的正确使用模式
4. **Result Processing** — 如何解析和使用返回的 JSON
5. **Caching** — 结果自动缓存，重复查询不走网络
6. **Common Workflows** — 基因→蛋白互作→通路富集 的完整示例

---

### Step 6: `omicverse/utils/filesystem_context.py` — 新增缓存类别

```python
CATEGORIES = {
    "notes": "General notes and observations",
    "results": "Intermediate computation results",
    "decisions": "Decision points and rationale",
    "snapshots": "Data state snapshots",
    "figures": "Generated figure paths",
    "errors": "Error logs and debugging info",
    "mcp_cache": "Cached MCP tool query results",   # ← 新增
}
```

---

### Step 7: `omicverse/agent/__init__.py` — 公开 API

```python
# 在现有 seeker() 之后添加

def mcp_connect(
    name: str,
    *,
    url: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Connect to an MCP server and return its tool inventory.

    Examples
    --------
    >>> import omicverse as ov
    >>> info = ov.agent.mcp_connect("biocontext",
    ...     url="https://mcp.biocontext.ai/mcp/")
    >>> print(f"Connected: {len(info['tools'])} tools")
    """
    from omicverse.utils.mcp_client import MCPClientManager
    mgr = MCPClientManager()
    server = mgr.connect(name, url=url, command=command, args=args)
    return {
        "name": server.name,
        "tools": [t.name for t in server.tools],
        "tool_count": len(server.tools),
        "transport": server.transport,
    }


def biocontext(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    mode: str = "remote",
) -> Any:
    """Quick one-shot query to BioContext MCP.

    Examples
    --------
    >>> import omicverse as ov
    >>> result = ov.agent.biocontext("string_interaction_partners",
    ...     {"identifiers": "TP53", "species": 9606})
    """
    from omicverse.utils.biocontext_bridge import BioContextBridge
    bridge = BioContextBridge(mode=mode)
    bridge.connect()
    return bridge.query(tool_name, arguments or {})


__all__ = ["seeker", "mcp_connect", "biocontext"]
```

---

### Step 8: `omicverse/utils/__init__.py` — 导出

```python
# 在现有 agent_reporter 导入之后
from .mcp_client import MCPClientManager, MCPTool, MCPServerInfo
from .biocontext_bridge import BioContextBridge
```

添加到 `__all__`:
```python
"MCPClientManager", "MCPTool", "MCPServerInfo", "BioContextBridge",
```

---

### Step 9: `tests/utils/test_mcp_client.py` — 单元测试

**测试策略**: 全部 mock，不依赖网络

```python
class TestMCPClientManager:
    def test_connect_http_initializes_client(self, mock_http)
    def test_connect_http_discovers_tools(self, mock_http)
    def test_list_tools_single_server(self)
    def test_list_tools_all_servers(self)
    def test_call_tool_routes_correctly(self, mock_http)
    def test_call_auto_routes_to_correct_server(self)
    def test_disconnect_removes_server(self)
    def test_tools_for_llm_prompt_format(self)
    def test_parse_tool_schema_from_dict(self)
    def test_parse_tool_schema_from_sdk_object(self)

class TestBioContextBridge:
    def test_connect_remote_mode(self, mock_mcp)
    def test_query_with_cache_hit(self, mock_context)
    def test_query_with_cache_miss(self, mock_mcp, mock_context)
    def test_convenience_methods(self, mock_mcp)
    def test_cache_key_deterministic(self)
    def test_cache_ttl_expired(self, mock_context)

class TestAgentMCPIntegration:
    def test_agent_init_with_biocontext(self, mock_mcp)
    def test_mcp_tools_in_system_prompt(self, mock_agent)
    def test_mcp_call_in_sandbox(self, mock_mcp)
    def test_agent_without_mcp_unchanged(self)
```

---

## 5. 依赖管理

### 新增依赖: 无

- HTTP 传输: `urllib.request` (标准库)
- JSON-RPC: `json` (标准库)
- MCP SDK (可选): 仅 stdio 传输时需要 `pip install "mcp[cli]"`
- BioContext (可选): `pip install biocontext-kb` (仅本地部署时需要)

### 依赖原则
- HTTP 模式零额外依赖，降低安装门槛
- MCP SDK 通过 `ImportError` 优雅降级
- BioContext 远程模式无需安装任何包

---

## 6. 用户体验设计

### 6.1 最简使用 (一行代码)

```python
import omicverse as ov
result = ov.agent.biocontext("string_interaction_partners",
    {"identifiers": "TP53", "species": 9606})
```

### 6.2 Agent 集成 (自动启用)

```python
agent = ov.Agent(model="gemini-2.5-flash", enable_biocontext=True)
# Agent 现在知道可以调用 BioContext 工具
adata = agent.run("找到 TP53 的蛋白互作伙伴并在我的数据中做子集分析", adata)
# → Agent 生成的代码会自动调用 mcp_call("string_interaction_partners", ...)
```

### 6.3 自定义 MCP 服务器

```python
agent = ov.Agent(
    model="gemini-2.5-flash",
    mcp_servers=[
        {"name": "biocontext", "url": "https://mcp.biocontext.ai/mcp/"},
        {"name": "my_rag", "command": "python", "args": ["my_mcp_server.py"]},
    ],
)
```

### 6.4 探索可用工具

```python
info = ov.agent.mcp_connect("biocontext",
    url="https://mcp.biocontext.ai/mcp/")
print(info["tools"])  # ['string_interaction_partners', 'uniprot_lookup', ...]
```

---

## 7. 数据流详解

```
用户: "查找 TP53 的蛋白互作网络并做通路富集"

1. 复杂度分析 → "complex" (涉及外部查询 + 分析)

2. Skill 匹配 → 匹配 "biocontext-mcp" + "gsea-enrichment"

3. 代码生成 (LLM 看到系统提示词中的 MCP 工具描述):
   ```python
   import omicverse as ov
   # Step 1: Query STRING for TP53 interactions
   interactions = mcp_call("string_interaction_partners",
       {"identifiers": "TP53", "species": 9606, "limit": 50})
   partner_genes = [p["preferredName"] for p in interactions["partners"]]
   print("Found " + str(len(partner_genes)) + " interaction partners")

   # Step 2: Subset adata to interaction network
   network_genes = [g for g in partner_genes if g in adata.var_names]
   adata_network = adata[:, network_genes].copy()

   # Step 3: Run pathway enrichment
   pathway_dict = ov.utils.geneset_prepare("pathway_file.gmt", organism="Human")
   ov.utils.bindea_bindea(adata_network, pathway_dict)
   ```

4. 沙箱执行:
   - mcp_call() → MCPClientManager.call() → HTTP POST to BioContext
   - 结果自动缓存到 FilesystemContextManager("mcp_cache")

5. 反思 + 结果审查 → 返回给用户
```

---

## 8. 实现顺序和优先级

| 阶段 | 步骤 | 优先级 | 预估改动 |
|------|------|--------|---------|
| P0 | Step 1: `mcp_client.py` | 必须 | 新增 ~400 行 |
| P0 | Step 2: `biocontext_bridge.py` | 必须 | 新增 ~250 行 |
| P0 | Step 3: `agent_config.py` MCPConfig | 必须 | +20 行 |
| P0 | Step 4a-4c: `smart_agent.py` 核心集成 | 必须 | +65 行 |
| P1 | Step 5: BioContext Skill | 重要 | 新增 ~200 行 |
| P1 | Step 6: FilesystemContext 缓存类别 | 重要 | +1 行 |
| P1 | Step 4d-4f: `smart_agent.py` 完整集成 | 重要 | +30 行 |
| P1 | Step 7: `agent/__init__.py` 公开 API | 重要 | +60 行 |
| P2 | Step 8: `utils/__init__.py` 导出 | 次要 | +5 行 |
| P2 | Step 9: 单元测试 | 次要 | 新增 ~300 行 |

**总计**: ~1330 行新代码 + ~120 行修改

---

## 9. 与现有组件的关系

### 与 FilesystemContextManager 的关系
- **Write**: MCP 结果写入 `"mcp_cache"` 类别
- **Select**: 代码生成前先 `search_context("mcp_*")` 检查缓存
- **Compress**: 过期 MCP 缓存被自动摘要
- **Isolate**: 子 Agent 共享 MCP 缓存

### 与 SkillRegistry 的关系
- 新增 `biocontext-mcp` Skill 遵循现有 SKILL.md 格式
- LLM Skill 匹配自动识别需要外部数据库查询的请求
- 渐进式加载: 元数据在启动时加载，完整内容按需加载

### 与 OvIntelligence 的关系
- OvIntelligence 的 `rag_mcp_server.py` 是 MCP **Server**
- 本次新增的是 MCP **Client** 能力
- 未来可以让 Agent 同时连接 BioContext (外部) + RAG (内部)

### 与 Inspector 的关系
- Inspector 验证本地前置条件 (adata 状态)
- MCP 提供外部数据注入 (不涉及 adata 前置条件)
- 两者互补，不冲突

### 与 ProactiveCodeTransformer 的关系
- 可能需要添加 MCP 相关的代码转换规则
- 例如: 确保 `mcp_call()` 的返回值被正确处理
- 这是可选的后续优化，不在本次 P0 范围内

---

## 10. 风险和缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| BioContext 远程服务器不可达 | 中 | 低 | `enable_biocontext=False` 是默认值; 连接失败只打印警告 |
| MCP 响应格式变化 | 低 | 中 | `_parse_tool_schema()` 兼容 dict + SDK 对象 |
| LLM 错误调用 MCP 工具 | 中 | 低 | Skill 中明确说明使用条件; 反思步骤检查 |
| 沙箱中 `mcp_call` 被滥用 | 低 | 中 | 只在有 MCP 连接时注入; rate limiting 在 BioContext 服务端 |
| 嵌套事件循环 (Jupyter) | 中 | 高 | `_run_sync()` 使用线程桥接，已验证模式 |

---

## 11. 不做的事情 (Out of Scope)

1. **不修改 LLM 后端** — MCP 不涉及 LLM 提供商
2. **不修改 SessionNotebookExecutor** — mcp_call 通过 sandbox globals 注入，notebook 无感知
3. **不添加 MCP Server 能力** — OvIntelligence 已有，本次只做 Client
4. **不修改 Verifier** — MCP Skill 的质量验证可后续添加
5. **不做 MCP-to-Skill 自动转换** — BioContext 的 skill-to-mcp 已有逆向工具，不重复
