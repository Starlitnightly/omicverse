# OV Agent 后端改进计划

> 借鉴 OpenAI Codex CLI 架构，针对 `agent_backend.py` / `smart_agent.py` / `model_config.py` 的具体改进方案。
> 每项改进包含：现状分析、目标设计、涉及文件与行号、迁移步骤、Before/After 代码示例、回归验证要点。

---

## P0-1: Provider 注册表模式

### 现状分析

当前 `agent_backend.py` 中存在 **三条并行的 if/elif 分发链**，每加一个 Provider 必须同步修改三处：

| 分发点 | 位置 | 职责 |
|--------|------|------|
| `_run_sync()` | `agent_backend.py:427-446` | 非流式调用分发 |
| `_stream_async()` | `agent_backend.py:397-422` | 流式调用分发 |
| `_resolve_api_key()` | `agent_backend.py:448-472` | API Key 解析 |

同时 `model_config.py` 中有 **四张平铺字典** (`AVAILABLE_MODELS`, `PROVIDER_API_KEYS`, `PROVIDER_ENDPOINTS`, `PROVIDER_DEFAULT_KEYS`) 也按 Provider 组织但彼此独立，新增模型需要在四处都加一条。

`OPENAI_COMPAT_BASE_URLS`（`agent_backend.py:50-58`）是第五个需要同步维护的数据结构。

**Codex 对应方案**: `ModelProviderInfo` 结构体将 `base_url`、`env_key`、`wire_api`、`retry_config`、`custom_headers` 合并为单一注册记录，新增 Provider 只需一条注册。

### 目标设计

```
model_config.py 中的 4 张字典 + agent_backend.py 中的 OPENAI_COMPAT_BASE_URLS
    ↓ 合并为
ProviderInfo 数据类注册表 (单一数据源)
    ↓ 消除
_run_sync / _stream_async / _resolve_api_key 中的 if/elif 链
    ↓ 替换为
provider_registry[provider_name].chat() / .stream() / .resolve_key()
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `omicverse/utils/model_config.py` | 重构：4 张字典 → `ProviderInfo` 注册表 |
| `omicverse/utils/agent_backend.py` | 重构：if/elif 链 → 注册表查找 + Protocol 分发 |

### 详细步骤

#### 步骤 1: 定义 ProviderInfo 数据类（新增于 `model_config.py`）

```python
# --- Before: 4 张散落的字典 ---
PROVIDER_ENDPOINTS = {"openai": "https://api.openai.com/v1", ...}
PROVIDER_DEFAULT_KEYS = {"openai": "OPENAI_API_KEY", ...}
OPENAI_COMPAT_BASE_URLS = {"openai": "https://api.openai.com/v1", ...}

# --- After: 单一注册表 ---
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class WireAPI(Enum):
    """LLM 通信协议类型"""
    CHAT_COMPLETIONS = "chat"        # /v1/chat/completions
    RESPONSES = "responses"          # /v1/responses (GPT-5)
    ANTHROPIC_MESSAGES = "anthropic" # Anthropic Messages API
    GEMINI_GENERATE = "gemini"       # Google Gemini generateContent
    DASHSCOPE = "dashscope"          # 阿里 DashScope
    LOCAL = "local"                  # 本地 Python 执行

@dataclass(frozen=True)
class ProviderInfo:
    """单个 Provider 的完整配置（Codex ModelProviderInfo 的 Python 等价物）"""
    name: str                                    # "openai", "anthropic", ...
    display_name: str                            # "OpenAI", "Anthropic", ...
    base_url: str                                # API endpoint
    env_key: str                                 # 主 API key 环境变量名
    wire_api: WireAPI                            # 通信协议
    alt_env_keys: List[str] = field(default_factory=list)  # 备用环境变量
    openai_compatible: bool = False              # 是否走 OpenAI SDK
    max_retry_attempts: int = 3
    stream_timeout_secs: int = 300
    models: dict = field(default_factory=dict)   # model_id -> description

# 全局注册表
PROVIDER_REGISTRY: dict[str, ProviderInfo] = {}

def register_provider(info: ProviderInfo) -> None:
    """注册一个 Provider（支持用户在运行时扩展）"""
    PROVIDER_REGISTRY[info.name] = info

def get_provider(name: str) -> ProviderInfo:
    """按名称查找 Provider"""
    if name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {name}. "
                         f"Available: {list(PROVIDER_REGISTRY.keys())}")
    return PROVIDER_REGISTRY[name]
```

#### 步骤 2: 注册所有内置 Provider

```python
# 替代原来分散在 4 张字典里的数据
register_provider(ProviderInfo(
    name="openai",
    display_name="OpenAI",
    base_url="https://api.openai.com/v1",
    env_key="OPENAI_API_KEY",
    wire_api=WireAPI.CHAT_COMPLETIONS,
    openai_compatible=True,
    models={
        "gpt-5": "OpenAI GPT-5 (Latest)",
        "gpt-5-mini": "OpenAI GPT-5 Mini",
        "gpt-4o": "OpenAI GPT-4o",
        "gpt-4o-mini": "OpenAI GPT-4o Mini",
        # ...
    },
))

register_provider(ProviderInfo(
    name="anthropic",
    display_name="Anthropic",
    base_url="https://api.anthropic.com",
    env_key="ANTHROPIC_API_KEY",
    wire_api=WireAPI.ANTHROPIC_MESSAGES,
    models={
        "anthropic/claude-opus-4-1-20250805": "Claude Opus 4.1",
        # ...
    },
))

register_provider(ProviderInfo(
    name="zhipu",
    display_name="Zhipu AI",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    env_key="ZAI_API_KEY",
    alt_env_keys=["ZHIPUAI_API_KEY"],  # 取代 _resolve_api_key 里的 or 特判
    wire_api=WireAPI.CHAT_COMPLETIONS,
    openai_compatible=True,
    models={...},
))
# ... 其余 Provider 同理
```

#### 步骤 3: 重构 `_resolve_api_key`（`agent_backend.py:448-472`）

```python
# --- Before: 12 行 if/elif ---
def _resolve_api_key(self) -> Optional[str]:
    if self.config.api_key:
        return self.config.api_key
    provider = self.config.provider
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "xai":
        return os.getenv("XAI_API_KEY")
    # ... 重复 8 次 ...

# --- After: 4 行 ---
def _resolve_api_key(self) -> Optional[str]:
    if self.config.api_key:
        return self.config.api_key
    info = get_provider(self.config.provider)
    return os.getenv(info.env_key) or next(
        (os.getenv(k) for k in info.alt_env_keys if os.getenv(k)), None
    )
```

#### 步骤 4: 重构 `_run_sync` 和 `_stream_async`（`agent_backend.py:397-446`）

```python
# --- Before: 两条并行 if/elif 链 ---
def _run_sync(self, user_prompt: str) -> str:
    provider = self.config.provider
    if provider in OPENAI_COMPAT_BASE_URLS:
        return self._chat_via_openai_compatible(user_prompt)
    if provider == "python":
        return self._run_python_local(user_prompt)
    if provider == "anthropic":
        return self._chat_via_anthropic(user_prompt)
    # ...

# --- After: 按 wire_api 分发 ---
# 映射表：WireAPI → (sync_method, stream_method)
_DISPATCH: dict[WireAPI, tuple[str, str]] = {
    WireAPI.CHAT_COMPLETIONS: ("_chat_via_openai_compatible", "_stream_openai_compatible"),
    WireAPI.RESPONSES:        ("_chat_via_openai_responses_dispatch", "_stream_openai_responses_dispatch"),
    WireAPI.ANTHROPIC_MESSAGES: ("_chat_via_anthropic", "_stream_anthropic"),
    WireAPI.GEMINI_GENERATE:   ("_chat_via_gemini", "_stream_gemini"),
    WireAPI.DASHSCOPE:         ("_chat_via_dashscope", "_stream_dashscope"),
    WireAPI.LOCAL:             ("_run_python_local", "_run_python_local_stream"),
}

def _run_sync(self, user_prompt: str) -> str:
    info = get_provider(self.config.provider)
    method_name = _DISPATCH[info.wire_api][0]
    return getattr(self, method_name)(user_prompt)

async def _stream_async(self, user_prompt: str):
    info = get_provider(self.config.provider)
    method_name = _DISPATCH[info.wire_api][1]
    async for chunk in getattr(self, method_name)(user_prompt):
        yield chunk
```

#### 步骤 5: 重构 ModelConfig 静态方法

`get_provider_from_model`（`model_config.py:257-278`）的 if/elif 前缀匹配改为遍历注册表：

```python
# --- Before: 前缀 if/elif 链 ---
@staticmethod
def get_provider_from_model(model: str) -> str:
    if model.startswith("anthropic/"):
        return "anthropic"
    elif model.startswith(("qwq-", "qwen-")):
        return "dashscope"
    # ...

# --- After: 从注册表反查 ---
@staticmethod
def get_provider_from_model(model: str) -> str:
    model = ModelConfig.normalize_model_id(model)
    for name, info in PROVIDER_REGISTRY.items():
        if model in info.models:
            return name
    # 前缀 fallback (向后兼容未注册的自定义模型)
    for name, info in PROVIDER_REGISTRY.items():
        prefix = name + "/"
        if model.startswith(prefix):
            return name
    return "openai"  # 默认
```

### 向后兼容

- 保留 `AVAILABLE_MODELS`、`PROVIDER_API_KEYS` 等字典作为**计算属性**，从 `PROVIDER_REGISTRY` 生成，不破坏外部代码对这些字典的引用。
- `OPENAI_COMPAT_BASE_URLS` 改为 `property`，从 `openai_compatible=True` 的 Provider 动态构建。

### 回归验证

- [ ] 所有已有模型仍能通过 `ModelConfig.is_model_supported()` 验证
- [ ] `_resolve_api_key()` 对每个 provider 返回值与原实现一致
- [ ] `_run_sync()` / `_stream_async()` 对每个 provider 的分发路径不变
- [ ] 新增自定义 Provider 只需调用 `register_provider()` 一次

---

## P0-2: 分组 Config 数据类

### 现状分析

`OmicVerseAgent.__init__`（`smart_agent.py:238`）接受 **15 个参数**，混合了四个不相关的关注点：

| 关注点 | 参数 |
|--------|------|
| LLM 配置 | `model`, `api_key`, `endpoint` |
| 反思策略 | `enable_reflection`, `reflection_iterations`, `enable_result_review` |
| Notebook 执行 | `use_notebook_execution`, `max_prompts_per_session`, `notebook_storage_dir`, `keep_execution_notebooks`, `notebook_timeout`, `strict_kernel_validation` |
| 文件系统上下文 | `enable_filesystem_context`, `context_storage_dir` |

加新功能 = 在这个构造函数上再追加参数。

**Codex 对应方案**: `Config` / `ConfigBuilder` 按职责分组，支持分层合并（CLI > env > profile > project > defaults）。

### 目标设计

```python
agent = ov.Agent(
    llm=LLMConfig(model="gpt-4o", api_key="sk-..."),
    reflection=ReflectionConfig(enabled=True, iterations=2),
    execution=ExecutionConfig(use_notebook=True, timeout=600),
    context=ContextConfig(enabled=True, storage_dir="~/.ovagent/context"),
)
# 或者保持向后兼容的平铺用法：
agent = ov.Agent(model="gpt-4o", api_key="sk-...")
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `omicverse/utils/smart_agent.py:238-412` | 重构构造函数 |
| 新增 `omicverse/utils/agent_config.py` | 新建配置模块 |

### 详细步骤

#### 步骤 1: 定义分组 Config 数据类（新文件 `agent_config.py`）

```python
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass
class LLMConfig:
    """LLM 连接配置"""
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

@dataclass
class ReflectionConfig:
    """代码反思与结果审查配置"""
    enabled: bool = True
    iterations: int = 1          # 自动 clamp 到 1-3
    result_review: bool = True

    def __post_init__(self):
        self.iterations = max(1, min(3, self.iterations))

@dataclass
class ExecutionConfig:
    """代码执行环境配置"""
    use_notebook: bool = True
    max_prompts_per_session: int = 5
    storage_dir: Optional[Path] = None
    keep_notebooks: bool = True
    timeout: int = 600
    strict_kernel_validation: bool = True

@dataclass
class ContextConfig:
    """文件系统上下文管理配置"""
    enabled: bool = True
    storage_dir: Optional[Path] = None

@dataclass
class AgentConfig:
    """Agent 完整配置（聚合各子配置）"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    verbose: bool = True  # 控制 print 输出（为 P1-1 做铺垫）

    @classmethod
    def from_flat_kwargs(cls, **kwargs) -> "AgentConfig":
        """从平铺关键字参数构建（向后兼容 OmicVerseAgent 原始签名）"""
        return cls(
            llm=LLMConfig(
                model=kwargs.get("model", "gemini-2.5-flash"),
                api_key=kwargs.get("api_key"),
                endpoint=kwargs.get("endpoint"),
            ),
            reflection=ReflectionConfig(
                enabled=kwargs.get("enable_reflection", True),
                iterations=kwargs.get("reflection_iterations", 1),
                result_review=kwargs.get("enable_result_review", True),
            ),
            execution=ExecutionConfig(
                use_notebook=kwargs.get("use_notebook_execution", True),
                max_prompts_per_session=kwargs.get("max_prompts_per_session", 5),
                storage_dir=Path(kwargs["notebook_storage_dir"]) if kwargs.get("notebook_storage_dir") else None,
                keep_notebooks=kwargs.get("keep_execution_notebooks", True),
                timeout=kwargs.get("notebook_timeout", 600),
                strict_kernel_validation=kwargs.get("strict_kernel_validation", True),
            ),
            context=ContextConfig(
                enabled=kwargs.get("enable_filesystem_context", True),
                storage_dir=Path(kwargs["context_storage_dir"]) if kwargs.get("context_storage_dir") else None,
            ),
        )
```

#### 步骤 2: 重构 `OmicVerseAgent.__init__`

```python
# --- Before: 15 个参数平铺 ---
def __init__(self, model="gemini-2.5-flash", api_key=None, endpoint=None,
             enable_reflection=True, reflection_iterations=1, ...):

# --- After: 接受 AgentConfig 或平铺参数 ---
def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
    if config is None:
        config = AgentConfig.from_flat_kwargs(**kwargs)
    self.config = config
    # 后续全部从 self.config.llm.model 等路径读取
```

#### 步骤 3: `Agent()` 工厂函数保持签名兼容

```python
def Agent(model="gemini-2.5-flash", api_key=None, endpoint=None, *,
          config: Optional[AgentConfig] = None, **kwargs) -> OmicVerseAgent:
    if config is not None:
        return OmicVerseAgent(config=config)
    return OmicVerseAgent(config=AgentConfig.from_flat_kwargs(
        model=model, api_key=api_key, endpoint=endpoint, **kwargs
    ))
```

### 回归验证

- [ ] `ov.Agent(model="gpt-4o", api_key="sk-...")` 仍然可用（向后兼容）
- [ ] `ov.Agent(config=AgentConfig(...))` 新用法可用
- [ ] 所有内部对 `self.enable_reflection` 等属性的引用改为 `self.config.reflection.enabled`
- [ ] 现有测试不受影响

---

## P0-3: 异常处理重构

### 现状分析

三个层面的问题：

**问题 1: `run_async` 中异常当控制流**（`smart_agent.py:2955-2976`）

```python
except ValueError as e:
    # Priority 1 failed → fall back
    fallback_occurred = True
except Exception as e:
    # 所有异常（含 TypeError/KeyError 等真 bug）也 fallback
    fallback_occurred = True
```

一个 `_run_registry_workflow` 里的 `KeyError`（代码 bug）被静默吞掉，当作"任务太复杂"处理。

**问题 2: `__init__` 中异常静默吞掉**（`smart_agent.py:281-285`）

```python
try:
    model = ModelConfig.normalize_model_id(model)
except Exception:
    model = model  # 所有异常包括 bug 都被吞掉
```

**问题 3: `agent_backend.py` 中 `_should_retry` 过宽匹配**（`agent_backend.py:164-170`）

```python
transient_exception_names = ['apierror', ...]
```

`'apierror'` 匹配 OpenAI SDK 的 `APIError`，包含 400 Bad Request 等不应重试的客户端错误。

**Codex 对应方案**: 结构化 `CodexErr` 枚举，每个变体明确 `is_retryable()`；命令结果用 `Result` 类型而非 exception-as-control-flow。

### 目标设计

新建 `omicverse/utils/agent_errors.py`：

```python
class OVAgentError(Exception):
    """Agent 错误基类"""
    pass

class WorkflowNeedsFallback(OVAgentError):
    """工作流判断需要升级到更复杂的策略（不是 bug，是正常流转）"""
    pass

class ProviderError(OVAgentError):
    """LLM Provider 相关错误"""
    def __init__(self, message: str, provider: str, status_code: int = 0,
                 retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable

class ExecutionError(OVAgentError):
    """代码执行错误"""
    pass

class ConfigError(OVAgentError):
    """配置错误（API key 缺失、模型不支持等）"""
    pass

class SandboxDeniedError(ExecutionError):
    """沙箱/Notebook 执行被拒绝"""
    pass
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| 新增 `omicverse/utils/agent_errors.py` | 新建错误层级模块 |
| `omicverse/utils/smart_agent.py:2955-2976` | `run_async` 改用 `WorkflowNeedsFallback` |
| `omicverse/utils/smart_agent.py:281-285` | `__init__` 中收窄 except |
| `omicverse/utils/agent_backend.py:133-195` | `_should_retry` 精确分类 |

### 详细步骤

#### 步骤 1: `run_async` 中区分 fallback 信号和真 bug

```python
# --- Before (smart_agent.py:2941-2976) ---
try:
    result = await self._run_registry_workflow(request, adata)
    return result
except ValueError as e:
    fallback_occurred = True
except Exception as e:        # ← bug 也被吞掉
    fallback_occurred = True

# --- After ---
try:
    result = await self._run_registry_workflow(request, adata)
    return result
except WorkflowNeedsFallback as e:
    # 正常流转：registry 工作流判断需要升级
    self._reporter.info(f"Registry workflow insufficient: {e}")
    fallback_occurred = True
except OVAgentError:
    raise  # Agent 已知错误直接传播
except Exception as e:
    # 未预期异常 → 记录但不静默
    logger.error("Unexpected error in registry workflow: %s", e, exc_info=True)
    raise
```

在 `_run_registry_workflow` 中，将"任务太复杂"的判断改为 `raise WorkflowNeedsFallback(reason)` 而非 `raise ValueError(reason)`。

#### 步骤 2: `__init__` 中收窄异常处理

```python
# --- Before (smart_agent.py:281-285) ---
try:
    model = ModelConfig.normalize_model_id(model)
except Exception:
    model = model

# --- After ---
try:
    model = ModelConfig.normalize_model_id(model)
except (KeyError, ValueError):
    # normalize_model_id 可能对未识别的模型名抛出，保留原值
    pass
```

#### 步骤 3: `_should_retry` 精确分类

```python
# --- Before (agent_backend.py:164-170) ---
transient_exception_names = [
    'timeout', 'connection', 'ratelimit', 'serviceunavailable',
    'apierror', ...  # ← apierror 太宽
]

# --- After ---
# 明确可重试的 SDK 异常类名
_RETRYABLE_EXCEPTION_NAMES = {
    'timeout', 'connection', 'ratelimit', 'serviceunavailable',
    'throttl', 'unavailable', 'overload', 'internalservererror',
}
# 明确不可重试的
_NON_RETRYABLE_EXCEPTION_NAMES = {
    'authenticationerror', 'permissiondenied', 'notfound',
    'badrequest', 'invalidrequest', 'validationerror',
}

def _should_retry(exc: Exception) -> bool:
    exc_type_name = type(exc).__name__.lower()
    if any(k in exc_type_name for k in _NON_RETRYABLE_EXCEPTION_NAMES):
        return False
    if any(k in exc_type_name for k in _RETRYABLE_EXCEPTION_NAMES):
        return True
    # 对 APIError 类：检查 status_code 属性
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if isinstance(status, int):
        if status == 429 or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    # ... 其余逻辑不变
```

### 回归验证

- [ ] `_run_registry_workflow` 抛 `WorkflowNeedsFallback` 时正确 fallback 到 Priority 2
- [ ] `_run_registry_workflow` 抛 `TypeError` 时不再被吞掉，传播给调用者
- [ ] OpenAI `BadRequestError`（400）不再被重试
- [ ] OpenAI `RateLimitError`（429）仍然被重试

---

## P1-1: Event/Reporter 替代 print

### 现状分析

`smart_agent.py` 中有 **76 处 `print()` 调用**，分布在 `__init__`（16 处）、`run_async`（约 30 处）和各工作流方法中。这导致：

1. **测试无法静默运行**：CI 日志被大量 emoji 输出污染
2. **无法集成到非终端环境**：Jupyter notebook、Web UI、API 服务无法自定义输出
3. **无结构化日志**：无法机器解析输出

**Codex 对应方案**: `deny(clippy::print_stdout)`，所有输出走结构化事件通道，由前端（TUI/CLI）决定如何渲染。

### 目标设计

```python
# 新文件: omicverse/utils/agent_reporter.py

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol

class EventLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class AgentEvent:
    """结构化 Agent 事件"""
    level: EventLevel
    message: str
    category: str = ""     # "init", "execution", "reflection", "result"
    data: dict = None      # 附加结构化数据

class Reporter(Protocol):
    """输出接口（Codex 事件通道的 Python 等价物）"""
    def emit(self, event: AgentEvent) -> None: ...

class PrintReporter:
    """默认实现：保持当前行为（print 到 stdout）"""
    def emit(self, event: AgentEvent) -> None:
        icon = {"info": "ℹ", "warning": "⚠️", "error": "❌",
                "success": "✅", "debug": "🔍"}.get(event.level.value, "")
        print(f"{icon} {event.message}")

class SilentReporter:
    """静默模式：只记录到 logger（测试/批处理用）"""
    def emit(self, event: AgentEvent) -> None:
        log_fn = getattr(logger, event.level.value, logger.info)
        log_fn(event.message)

class CallbackReporter:
    """回调模式：转发给用户自定义函数（Web UI/Jupyter 集成用）"""
    def __init__(self, callback: Callable[[AgentEvent], None]):
        self._callback = callback

    def emit(self, event: AgentEvent) -> None:
        self._callback(event)
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| 新增 `omicverse/utils/agent_reporter.py` | 新建 Reporter 模块 |
| `omicverse/utils/smart_agent.py` | 所有 `print()` → `self._reporter.emit()` |
| `omicverse/utils/agent_config.py` | `AgentConfig` 增加 `verbose` / `reporter` 字段 |

### 详细步骤

#### 步骤 1: 在 `AgentConfig` 中加入 Reporter 配置

```python
@dataclass
class AgentConfig:
    # ...
    verbose: bool = True
    reporter: Optional[Reporter] = None  # None = 根据 verbose 自动选择
```

#### 步骤 2: `OmicVerseAgent.__init__` 中初始化 Reporter

```python
def __init__(self, config=None, **kwargs):
    # ...
    if config.reporter:
        self._reporter = config.reporter
    elif config.verbose:
        self._reporter = PrintReporter()
    else:
        self._reporter = SilentReporter()
```

#### 步骤 3: 逐步替换 print（示例）

```python
# --- Before (smart_agent.py:277) ---
print(f" Initializing OmicVerse Smart Agent (internal backend)...")

# --- After ---
self._reporter.emit(AgentEvent(
    level=EventLevel.INFO,
    message="Initializing OmicVerse Smart Agent (internal backend)",
    category="init",
))

# --- Before (smart_agent.py:2948) ---
print(f"✅ SUCCESS - Priority 1 completed successfully!")

# --- After ---
self._reporter.emit(AgentEvent(
    level=EventLevel.SUCCESS,
    message="Priority 1 completed successfully",
    category="execution",
    data={"priority": 1, "strategy": "registry"},
))
```

### 回归验证

- [ ] 默认行为（`verbose=True`）输出与当前一致
- [ ] `ov.Agent(verbose=False)` 完全静默
- [ ] `CallbackReporter` 能接收所有事件

---

## P1-2: 执行方法拆分

### 现状分析

`_execute_generated_code`（`smart_agent.py:2318-2469`）在 150 行内混合了 **10+ 个职责**：

| 行号范围 | 职责 |
|----------|------|
| 2331-2354 | Notebook 执行分发 + 失败降级 |
| 2356-2360 | Legacy compile/exec |
| 2362-2365 | HVG 列重命名 |
| 2367-2370 | initial_cells/genes 存储 |
| 2372-2381 | raw/scaled layer 初始化 |
| 2383-2390 | NaN 填充 |
| 2392-2420 | QC 别名生成 (total_counts, n_counts, n_genes_by_counts, pct_counts_mito) |
| 2421-2426 | mito/mt var 列 |
| 2428-2437 | pandas.cut monkey-patch |
| 2441-2445 | 输出目录创建 |
| 2460 | exec() 执行 |
| 2463 | doublet 归一化 |
| 2466-2467 | context 指令处理 |

**Codex 对应方案**: 执行策略 (Sandbox / In-process) 与数据准备完全分离，用 `ToolOrchestrator` 做三阶段流水线。

### 目标设计

拆分为三个独立组件：

```
_execute_generated_code()
    ↓ 拆分为
1. AnnDataPreprocessor.prepare(adata)     → 数据标准化
2. CodeExecutor.execute(code, adata)      → 执行策略选择
3. AnnDataPostprocessor.finalize(adata)   → 后处理 (doublet, context)
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `omicverse/utils/smart_agent.py:2318-2469` | 拆分方法 |
| 可选新增 `omicverse/utils/adata_normalizer.py` | 提取 adata 标准化逻辑 |

### 详细步骤

#### 步骤 1: 提取 AnnData 预处理（`smart_agent.py:2362-2445`）

```python
# 新增常量（取代硬编码字符串）
_QC_COLUMN_ALIASES = {
    "total_counts": lambda adata: _compute_total_counts(adata),
    "n_counts": lambda adata: adata.obs.get("total_counts", 0),
    "n_genes_by_counts": lambda adata: _compute_n_genes(adata),
}
_MITO_ALIASES = ("pct_counts_mito", "pct_counts_mt")

def _prepare_adata_for_execution(self, adata: Any) -> None:
    """在代码执行前标准化 AnnData 结构。

    从 _execute_generated_code 提取出来的纯数据操作，
    不涉及代码编译或执行逻辑。
    """
    self._normalize_hvg_columns(adata)
    self._store_initial_sizes(adata)
    self._ensure_raw_and_scaled(adata)
    self._fill_missing_numeric(adata)
    self._ensure_qc_aliases(adata)
    self._normalize_mito_columns(adata)

# 每个子方法 10-15 行，职责单一，可独立测试
def _normalize_hvg_columns(self, adata):
    if hasattr(adata, "var") and adata.var is not None:
        if "highly_variable" not in adata.var.columns and "highly_variable_features" in adata.var.columns:
            adata.var["highly_variable"] = adata.var["highly_variable_features"]

def _store_initial_sizes(self, adata):
    if hasattr(adata, "uns"):
        adata.uns.setdefault("initial_cells", adata.n_obs if hasattr(adata, "n_obs") else None)
        adata.uns.setdefault("initial_genes", adata.n_vars if hasattr(adata, "n_vars") else None)

# ... 其余子方法同理
```

#### 步骤 2: 删除 pandas.cut monkey-patch

```python
# --- Before (smart_agent.py:2428-2437): 全局 monkey-patch ---
if not getattr(_pd.cut, "_ov_wrapped", False):
    _orig_cut = _pd.cut
    def _safe_cut(*args, **kwargs):
        kwargs.setdefault("duplicates", "drop")
        return _orig_cut(*args, **kwargs)
    _pd.cut = _safe_cut

# --- After: 在 ProactiveCodeTransformer 中处理 ---
# 在 transform() 中加一条规则：
# 把生成代码里的 pd.cut(...) 自动加上 duplicates="drop" 参数
# 这样不会污染全局 pandas
```

#### 步骤 3: 简化 `_execute_generated_code`

```python
def _execute_generated_code(self, code: str, adata: Any) -> Any:
    # 1. 预处理
    self._prepare_adata_for_execution(adata)

    # 2. 执行
    if self.use_notebook_execution and self._notebook_executor is not None:
        result_adata = self._execute_via_notebook(code, adata)
    else:
        result_adata = self._execute_in_process(code, adata)

    # 3. 后处理
    self._normalize_doublet_obs(result_adata)
    if self.enable_filesystem_context and self._filesystem_context:
        self._process_context_directives(code, {})

    return result_adata
```

#### 步骤 4: Notebook 降级不再静默（衔接 P2-3）

```python
def _execute_via_notebook(self, code, adata):
    try:
        return self._notebook_executor.execute(code, adata)
    except Exception as e:
        # 不再静默 fallthrough，抛出明确异常
        raise SandboxDeniedError(
            f"Notebook execution failed: {e}. "
            "Set use_notebook_execution=False to use in-process execution."
        ) from e
```

### 回归验证

- [ ] 提取后的 `_prepare_adata_for_execution` 对相同输入产生相同 adata 变换
- [ ] 删除 pandas.cut monkey-patch 后，生成代码中含 `pd.cut()` 仍能正确执行
- [ ] Notebook 执行失败不再静默降级

---

## P1-3: retry 去重 + GPT-5 响应提取器去重

### 现状分析

**问题 1: retry 参数展开重复 12 次**

`agent_backend.py` 中 `_retry_with_backoff(func, max_attempts=self.config.max_retry_attempts, base_delay=self.config.retry_base_delay, factor=self.config.retry_backoff_factor, jitter=self.config.retry_jitter)` 这 5 行在以下 12 个位置出现：

- 行 527-533, 602-608, 758-764, 972-978, 1082-1088, 1152-1158
- 行 1213-1219, 1321-1327, 1413-1419, 1537-1543, 1604-1610, 1678-1684

**问题 2: GPT-5 Responses API 文本提取逻辑重复 4 处**

~40 行的嵌套 if/elif 文本提取链出现在：
- `_chat_via_openai_responses`（SDK 路径）
- `_chat_via_openai_responses_http`（HTTP 路径）
- `_stream_openai_responses`（流式路径）
- HTTP 400 重试路径

**问题 3: Gemini 流式伪流式**（`agent_backend.py:1612-1617`）

```python
response_list = list(response)  # 收集全部再逐个 yield，失去流式意义
```

### 目标设计

#### retry 去重

```python
# --- Before: 5 行 × 12 处 ---
return _retry_with_backoff(
    _make_sdk_call,
    max_attempts=self.config.max_retry_attempts,
    base_delay=self.config.retry_base_delay,
    factor=self.config.retry_backoff_factor,
    jitter=self.config.retry_jitter
)

# --- After: 1 行 × 12 处 ---
return self._retry(_make_sdk_call)
```

实现：

```python
def _retry(self, func: Callable[..., T], *args, **kwargs) -> T:
    """使用当前 config 中的重试参数执行 func"""
    return _retry_with_backoff(
        func, *args,
        max_attempts=self.config.max_retry_attempts,
        base_delay=self.config.retry_base_delay,
        factor=self.config.retry_backoff_factor,
        jitter=self.config.retry_jitter,
        **kwargs,
    )
```

#### GPT-5 响应提取器去重

```python
def _extract_responses_api_text(self, response_or_event) -> str:
    """从 Responses API 的各种响应格式中提取文本。

    统一处理 SDK 对象、HTTP dict、流式事件三种来源。
    """
    # 1. SDK 对象 (有 output 属性)
    if hasattr(response_or_event, 'output'):
        for item in response_or_event.output:
            if hasattr(item, 'content'):
                for block in item.content:
                    if hasattr(block, 'text'):
                        return block.text
    # 2. HTTP dict
    if isinstance(response_or_event, dict):
        for item in response_or_event.get('output', []):
            for block in item.get('content', []):
                if block.get('type') == 'output_text':
                    return block.get('text', '')
    # 3. 流式 delta
    if hasattr(response_or_event, 'delta'):
        return getattr(response_or_event.delta, 'text', '') or ''
    return ''
```

#### Gemini 真流式

```python
# --- Before ---
response_list = list(response)
for chunk in response_list:
    if hasattr(chunk, 'text') and chunk.text:
        yield chunk.text

# --- After ---
for chunk in response:  # 直接迭代，不收集
    if hasattr(chunk, 'text') and chunk.text:
        yield chunk.text
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `omicverse/utils/agent_backend.py` | 全局：12 处 retry 缩减、4 处提取器合并、Gemini 流式修复 |

### 回归验证

- [ ] 所有 12 个 retry 调用点行为不变
- [ ] GPT-5 SDK / HTTP / 流式三条路径返回相同文本
- [ ] Gemini 流式用户可以看到逐 chunk 输出（不再等全部完成）

---

## P2-1: 上下文窗口压缩

### 现状分析

当前 `OmicVerseAgent` 每次调用都是无状态的（`run_async` 不维护对话历史），但 `_run_skills_workflow` 和 `_run_registry_workflow` 的 system prompt 本身非常大（包含完整函数注册表 + skill 指导），已接近部分模型的上下文限制。

此外，`FilesystemContextManager` 虽提供了上下文持久化，但没有自动压缩机制 —— 随着 context 文件增长，注入 system prompt 的内容无限膨胀。

**Codex 对应方案**: `compact.rs` 在上下文超限时自动调用 LLM 做摘要压缩。

### 目标设计

新增 `ContextCompactor` 模块：

```
系统 prompt 组装
    ↓
检查 token 估算 > 模型上下文限制的 80%?
    ↓ 是
调用 LLM 压缩: 保留关键函数签名 + 压缩描述文本
    ↓
用压缩后的 prompt 替换原始 prompt
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| 新增 `omicverse/utils/context_compactor.py` | 新建模块 |
| `omicverse/utils/smart_agent.py` | `_setup_agent` / `_run_*_workflow` 中集成 |

### 详细步骤

#### 步骤 1: Token 估算工具

```python
# context_compactor.py

import re

# 粗略估算：1 token ≈ 4 字符（英文）/ 2 字符（中文）
def estimate_tokens(text: str) -> int:
    """快速 token 估算（无需 tiktoken 依赖）"""
    # 混合语言估算
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
    ascii_chars = len(text) - cjk_chars
    return cjk_chars // 2 + ascii_chars // 4

# 模型上下文窗口（从 ProviderInfo 获取或硬编码常用值）
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5": 256_000,
    "gemini-2.5-flash": 1_000_000,
    "anthropic/claude-sonnet-4-20250514": 200_000,
    "deepseek/deepseek-chat": 64_000,
    # ...
}

def get_context_window(model: str) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model, 32_000)
```

#### 步骤 2: Compactor 实现

```python
class ContextCompactor:
    """上下文压缩器（借鉴 Codex compact.rs）"""

    COMPACT_THRESHOLD = 0.75  # 超过上下文窗口的 75% 触发压缩
    MAX_COMPACT_INPUT = 20_000  # 压缩请求本身的 token 限制

    def __init__(self, llm_backend, model: str):
        self._llm = llm_backend
        self._model = model
        self._context_window = get_context_window(model)

    def needs_compaction(self, system_prompt: str, user_prompt: str) -> bool:
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        return total > self._context_window * self.COMPACT_THRESHOLD

    async def compact(self, system_prompt: str) -> str:
        """压缩 system prompt，保留关键信息"""
        prompt = (
            "Summarize the following OmicVerse function registry and skill instructions. "
            "Keep: function names, parameter signatures, prerequisite chains. "
            "Remove: verbose descriptions, examples, related functions.\n\n"
            f"{system_prompt[:self.MAX_COMPACT_INPUT * 4]}"  # 截断到 ~MAX tokens
        )
        compacted = await self._llm.run(prompt)
        return compacted
```

#### 步骤 3: 集成到工作流

```python
# 在 _run_registry_workflow / _run_skills_workflow 中
if self._compactor and self._compactor.needs_compaction(system_prompt, user_prompt):
    system_prompt = await self._compactor.compact(system_prompt)
    self._reporter.emit(AgentEvent(
        level=EventLevel.INFO,
        message="System prompt compressed to fit context window",
        category="execution",
    ))
```

### 回归验证

- [ ] 小 prompt 不触发压缩（行为不变）
- [ ] 大 prompt 压缩后仍包含所有函数名和签名
- [ ] 压缩后的 prompt 不超过上下文窗口限制

---

## P2-2: 会话持久化与恢复

### 现状分析

当前 `OmicVerseAgent` 的每次 `run()` 调用是无状态的，没有跨调用的历史记录。`FilesystemContextManager` 提供了文件级别的 context 存储，但没有对话历史（之前的 request → code → result 记录）。

**Codex 对应方案**: `message_history.rs` 用 append-only JSONL 持久化，支持 `fork_thread()` / `resume_thread()`。

### 目标设计

```python
# 新文件: omicverse/utils/session_history.py

@dataclass
class HistoryEntry:
    session_id: str
    timestamp: float
    request: str
    generated_code: str
    result_summary: str  # 而非完整 adata
    usage: Optional[dict]
    priority_used: int
    success: bool

class SessionHistory:
    """Append-only JSONL 会话历史（借鉴 Codex message_history.rs）"""

    def __init__(self, path: Path = None):
        self._path = path or Path.home() / ".ovagent" / "history.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: HistoryEntry) -> None:
        """原子写入一条记录"""
        line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def get_session(self, session_id: str) -> list[HistoryEntry]:
        """获取某个 session 的完整历史"""
        # ...

    def get_recent(self, n: int = 10) -> list[HistoryEntry]:
        """获取最近 n 条记录"""
        # ...

    def build_context_for_llm(self, session_id: str, max_entries: int = 5) -> str:
        """为 LLM 构建历史上下文摘要"""
        entries = self.get_session(session_id)[-max_entries:]
        lines = []
        for e in entries:
            lines.append(f"[Previous request]: {e.request}")
            lines.append(f"[Result]: {'Success' if e.success else 'Failed'} - {e.result_summary}")
        return "\n".join(lines)
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| 新增 `omicverse/utils/session_history.py` | 新建模块 |
| `omicverse/utils/smart_agent.py` | `run_async` 末尾追加 history 记录 |
| `omicverse/utils/agent_config.py` | 增加 `history_enabled` / `history_path` 配置 |

### 集成方式

```python
# 在 run_async 成功返回前
if self._history:
    self._history.append(HistoryEntry(
        session_id=self._session_id,
        timestamp=time.time(),
        request=request,
        generated_code=generated_code,
        result_summary=f"{result.shape[0]} cells x {result.shape[1]} genes",
        usage=self.last_usage.__dict__ if self.last_usage else None,
        priority_used=priority_used,
        success=True,
    ))
```

### 回归验证

- [ ] `history_enabled=False` 时无文件 I/O（默认行为）
- [ ] JSONL 文件格式正确，可被 `jq` 解析
- [ ] `build_context_for_llm` 输出可注入 system prompt

---

## P2-3: Notebook 降级需显式确认

### 现状分析

`smart_agent.py:2351-2354`：

```python
except Exception as e:
    print(f"⚠️  Session execution failed: {e}")
    print(f"   Falling back to in-process execution...")
    # Fall through to legacy execution  ← 静默降级
```

用户选择 Notebook 执行是为了隔离和安全，静默降级到裸 `exec()` 违背了这个意图。类似地，`__init__`（`smart_agent.py:385-389`）中 Notebook 初始化失败也会静默降级。

**Codex 对应方案**: 沙箱降级需要经过 Approval 审批，不会自动发生。

### 目标设计

两种策略（通过配置选择）：

```python
class SandboxFallbackPolicy(Enum):
    RAISE = "raise"              # 不降级，直接抛异常
    WARN_AND_FALLBACK = "warn"   # 发出警告事件后降级（当前行为但有结构化通知）
    SILENT = "silent"            # 静默降级（不推荐，保留给极端向后兼容场景）
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `omicverse/utils/smart_agent.py:2351-2354` | 执行时降级 |
| `omicverse/utils/smart_agent.py:385-389` | 初始化时降级 |
| `omicverse/utils/agent_config.py` | 增加 `sandbox_fallback_policy` |

### 详细步骤

```python
# --- Before (smart_agent.py:2351-2354) ---
except Exception as e:
    print(f"⚠️  Session execution failed: {e}")
    print(f"   Falling back to in-process execution...")
    # Fall through

# --- After ---
except Exception as e:
    policy = self.config.execution.sandbox_fallback_policy
    if policy == SandboxFallbackPolicy.RAISE:
        raise SandboxDeniedError(
            f"Notebook execution failed: {e}. "
            "Set sandbox_fallback_policy='warn' to allow in-process fallback."
        ) from e
    elif policy == SandboxFallbackPolicy.WARN_AND_FALLBACK:
        self._reporter.emit(AgentEvent(
            level=EventLevel.WARNING,
            message=f"Notebook execution failed ({e}), falling back to in-process execution",
            category="execution",
            data={"error": str(e), "fallback": "in_process"},
        ))
        # Continue to in-process execution below
    # SILENT: no notification, just fall through
```

### 默认值选择

- 新用户：`WARN_AND_FALLBACK`（保持当前行为但有结构化告警）
- 生产环境推荐：`RAISE`（不允许静默降级）

### 回归验证

- [ ] `WARN_AND_FALLBACK` 行为与当前一致（但通过 Reporter 输出）
- [ ] `RAISE` 在 Notebook 失败时抛 `SandboxDeniedError`
- [ ] `__init__` 中的初始化降级也遵循同一策略

---

## P3-1: 共享 ThreadPoolExecutor

### 现状分析

`agent_backend.py:1264-1267`：

```python
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(_run_stream)
```

每次 `_run_generator_in_thread` 调用都创建并销毁一个 `ThreadPoolExecutor`，在高频流式调用下造成线程创建/销毁开销。

### 目标设计

```python
# --- Before: 每次新建 ---
with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(_run_stream)

# --- After: 类级别共享 ---
import atexit
import concurrent.futures

# 模块级单例（懒初始化）
_SHARED_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None

def _get_shared_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None or _SHARED_EXECUTOR._shutdown:
        _SHARED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="ovagent-stream",
        )
        atexit.register(_SHARED_EXECUTOR.shutdown, wait=False)
    return _SHARED_EXECUTOR
```

在 `_run_generator_in_thread` 中使用：

```python
async def _run_generator_in_thread(self, generator_func):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    exception_holder = []

    def _run_stream():
        try:
            for item in generator_func():
                asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        except Exception as exc:
            exception_holder.append(exc)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    executor = _get_shared_executor()
    executor.submit(_run_stream)  # 不再 with ... as，不销毁

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk

    if exception_holder:
        raise exception_holder[0]
```

### 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `omicverse/utils/agent_backend.py:1264-1278` | 替换为共享 executor |

### 回归验证

- [ ] 连续多次流式调用不再创建新线程池
- [ ] 进程退出时 executor 正常 shutdown（`atexit`）
- [ ] 异常仍然正确传播

---

## 附录: 额外清理项

### 删除死代码

| 位置 | 内容 |
|------|------|
| `smart_agent.py:3010-3060` | `run_async_LEGACY` 方法（docstring 明确说"不再使用"） |
| `smart_agent.py:2874-2893` | `run_async` docstring 中的性能声明（"60-70% faster"、"~2-3 seconds"） |

### 删除 `usage_breakdown` 重复重置

```python
# 出现在多处的相同字典字面量
self.last_usage_breakdown = {
    'generation': None,
    'reflection': [],
    'review': [],
    'total': None
}
# → 提取为
def _reset_usage_tracking(self):
    self.last_usage = None
    self.last_usage_breakdown = UsageBreakdown()
```

### `reasoning={"effort": "high"}` 可配置化

```python
# --- Before: 硬编码在 4 处 ---
reasoning={"effort": "high"}

# --- After: 从 config 读取 ---
reasoning={"effort": self.config.llm.reasoning_effort}

# 在 LLMConfig 中添加
@dataclass
class LLMConfig:
    # ...
    reasoning_effort: str = "high"  # "low" | "medium" | "high"
```

---

## 实施顺序建议

```
Phase 1 (P0): 基础架构
  ├─ P0-1: Provider 注册表   ← 最大去重收益
  ├─ P0-2: Config 数据类     ← 为后续改动提供配置支点
  └─ P0-3: 异常处理层级      ← 消除静默 bug 吞掉

Phase 2 (P1): 质量改进
  ├─ P1-1: Event/Reporter    ← 依赖 P0-2 的 verbose 配置
  ├─ P1-2: 执行方法拆分      ← 独立，可并行推进
  └─ P1-3: retry + 提取器去重 ← 依赖 P0-1 的 Provider 注册表

Phase 3 (P2): 高级特性
  ├─ P2-1: 上下文压缩        ← 依赖 P1-3 的共享 LLM 调用
  ├─ P2-2: 会话持久化        ← 独立
  └─ P2-3: Notebook 降级策略  ← 依赖 P0-2 / P0-3

Phase 4 (P3): 优化
  └─ P3-1: 共享线程池         ← 独立，低风险
```
