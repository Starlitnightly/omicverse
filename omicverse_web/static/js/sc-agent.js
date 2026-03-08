/**
 * OmicVerse Single Cell Analysis — AI Agent Chat & Configuration
 *
 * Phase 3: session management, server-side cancel, session indicator,
 * streaming SSE consumer, inline tool/code/result cards,
 * stop button, new-chat button, reconnection via turn buffer.
 */

Object.assign(SingleCellAnalysis.prototype, {

    // =====================================================================
    // Agent state
    // =====================================================================

    /** @type {'idle'|'streaming'|'reconnecting'} */
    _agentState: 'idle',

    /** @type {AbortController|null} */
    _agentAbort: null,

    /** Current turn ID (set by the first status event) */
    _agentTurnId: null,

    /** Current session ID (persists across turns) */
    _agentSessionId: null,

    /** Harness capability handshake payload */
    _agentHarnessCaps: null,

    /** Latest trace id seen in the current session */
    _agentLastTraceId: null,

    /** Pending approvals for the current session */
    _agentPendingApprovals: [],

    /** Pending questions for the current session */
    _agentPendingQuestions: [],

    /** Runtime task list for the current session */
    _agentTasks: [],

    /** Latest runtime state snapshot */
    _agentRuntimeState: null,

    /** Latest loaded trace payload */
    _agentTracePayload: null,

    /** DOM element for the pending assistant bubble */
    _agentPendingBubble: null,

    /** Accumulated LLM text for the current turn */
    _agentLlmText: '',

    // =====================================================================
    // Setup
    // =====================================================================

    setupAgentConfig() {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        this.loadAgentConfig();
        Object.values(fields).forEach(field => {
            field.addEventListener('change', () => this.saveAgentConfig(true));
        });
    },

    setupAgentChat() {
        const input = document.getElementById('agent-input');
        if (!input) return;
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.sendAgentMessage();
            }
        });
        // Initialize session
        this._ensureAgentSession();
        this._initializeAgentHarness();
    },

    // =====================================================================
    // Session management
    // =====================================================================

    /** Generate a random session ID. */
    _generateSessionId() {
        return 'ses_' + Math.random().toString(36).slice(2, 14);
    },

    /** Ensure we have an active session ID. */
    _ensureAgentSession() {
        if (!this._agentSessionId) {
            // Try to restore from sessionStorage (survives page refresh)
            this._agentSessionId = sessionStorage.getItem('omicverse.agentSessionId');
            if (!this._agentSessionId) {
                this._agentSessionId = this._generateSessionId();
                sessionStorage.setItem('omicverse.agentSessionId', this._agentSessionId);
            }
        }
        this._updateSessionIndicator();
        return this._agentSessionId;
    },

    /** Update the session indicator in the UI. */
    _updateSessionIndicator() {
        const indicator = document.getElementById('agent-session-indicator');
        if (indicator && this._agentSessionId) {
            indicator.textContent = this._agentSessionId.slice(0, 12);
            indicator.title = this.t('agent.sessionId') + ': ' + this._agentSessionId;
        }
        this._renderAgentHarnessMeta();
    },

    async _initializeAgentHarness() {
        try {
            const resp = await fetch('/api/agent/harness/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Agent-Session-Id': this._agentSessionId || '',
                },
                body: JSON.stringify({ session_id: this._agentSessionId || '' }),
            });
            if (!resp.ok) return;
            this._applyAgentHarnessSnapshot(await resp.json());
        } catch (_) {
            this._agentHarnessCaps = null;
        }
        await this.refreshAgentHarnessPanel();
    },

    _applyAgentHarnessSnapshot(payload) {
        const session = payload && payload.session ? payload.session : null;
        const runtime = payload && payload.runtime ? payload.runtime : null;
        this._agentHarnessCaps = payload && payload.capabilities ? payload.capabilities : null;

        if (session && session.last_trace_id) {
            this._agentLastTraceId = session.last_trace_id;
        }
        if (session && session.active_turn_id) {
            this._agentTurnId = session.active_turn_id;
        }

        this._agentRuntimeState = runtime || {
            loaded_tools: [],
            plan_mode: 'off',
            worktree: {},
            worktree_label: '',
            tasks: [],
            pending_questions: [],
            pending_approvals: [],
        };
        this._agentTasks = Array.isArray(this._agentRuntimeState.tasks) ? this._agentRuntimeState.tasks : [];
        this._agentPendingQuestions = Array.isArray(this._agentRuntimeState.pending_questions)
            ? this._agentRuntimeState.pending_questions
            : [];
        if (Array.isArray(this._agentRuntimeState.pending_approvals)) {
            this._agentPendingApprovals = this._agentRuntimeState.pending_approvals;
        }
    },

    _renderAgentHarnessMeta() {
        const turnEl = document.getElementById('agent-turn-indicator');
        const traceEl = document.getElementById('agent-trace-indicator');
        const approvalEl = document.getElementById('agent-approval-indicator');
        const questionEl = document.getElementById('agent-question-indicator');
        const taskEl = document.getElementById('agent-task-indicator');
        const planEl = document.getElementById('agent-plan-indicator');
        const worktreeEl = document.getElementById('agent-worktree-indicator');
        const runtime = this._agentRuntimeState || {};
        if (turnEl) {
            turnEl.textContent = this._agentTurnId ? `turn:${String(this._agentTurnId).slice(0, 12)}` : 'turn:idle';
            turnEl.title = this._agentTurnId || 'No active turn';
        }
        if (traceEl) {
            traceEl.textContent = this._agentLastTraceId ? `trace:${String(this._agentLastTraceId).slice(0, 12)}` : 'trace:none';
            traceEl.title = this._agentLastTraceId || 'No trace loaded';
        }
        if (approvalEl) {
            const count = Array.isArray(this._agentPendingApprovals) ? this._agentPendingApprovals.length : 0;
            approvalEl.textContent = `approvals:${count}`;
            approvalEl.title = `${count} pending approval${count === 1 ? '' : 's'}`;
        }
        if (questionEl) {
            const count = Array.isArray(this._agentPendingQuestions) ? this._agentPendingQuestions.length : 0;
            questionEl.textContent = `questions:${count}`;
            questionEl.title = `${count} pending question${count === 1 ? '' : 's'}`;
        }
        if (taskEl) {
            const count = Array.isArray(this._agentTasks) ? this._agentTasks.length : 0;
            taskEl.textContent = `tasks:${count}`;
            taskEl.title = `${count} recent task${count === 1 ? '' : 's'}`;
        }
        if (planEl) {
            const planMode = runtime.plan_mode || 'off';
            planEl.textContent = `plan:${planMode}`;
            planEl.title = `Plan mode: ${planMode}`;
        }
        if (worktreeEl) {
            const label = runtime.worktree_label
                || (runtime.worktree && (runtime.worktree.label || runtime.worktree.path || runtime.worktree.name))
                || '';
            worktreeEl.textContent = label ? `worktree:${String(label).slice(0, 12)}` : 'worktree:none';
            worktreeEl.title = label || 'No worktree selected';
        }
    },

    _renderHarnessCapabilities() {
        const wrap = document.getElementById('agent-harness-capabilities');
        if (!wrap) return;
        wrap.innerHTML = '';
        const caps = this._agentHarnessCaps;
        if (!caps || !caps.supports) {
            wrap.innerHTML = '<span class="agent-harness-empty">Harness capabilities unavailable</span>';
            return;
        }
        Object.entries(caps.supports).forEach(([key, enabled]) => {
            const badge = document.createElement('span');
            badge.className = 'agent-harness-badge';
            badge.textContent = `${key}:${enabled ? 'on' : 'off'}`;
            wrap.appendChild(badge);
        });
    },

    _renderRuntimeState() {
        const wrap = document.getElementById('agent-runtime-state');
        if (!wrap) return;
        wrap.innerHTML = '';
        const runtime = this._agentRuntimeState || {};
        const loadedTools = Array.isArray(runtime.loaded_tools) ? runtime.loaded_tools : [];
        const hasState = loadedTools.length > 0
            || runtime.active_tool_name
            || runtime.last_tool_name
            || runtime.plan_mode
            || runtime.worktree_label
            || (runtime.worktree && Object.keys(runtime.worktree).length > 0);

        if (!hasState) {
            wrap.innerHTML = '<div class="agent-harness-empty">No runtime state yet</div>';
            return;
        }

        const summary = document.createElement('div');
        summary.className = 'agent-card-detail';
        summary.textContent = `Active: ${runtime.active_tool_name || 'none'} | Last: ${runtime.last_tool_name || 'none'} | Plan: ${runtime.plan_mode || 'off'}`;
        wrap.appendChild(summary);

        const worktree = document.createElement('div');
        worktree.className = 'agent-card-detail';
        worktree.textContent = `Worktree: ${runtime.worktree_label || 'none'}`;
        wrap.appendChild(worktree);

        const toolsWrap = document.createElement('div');
        loadedTools.forEach((tool) => {
            const badge = document.createElement('span');
            badge.className = 'agent-harness-badge';
            badge.textContent = tool;
            toolsWrap.appendChild(badge);
        });
        if (loadedTools.length > 0) {
            wrap.appendChild(toolsWrap);
        }
    },

    _renderApprovalList() {
        const wrap = document.getElementById('agent-approval-list');
        if (!wrap) return;
        wrap.innerHTML = '';
        const approvals = Array.isArray(this._agentPendingApprovals) ? this._agentPendingApprovals : [];
        if (approvals.length === 0) {
            wrap.innerHTML = '<div class="agent-harness-empty">No pending approvals</div>';
            return;
        }
        approvals.forEach((approval) => {
            const card = document.createElement('div');
            card.className = 'agent-harness-approval';

            const title = document.createElement('div');
            title.className = 'agent-harness-approval-title';
            title.textContent = approval.title || 'Approval required';
            card.appendChild(title);

            if (approval.message) {
                const detail = document.createElement('div');
                detail.className = 'agent-card-detail';
                detail.textContent = approval.message;
                card.appendChild(detail);
            }

            const actions = document.createElement('div');
            actions.className = 'agent-harness-approval-actions';

            const allowBtn = document.createElement('button');
            allowBtn.type = 'button';
            allowBtn.className = 'btn btn-sm btn-success';
            allowBtn.textContent = 'Allow';
            allowBtn.onclick = () => this._respondAgentApproval(approval.approval_id, 'approve', card);

            const denyBtn = document.createElement('button');
            denyBtn.type = 'button';
            denyBtn.className = 'btn btn-sm btn-outline-danger';
            denyBtn.textContent = 'Deny';
            denyBtn.onclick = () => this._respondAgentApproval(approval.approval_id, 'deny', card);

            actions.appendChild(allowBtn);
            actions.appendChild(denyBtn);
            card.appendChild(actions);
            wrap.appendChild(card);
        });
    },

    _renderQuestionList() {
        const wrap = document.getElementById('agent-question-list');
        if (!wrap) return;
        wrap.innerHTML = '';
        const questions = Array.isArray(this._agentPendingQuestions) ? this._agentPendingQuestions : [];
        if (questions.length === 0) {
            wrap.innerHTML = '<div class="agent-harness-empty">No pending questions</div>';
            return;
        }

        questions.forEach((question) => {
            const card = document.createElement('div');
            card.className = 'agent-harness-approval';

            const title = document.createElement('div');
            title.className = 'agent-harness-approval-title';
            title.textContent = question.title || 'Question for user';
            card.appendChild(title);

            if (question.message) {
                const detail = document.createElement('div');
                detail.className = 'agent-card-detail';
                detail.textContent = question.message;
                card.appendChild(detail);
            }

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'form-control form-control-sm';
            input.placeholder = question.placeholder || 'Type your answer';
            card.appendChild(input);

            if (Array.isArray(question.options) && question.options.length > 0) {
                const options = document.createElement('div');
                options.className = 'agent-harness-approval-actions';
                question.options.forEach((option) => {
                    const optionBtn = document.createElement('button');
                    optionBtn.type = 'button';
                    optionBtn.className = 'btn btn-sm btn-outline-secondary';
                    optionBtn.textContent = option;
                    optionBtn.onclick = () => this._respondAgentQuestion(question.question_id, option, card, input);
                    options.appendChild(optionBtn);
                });
                card.appendChild(options);
            }

            const actions = document.createElement('div');
            actions.className = 'agent-harness-approval-actions';

            const sendBtn = document.createElement('button');
            sendBtn.type = 'button';
            sendBtn.className = 'btn btn-sm btn-primary';
            sendBtn.textContent = 'Send answer';
            sendBtn.onclick = () => this._respondAgentQuestion(question.question_id, input.value, card, input);
            actions.appendChild(sendBtn);

            card.appendChild(actions);
            wrap.appendChild(card);
        });
    },

    _renderTaskList() {
        const wrap = document.getElementById('agent-task-list');
        if (!wrap) return;
        wrap.innerHTML = '';
        const tasks = Array.isArray(this._agentTasks) ? this._agentTasks : [];
        if (tasks.length === 0) {
            wrap.innerHTML = '<div class="agent-harness-empty">No tasks yet</div>';
            return;
        }

        tasks.forEach((task) => {
            const card = document.createElement('div');
            card.className = 'agent-harness-approval';

            const title = document.createElement('div');
            title.className = 'agent-harness-approval-title';
            title.textContent = `${task.title || task.item_type || 'task'} [${task.status || 'pending'}]`;
            card.appendChild(title);

            const detail = document.createElement('div');
            detail.className = 'agent-card-detail';
            detail.textContent = task.output_summary || task.summary || task.error || '';
            card.appendChild(detail);

            wrap.appendChild(card);
        });
    },

    _renderTracePreview() {
        const wrap = document.getElementById('agent-trace-preview');
        if (!wrap) return;
        wrap.innerHTML = '';
        const trace = this._agentTracePayload;
        if (!trace) {
            wrap.innerHTML = '<div class="agent-harness-empty">No trace loaded</div>';
            return;
        }
        const tools = (trace.steps || [])
            .map((step) => step.name || step.tool_name)
            .filter(Boolean);
        const summary = {
            trace_id: trace.trace_id || this._agentLastTraceId || '',
            status: trace.status || '',
            request: trace.request || '',
            result_summary: trace.result_summary || '',
            tool_names: tools,
            event_count: (trace.events || []).length,
            step_count: (trace.steps || []).length,
            artifact_count: (trace.artifacts || []).length,
        };
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(summary, null, 2);
        wrap.appendChild(pre);
    },

    async refreshAgentApprovals() {
        if (!this._agentSessionId) return [];
        try {
            const resp = await fetch(`/api/agent/session/${this._agentSessionId}/approvals`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            this._agentPendingApprovals = payload.approvals || [];
            if (!this._agentRuntimeState) this._agentRuntimeState = { loaded_tools: [], plan_mode: 'off', worktree: {} };
            this._agentRuntimeState.pending_approvals = this._agentPendingApprovals;
        } catch (_) {
            this._agentPendingApprovals = [];
        }
        this._renderAgentHarnessMeta();
        this._renderApprovalList();
        return this._agentPendingApprovals;
    },

    async refreshAgentQuestions() {
        if (!this._agentSessionId) return [];
        try {
            const resp = await fetch(`/api/agent/session/${this._agentSessionId}/questions`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            this._agentPendingQuestions = payload.questions || [];
            if (!this._agentRuntimeState) this._agentRuntimeState = { loaded_tools: [], plan_mode: 'off', worktree: {} };
            this._agentRuntimeState.pending_questions = this._agentPendingQuestions;
        } catch (_) {
            this._agentPendingQuestions = [];
        }
        this._renderAgentHarnessMeta();
        this._renderQuestionList();
        return this._agentPendingQuestions;
    },

    async refreshAgentTasks() {
        if (!this._agentSessionId) return [];
        try {
            const resp = await fetch(`/api/agent/session/${this._agentSessionId}/tasks`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const payload = await resp.json();
            this._agentTasks = payload.tasks || [];
            if (payload.runtime) {
                this._agentRuntimeState = payload.runtime;
            } else if (this._agentRuntimeState) {
                this._agentRuntimeState.tasks = this._agentTasks;
            }
        } catch (_) {
            this._agentTasks = [];
        }
        this._renderAgentHarnessMeta();
        this._renderRuntimeState();
        this._renderTaskList();
        return this._agentTasks;
    },

    _recordLoadedTool(name, stepId) {
        if (!name) return;
        if (!this._agentRuntimeState) {
            this._agentRuntimeState = { loaded_tools: [], plan_mode: 'off', worktree: {} };
        }
        const loaded = Array.isArray(this._agentRuntimeState.loaded_tools) ? this._agentRuntimeState.loaded_tools : [];
        if (!loaded.includes(name)) {
            loaded.push(name);
        }
        this._agentRuntimeState.loaded_tools = loaded;
        this._agentRuntimeState.active_tool_name = name;
        this._agentRuntimeState.last_tool_name = name;
        if (stepId) {
            this._agentRuntimeState.active_step_id = stepId;
        }
    },

    _upsertTask(taskId, patch) {
        const tasks = Array.isArray(this._agentTasks) ? [...this._agentTasks] : [];
        const idx = tasks.findIndex((task) => task.task_id === taskId);
        if (idx >= 0) {
            tasks[idx] = { ...tasks[idx], ...patch };
        } else {
            tasks.unshift({ task_id: taskId, ...patch });
        }
        this._agentTasks = tasks.slice(0, 12);
        if (!this._agentRuntimeState) {
            this._agentRuntimeState = { loaded_tools: [], plan_mode: 'off', worktree: {} };
        }
        this._agentRuntimeState.tasks = this._agentTasks;
    },

    _applyAgentRuntimeEvent(event) {
        const content = event.content || {};
        const taskId = event.step_id || `${event.type}:${content.name || content.item_type || 'task'}`;
        if (!this._agentRuntimeState) {
            this._agentRuntimeState = { loaded_tools: [], plan_mode: 'off', worktree: {} };
        }

        if (Array.isArray(content.loaded_tools)) {
            content.loaded_tools.forEach((tool) => this._recordLoadedTool(tool, event.step_id));
        }
        if (Object.prototype.hasOwnProperty.call(content, 'plan_mode')) {
            this._agentRuntimeState.plan_mode = content.plan_mode || 'off';
        }
        if (Object.prototype.hasOwnProperty.call(content, 'worktree')) {
            this._agentRuntimeState.worktree = content.worktree || {};
            this._agentRuntimeState.worktree_label = (content.worktree && (content.worktree.label || content.worktree.path || content.worktree.name)) || '';
        }

        if (event.type === 'tool_call') {
            this._recordLoadedTool(content.name, event.step_id);
            this._upsertTask(taskId, {
                turn_id: event.turn_id || '',
                step_id: event.step_id || '',
                title: content.name || 'tool_call',
                item_type: 'tool_call',
                status: 'in_progress',
                summary: `${content.name || 'tool'} dispatched`,
            });
        } else if (event.type === 'item_started') {
            this._recordLoadedTool(content.name, event.step_id);
            this._upsertTask(taskId, {
                turn_id: event.turn_id || '',
                step_id: event.step_id || '',
                title: content.name || content.item_type || 'task',
                item_type: content.item_type || 'task',
                status: 'in_progress',
                summary: `${content.name || content.item_type || 'task'} started`,
            });
        } else if (event.type === 'item_completed') {
            this._upsertTask(taskId, {
                turn_id: event.turn_id || '',
                step_id: event.step_id || '',
                title: content.name || content.item_type || 'task',
                item_type: content.item_type || 'task',
                status: content.status || 'completed',
                output_summary: `${content.name || content.item_type || 'task'} ${content.status || 'completed'}`,
            });
            if (this._agentRuntimeState && this._agentRuntimeState.active_step_id === event.step_id) {
                this._agentRuntimeState.active_step_id = '';
                this._agentRuntimeState.active_tool_name = '';
            }
        } else if (event.type === 'question_request') {
            const existing = Array.isArray(this._agentPendingQuestions) ? [...this._agentPendingQuestions] : [];
            const questionId = content.question_id || content.request_id;
            if (questionId && !existing.some((question) => question.question_id === questionId)) {
                existing.push({ ...content, question_id: questionId });
            }
            this._agentPendingQuestions = existing;
        } else if (event.type === 'question_resolved') {
            const questionId = content.question_id || content.request_id;
            this._agentPendingQuestions = (this._agentPendingQuestions || []).filter((question) => question.question_id !== questionId);
        }
    },

    async refreshAgentHarnessPanel() {
        if (this._agentSessionId) {
            try {
                const resp = await fetch('/api/agent/harness/initialize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Agent-Session-Id': this._agentSessionId || '',
                    },
                    body: JSON.stringify({ session_id: this._agentSessionId || '' }),
                });
                if (resp.ok) {
                    this._applyAgentHarnessSnapshot(await resp.json());
                }
            } catch (_) {}
        }
        this._renderHarnessCapabilities();
        this._renderAgentHarnessMeta();
        this._renderRuntimeState();
        this._renderApprovalList();
        this._renderQuestionList();
        this._renderTaskList();
        await this.refreshAgentApprovals();
        await this.refreshAgentQuestions();
        await this.refreshAgentTasks();
        if (this._agentLastTraceId) {
            try {
                this._agentTracePayload = await this.loadAgentTrace(this._agentLastTraceId);
            } catch (_) {
                this._agentTracePayload = null;
            }
        }
        this._renderTracePreview();
    },

    // =====================================================================
    // Config persistence (Phase 4: API key in sessionStorage by default)
    // =====================================================================

    /** Whether the user opted in to persisting the API key in localStorage. */
    _isRememberKeyEnabled() {
        const cb = document.getElementById('agent-remember-key');
        return cb ? cb.checked : false;
    },

    getAgentConfigFields() {
        const fields = {
            apiBase: document.getElementById('agent-api-base'),
            apiKey: document.getElementById('agent-api-key'),
            model: document.getElementById('agent-model'),
            temperature: document.getElementById('agent-temperature'),
            topP: document.getElementById('agent-top-p'),
            maxTokens: document.getElementById('agent-max-tokens'),
            timeout: document.getElementById('agent-timeout'),
            systemPrompt: document.getElementById('agent-system-prompt')
        };
        const hasAll = Object.values(fields).every(Boolean);
        return hasAll ? fields : null;
    },

    loadAgentConfig() {
        const fields = this.getAgentConfigFields();
        if (!fields) return;

        // Load non-secret config from localStorage
        let stored = null;
        try {
            stored = JSON.parse(localStorage.getItem('omicverse.agentConfig') || 'null');
        } catch (e) {
            stored = null;
        }
        if (!stored) {
            fields.apiBase.value = fields.apiBase.value || 'https://api.openai.com/v1';
            fields.model.value = fields.model.value || 'gpt-5';
        } else {
            fields.apiBase.value = stored.apiBase || fields.apiBase.value || 'https://api.openai.com/v1';
            fields.model.value = stored.model || fields.model.value || 'gpt-5';
            fields.temperature.value = stored.temperature ?? fields.temperature.value;
            fields.topP.value = stored.topP ?? fields.topP.value;
            fields.maxTokens.value = stored.maxTokens ?? fields.maxTokens.value;
            fields.timeout.value = stored.timeout ?? fields.timeout.value;
            fields.systemPrompt.value = stored.systemPrompt || '';
        }

        // API key: try localStorage first (user opted in), then sessionStorage
        const rememberedKey = localStorage.getItem('omicverse.agentApiKey');
        const sessionKey = sessionStorage.getItem('omicverse.agentApiKey');
        fields.apiKey.value = rememberedKey || sessionKey || '';

        // Restore "remember key" checkbox state
        const cb = document.getElementById('agent-remember-key');
        if (cb) cb.checked = !!rememberedKey;
    },

    saveAgentConfig(silent = false) {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        const apiKey = fields.apiKey.value.trim();
        const rememberKey = this._isRememberKeyEnabled();

        // Save non-secret config to localStorage (never includes API key)
        const payload = {
            apiBase: fields.apiBase.value.trim(),
            model: fields.model.value.trim(),
            temperature: fields.temperature.value,
            topP: fields.topP.value,
            maxTokens: fields.maxTokens.value,
            timeout: fields.timeout.value,
            systemPrompt: fields.systemPrompt.value.trim()
        };
        localStorage.setItem('omicverse.agentConfig', JSON.stringify(payload));

        // API key: sessionStorage always (available this tab), localStorage
        // only if explicitly opted in via the "remember key" checkbox.
        sessionStorage.setItem('omicverse.agentApiKey', apiKey);
        if (rememberKey) {
            localStorage.setItem('omicverse.agentApiKey', apiKey);
        } else {
            localStorage.removeItem('omicverse.agentApiKey');
        }

        if (!silent) {
            this.showStatus(this.t('status.agentSaved'), false);
            setTimeout(() => this.hideStatus(), 1200);
        }
    },

    resetAgentConfig() {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        localStorage.removeItem('omicverse.agentConfig');
        localStorage.removeItem('omicverse.agentApiKey');
        sessionStorage.removeItem('omicverse.agentApiKey');
        fields.apiBase.value = 'https://api.openai.com/v1';
        fields.apiKey.value = '';
        fields.model.value = 'gpt-5';
        fields.temperature.value = 0.3;
        fields.topP.value = 1;
        fields.maxTokens.value = 2048;
        fields.timeout.value = 60;
        fields.systemPrompt.value = '';
        const cb = document.getElementById('agent-remember-key');
        if (cb) cb.checked = false;
        this.showStatus(this.t('status.agentReset'), false);
        setTimeout(() => this.hideStatus(), 1200);
    },

    getAgentConfig() {
        // Try stored config first
        let stored = null;
        try {
            stored = JSON.parse(localStorage.getItem('omicverse.agentConfig') || 'null');
        } catch (e) {
            stored = null;
        }

        // Resolve API key: localStorage (remembered) > sessionStorage > field
        const apiKey = localStorage.getItem('omicverse.agentApiKey')
            || sessionStorage.getItem('omicverse.agentApiKey')
            || '';

        if (stored) {
            stored.apiKey = apiKey;
            return stored;
        }
        const fields = this.getAgentConfigFields();
        if (!fields) return { apiKey };
        return {
            apiBase: fields.apiBase.value.trim(),
            apiKey: apiKey || fields.apiKey.value.trim(),
            model: fields.model.value.trim(),
            temperature: fields.temperature.value,
            topP: fields.topP.value,
            maxTokens: fields.maxTokens.value,
            timeout: fields.timeout.value,
            systemPrompt: fields.systemPrompt.value.trim()
        };
    },

    // =====================================================================
    // DOM helpers
    // =====================================================================

    appendAgentMessage(text, role = 'assistant', useMarkdown = false) {
        const container = document.getElementById('agent-messages');
        if (!container) return null;
        const item = document.createElement('div');
        item.className = `agent-message ${role}`;
        if (useMarkdown) {
            item.innerHTML = this.renderMarkdown(text);
        } else {
            item.textContent = text;
        }
        container.appendChild(item);
        container.scrollTop = container.scrollHeight;
        return item;
    },

    updateAgentMessageContent(target, text, code) {
        if (!target) return;
        target.innerHTML = this.renderMarkdown(text || '');
        if (code) {
            const pre = document.createElement('pre');
            pre.textContent = code;
            target.appendChild(pre);
        }
    },

    /** Set the agent UI into idle / streaming / reconnecting state. */
    _setAgentState(state) {
        this._agentState = state;
        const sendBtn = document.getElementById('agent-send-btn');
        const stopBtn = document.getElementById('agent-stop-btn');
        const input = document.getElementById('agent-input');

        if (sendBtn) sendBtn.style.display = state === 'idle' ? '' : 'none';
        if (stopBtn) stopBtn.style.display = state === 'streaming' ? '' : 'none';
        if (input) input.disabled = state !== 'idle';
    },

    /** Scroll the messages container to the bottom. */
    _scrollAgentMessages() {
        const container = document.getElementById('agent-messages');
        if (container) container.scrollTop = container.scrollHeight;
    },

    // =====================================================================
    // Inline card builders (tool_call, code, result)
    // =====================================================================

    _appendToolCallCard(bubble, name, args) {
        const card = document.createElement('div');
        card.className = 'agent-card agent-card-tool';
        const label = document.createElement('span');
        label.className = 'agent-card-label';
        label.textContent = name;
        card.appendChild(label);
        if (args && Object.keys(args).length > 0) {
            const detail = document.createElement('span');
            detail.className = 'agent-card-detail';
            const parts = [];
            for (const [k, v] of Object.entries(args)) {
                if (k === 'code') continue; // shown separately
                const s = String(v);
                parts.push(`${k}=${s.length > 60 ? s.slice(0, 60) + '...' : s}`);
            }
            detail.textContent = parts.join(', ');
            card.appendChild(detail);
        }
        bubble.appendChild(card);
        this._scrollAgentMessages();
    },

    _appendCodeCard(bubble, code) {
        const card = document.createElement('div');
        card.className = 'agent-card agent-card-code';
        const pre = document.createElement('pre');
        pre.textContent = code;
        card.appendChild(pre);
        bubble.appendChild(card);
        this._scrollAgentMessages();
    },

    _appendResultCard(bubble, content) {
        const card = document.createElement('div');
        card.className = 'agent-card agent-card-result';
        const shape = content && content.shape;
        card.textContent = shape
            ? `${this.t('agent.dataUpdated')}: ${shape[0]} x ${shape[1]}`
            : this.t('agent.dataUpdated');
        bubble.appendChild(card);
        this._scrollAgentMessages();
    },

    _appendErrorCard(bubble, message) {
        const card = document.createElement('div');
        card.className = 'agent-card agent-card-error';
        card.textContent = message;
        bubble.appendChild(card);
        this._scrollAgentMessages();
    },

    _appendApprovalCard(bubble, content) {
        const card = document.createElement('div');
        card.className = 'agent-card agent-card-tool';
        const title = document.createElement('div');
        title.className = 'agent-card-label';
        title.textContent = content.title || 'Approval required';
        card.appendChild(title);

        if (content.message) {
            const detail = document.createElement('div');
            detail.className = 'agent-card-detail';
            detail.textContent = content.message;
            card.appendChild(detail);
        }

        if (content.code) {
            const pre = document.createElement('pre');
            pre.textContent = content.code.length > 800 ? content.code.slice(0, 800) + '\n...' : content.code;
            card.appendChild(pre);
        }

        const actions = document.createElement('div');
        actions.className = 'agent-card-detail';

        const allowBtn = document.createElement('button');
        allowBtn.type = 'button';
        allowBtn.className = 'btn btn-sm btn-success me-2';
        allowBtn.textContent = 'Allow';
        allowBtn.onclick = () => this._respondAgentApproval(content.approval_id, 'approve', card);

        const denyBtn = document.createElement('button');
        denyBtn.type = 'button';
        denyBtn.className = 'btn btn-sm btn-outline-danger';
        denyBtn.textContent = 'Deny';
        denyBtn.onclick = () => this._respondAgentApproval(content.approval_id, 'deny', card);

        actions.appendChild(allowBtn);
        actions.appendChild(denyBtn);
        card.appendChild(actions);

        bubble.appendChild(card);
        this._scrollAgentMessages();
    },

    _appendQuestionCard(bubble, content) {
        const card = document.createElement('div');
        card.className = 'agent-card agent-card-tool';

        const title = document.createElement('div');
        title.className = 'agent-card-label';
        title.textContent = content.title || 'Question for user';
        card.appendChild(title);

        if (content.message) {
            const detail = document.createElement('div');
            detail.className = 'agent-card-detail';
            detail.textContent = content.message;
            card.appendChild(detail);
        }

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'form-control form-control-sm';
        input.placeholder = content.placeholder || 'Type your answer';
        card.appendChild(input);

        if (Array.isArray(content.options) && content.options.length > 0) {
            const options = document.createElement('div');
            options.className = 'agent-card-detail';
            content.options.forEach((option) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'btn btn-sm btn-outline-secondary me-2';
                btn.textContent = option;
                btn.onclick = () => this._respondAgentQuestion(content.question_id || content.request_id, option, card, input);
                options.appendChild(btn);
            });
            card.appendChild(options);
        }

        const actions = document.createElement('div');
        actions.className = 'agent-card-detail';

        const sendBtn = document.createElement('button');
        sendBtn.type = 'button';
        sendBtn.className = 'btn btn-sm btn-primary';
        sendBtn.textContent = 'Send answer';
        sendBtn.onclick = () => this._respondAgentQuestion(content.question_id || content.request_id, input.value, card, input);

        actions.appendChild(sendBtn);
        card.appendChild(actions);

        bubble.appendChild(card);
        this._scrollAgentMessages();
    },

    async _respondAgentApproval(approvalId, decision, card) {
        if (!approvalId) return;
        try {
            const resp = await fetch('/api/agent/chat/approval', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this._agentSessionId || '',
                    approval_id: approvalId,
                    decision,
                }),
            });
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            if (card) {
                const detail = document.createElement('div');
                detail.className = 'agent-card-detail';
                detail.textContent = decision === 'approve' ? 'Approved' : 'Denied';
                card.appendChild(detail);
            }
            await this.refreshAgentApprovals();
        } catch (err) {
            this.appendAgentMessage(`Approval response failed: ${err.message}`, 'assistant');
        }
    },

    async _respondAgentQuestion(questionId, answer, card, inputEl) {
        if (!questionId) return;
        const finalAnswer = String(answer || '').trim();
        if (!finalAnswer) {
            this.appendAgentMessage('Please provide an answer before submitting.', 'assistant');
            return;
        }
        try {
            const resp = await fetch('/api/agent/chat/question', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this._agentSessionId || '',
                    question_id: questionId,
                    answer: finalAnswer,
                }),
            });
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            if (card) {
                const detail = document.createElement('div');
                detail.className = 'agent-card-detail';
                detail.textContent = `Answered: ${finalAnswer}`;
                card.appendChild(detail);
            }
            if (inputEl) {
                inputEl.value = '';
            }
            await this.refreshAgentQuestions();
            this._renderAgentHarnessMeta();
        } catch (err) {
            this.appendAgentMessage(`Question response failed: ${err.message}`, 'assistant');
        }
    },

    // =====================================================================
    // SSE streaming: send message
    // =====================================================================

    sendAgentMessage() {
        if (this._agentState !== 'idle') return;

        const input = document.getElementById('agent-input');
        if (!input) return;
        const message = input.value.trim();
        if (!message) return;
        input.value = '';

        // Ensure session exists
        this._ensureAgentSession();

        // User bubble
        this.appendAgentMessage(message, 'user');

        // Pending assistant bubble with streaming cursor
        const bubble = document.createElement('div');
        bubble.className = 'agent-message assistant agent-streaming';
        const textSpan = document.createElement('span');
        textSpan.className = 'agent-llm-text';
        bubble.appendChild(textSpan);
        const container = document.getElementById('agent-messages');
        if (container) {
            container.appendChild(bubble);
            container.scrollTop = container.scrollHeight;
        }

        this._agentPendingBubble = bubble;
        this._agentLlmText = '';
        this._agentTurnId = null;
        this._agentTracePayload = null;

        this._setAgentState('streaming');
        this._renderAgentHarnessMeta();
        this._startAgentStream(message, bubble, textSpan);
    },

    async _startAgentStream(message, bubble, textSpan) {
        const abort = new AbortController();
        this._agentAbort = abort;

        try {
            const resp = await fetch('/api/agent/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Agent-Session-Id': this._agentSessionId || '',
                },
                body: JSON.stringify({
                    message,
                    config: this.getAgentConfig()
                }),
                signal: abort.signal
            });

            if (!resp.ok) {
                let detail = `HTTP ${resp.status}`;
                try {
                    const body = await resp.json();
                    if (body && body.error) detail = body.error;
                } catch (_) {}
                throw new Error(detail);
            }

            // Capture turn ID from response header
            const turnId = resp.headers.get('X-Agent-Turn-Id');
            if (turnId) this._agentTurnId = turnId;

            await this._consumeSSEStream(resp.body, bubble, textSpan);

        } catch (err) {
            if (err.name === 'AbortError') {
                // User pressed stop — send server-side cancel, then finalize
                this._sendCancelRequest();
                this._appendErrorCard(bubble, this.t('agent.stopped'));
                this._finalizeBubble(bubble, textSpan);
            } else {
                const detail = err.message === 'Failed to fetch'
                    ? this.t('status.backendUnavailable')
                    : err.message;
                this._appendErrorCard(bubble, `${this.t('common.failed')}: ${detail}`);
                this._finalizeBubble(bubble, textSpan);
            }
        } finally {
            this._agentAbort = null;
            this._setAgentState('idle');
        }
    },

    /** Send a server-side cancel request (best-effort, fire-and-forget). */
    _sendCancelRequest() {
        const payload = {};
        if (this._agentTurnId) {
            payload.turn_id = this._agentTurnId;
        } else if (this._agentSessionId) {
            payload.session_id = this._agentSessionId;
        } else {
            return; // nothing to cancel
        }
        fetch('/api/agent/chat/cancel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        }).catch(() => {}); // fire-and-forget
    },

    // =====================================================================
    // SSE line parser
    // =====================================================================

    async _consumeSSEStream(body, bubble, textSpan) {
        const reader = body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            // Keep the last potentially incomplete line in the buffer
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6);
                if (!jsonStr) continue;

                let event;
                try {
                    event = JSON.parse(jsonStr);
                } catch (_) {
                    continue;
                }

                this._handleAgentEvent(event, bubble, textSpan);
            }
        }

        // Process any remaining data in the buffer
        if (buffer.startsWith('data: ')) {
            try {
                const event = JSON.parse(buffer.slice(6));
                this._handleAgentEvent(event, bubble, textSpan);
            } catch (_) {}
        }
    },

    // =====================================================================
    // Event dispatcher
    // =====================================================================

    _handleAgentEvent(event, bubble, textSpan) {
        const type = event.type;

        switch (type) {
            case 'status':
                if (event.turn_id) this._agentTurnId = event.turn_id;
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                this._applyAgentRuntimeEvent(event);
                this._renderRuntimeState();
                this._renderAgentHarnessMeta();
                break;

            case 'llm_chunk':
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                this._agentLlmText += (event.content || '');
                textSpan.innerHTML = this.renderMarkdown(this._agentLlmText);
                this._scrollAgentMessages();
                break;

            case 'tool_call': {
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                const tc = event.content || {};
                this._applyAgentRuntimeEvent(event);
                this._appendToolCallCard(bubble, tc.name, tc.arguments);
                this._renderRuntimeState();
                this._renderTaskList();
                this._renderAgentHarnessMeta();
                break;
            }

            case 'code':
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                this._appendCodeCard(bubble, event.content || '');
                break;

            case 'result': {
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                const rc = event.content || {};
                this._appendResultCard(bubble, rc);
                // Refresh sidebar with data_info from the result event
                const info = rc.data_info;
                if (info && typeof this.refreshDataFromKernel === 'function') {
                    this.refreshDataFromKernel(info);
                }
                break;
            }

            case 'done':
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                // If no LLM text was streamed, show the summary
                if (!this._agentLlmText && event.content) {
                    this._agentLlmText = event.content;
                    textSpan.innerHTML = this.renderMarkdown(this._agentLlmText);
                }
                break;

            case 'error':
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                this._appendErrorCard(bubble, event.content || this.t('common.unknownError'));
                break;

            case 'approval_request':
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                this._applyAgentRuntimeEvent(event);
                this._appendApprovalCard(bubble, event.content || {});
                this.refreshAgentApprovals();
                break;

            case 'approval_resolved':
                this.refreshAgentApprovals();
                break;

            case 'question_request':
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                this._applyAgentRuntimeEvent(event);
                this._appendQuestionCard(bubble, event.content || {});
                this._renderQuestionList();
                this._renderAgentHarnessMeta();
                break;

            case 'question_resolved':
                this._applyAgentRuntimeEvent(event);
                this.refreshAgentQuestions();
                break;

            case 'item_started':
            case 'item_completed':
                if (event.trace_id) this._agentLastTraceId = event.trace_id;
                this._applyAgentRuntimeEvent(event);
                this._renderRuntimeState();
                this._renderTaskList();
                this._renderAgentHarnessMeta();
                break;

            case 'stream_end':
                this.refreshAgentHarnessPanel();
                this._finalizeBubble(bubble, textSpan);
                break;

            case 'usage':
            case 'heartbeat':
                // Silently consumed
                break;

            default:
                break;
        }
    },

    /** Remove streaming cursor, mark bubble as final. */
    _finalizeBubble(bubble, textSpan) {
        if (!bubble) return;
        bubble.classList.remove('agent-streaming');
        // Show default "done" only when there is no text AND no cards at all.
        // Cards (tool, code, result, error) are already appended before this
        // is called, so an error-only bubble will NOT get a misleading "done".
        const hasCards = bubble.querySelectorAll('.agent-card').length > 0;
        if (!this._agentLlmText && !hasCards) {
            textSpan.textContent = this.t('agent.done');
        }
    },

    // =====================================================================
    // Stop / New Chat / Reconnect
    // =====================================================================

    stopAgentStream() {
        if (this._agentAbort) {
            this._agentAbort.abort();
        }
    },

    newAgentChat() {
        // Abort any running stream
        this.stopAgentStream();

        // Delete the old session on the server (best-effort)
        if (this._agentSessionId) {
            fetch(`/api/agent/session/${this._agentSessionId}`, {
                method: 'DELETE',
            }).catch(() => {});
        }

        // Clear messages
        const container = document.getElementById('agent-messages');
        if (container) {
            container.innerHTML = '';
            // Re-add greeting
            const greeting = document.createElement('div');
            greeting.className = 'agent-message assistant';
            greeting.textContent = this.t('agent.greeting');
            container.appendChild(greeting);
        }
        this._agentLlmText = '';
        this._agentPendingBubble = null;
        this._agentTurnId = null;
        this._agentLastTraceId = null;
        this._agentPendingApprovals = [];
        this._agentPendingQuestions = [];
        this._agentTasks = [];
        this._agentRuntimeState = null;
        this._agentTracePayload = null;

        // Create a new session
        this._agentSessionId = this._generateSessionId();
        sessionStorage.setItem('omicverse.agentSessionId', this._agentSessionId);
        this._updateSessionIndicator();
        this._renderHarnessCapabilities();
        this._renderRuntimeState();
        this._renderApprovalList();
        this._renderQuestionList();
        this._renderTaskList();
        this._renderTracePreview();

        this._setAgentState('idle');
    },

    /** Reconnect a dropped turn by fetching buffered events. */
    async reconnectAgentTurn(turnId) {
        if (!turnId) return;
        this._setAgentState('reconnecting');
        try {
            const resp = await fetch(`/api/agent/chat/turn/${turnId}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const events = data.events || [];

            // Replay into a fresh bubble
            const bubble = document.createElement('div');
            bubble.className = 'agent-message assistant';
            const textSpan = document.createElement('span');
            textSpan.className = 'agent-llm-text';
            bubble.appendChild(textSpan);

            const container = document.getElementById('agent-messages');
            if (container) container.appendChild(bubble);

            this._agentLlmText = '';
            for (const event of events) {
                this._handleAgentEvent(event, bubble, textSpan);
            }
            this._finalizeBubble(bubble, textSpan);
        } catch (err) {
            this.appendAgentMessage(
                `${this.t('agent.reconnectFailed')}: ${err.message}`, 'assistant'
            );
        } finally {
            this._setAgentState('idle');
        }
    },

    async loadAgentTrace(traceId) {
        if (!traceId) return null;
        const resp = await fetch(`/api/agent/trace/${traceId}`);
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }
        return await resp.json();
    },

    async replayLatestAgentTrace() {
        if (!this._agentLastTraceId) {
            this.appendAgentMessage('No trace available for replay yet.', 'assistant');
            return;
        }
        try {
            this._agentTracePayload = await this.loadAgentTrace(this._agentLastTraceId);
            this._renderTracePreview();
        } catch (err) {
            this.appendAgentMessage(`Trace replay failed: ${err.message}`, 'assistant');
        }
    },

    // =====================================================================
    // Config panel (unchanged)
    // =====================================================================

    showAgentConfig() {
        const panel = document.getElementById('agent-config-nav');
        if (panel) {
            panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

});
