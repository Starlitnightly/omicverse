/**
 * OmicVerse Single Cell Analysis — AI Agent Chat & Configuration
 */

Object.assign(SingleCellAnalysis.prototype, {

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
        let stored = null;
        try {
            stored = JSON.parse(localStorage.getItem('omicverse.agentConfig') || 'null');
        } catch (e) {
            stored = null;
        }
        if (!stored) {
            fields.apiBase.value = fields.apiBase.value || 'https://api.openai.com/v1';
            fields.model.value = fields.model.value || 'gpt-5';
            return;
        }
        fields.apiBase.value = stored.apiBase || fields.apiBase.value || 'https://api.openai.com/v1';
        fields.apiKey.value = stored.apiKey || '';
        fields.model.value = stored.model || fields.model.value || 'gpt-5';
        fields.temperature.value = stored.temperature ?? fields.temperature.value;
        fields.topP.value = stored.topP ?? fields.topP.value;
        fields.maxTokens.value = stored.maxTokens ?? fields.maxTokens.value;
        fields.timeout.value = stored.timeout ?? fields.timeout.value;
        fields.systemPrompt.value = stored.systemPrompt || '';
    },

    saveAgentConfig(silent = false) {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        const payload = {
            apiBase: fields.apiBase.value.trim(),
            apiKey: fields.apiKey.value.trim(),
            model: fields.model.value.trim(),
            temperature: fields.temperature.value,
            topP: fields.topP.value,
            maxTokens: fields.maxTokens.value,
            timeout: fields.timeout.value,
            systemPrompt: fields.systemPrompt.value.trim()
        };
        localStorage.setItem('omicverse.agentConfig', JSON.stringify(payload));
        if (!silent) {
            this.showStatus(this.t('status.agentSaved'), false);
            setTimeout(() => this.hideStatus(), 1200);
        }
    },

    resetAgentConfig() {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        localStorage.removeItem('omicverse.agentConfig');
        fields.apiBase.value = 'https://api.openai.com/v1';
        fields.apiKey.value = '';
        fields.model.value = 'gpt-5';
        fields.temperature.value = 0.3;
        fields.topP.value = 1;
        fields.maxTokens.value = 2048;
        fields.timeout.value = 60;
        fields.systemPrompt.value = '';
        this.showStatus(this.t('status.agentReset'), false);
        setTimeout(() => this.hideStatus(), 1200);
    },

    getAgentConfig() {
        let stored = null;
        try {
            stored = JSON.parse(localStorage.getItem('omicverse.agentConfig') || 'null');
        } catch (e) {
            stored = null;
        }
        if (stored) {
            return stored;
        }
        const fields = this.getAgentConfigFields();
        if (!fields) return {};
        return {
            apiBase: fields.apiBase.value.trim(),
            apiKey: fields.apiKey.value.trim(),
            model: fields.model.value.trim(),
            temperature: fields.temperature.value,
            topP: fields.topP.value,
            maxTokens: fields.maxTokens.value,
            timeout: fields.timeout.value,
            systemPrompt: fields.systemPrompt.value.trim()
        };
    },

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

    sendAgentMessage() {
        const input = document.getElementById('agent-input');
        if (!input) return;
        const message = input.value.trim();
        if (!message) return;
        input.value = '';
        this.appendAgentMessage(message, 'user');
        const pending = this.appendAgentMessage(this.t('agent.analyzing'), 'assistant');
        fetch('/api/agent/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                config: this.getAgentConfig()
            })
        })
        .then(async response => {
            let data = null;
            try {
                data = await response.json();
            } catch (e) {
                data = null;
            }
            if (!response.ok) {
                const message = (data && data.error) ? data.error : `HTTP ${response.status}`;
                throw new Error(message);
            }
            return data || {};
        })
        .then(data => {
            if (data.error) {
                this.updateAgentMessageContent(pending, `${this.t('common.failed')}: ${data.error}`);
                return;
            }
            this.updateAgentMessageContent(pending, data.reply || this.t('agent.done'), data.code);
            if (data.data_updated) {
                this.refreshDataFromKernel(data.data_info);
            }
        })
        .catch(error => {
            const detail = error && error.message ? error.message : this.t('common.unknownError');
            const message = detail === 'Failed to fetch'
                ? this.t('status.backendUnavailable')
                : detail;
            this.updateAgentMessageContent(pending, `${this.t('common.failed')}: ${message}`);
        });
    },

    showAgentConfig() {
        const panel = document.getElementById('agent-config-nav');
        if (panel) {
            panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

});
