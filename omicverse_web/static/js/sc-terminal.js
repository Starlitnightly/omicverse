/**
 * OmicVerse — Multi-session Terminal Manager
 *
 * Architecture:
 *   TerminalManager  — manages a list of OVTerminalSession instances,
 *                      renders the sidebar, handles create / activate / delete.
 *   OVTerminalSession — one xterm.js instance + one PTY backend session.
 */

/* ── Colour palettes (Catppuccin) ────────────────────────────────────────── */
const _TERM_THEME_DARK = {
    background:    '#1a1c20',
    foreground:    '#e5e7eb',
    cursor:        '#89b4fa',
    cursorAccent:  '#1a1c20',
    selectionBackground: 'rgba(137,180,250,0.3)',
    black:  '#313244', red:    '#f38ba8', green:  '#a6e3a1', yellow: '#f9e2af',
    blue:   '#89b4fa', magenta:'#cba6f7', cyan:   '#89dceb', white:  '#cdd6f4',
    brightBlack: '#45475a', brightRed: '#f38ba8', brightGreen: '#a6e3a1',
    brightYellow:'#f9e2af', brightBlue:'#89b4fa', brightMagenta:'#cba6f7',
    brightCyan:  '#89dceb', brightWhite:'#cdd6f4',
};

const _TERM_THEME_LIGHT = {
    background:    '#1e1e2e',   // keep terminals dark regardless of app theme
    foreground:    '#cdd6f4',
    cursor:        '#cba6f7',
    cursorAccent:  '#1e1e2e',
    selectionBackground: 'rgba(203,166,247,0.25)',
    black:  '#45475a', red:    '#f38ba8', green:  '#a6e3a1', yellow: '#f9e2af',
    blue:   '#89b4fa', magenta:'#cba6f7', cyan:   '#89dceb', white:  '#cdd6f4',
    brightBlack: '#585b70', brightRed: '#f38ba8', brightGreen: '#a6e3a1',
    brightYellow:'#f9e2af', brightBlue:'#89b4fa', brightMagenta:'#cba6f7',
    brightCyan:  '#89dceb', brightWhite:'#cdd6f4',
};

function _termTheme() {
    return _TERM_THEME_DARK;   // always dark; PTY output looks best on dark bg
}

/* ── OVTerminalSession ───────────────────────────────────────────────────── */
class OVTerminalSession {
    /**
     * @param {HTMLElement} panelEl  — the panel element to mount xterm.js into
     */
    constructor(panelEl) {
        this.panelEl   = panelEl;
        this.sessionId = null;
        this.term      = null;
        this.fitAddon  = null;
        this.evtSrc    = null;
        this.alive     = false;
        this._ro       = null;
        this._ready    = false;
    }

    /* ── public ─────────────────────────────────────────────────────────── */

    async open() {
        if (this._ready) { this._fit(); return; }
        if (!window.Terminal || !window.FitAddon) {
            this.panelEl.innerHTML =
                '<div style="color:#f38ba8;padding:16px">xterm.js not loaded. Check CDN.</div>';
            return;
        }

        this.panelEl.innerHTML = '';

        this.term = new window.Terminal({
            cursorBlink: true,
            fontSize: 13,
            fontFamily: '"JetBrains Mono","Fira Code","Cascadia Code",Menlo,monospace',
            theme: _termTheme(),
            scrollback: 10000,
            convertEol: false,
            allowTransparency: false,
        });

        this.fitAddon = new window.FitAddon.FitAddon();
        this.term.loadAddon(this.fitAddon);
        this.term.open(this.panelEl);
        this._fit();

        this.term.onData(data => this._input(data));
        this.term.onResize(({ rows, cols }) => this._sendResize(rows, cols));

        this._ro = new ResizeObserver(() => this._fit());
        this._ro.observe(this.panelEl);

        this._ready = true;
        await this._createSession();
    }

    /** Called when this panel becomes visible after being hidden. */
    refit() {
        setTimeout(() => this._fit(), 30);
    }

    updateTheme() {
        if (this.term) this.term.options.theme = _termTheme();
    }

    async destroy() {
        this.alive = false;
        if (this.evtSrc) { this.evtSrc.close(); this.evtSrc = null; }
        if (this._ro)    { this._ro.disconnect(); this._ro = null; }
        if (this.sessionId) {
            fetch('/api/terminal/kill', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId }),
            }).catch(() => {});
            this.sessionId = null;
        }
        if (this.term) { this.term.dispose(); this.term = null; }
        this._ready = false;
    }

    /* ── private ─────────────────────────────────────────────────────────── */

    _fit() {
        try { if (this.fitAddon) this.fitAddon.fit(); } catch (_) {}
    }

    async _createSession() {
        try {
            const r = await fetch('/api/terminal/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cwd: window._ovWorkingDir || null }),
            });
            const d = await r.json();
            if (d.error) throw new Error(d.error);
            this.sessionId = d.session_id;
            this.alive     = true;
            this._startStream();
            setTimeout(() => {
                this._fit();
                if (this.term) this._sendResize(this.term.rows, this.term.cols);
            }, 250);
        } catch (e) {
            if (this.term)
                this.term.write(`\r\n\x1b[31mFailed to connect: ${e.message}\x1b[0m\r\n`);
        }
    }

    _startStream() {
        if (!this.sessionId) return;
        this.evtSrc = new EventSource(`/api/terminal/stream/${this.sessionId}`);

        this.evtSrc.onmessage = (evt) => {
            const msg = JSON.parse(evt.data);
            if (msg.type === 'output' && this.term) {
                const bin = atob(msg.data);
                const u8  = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
                this.term.write(new TextDecoder('utf-8', { fatal: false }).decode(u8));
            } else if (msg.type === 'exit') {
                if (this.term) this.term.write('\r\n\x1b[33m[Process exited]\x1b[0m\r\n');
                this.alive = false;
                this.evtSrc.close();
            }
        };

        this.evtSrc.onerror = () => {
            if (!this.alive) return;
            if (this.term) this.term.write('\r\n\x1b[33m[Reconnecting…]\x1b[0m\r\n');
            this.evtSrc.close();
            setTimeout(() => { if (this.alive) this._startStream(); }, 2000);
        };
    }

    _input(data) {
        if (!this.sessionId || !this.alive) return;
        fetch('/api/terminal/input', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId, data }),
        }).catch(() => {});
    }

    _sendResize(rows, cols) {
        if (!this.sessionId) return;
        fetch('/api/terminal/resize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId, rows, cols }),
        }).catch(() => {});
    }
}

/* ── TerminalManager ─────────────────────────────────────────────────────── */
class TerminalManager {
    constructor() {
        this.sessions  = new Map();  // id → { sess: OVTerminalSession, panelEl, name }
        this.activeId  = null;
        this._counter  = 0;
        this._inited   = false;
    }

    /* ── public ─────────────────────────────────────────────────────────── */

    async init() {
        if (this._inited) {
            // Already initialised; just refit the active session
            if (this.activeId) {
                const s = this.sessions.get(this.activeId);
                if (s) s.sess.refit();
            }
            return;
        }
        this._inited = true;
        await this.newSession();
    }

    async newSession() {
        this._counter++;
        const id   = `ts-${Date.now()}-${this._counter}`;
        const name = `Terminal ${this._counter}`;

        const panels = document.getElementById('term-panels');
        if (!panels) return;

        const panelEl = document.createElement('div');
        panelEl.className = 'term-panel';
        panelEl.dataset.id = id;
        panels.appendChild(panelEl);

        const sess = new OVTerminalSession(panelEl);
        this.sessions.set(id, { sess, panelEl, name, id });

        this._renderSidebar();
        await this._activate(id);   // show panel first (so xterm can measure)
        await sess.open();
    }

    async activateSession(id) {
        await this._activate(id);
    }

    async deleteSession(id) {
        const entry = this.sessions.get(id);
        if (!entry) return;

        // Kill PTY + dispose xterm
        await entry.sess.destroy();
        entry.panelEl.remove();
        this.sessions.delete(id);

        if (this.activeId === id) {
            this.activeId = null;
            const remaining = [...this.sessions.keys()];
            if (remaining.length > 0) {
                await this._activate(remaining[remaining.length - 1]);
            } else {
                // Auto-create a new session when the last one is closed
                await this.newSession();
                return;
            }
        }
        this._renderSidebar();
    }

    updateTheme() {
        this.sessions.forEach(({ sess }) => sess.updateTheme());
    }

    /* ── private ─────────────────────────────────────────────────────────── */

    async _activate(id) {
        const entry = this.sessions.get(id);
        if (!entry) return;

        // Hide all panels
        this.sessions.forEach(({ panelEl }) => {
            panelEl.style.display = 'none';
        });

        // Show target panel
        entry.panelEl.style.display = 'block';
        this.activeId = id;

        // Refit after the panel is visible
        entry.sess.refit();

        this._renderSidebar();
    }

    _renderSidebar() {
        const list = document.getElementById('term-session-list');
        if (!list) return;
        list.innerHTML = '';

        this.sessions.forEach(({ name, id }) => {
            const isActive = id === this.activeId;
            const item = document.createElement('div');
            item.className = 'term-session-item' + (isActive ? ' active' : '');
            item.innerHTML = `
                <span class="term-session-indicator"></span>
                <span class="term-session-name"
                      onclick="singleCellApp._termMgr.activateSession('${id}')">
                    <i class="feather-terminal" style="width:12px;height:12px;margin-right:5px;"></i>${this._esc(name)}
                </span>
                <button class="term-session-del"
                        title="Close"
                        onclick="event.stopPropagation();singleCellApp._termMgr.deleteSession('${id}')">
                    <i class="feather-x" style="width:11px;height:11px;"></i>
                </button>`;
            // clicking the row itself also activates
            item.addEventListener('click', () => this.activateSession(id));
            list.appendChild(item);
        });

        // re-apply feather icons
        if (window.feather) feather.replace({ 'stroke-width': 2 });
    }

    _esc(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
}

/* ── Mix into SingleCellAnalysis ─────────────────────────────────────────── */
Object.assign(SingleCellAnalysis.prototype, {

    _termMgr: null,

    /** Called by switchView('terminal') */
    async openTerminalView() {
        if (!this._termMgr) {
            this._termMgr = new TerminalManager();
        }
        await this._termMgr.init();
    },

    /** Called after theme toggle */
    updateTerminalTheme() {
        if (this._termMgr) this._termMgr.updateTheme();
    },
});
