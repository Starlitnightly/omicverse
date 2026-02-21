/**
 * OmicVerse Single Cell Analysis — Python Environment Manager
 */

Object.assign(SingleCellAnalysis.prototype, {

    showEnvManager() {
        const modal = document.getElementById('env-manager-modal');
        if (!modal) return;
        modal.style.display = '';
        this._envInitChannels();
        this._envUpdatePipPreview();
        this._envUpdateCondaPreview();
        this.applyLanguage(this.currentLang);
    },

    hideEnvManager() {
        const modal = document.getElementById('env-manager-modal');
        if (modal) modal.style.display = 'none';
    },

    switchEnvTab(tab) {
        document.getElementById('env-panel-pip').style.display   = tab === 'pip'   ? '' : 'none';
        document.getElementById('env-panel-conda').style.display = tab === 'conda' ? '' : 'none';
        document.getElementById('env-tab-pip').classList.toggle('active',   tab === 'pip');
        document.getElementById('env-tab-conda').classList.toggle('active', tab === 'conda');
    },

    _envInitChannels() {
        const list = document.getElementById('env-channel-list');
        if (!list || list.dataset.initialized) return;
        list.dataset.initialized = '1';
        this._envDefaultChannels.forEach(ch => {
            const id = `ch-${ch.name}`;
            const wrap = document.createElement('div');
            wrap.className = 'form-check form-check-inline';
            wrap.innerHTML = `<input class="form-check-input" type="checkbox" id="${id}"
                                     value="${ch.name}" ${ch.checked ? 'checked' : ''}
                                     onchange="singleCellApp._envUpdateCondaPreview()">
                              <label class="form-check-label small" for="${id}">${ch.name}</label>`;
            list.appendChild(wrap);
        });
    },

    envAddChannel() {
        const inp = document.getElementById('env-custom-channel');
        const name = inp ? inp.value.trim() : '';
        if (!name) return;
        const list = document.getElementById('env-channel-list');
        const id = `ch-${name}`;
        if (document.getElementById(id)) { inp.value = ''; return; }
        const wrap = document.createElement('div');
        wrap.className = 'form-check form-check-inline';
        wrap.innerHTML = `<input class="form-check-input" type="checkbox" id="${id}"
                                 value="${name}" checked
                                 onchange="singleCellApp._envUpdateCondaPreview()">
                          <label class="form-check-label small" for="${id}">${name}</label>`;
        list.appendChild(wrap);
        inp.value = '';
        this._envUpdateCondaPreview();
    },

    _envGetPkg() {
        return (document.getElementById('env-pkg-input') || {}).value?.trim() || '<package>';
    },

    _envGetChannels() {
        return [...document.querySelectorAll('#env-channel-list input:checked')].map(i => i.value);
    },

    _envUpdatePipPreview() {
        const pkg    = this._envGetPkg();
        const mirror = (document.getElementById('env-mirror-select') || {}).value || '';
        const extra  = (document.getElementById('env-pip-extra') || {}).value?.trim() || '';
        let cmd = `uv pip install ${pkg}`;
        if (mirror) cmd += ` --index-url ${mirror}`;
        if (extra)  cmd += ` ${extra}`;
        const el = document.getElementById('env-pip-preview');
        if (el) el.textContent = cmd;
    },

    _envUpdateCondaPreview() {
        const pkg      = this._envGetPkg();
        const channels = this._envGetChannels();
        const extra    = (document.getElementById('env-conda-extra') || {}).value?.trim() || '';
        const chStr    = channels.map(c => `-c ${c}`).join(' ');
        let cmd = `mamba install -y ${chStr} ${pkg}`;
        if (extra) cmd += ` ${extra}`;
        const el = document.getElementById('env-conda-preview');
        if (el) el.textContent = cmd;
    },

    _envLog(text, type = 'output') {
        const con = document.getElementById('env-console');
        if (!con) return;
        const color = type === 'error' ? '#f38ba8' : type === 'cmd' ? '#89b4fa' : type === 'ok' ? '#a6e3a1' : '#cdd6f4';
        con.innerHTML += `<span style="color:${color}">${this._escHtml(text)}</span>`;
        con.scrollTop = con.scrollHeight;
    },

    _escHtml(s) {
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    },

    async envSearch() {
        const pkg = this._envGetPkg();
        if (pkg === '<package>' || !pkg) return;
        const infoEl    = document.getElementById('env-pkg-info');
        const notFound  = document.getElementById('env-pkg-notfound');
        infoEl.style.display = 'none';
        notFound.style.display = 'none';
        this._envUpdatePipPreview();
        this._envUpdateCondaPreview();
        try {
            const r = await fetch(`/api/env/search_pypi?package=${encodeURIComponent(pkg)}`);
            const d = await r.json();
            if (d.found) {
                document.getElementById('env-pkg-name').textContent    = d.name;
                document.getElementById('env-pkg-version').textContent = d.version || '';
                document.getElementById('env-pkg-pyver').textContent   = d.requires_python ? `Python ${d.requires_python}` : '';
                document.getElementById('env-pkg-summary').textContent = d.summary || '';
                document.getElementById('env-pkg-found-badge').innerHTML =
                    `<span class="badge text-bg-success">${this.t('env.foundOnPypi')}</span>`;
                infoEl.style.display = '';
            } else {
                notFound.style.display = '';
            }
        } catch (e) {
            this._envLog('Search error: ' + e.message, 'error');
        }
    },

    async envTestMirrors() {
        const sel = document.getElementById('env-mirror-select');
        this._envLog(this.t('env.testingMirrors') + '\n', 'cmd');
        try {
            const r = await fetch('/api/env/test_mirrors');
            const d = await r.json();
            sel.innerHTML = '';
            (d.mirrors || []).forEach((m, i) => {
                const opt = document.createElement('option');
                opt.value = m.url;
                const tag = m.ok ? `${m.latency_ms}ms` : 'timeout';
                opt.textContent = `${m.name} — ${tag}`;
                if (i === 0 && m.ok) opt.selected = true;
                sel.appendChild(opt);
                this._envLog(`  ${m.name}: ${tag}\n`, m.ok ? 'output' : 'error');
            });
            this._envUpdatePipPreview();
        } catch (e) {
            this._envLog('Mirror test error: ' + e.message, 'error');
        }
    },

    envInstallPip() {
        const pkg   = this._envGetPkg();
        const mirror = (document.getElementById('env-mirror-select') || {}).value || '';
        const extra  = (document.getElementById('env-pip-extra') || {}).value?.trim() || '';
        if (!pkg || pkg === '<package>') { alert(this.t('env.enterPkg')); return; }

        const con = document.getElementById('env-console');
        if (con) con.textContent = '';
        const es = new EventSource(`/api/env/install_pip?_dummy=${Date.now()}`);
        // Use POST via fetch + ReadableStream instead
        es.close();
        this._envStreamInstall('/api/env/install_pip', { package: pkg, mirror, extra_args: extra });
    },

    envInstallConda() {
        const pkg      = this._envGetPkg();
        const channels = this._envGetChannels();
        const extra    = (document.getElementById('env-conda-extra') || {}).value?.trim() || '';
        if (!pkg || pkg === '<package>') { alert(this.t('env.enterPkg')); return; }

        const con = document.getElementById('env-console');
        if (con) con.textContent = '';
        this._envStreamInstall('/api/env/install_conda', { package: pkg, channels, extra_args: extra });
    },

    async _envStreamInstall(url, body) {
        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const evt = JSON.parse(line.slice(6));
                        if (evt.type === 'cmd')      this._envLog('$ ' + evt.text + '\n', 'cmd');
                        else if (evt.type === 'output') this._envLog(evt.text);
                        else if (evt.type === 'error')  this._envLog(evt.text + '\n', 'error');
                        else if (evt.type === 'complete') {
                            const msg = evt.success
                                ? this.t('env.installSuccess')
                                : `${this.t('env.installFailed')} (code ${evt.returncode})`;
                            this._envLog('\n' + msg + '\n', evt.success ? 'ok' : 'error');
                        }
                    } catch {}
                }
            }
        } catch (e) {
            this._envLog('Stream error: ' + e.message + '\n', 'error');
        }
    },

    showEnvInfo() {
        const modal = document.getElementById('env-info-modal');
        if (!modal) return;
        modal.style.display = '';
        this.applyLanguage(this.currentLang);
        this.loadEnvInfo();
    },

    hideEnvInfo() {
        const modal = document.getElementById('env-info-modal');
        if (modal) modal.style.display = 'none';
    },

    async loadEnvInfo() {
        const loading = document.getElementById('env-info-loading');
        const content = document.getElementById('env-info-content');
        const errEl   = document.getElementById('env-info-error');
        loading.style.display = '';
        content.style.display = 'none';
        errEl.style.display   = 'none';

        try {
            const r = await fetch('/api/env/info');
            const d = await r.json();

            this._renderEnvSystem(d.system || {});
            this._renderEnvPython(d.python || {});
            this._renderEnvGpu(d.gpu || {});
            this._renderEnvKeyPkgs(d.key_pkgs || []);
            this._renderEnvAllPkgs(d.all_pkgs || []);

            loading.style.display = 'none';
            content.style.display = '';
        } catch (e) {
            loading.style.display = 'none';
            errEl.textContent = 'Failed to load environment info: ' + e.message;
            errEl.style.display = '';
        }
    },

    _kv(label, value) {
        if (value === null || value === undefined || value === '') return '';
        return `<div class="d-flex gap-2 mb-1">
            <span class="text-muted small" style="min-width:90px;flex-shrink:0">${this._escHtml(label)}</span>
            <span class="small fw-medium" style="word-break:break-all">${this._escHtml(String(value))}</span>
        </div>`;
    },

    _renderEnvSystem(s) {
        const el = document.getElementById('env-info-system');
        if (!el) return;
        el.innerHTML =
            this._kv('OS', `${s.os} ${s.machine}`) +
            this._kv('Version', s.os_ver) +
            this._kv('Host', s.hostname) +
            (s.cpu_count   ? this._kv('CPU cores', s.cpu_count) : '') +
            (s.ram_total_gb ? this._kv('RAM', `${s.ram_used_gb} / ${s.ram_total_gb} GB`) : '');
    },

    _renderEnvPython(p) {
        const el = document.getElementById('env-info-python');
        if (!el) return;
        const ver = p.version ? p.version.split(' ')[0] : '';
        el.innerHTML =
            this._kv('Python', ver) +
            this._kv('Executable', p.executable) +
            this._kv('Prefix', p.prefix) +
            (p.uv     ? this._kv('uv',     p.uv)     : '') +
            (p.mamba  ? this._kv('mamba',  p.mamba)  : '') +
            (p.conda  ? this._kv('conda',  p.conda)  : '') +
            (p.pip    ? this._kv('pip',    p.pip)     : '');
    },

    _renderEnvGpu(g) {
        const el = document.getElementById('env-info-gpu');
        if (!el) return;
        let html = g.torch_version ? this._kv('PyTorch', g.torch_version) : '';
        if (g.cuda_available) {
            html += this._kv('CUDA', g.cuda_version);
            (g.devices || []).forEach(dev => {
                html += this._kv(`GPU ${dev.index}`, `${dev.name}${dev.mem_gb ? ' — ' + dev.mem_gb + ' GB' : ''}`);
            });
        } else if (g.mps_available) {
            html += `<span class="badge text-bg-success">Apple MPS</span>`;
        } else {
            html += `<span class="text-muted small">${this.t('envInfo.noGpu')}</span>`;
        }
        el.innerHTML = html || `<span class="text-muted small">${this.t('envInfo.noGpu')}</span>`;
    },

    _renderEnvKeyPkgs(pkgs) {
        const tbody = document.getElementById('env-info-pkgs');
        if (!tbody) return;
        tbody.innerHTML = pkgs.map(p => {
            const badge = p.installed
                ? `<span class="badge text-bg-success">✓ ${this._escHtml(p.version)}</span>`
                : `<span class="badge text-bg-secondary">${this.t('envInfo.notInstalled')}</span>`;
            return `<tr>
                <td class="ps-3 fw-medium">${this._escHtml(p.name)}</td>
                <td>${p.installed ? this._escHtml(p.version) : '—'}</td>
                <td>${badge}</td>
            </tr>`;
        }).join('');
    },

    _renderEnvAllPkgs(pkgs) {
        this._envAllPkgsData = pkgs;
        const count = document.getElementById('env-info-pkg-count');
        if (count) count.textContent = pkgs.length;
        this._renderEnvAllPkgsTable(pkgs);
    },

    _renderEnvAllPkgsTable(pkgs) {
        const tbody = document.getElementById('env-allpkgs-table');
        if (!tbody) return;
        tbody.innerHTML = pkgs.map(p =>
            `<tr><td class="ps-3">${this._escHtml(p.name)}</td><td class="text-muted">${this._escHtml(p.version)}</td></tr>`
        ).join('');
    },

    filterEnvPkgs(query) {
        const q = query.toLowerCase();
        const filtered = q
            ? this._envAllPkgsData.filter(p => p.name.toLowerCase().includes(q))
            : this._envAllPkgsData;
        this._renderEnvAllPkgsTable(filtered);
    },

    toggleEnvAllPkgs() {
        const body = document.getElementById('env-allpkgs-body');
        const icon = document.getElementById('env-allpkgs-icon');
        if (!body) return;
        const collapsed = body.style.display === 'none';
        body.style.display = collapsed ? '' : 'none';
        if (icon) icon.className = collapsed ? 'feather-chevron-up' : 'feather-chevron-down';
    }

});
