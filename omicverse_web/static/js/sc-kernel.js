/**
 * OmicVerse Single Cell Analysis — Kernel Management & Variable Inspector
 */

Object.assign(SingleCellAnalysis.prototype, {

    setupKernelSelector() {
        const kernelSelect = document.getElementById('kernel-select');
        if (!kernelSelect) return;
        kernelSelect.addEventListener('change', () => {
            const activeTab = this.getActiveTab();
            if (!activeTab || activeTab.type !== 'notebook') {
                return;
            }
            const selected = kernelSelect.value;
            if (selected === activeTab.kernelName) {
                return;
            }
            this.changeKernel(activeTab, selected, kernelSelect);
        });
    },

    loadKernelOptions(kernelSelect, tab) {
        const kernelId = tab?.kernelId || 'default.ipynb';
        fetch(`/api/kernel/list?kernel_id=${encodeURIComponent(kernelId)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                return;
            }
            const kernels = data.kernels || [];
            if (!kernels.length) {
                return;
            }
            kernelSelect.innerHTML = '';
            kernels.forEach(kernel => {
                const option = document.createElement('option');
                option.value = kernel.name;
                option.textContent = kernel.display_name || kernel.name;
                kernelSelect.appendChild(option);
            });
            const current = data.current || data.default;
            if (current) {
                kernelSelect.value = current;
            }
            if (tab) {
                tab.kernelName = kernelSelect.value;
            }
        })
        .catch(() => {});
    },

    changeKernel(tab, name, kernelSelect) {
        const previous = tab.kernelName || kernelSelect.value;
        const label = kernelSelect.options[kernelSelect.selectedIndex]?.text || name;
        this.showStatus(this.t('kernel.switching') + ` ${label}...`, true);
        fetch('/api/kernel/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, kernel_id: tab.kernelId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            tab.kernelName = data.current || name;
            this.updateStatus(this.t('kernel.switched') + ` ${label}`);
            setTimeout(() => this.hideStatus(), 1200);
            this.fetchKernelStats(tab.kernelId);
            this.fetchKernelVars(tab.kernelId);
        })
        .catch(error => {
            kernelSelect.value = previous;
            this.updateStatus(this.t('kernel.switchFailed') + ` ${error.message}`);
            setTimeout(() => this.hideStatus(), 2000);
        });
    },

    updateKernelSelectorForTab(tab) {
        const kernelSelect = document.getElementById('kernel-select');
        if (!kernelSelect) return;
        if (!tab || tab.type !== 'notebook') {
            kernelSelect.disabled = true;
            return;
        }
        kernelSelect.disabled = false;
        this.loadKernelOptions(kernelSelect, tab);
    },

    interruptExecution() {
        if (!this.isExecuting) {
            return;
        }

        // Disable interrupt button to prevent double-clicks
        const interruptBtn = document.getElementById('interrupt-btn');
        if (interruptBtn) {
            interruptBtn.disabled = true;
            interruptBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Interrupting...';
        }

        // Send interrupt request to backend
        fetch('/api/kernel/interrupt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Interrupt failed:', data.error);
                this.addToLog('Interrupt failed: ' + data.error, 'error');
            } else {
                this.addToLog('Execution interrupted');
            }
        })
        .catch(error => {
            console.error('Interrupt request failed:', error);
            this.addToLog('Interrupt request failed', 'error');
        })
        .finally(() => {
            // Abort the fetch request immediately for faster UI response
            if (this.executionAbortController) {
                this.executionAbortController.abort();
            }
        });
    },

    restartKernel() {
        // Confirm restart
        if (!confirm('Are you sure you want to restart the kernel? All variables will be lost.')) {
            return;
        }

        const restartBtn = document.getElementById('restart-kernel-btn');
        if (restartBtn) {
            restartBtn.disabled = true;
            restartBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        }

        this.addToLog('Restarting kernel...');

        const kernelId = this.getActiveKernelId();

        fetch('/api/kernel/restart', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                kernel_id: kernelId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Kernel restart failed:', data.error);
                this.addToLog('Kernel restart failed: ' + data.error, 'error');
            } else {
                this.addToLog('Kernel restarted successfully');

                // Clear all cell execution numbers
                this.codeCells.forEach(cellId => {
                    this.updateCellNumber(cellId, 'idle');
                });

                // Refresh variable viewer and kernel stats
                this.refreshKernelInfo();
            }
        })
        .catch(error => {
            console.error('Kernel restart request failed:', error);
            this.addToLog('Kernel restart failed: ' + error.message, 'error');
        })
        .finally(() => {
            if (restartBtn) {
                restartBtn.disabled = false;
                restartBtn.innerHTML = '<i class="fas fa-redo"></i>';
            }
        });
    },

    refreshKernelInfo() {
        // Refresh both variable viewer and kernel stats
        this.fetchKernelVars();
        this.fetchKernelStats();
    },

    showInterruptButton() {
        this.isExecuting = true;
        const btn = document.getElementById('interrupt-btn');
        if (btn) {
            btn.style.display = 'inline-block';
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-stop"></i> <span data-i18n="toolbar.interrupt">Interrupt</span>';
        }
    },

    hideInterruptButton() {
        this.isExecuting = false;
        const btn = document.getElementById('interrupt-btn');
        if (btn) {
            btn.style.display = 'none';
        }
    },

    startExecutionStatusPolling() {
        // Poll execution status every 500ms to update UI
        this.executionStatusPollInterval = setInterval(() => {
            fetch('/api/kernel/status')
                .then(response => response.json())
                .then(data => {
                    if (!data.is_executing) {
                        // Execution finished, stop polling
                        this.stopExecutionStatusPolling();
                        this.hideInterruptButton();
                    } else if (data.elapsed_seconds) {
                        // Update elapsed time display
                        const btn = document.getElementById('interrupt-btn');
                        if (btn && data.elapsed_seconds > 5) {
                            btn.title = `Running for ${Math.floor(data.elapsed_seconds)}s`;
                        }
                    }
                })
                .catch(err => {
                    console.error('Status poll failed:', err);
                });
        }, 500);
    },

    stopExecutionStatusPolling() {
        if (this.executionStatusPollInterval) {
            clearInterval(this.executionStatusPollInterval);
            this.executionStatusPollInterval = null;
        }
    },

    showVarDetail(detail) {
        const container = document.getElementById('code-cells-container');
        const textView = document.getElementById('text-file-view');
        const varView = document.getElementById('var-detail-view');
        const mdView = document.getElementById('md-file-view');
        const imageView = document.getElementById('image-file-view');
        if (container) container.style.display = 'none';
        if (textView) textView.style.display = 'none';
        if (mdView) mdView.style.display = 'none';
        if (imageView) imageView.style.display = 'none';
        if (!varView) return;
        varView.style.display = 'block';
        varView.scrollTop = 0;
        varView.innerHTML = '';
        const meta = document.createElement('div');
        meta.className = 'var-detail-meta';
        if (detail.type === 'dataframe') {
            meta.innerHTML = `<strong>${detail.name}</strong> &nbsp;·&nbsp; DataFrame &nbsp;
                <span class="df-card-shape" style="margin-left:4px">${detail.shape[0].toLocaleString()} rows × ${detail.shape[1]} cols</span>`;
        } else if (detail.type === 'anndata') {
            meta.textContent = `${detail.name} · ${this.t('var.anndataLabel')} ${detail.summary.shape.join('x')}`;
        } else {
            meta.textContent = `${detail.name}`;
        }
        varView.appendChild(meta);

        if (detail.type === 'dataframe' && detail.table) {
            // Dtype header chips
            if (detail.dtypes) {
                const dtypeWrap = document.createElement('div');
                dtypeWrap.className = 'df-dtype-header-chips';
                detail.table.columns.forEach((col, colIdx) => {
                    const dtype = detail.dtypes[col] || '';
                    const chip = document.createElement('span');
                    chip.className = 'df-col-chip';
                    chip.textContent = col;
                    this._applyDfColumnTheme(chip, colIdx, 'chip');
                    if (dtype) {
                        const badge = document.createElement('span');
                        badge.className = `df-dtype-badge ${this._dtypeClass(dtype)}`;
                        badge.textContent = dtype;
                        chip.appendChild(badge);
                    }
                    dtypeWrap.appendChild(chip);
                });
                varView.appendChild(dtypeWrap);
            }

            // Scrollable table
            const wrap = document.createElement('div');
            wrap.className = 'df-viewer-wrap';

            const table = document.createElement('table');
            table.className = 'df-viewer-table';

            // Header
            const thead = document.createElement('thead');
            const headRow = document.createElement('tr');
            const corner = document.createElement('th');
            corner.className = 'df-th-index';
            corner.textContent = '#';
            headRow.appendChild(corner);
            detail.table.columns.forEach((col, colIdx) => {
                const th = document.createElement('th');
                th.textContent = col;
                this._applyDfColumnTheme(th, colIdx, 'header');
                if (detail.dtypes && detail.dtypes[col]) {
                    const badge = document.createElement('span');
                    badge.className = `df-dtype-badge ${this._dtypeClass(detail.dtypes[col])}`;
                    badge.textContent = detail.dtypes[col];
                    th.appendChild(badge);
                }
                headRow.appendChild(th);
            });
            thead.appendChild(headRow);
            table.appendChild(thead);

            // Body
            const tbody = document.createElement('tbody');
            detail.table.data.forEach((row, idx) => {
                const tr = document.createElement('tr');
                const idxCell = document.createElement('td');
                idxCell.textContent = detail.table.index[idx];
                tr.appendChild(idxCell);
                row.forEach((cell, colIdx) => {
                    const td = document.createElement('td');
                    const val = cell !== null && cell !== undefined ? String(cell) : '';
                    td.textContent = val;
                    td.title = val;
                    this._applyDfColumnTheme(td, colIdx, 'cell');
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            wrap.appendChild(table);
            varView.appendChild(wrap);

            // Footer: row count info
            const footer = document.createElement('div');
            footer.className = 'df-viewer-footer';
            const total = detail.shape[0];
            const shown = detail.table.data.length;
            footer.textContent = shown < total
                ? `Showing ${shown} of ${total.toLocaleString()} rows × ${detail.shape[1]} columns`
                : `${total.toLocaleString()} rows × ${detail.shape[1]} columns`;
            varView.appendChild(footer);
            return;
        }

        if (detail.type === 'anndata') {
            const summary = detail.summary || {};
            const section = (label, slot, keys, total, more) => {
                const wrap = document.createElement('div');
                wrap.className = 'adata-viewer-section';

                const title = document.createElement('div');
                title.className = 'adata-viewer-section-title';
                title.textContent = `${label} (${Number.isFinite(total) ? total : (keys || []).length})`;
                wrap.appendChild(title);

                const chips = document.createElement('div');
                chips.className = 'df-card-cols';
                (keys || []).forEach((k) => {
                    const chip = document.createElement('button');
                    chip.type = 'button';
                    chip.className = 'adata-key-chip adata-chip-clickable';
                    chip.dataset.slot = slot;
                    chip.dataset.key = String(k);
                    chip.textContent = String(k);
                    chips.appendChild(chip);
                });
                if ((more || 0) > 0) {
                    const badge = document.createElement('span');
                    badge.className = 'df-col-more';
                    badge.textContent = `+${more} more`;
                    chips.appendChild(badge);
                }
                if (!keys || keys.length === 0) {
                    const empty = document.createElement('span');
                    empty.className = 'text-muted';
                    empty.textContent = '—';
                    chips.appendChild(empty);
                }
                wrap.appendChild(chips);
                return wrap;
            };

            const actionBar = document.createElement('div');
            actionBar.className = 'adata-viewer-actions';
            const obsBtn = document.createElement('button');
            obsBtn.type = 'button';
            obsBtn.className = 'btn btn-sm btn-outline-secondary';
            obsBtn.textContent = 'Open .obs';
            obsBtn.onclick = () => this.openVarTab(`${detail.name}.obs`);
            const varBtn = document.createElement('button');
            varBtn.type = 'button';
            varBtn.className = 'btn btn-sm btn-outline-secondary';
            varBtn.textContent = 'Open .var';
            varBtn.onclick = () => this.openVarTab(`${detail.name}.var`);
            const loadBtn = document.createElement('button');
            loadBtn.className = 'btn btn-sm btn-primary';
            loadBtn.innerHTML = `<i class="feather-eye"></i> ${this.t('var.loadToViz')}`;
            loadBtn.onclick = () => this.loadAnndataToVisualization(detail.name);
            actionBar.appendChild(obsBtn);
            actionBar.appendChild(varBtn);
            actionBar.appendChild(loadBtn);
            varView.appendChild(actionBar);

            const grid = document.createElement('div');
            grid.className = 'adata-viewer-grid';
            grid.appendChild(section('obs', 'obs', summary.obs_columns || [], summary.obs_columns_total, summary.obs_columns_more));
            grid.appendChild(section('var', 'var', summary.var_columns || [], summary.var_columns_total, summary.var_columns_more));
            grid.appendChild(section('uns', 'uns', summary.uns_keys || [], summary.uns_keys_total, summary.uns_keys_more));
            grid.appendChild(section('obsm', 'obsm', summary.obsm_keys || [], summary.obsm_keys_total, summary.obsm_keys_more));
            grid.appendChild(section('layers', 'layers', summary.layers || [], summary.layers_total, summary.layers_more));
            grid.addEventListener('click', (e) => {
                const chip = e.target.closest('.adata-chip-clickable');
                if (!chip) return;
                this.showAdataSlotDetail(detail.name, chip.dataset.slot, chip.dataset.key);
            });
            varView.appendChild(grid);
            return;
        }

        const pre = document.createElement('pre');
        pre.className = 'code-output-text';
        pre.textContent = detail.content || '';
        varView.appendChild(pre);
    },

    _dtypeClass(dtype) {
        if (!dtype) return 'df-dtype-other';
        const d = dtype.toLowerCase();
        if (d.includes('int')) return 'df-dtype-int';
        if (d.includes('float')) return 'df-dtype-float';
        if (d === 'object' || d === 'string' || d === 'str' || d.startsWith('string')) return 'df-dtype-object';
        if (d === 'bool') return 'df-dtype-bool';
        if (d.includes('datetime') || d.includes('timedelta')) return 'df-dtype-datetime';
        if (d === 'category') return 'df-dtype-category';
        return 'df-dtype-other';
    },

    _dfColumnTheme(colIdx) {
        const hue = (colIdx * 47 + 18) % 360;
        const dark = document.documentElement.classList.contains('app-skin-dark');
        if (dark) {
            return {
                bg: `hsla(${hue}, 72%, 28%, 0.42)`,
                border: `hsla(${hue}, 70%, 58%, 0.45)`,
                fg: `hsl(${hue}, 78%, 84%)`
            };
        }
        return {
            // Day mode: stronger contrast for readability while keeping column colors.
            bg: `hsla(${hue}, 88%, 84%, 0.95)`,
            border: `hsla(${hue}, 78%, 42%, 0.55)`,
            fg: `hsl(${hue}, 72%, 18%)`
        };
    },

    _applyDfColumnTheme(el, colIdx, role = 'cell') {
        if (!el) return;
        const theme = this._dfColumnTheme(colIdx);
        if (role === 'header' || role === 'chip') {
            el.style.background = theme.bg;
            el.style.color = theme.fg;
            el.style.borderColor = theme.border;
        } else {
            // Keep cells readable in day mode: lighter tint + neutral text.
            const dark = document.documentElement.classList.contains('app-skin-dark');
            el.style.background = dark ? theme.bg : `hsla(${(colIdx * 47 + 18) % 360}, 88%, 90%, 0.7)`;
            el.style.borderLeft = `1px solid ${theme.border}`;
            el.style.color = dark ? theme.fg : '#1f2937';
        }
    },

    loadAnndataToVisualization(varName) {
        console.log('=== loadAnndataToVisualization CALLED ===');
        console.log('varName:', varName);

        if (!varName) {
            console.error('No varName provided');
            return;
        }

        const kernelId = this.getActiveKernelId();
        console.log('kernelId:', kernelId);

        if (!kernelId) {
            console.error('No active kernel');
            this.addToLog('No active kernel', 'error');
            return;
        }

        // Show loading notification
        this.addToLog('Loading AnnData to visualization...');
        console.log('Sending request to /api/kernel/load_adata');

        fetch('/api/kernel/load_adata', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                var_name: varName,
                kernel_id: kernelId
            })
        })
        .then(response => {
            console.log('Response status:', response.status);
            console.log('Response ok:', response.ok);
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('Load AnnData error:', data.error);
                this.addToLog(`Failed to load: ${data.error}`, 'error');
                return;
            }

            if (data.success && data.data_info) {
                console.log('Successfully loaded AnnData:', data.data_info);

                // Update UI with loaded data
                this.addToLog(`Successfully loaded ${varName} to visualization`);

                // Switch to visualization view FIRST
                this.switchView('visualization');

                // Ensure visualization controls are shown (like updateUI does)
                const uploadSection = document.getElementById('upload-section');
                const vizControls = document.getElementById('viz-controls');
                const vizPanel = document.getElementById('viz-panel');

                if (uploadSection) {
                    uploadSection.style.display = 'none';
                    console.log('Hid upload section');
                }
                if (vizControls) {
                    vizControls.style.display = 'block';
                    console.log('Showed viz controls');
                }
                if (vizPanel) {
                    vizPanel.style.display = 'block';
                    console.log('Showed viz panel');
                }

                // Refresh data info display
                this.refreshDataFromKernel(data.data_info);

                // Fetch gene list for autocomplete
                if (this.fetchGeneList) {
                    this.fetchGeneList();
                }

                // Manually trigger plot update after view switch
                setTimeout(() => {
                    const embeddingSelect = document.getElementById('embedding-select');
                    if (embeddingSelect && embeddingSelect.value) {
                        console.log('Triggering plot update with embedding:', embeddingSelect.value);
                        this.updatePlot();
                    } else if (data.data_info.embeddings && data.data_info.embeddings.length > 0) {
                        // Auto-select first embedding if available
                        embeddingSelect.value = data.data_info.embeddings[0];
                        console.log('Auto-selected embedding:', data.data_info.embeddings[0]);
                        this.updatePlot();
                    }
                }, 300);
            } else {
                console.error('Unexpected response:', data);
                this.addToLog('Unexpected response from server', 'error');
            }
        })
        .catch(error => {
            console.error('Load AnnData exception:', error);
            this.addToLog(`Error: ${error.message}`, 'error');
        });
    },

    toggleSection(sectionId) {
        const section = document.getElementById(sectionId);
        const toggle = document.getElementById(`${sectionId}-toggle`);
        if (!section || !toggle) return;
        const isHidden = section.style.display === 'none';
        section.style.display = isHidden ? 'block' : 'none';
        toggle.textContent = isHidden ? this.t('common.collapse') : this.t('common.expand');
    },

    fetchKernelStats(kernelId = null) {
        const activeKernelId = kernelId || this.getActiveKernelId();
        if (!activeKernelId) return;
        fetch(`/api/kernel/stats?kernel_id=${encodeURIComponent(activeKernelId)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                return;
            }
            if (activeKernelId !== this.getActiveKernelId()) {
                return;
            }
            const mem = document.getElementById('kernel-memory-value');
            if (mem) {
                mem.textContent = data.memory_mb ? `${data.memory_mb.toFixed(1)} MB` : '--';
            }
            const list = document.getElementById('kernel-var-list');
            if (!list) return;
            list.innerHTML = '';
            const vars = data.vars || [];
            if (!vars.length) {
                const li = document.createElement('li');
                li.className = 'kernel-var-item';
                li.textContent = this.t('common.noData');
                list.appendChild(li);
                return;
            }
            const max = Math.max(...vars.map(v => v.size_mb || 0), 1);
            vars.forEach(v => {
                const li = document.createElement('li');
                li.className = 'kernel-var-item';
                const label = document.createElement('span');
                label.textContent = v.name;
                const bar = document.createElement('div');
                bar.className = 'kernel-var-bar';
                const fill = document.createElement('span');
                fill.style.width = `${Math.round((v.size_mb / max) * 100)}%`;
                bar.appendChild(fill);
                const size = document.createElement('span');
                size.textContent = `${v.size_mb.toFixed(1)} MB`;
                li.appendChild(label);
                li.appendChild(bar);
                li.appendChild(size);
                list.appendChild(li);
            });
        })
        .catch(() => {});
    },

    fetchKernelVars(kernelId = null) {
        const activeKernelId = kernelId || this.getActiveKernelId();
        if (!activeKernelId) return;
        fetch(`/api/kernel/vars?kernel_id=${encodeURIComponent(activeKernelId)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                return;
            }
            if (activeKernelId !== this.getActiveKernelId()) {
                return;
            }
            const table = document.getElementById('kernel-vars-table');
            if (!table) return;
            table.innerHTML = '';
            const vars = data.vars || [];
            if (!vars.length) {
                const row = document.createElement('tr');
                row.innerHTML = `<td colspan="3" class="text-muted">${this.t('common.noData')}</td>`;
                table.appendChild(row);
                return;
            }
            vars.forEach(v => {
                const row = document.createElement('tr');
                row.className = `var-row ${v.is_child ? 'var-child' : ''}`;
                const name = document.createElement('td');
                name.textContent = v.name;
                const type = document.createElement('td');
                type.textContent = v.type;
                const preview = document.createElement('td');
                preview.className = 'var-preview';
                preview.textContent = v.preview || '';
                preview.title = v.preview || '';
                row.appendChild(name);
                row.appendChild(type);
                row.appendChild(preview);
                row.addEventListener('click', () => this.openVarTab(v.name));
                table.appendChild(row);
            });
        })
        .catch(() => {});
    },

    openVarTab(name) {
        if (!name) return;
        const kernelId = this.getActiveKernelId();
        if (!kernelId) return;
        const dfLimits = this.getDataFramePreviewLimits ? this.getDataFramePreviewLimits() : { rows: 50, cols: 20 };
        const existing = this.openTabs.find(t => t.type === 'var' && t.name === name && t.kernelId === kernelId);
        if (existing) {
            this.setActiveTab(existing.id);
            return;
        }
        fetch(`/api/kernel/var_detail?name=${encodeURIComponent(name)}&kernel_id=${encodeURIComponent(kernelId)}&df_max_rows=${encodeURIComponent(dfLimits.rows)}&df_max_cols=${encodeURIComponent(dfLimits.cols)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.openFailed')}: ${data.error}`);
                return;
            }
            const id = `tab-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
            this.openTabs.push({
                id,
                name: data.name || name,
                path: data.name || name,
                type: 'var',
                detail: data,
                kernelId
            });
            this.setActiveTab(id);
        })
        .catch(error => {
            alert(`${this.t('status.openFailed')}: ${error.message}`);
        });
    }

});
