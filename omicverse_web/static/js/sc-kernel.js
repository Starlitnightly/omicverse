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
            meta.textContent = `${detail.name} · ${this.t('var.dataframeLabel')} ${detail.shape[0]}x${detail.shape[1]} (${this.t('var.preview50')})`;
        } else if (detail.type === 'anndata') {
            meta.textContent = `${detail.name} · ${this.t('var.anndataLabel')} ${detail.summary.shape.join('x')}`;
        } else {
            meta.textContent = `${detail.name}`;
        }
        varView.appendChild(meta);

        if (detail.type === 'dataframe' && detail.table) {
            const table = document.createElement('table');
            table.className = 'var-table-view';
            const thead = document.createElement('thead');
            const headRow = document.createElement('tr');
            const corner = document.createElement('th');
            corner.textContent = '';
            headRow.appendChild(corner);
            detail.table.columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headRow.appendChild(th);
            });
            thead.appendChild(headRow);
            table.appendChild(thead);
            const tbody = document.createElement('tbody');
            detail.table.data.forEach((row, idx) => {
                const tr = document.createElement('tr');
                const idxCell = document.createElement('td');
                idxCell.textContent = detail.table.index[idx];
                tr.appendChild(idxCell);
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = cell;
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            varView.appendChild(table);
            return;
        }

        if (detail.type === 'anndata') {
            const list = document.createElement('div');
            list.className = 'var-detail-meta';
            list.innerHTML = `
                obs columns: ${detail.summary.obs_columns.length}<br/>
                var columns: ${detail.summary.var_columns.length}<br/>
                obsm: ${(detail.summary.obsm_keys || []).join(', ') || '—'}<br/>
                layers: ${(detail.summary.layers || []).join(', ') || '—'}
            `;
            varView.appendChild(list);

            // Add "Load to Visualization" button
            const loadBtn = document.createElement('button');
            loadBtn.className = 'btn btn-sm btn-primary mt-3';
            loadBtn.innerHTML = `<i class="feather-eye"></i> ${this.t('var.loadToViz')}`;
            loadBtn.onclick = () => {
                console.log('=== Load to Visualization button CLICKED ===');
                console.log('Variable name:', detail.name);
                this.loadAnndataToVisualization(detail.name);
            };
            varView.appendChild(loadBtn);
            console.log('Load to Visualization button created for:', detail.name);
            return;
        }

        const pre = document.createElement('pre');
        pre.className = 'code-output-text';
        pre.textContent = detail.content || '';
        varView.appendChild(pre);
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
        const existing = this.openTabs.find(t => t.type === 'var' && t.name === name && t.kernelId === kernelId);
        if (existing) {
            this.setActiveTab(existing.id);
            return;
        }
        fetch(`/api/kernel/var_detail?name=${encodeURIComponent(name)}&kernel_id=${encodeURIComponent(kernelId)}`)
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
