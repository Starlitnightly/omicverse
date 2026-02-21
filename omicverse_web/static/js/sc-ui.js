/**
 * OmicVerse Single Cell Analysis — UI Components, Navigation & Status
 */

Object.assign(SingleCellAnalysis.prototype, {

    setupNavigation() {
        // Setup navigation menu toggle functionality
        const navItems = document.querySelectorAll('.nxl-item.nxl-hasmenu');
        
        navItems.forEach(item => {
            const link = item.querySelector('.nxl-link');
            const submenu = item.querySelector('.nxl-submenu');
            
            if (link && submenu) {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.toggleSubmenu(item);
                });
            }
        });

        // Mobile menu toggle
        const mobileToggle = document.getElementById('mobile-collapse');

        if (mobileToggle) {
            mobileToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
        }
    },

    setupSidebarResize() {
        // JupyterLab-like resizable sidebar using CSS variables
        const handle = document.getElementById('sidebar-resize-handle');
        const sidebar = document.querySelector('.nxl-navigation');

        if (!handle || !sidebar) return;

        let isResizing = false;
        let startX = 0;
        let startWidth = 0;
        const minWidth = 200;  // Minimum sidebar width
        const maxWidth = 600;  // Maximum sidebar width

        // Function to update CSS variable for sidebar width
        const setSidebarWidth = (width) => {
            document.documentElement.style.setProperty('--sidebar-width', width + 'px');
        };

        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;

            // Get current width from CSS variable
            const currentWidth = getComputedStyle(document.documentElement)
                .getPropertyValue('--sidebar-width');
            startWidth = parseInt(currentWidth);

            // Add visual feedback
            handle.classList.add('resizing');
            document.body.classList.add('resizing-sidebar');

            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;

            const delta = e.clientX - startX;
            const newWidth = Math.min(Math.max(startWidth + delta, minWidth), maxWidth);

            // Update CSS variable - this updates all elements using var(--sidebar-width)
            setSidebarWidth(newWidth);

            e.preventDefault();
        });

        document.addEventListener('mouseup', () => {
            if (!isResizing) return;

            isResizing = false;
            handle.classList.remove('resizing');
            document.body.classList.remove('resizing-sidebar');

            // Save the width to localStorage
            const currentWidth = getComputedStyle(document.documentElement)
                .getPropertyValue('--sidebar-width');
            localStorage.setItem('omicverse.sidebarWidth', parseInt(currentWidth));
        });

        // Restore saved width on load
        const savedWidth = localStorage.getItem('omicverse.sidebarWidth');
        if (savedWidth) {
            const width = parseInt(savedWidth);
            if (width >= minWidth && width <= maxWidth) {
                setSidebarWidth(width);
            }
        }
    },

    setupNotebookManager() {
        const fileInput = document.getElementById('notebook-file-input');
        if (!fileInput) return;
        fileInput.addEventListener('change', (e) => {
            const files = e.target.files;
            if (!files || files.length === 0) return;
            this.importNotebookFile(files[0]);
            fileInput.value = '';
        });
        this.fetchFileTree();
        this.fetchKernelStats();
        this.fetchKernelVars();
        document.addEventListener('click', () => this.hideContextMenu());
    },

    setupThemeToggle() {
        // Setup click handlers for existing theme toggle buttons
        const themeToggle = document.getElementById('theme-toggle');

        if (themeToggle) {
            themeToggle.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleTheme();
            });
        }
    },

    setupGeneAutocomplete() {
        const geneInput = document.getElementById('gene-input');
        const autocompleteDiv = document.getElementById('gene-autocomplete');

        if (!geneInput || !autocompleteDiv) return;

        let geneList = [];
        let selectedIndex = -1;

        // Fetch gene list when data is loaded
        const fetchGeneList = () => {
            if (!this.currentData) return;

            fetch('/api/genes')
                .then(response => response.json())
                .then(data => {
                    if (data.genes) {
                        geneList = data.genes;
                    }
                })
                .catch(error => {
                    console.log('Failed to fetch gene list:', error);
                });
        };

        // Input event listener
        geneInput.addEventListener('input', (e) => {
            const value = e.target.value.trim().toLowerCase();
            selectedIndex = -1;

            if (value.length < 1) {
                autocompleteDiv.style.display = 'none';
                return;
            }

            if (geneList.length === 0) {
                fetchGeneList();
                return;
            }

            // Filter genes
            const matches = geneList.filter(gene =>
                gene.toLowerCase().includes(value)
            ).slice(0, 20); // Limit to 20 results

            if (matches.length === 0) {
                autocompleteDiv.style.display = 'none';
                return;
            }

            // Display matches
            autocompleteDiv.innerHTML = matches.map((gene, index) =>
                `<button type="button" class="list-group-item list-group-item-action" data-index="${index}" data-gene="${gene}">
                    ${gene}
                </button>`
            ).join('');
            autocompleteDiv.style.display = 'block';

            // Add click handlers
            autocompleteDiv.querySelectorAll('button').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const gene = e.currentTarget.getAttribute('data-gene');
                    geneInput.value = gene;
                    autocompleteDiv.style.display = 'none';
                    this.colorByGene();
                });
            });
        });

        // Keyboard navigation
        geneInput.addEventListener('keydown', (e) => {
            const items = autocompleteDiv.querySelectorAll('button');

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                updateSelection(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, -1);
                updateSelection(items);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (selectedIndex >= 0 && items[selectedIndex]) {
                    const gene = items[selectedIndex].getAttribute('data-gene');
                    geneInput.value = gene;
                    autocompleteDiv.style.display = 'none';
                    this.colorByGene();
                } else {
                    this.colorByGene();
                }
            } else if (e.key === 'Escape') {
                autocompleteDiv.style.display = 'none';
                selectedIndex = -1;
            }
        });

        const updateSelection = (items) => {
            items.forEach((item, index) => {
                if (index === selectedIndex) {
                    item.classList.add('active');
                    item.scrollIntoView({ block: 'nearest' });
                } else {
                    item.classList.remove('active');
                }
            });
        };

        // Close autocomplete when clicking outside
        document.addEventListener('click', (e) => {
            if (!geneInput.contains(e.target) && !autocompleteDiv.contains(e.target)) {
                autocompleteDiv.style.display = 'none';
                selectedIndex = -1;
            }
        });

        // Store reference for later use
        this.fetchGeneList = fetchGeneList;
    },

    setupBeforeUnloadWarning() {
        // Warn user before leaving/refreshing if data is loaded
        window.addEventListener('beforeunload', (e) => {
            if (this.currentData) {
                // Modern browsers require returnValue to be set
                e.preventDefault();
                // Chrome requires returnValue to be set
                e.returnValue = '';
                // Some browsers show a custom message (though most modern browsers ignore it)
                return this.t('status.beforeLeave');
            }
        });
    },

    toggleSubmenu(item) {
        const isOpen = item.classList.contains('open');
        
        // Close all other submenus
        document.querySelectorAll('.nxl-item.nxl-hasmenu.open').forEach(openItem => {
            if (openItem !== item) {
                openItem.classList.remove('open');
            }
        });
        
        // Toggle current submenu
        if (isOpen) {
            item.classList.remove('open');
        } else {
            item.classList.add('open');
        }
    },

    toggleMobileMenu() {
        const navigation = document.querySelector('.nxl-navigation');
        navigation.classList.toggle('open');
    },

    toggleTheme() {
        const html = document.documentElement;
        const icon = document.getElementById('theme-toggle-icon');
        
        if (html.classList.contains('app-skin-dark')) {
            // Switch to light mode
            html.classList.remove('app-skin-dark');
            localStorage.setItem('app-skin-dark', 'app-skin-light');
            this.currentTheme = 'light';
            if (icon) {
                icon.classList.remove('feather-sun');
                icon.classList.add('feather-moon');
            }
        } else {
            // Switch to dark mode
            html.classList.add('app-skin-dark');
            localStorage.setItem('app-skin-dark', 'app-skin-dark');
            this.currentTheme = 'dark';
            if (icon) {
                icon.classList.remove('feather-moon');
                icon.classList.add('feather-sun');
            }
        }
        
        // Update Plotly theme and status bar theme
        this.updatePlotlyTheme();
        this.updateStatusBarTheme();
    },

    updateUI(data) {
        // ── Reset all controls when switching datasets ───────────────────────
        const geneInput = document.getElementById('gene-input');
        if (geneInput) geneInput.value = '';

        const paletteSelect = document.getElementById('palette-select');
        if (paletteSelect) paletteSelect.value = 'default';

        const catPaletteSelect = document.getElementById('category-palette-select');
        if (catPaletteSelect) catPaletteSelect.value = 'default';

        // Clear any stale Plotly chart from the previous dataset
        const plotDiv = document.getElementById('plotly-div');
        if (plotDiv && typeof Plotly !== 'undefined') Plotly.purge(plotDiv);

        // Reset point-size slider to auto mode
        const sizeSlider = document.getElementById('point-size-slider');
        if (sizeSlider) sizeSlider.dataset.auto = 'true';

        // Hide palette visibility rows (will be re-evaluated after plot)
        this.updatePaletteVisibility('');

        // ── Hide upload section ──────────────────────────────────────────────
        document.getElementById('upload-section').style.display = 'none';

        // Show data status
        const statusDiv = document.getElementById('data-status');
        statusDiv.classList.remove('d-none');
        document.getElementById('filename-display').textContent = data.filename;
        document.getElementById('cell-count').textContent = data.n_cells;
        document.getElementById('gene-count').textContent = data.n_genes;

        // Show controls and visualization
        document.getElementById('viz-controls').style.display = 'block';
        document.getElementById('viz-panel').style.display = 'block';

        // Init collapsible cards that were inside hidden containers on page load
        requestAnimationFrame(() => {
            const vizCtrl = document.getElementById('viz-controls');
            if (vizCtrl) this.initCollapsibleCards(vizCtrl);
        });

        // Sync left-panel height to match data-status + viz-panel
        requestAnimationFrame(() => this.syncPanelHeight());

        // Initialise point-size slider to auto default for this dataset
        this.initPointSizeSlider();

        // Update embedding options
        // data.embeddings now contains actual obsm keys (e.g. 'X_umap', 'UMAP').
        // Use the full key as the option value so the backend can look it up
        // exactly; strip the leading 'X_' only for the human-readable label.
        const embeddingSelect = document.getElementById('embedding-select');
        embeddingSelect.innerHTML = `<option value="">${this.t('controls.embeddingPlaceholder')}</option>`;
        data.embeddings.forEach(emb => {
            const option = document.createElement('option');
            option.value = emb;
            option.textContent = (emb.startsWith('X_') ? emb.slice(2) : emb).toUpperCase();
            embeddingSelect.appendChild(option);
        });

        // Update color options
        const colorSelect = document.getElementById('color-select');
        colorSelect.innerHTML = `<option value="">${this.t('controls.colorNone')}</option>`;
        data.obs_columns.forEach(col => {
            const option = document.createElement('option');
            option.value = 'obs:' + col;
            option.textContent = col;
            colorSelect.appendChild(option);
        });

        // Update parameter panel to enable buttons
        this.updateParameterPanel();
        if (this.fetchGeneList) {
            this.fetchGeneList();
        }

        // Fetch gene list for autocomplete
        if (this.fetchGeneList) {
            this.fetchGeneList();
        }

        // Reset parameter panel back to tool-list view (clear any open tool form)
        if (this.currentCategory) {
            this.selectAnalysisCategory(this.currentCategory, { silent: true });
        } else {
            this.showParameterPlaceholder();
        }

        // Auto-select first embedding and update plot
        if (data.embeddings.length > 0) {
            embeddingSelect.value = data.embeddings[0];
            this.updatePlot();
        }
    },

    refreshDataFromKernel(data) {
        if (!data) return;
        this.currentData = data;
        const statusDiv = document.getElementById('data-status');
        if (statusDiv) statusDiv.classList.remove('d-none');
        const filenameDisplay = document.getElementById('filename-display');
        if (filenameDisplay) filenameDisplay.textContent = data.filename || '';
        const cellCount = document.getElementById('cell-count');
        if (cellCount) cellCount.textContent = data.n_cells;
        const geneCount = document.getElementById('gene-count');
        if (geneCount) geneCount.textContent = data.n_genes;

        const embeddingSelect = document.getElementById('embedding-select');
        const colorSelect = document.getElementById('color-select');
        const prevEmbedding = embeddingSelect ? embeddingSelect.value : '';
        const prevColor = colorSelect ? colorSelect.value : '';

        if (embeddingSelect) {
            embeddingSelect.innerHTML = `<option value="">${this.t('controls.embeddingPlaceholder')}</option>`;
            data.embeddings.forEach(emb => {
                const option = document.createElement('option');
                option.value = emb;
                option.textContent = (emb.startsWith('X_') ? emb.slice(2) : emb).toUpperCase();
                embeddingSelect.appendChild(option);
            });
            if (data.embeddings.includes(prevEmbedding)) {
                embeddingSelect.value = prevEmbedding;
            } else if (data.embeddings.length > 0) {
                embeddingSelect.value = data.embeddings[0];
            }
        }

        if (colorSelect) {
            colorSelect.innerHTML = `<option value="">${this.t('controls.colorNone')}</option>`;
            data.obs_columns.forEach(col => {
                const option = document.createElement('option');
                option.value = 'obs:' + col;
                option.textContent = col;
                colorSelect.appendChild(option);
            });
            if (prevColor && prevColor.startsWith('obs:')) {
                const rawCol = prevColor.replace('obs:', '');
                if (data.obs_columns.includes(rawCol)) {
                    colorSelect.value = prevColor;
                }
            }
        }

        this.updateParameterPanel();

        if (embeddingSelect && embeddingSelect.value) {
            if (this.currentView === 'visualization') {
                this.updatePlot();
                this.pendingPlotRefresh = false;
            } else {
                this.pendingPlotRefresh = true;
            }
        }

        // Keep adata status panel in sync
        this.updateAdataStatus(data);
    },

    updateParameterPanel() {
        // Re-enable all parameter buttons now that data is loaded
        const buttons = document.querySelectorAll('#parameter-content button');
        buttons.forEach(button => {
            if (!button.onclick || !button.onclick.toString().includes('showComingSoon')) {
                button.disabled = false;
            }
        });
    },

    updatePlotlyTheme() {
        // If there's an existing plot, update it with new theme
        const plotDiv = document.getElementById('plotly-div');
        if (plotDiv && plotDiv.data) {
            const layout = this.getPlotlyLayout();
            Plotly.relayout(plotDiv, layout);
        }
    },

    syncPanelHeight() {
        const leftMain    = document.getElementById('left-main-panel');
        const dataStatus  = document.getElementById('data-status');
        const vizPanel    = document.getElementById('viz-panel');
        const adataCard   = document.getElementById('adata-status-section');
        if (!leftMain || !vizPanel) return;

        // If adata-status is collapsed (flex shrunken to auto), let the left
        // panel shrink naturally so the Analysis-Status card moves up and no
        // blank space remains at the bottom.
        const adataCollapsed = adataCard &&
            adataCard.querySelector('.card-collapse-btn.collapsed') !== null;

        if (adataCollapsed) {
            leftMain.style.minHeight = '';
            return;
        }

        const dsH = (dataStatus && !dataStatus.classList.contains('d-none'))
            ? dataStatus.offsetHeight : 0;
        const vpH = vizPanel.offsetHeight;
        leftMain.style.minHeight = (dsH + vpH > 0) ? (dsH + vpH) + 'px' : '';
    },

    checkStatus() {
        fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            if (!data.loaded) return;
            // Guard: ensure the response has the fields updateUI needs
            if (!Array.isArray(data.embeddings)) {
                console.warn('checkStatus: /api/status response missing embeddings field', data);
                return;
            }
            // Restore preview mode flag from server state
            this.isPreviewMode = !!data.preview_mode;
            // Server has adata in memory — restore full UI without re-uploading
            this.currentData = data;
            this.updateUI(data);
            this.updateAdataStatus(data);
            this.updatePreviewModeBanner(this.isPreviewMode);
            requestAnimationFrame(() => this.syncPanelHeight());
            this.addToLog(
                `${this.t('upload.successDetail')}: ${data.n_cells} ${this.t('status.cells')}, ${data.n_genes} ${this.t('status.genes')}`
            );
        })
        .catch(err => { console.warn('checkStatus error:', err); });
    },

    showLoading(text = null) {
        const loadingText = document.getElementById('loading-text');
        const loadingOverlay = document.getElementById('loading-overlay');

        const resolved = text || this.t('loading.processing');
        if (loadingText) loadingText.textContent = resolved;
        if (loadingOverlay) loadingOverlay.style.display = 'flex';
    },

    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) loadingOverlay.style.display = 'none';
    },

    addToLog(message, type = 'info') {
        const log = document.getElementById('analysis-log');
        if (!log) return;

        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');

        if (type === 'error') {
            logEntry.className = 'mb-1 text-danger';
            logEntry.innerHTML = `<small class="text-muted">[${timestamp}]</small> ${message}`;
        } else if (type === 'stdout') {
            // Render captured Python print output as monospace terminal block
            logEntry.className = 'mb-1';
            const pre = document.createElement('pre');
            pre.style.cssText = [
                'font-size:0.75rem',
                'margin:2px 0 2px 0',
                'padding:4px 8px',
                'background:var(--bs-light, #f8f9fa)',
                'border-left:3px solid #6c757d',
                'border-radius:0 4px 4px 0',
                'white-space:pre-wrap',
                'word-break:break-all',
                'color:#495057'
            ].join(';');
            pre.textContent = message;
            logEntry.appendChild(pre);
        } else {
            logEntry.className = 'mb-1 text-dark';
            logEntry.innerHTML = `<small class="text-muted">[${timestamp}]</small> ${message}`;
        }

        log.appendChild(logEntry);
        log.scrollTop = log.scrollHeight;
    },

    showStatus(message, showSpinner = false) {
        const statusBar = document.getElementById('status-bar');
        const statusText = document.getElementById('status-text');
        const statusSpinner = document.getElementById('status-spinner');
        const statusTime = document.getElementById('status-time');
        
        if (!statusBar || !statusText || !statusSpinner || !statusTime) return;
        
        // 应用主题样式
        this.updateStatusBarTheme();
        
        statusText.textContent = message;
        statusTime.textContent = new Date().toLocaleTimeString();
        
        if (showSpinner) {
            statusSpinner.style.display = 'inline-block';
        } else {
            statusSpinner.style.display = 'none';
        }
        
        statusBar.style.display = 'block';
    },

    hideStatus() {
        const statusBar = document.getElementById('status-bar');
        if (statusBar) {
            statusBar.style.display = 'none';
        }
    },

    updateStatus(message, showSpinner = false) {
        const statusText = document.getElementById('status-text');
        const statusSpinner = document.getElementById('status-spinner');
        const statusTime = document.getElementById('status-time');
        
        if (!statusText || !statusSpinner || !statusTime) return;
        
        statusText.textContent = message;
        statusTime.textContent = new Date().toLocaleTimeString();
        
        if (showSpinner) {
            statusSpinner.style.display = 'inline-block';
        } else {
            statusSpinner.style.display = 'none';
        }
    },

    updateStatusBarTheme() {
        const statusBar = document.getElementById('status-bar');
        const statusText = document.getElementById('status-text');
        const statusTime = document.getElementById('status-time');
        
        if (!statusBar || !statusText || !statusTime) return;
        
        const isDark = document.documentElement.classList.contains('app-skin-dark');
        
        if (isDark) {
            statusBar.style.backgroundColor = '#1f2937';
            statusBar.style.borderColor = '#374151';
            statusText.style.color = '#e5e7eb';
            statusTime.style.color = '#9ca3af';
        } else {
            statusBar.style.backgroundColor = '#ffffff';
            statusBar.style.borderColor = '#e5e7eb';
            statusText.style.color = '#283c50';
            statusTime.style.color = '#6b7280';
        }
    },

    showComingSoon() {
        alert(this.t('common.comingSoon'));
    },

    showPreviewModeAlert() {
        alert(this.t('preview.toolDisabledAlert') ||
            '⚠️ 预览模式下无法进行分析操作。\n\n如需分析，请点击数据状态栏中的「切换分析读取」按钮，\n以完整加载模式重新打开文件。');
    },

    switchView(view) {
        this.currentView = view;

        const vizView = document.getElementById('visualization-view');
        const codeView = document.getElementById('code-editor-view');
        const agentView = document.getElementById('agent-view');
        const vizBtn = document.getElementById('view-viz-btn');
        const codeBtn = document.getElementById('view-code-btn');
        const agentBtn = document.getElementById('view-agent-btn');
        const vizToolbar = document.getElementById('viz-toolbar');
        const codeToolbarRow = document.getElementById('code-editor-toolbar-row');
        const pageTitle = document.getElementById('page-title');
        const breadcrumbTitle = document.getElementById('breadcrumb-title');
        const analysisNav = document.getElementById('analysis-nav');
        const agentConfigNav = document.getElementById('agent-config-nav');
        const fileManager = document.getElementById('file-manager');

        if (view === 'visualization') {
            vizView.style.display = 'block';
            codeView.style.display = 'none';
            if (agentView) agentView.style.display = 'none';
            vizBtn.classList.remove('btn-outline-primary');
            vizBtn.classList.add('btn-primary');
            codeBtn.classList.remove('btn-primary');
            codeBtn.classList.add('btn-outline-primary');
            if (agentBtn) {
                agentBtn.classList.remove('btn-primary');
                agentBtn.classList.add('btn-outline-primary');
            }

            // Toggle toolbars
            if (vizToolbar) vizToolbar.style.display = 'flex';
            if (codeToolbarRow) codeToolbarRow.style.display = 'none';

            // Scroll to top when switching to visualization view (JupyterLab-like behavior)
            window.scrollTo({ top: 0, behavior: 'smooth' });

            // Update page title
            if (pageTitle) pageTitle.textContent = this.t('breadcrumb.title');
            if (breadcrumbTitle) breadcrumbTitle.textContent = this.t('breadcrumb.title');
            if (analysisNav) analysisNav.style.display = 'block';
            if (agentConfigNav) agentConfigNav.style.display = 'none';
            if (fileManager) fileManager.style.display = 'none';
            if (this.pendingPlotRefresh) {
                const embeddingSelect = document.getElementById('embedding-select');
                if (embeddingSelect && embeddingSelect.value) {
                    this.updatePlot();
                }
                this.pendingPlotRefresh = false;
            }
        } else if (view === 'code') {
            vizView.style.display = 'none';
            codeView.style.display = 'block';
            if (agentView) agentView.style.display = 'none';
            vizBtn.classList.remove('btn-primary');
            vizBtn.classList.add('btn-outline-primary');
            codeBtn.classList.remove('btn-outline-primary');
            codeBtn.classList.add('btn-primary');
            if (agentBtn) {
                agentBtn.classList.remove('btn-outline-primary');
                agentBtn.classList.add('btn-primary');
            }

            // Toggle toolbars
            if (vizToolbar) vizToolbar.style.display = 'none';
            if (codeToolbarRow) codeToolbarRow.style.display = 'flex';

            // Update page title
            if (pageTitle) pageTitle.innerHTML = `<i class="feather-code me-2"></i>${this.t('view.codeTitle')}`;
            if (breadcrumbTitle) breadcrumbTitle.textContent = this.t('breadcrumb.code');
            if (analysisNav) analysisNav.style.display = 'none';
            if (agentConfigNav) agentConfigNav.style.display = 'none';
            if (fileManager) fileManager.style.display = 'block';
            if (!this.fileTreeLoaded) {
                this.fetchFileTree();
                this.fileTreeLoaded = true;
            }
            this.fetchKernelStats();
            this.fetchKernelVars();
            // Ensure visualization adata is synced to kernel as odata when entering code view
            if (this.currentData) {
                fetch('/api/kernel/sync_odata', { method: 'POST' }).catch(() => {});
            }
            if (this.openTabs.length === 0) {
                this.openFileFromServer('default.ipynb');
            }

            // Add a default cell if none exists
            if (this.codeCells.length === 0) {
                this.addCodeCell();
            }
        } else if (view === 'agent') {
            vizView.style.display = 'none';
            codeView.style.display = 'none';
            if (agentView) agentView.style.display = 'block';
            vizBtn.classList.remove('btn-primary');
            vizBtn.classList.add('btn-outline-primary');
            codeBtn.classList.remove('btn-primary');
            codeBtn.classList.add('btn-outline-primary');
            if (agentBtn) {
                agentBtn.classList.remove('btn-outline-primary');
                agentBtn.classList.add('btn-primary');
            }

            // Toggle toolbars
            if (vizToolbar) vizToolbar.style.display = 'none';
            if (codeToolbarRow) codeToolbarRow.style.display = 'none';

            // Update page title
            if (pageTitle) pageTitle.innerHTML = `<i class="feather-message-circle me-2"></i>${this.t('view.agentTitle')}`;
            if (breadcrumbTitle) breadcrumbTitle.textContent = this.t('breadcrumb.agent');
            if (analysisNav) analysisNav.style.display = 'none';
            if (agentConfigNav) agentConfigNav.style.display = 'block';
            if (fileManager) fileManager.style.display = 'none';
        }
    },

    applyCodeFontSize() {
        const size = `${this.codeFontSize}px`;
        document.querySelectorAll('.code-input').forEach(el => {
            el.style.fontSize = size;
        });
        document.querySelectorAll('.code-highlight').forEach(el => {
            el.style.fontSize = size;
        });
        document.querySelectorAll('.code-highlight code').forEach(el => {
            el.style.fontSize = size;
        });
        document.querySelectorAll('.code-cell-output').forEach(el => {
            el.style.fontSize = size;
        });
        const textEditor = document.getElementById('text-file-editor');
        if (textEditor) {
            textEditor.style.fontSize = size;
        }
    },

    adjustFontSize(delta) {
        const next = Math.min(20, Math.max(10, this.codeFontSize + delta));
        this.codeFontSize = next;
        this.applyCodeFontSize();
    },

    // ── Memory Bar ──────────────────────────────────────────────────────────

    startMemoryMonitor() {
        // Fetch immediately, then poll every 5 seconds
        this.updateMemoryBar();
        setInterval(() => this.updateMemoryBar(), 5000);
    },

    updateMemoryBar() {
        fetch('/api/memory')
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (!data) return;
                const processMb = data.process_mb;
                const totalMb   = data.total_mb;
                const usedMb    = data.used_mb;   // total system used (includes this process)

                const fmtGb = mb => mb != null ? (mb / 1024).toFixed(1) + ' GB' : '--';

                const barProcess = document.getElementById('memory-bar-process');
                const barOther   = document.getElementById('memory-bar-other');
                const barText    = document.getElementById('memory-bar-text');
                const lblProcess = document.getElementById('memory-label-process');
                const lblOther   = document.getElementById('memory-label-other');

                if (!barProcess || !barOther || !barText) return;

                const isDark = document.documentElement.classList.contains('app-skin-dark');
                const otherColor = isDark ? '#5a6172' : '#adb5bd';

                // Sync legend dot color with current theme
                const otherDot = document.getElementById('memory-other-dot');
                if (otherDot) otherDot.style.background = otherColor;
                if (barOther) barOther.style.background = otherColor;

                if (totalMb && processMb != null) {
                    // Green  = this process
                    // Gray   = rest of system used (usedMb - processMb)
                    const otherMb = Math.max(0, (usedMb || 0) - processMb);
                    const pPct = Math.min(100, (processMb / totalMb) * 100);
                    const oPct = Math.min(100 - pPct, (otherMb  / totalMb) * 100);

                    barProcess.style.width = pPct.toFixed(1) + '%';
                    barOther.style.left    = pPct.toFixed(1) + '%';
                    barOther.style.width   = oPct.toFixed(1) + '%';

                    // Top text: "已用 Y.Y GB / 总 Z.Z GB"
                    barText.textContent = fmtGb(usedMb) + ' / ' + fmtGb(totalMb);

                    // Bottom labels:
                    //   绿色 → 本程序占用 (e.g. "0.8 GB")
                    //   灰色 → 本机已用   (total used, e.g. "6.2 GB")
                    if (lblProcess) lblProcess.textContent = fmtGb(processMb);
                    if (lblOther)   lblOther.textContent   = fmtGb(usedMb);
                } else if (processMb != null) {
                    // No system total available — just show process usage
                    barProcess.style.width = '100%';
                    barOther.style.width   = '0%';
                    barText.textContent    = fmtGb(processMb);
                    if (lblProcess) lblProcess.textContent = fmtGb(processMb);
                    if (lblOther)   lblOther.textContent   = '--';
                }
            })
            .catch(() => {/* silent fail */});
    },

    // ── Collapsible Cards ─────────────────────────────────────────────────────
    /**
     * initCollapsibleCards([root])
     *
     * Scans every `.card` inside `root` (default: document) that has both a
     * `.card-header` and a `.card-body`, and injects a toggle button that lets
     * the user collapse / expand the card body.
     *
     * Cards opt-out with the  data-no-collapse  attribute.
     * State is persisted in sessionStorage so panels remember their position
     * after a soft refresh.
     *
     * Safe to call multiple times — already-initialised cards are skipped.
     */
    initCollapsibleCards(root) {
        root = root || document;
        root.querySelectorAll('.card').forEach((card, idx) => {
            // ── opt-out ────────────────────────────────────────────────────
            if ('noCollapse' in card.dataset) return;

            const header = card.querySelector(':scope > .card-header');
            const body   = card.querySelector(':scope > .card-body');
            if (!header || !body) return;

            // Already initialised?
            if (header.querySelector('.card-collapse-btn')) return;

            // ── stable unique ID for sessionStorage ────────────────────────
            // Walk up to find a real id, or assign a data-cc-id to the card.
            if (!card.dataset.ccId) {
                card.dataset.ccId = card.id || `cc-${Date.now()}-${idx}`;
            }
            const storageKey = `cc-${card.dataset.ccId}`;

            // ── inject toggle button ───────────────────────────────────────
            const btn = document.createElement('button');
            btn.type      = 'button';
            btn.className = 'card-collapse-btn';
            btn.setAttribute('aria-label', 'Toggle card');
            btn.innerHTML = '<span class="cc-chevron"><i class="fas fa-chevron-up"></i></span>';

            header.classList.add('cc-header');
            header.appendChild(btn);

            // ── prepare body for CSS transition ───────────────────────────
            body.classList.add('cc-animated');

            // ── save original flex value so we can restore it on expand ──
            // Cards with flex:1 (e.g. adata-status) must shrink when collapsed
            // so sibling cards below them can move up.
            const origFlex = card.style.flex || '';

            // ── toggle logic (shared by button and header click) ──────────
            const toggle = () => {
                const isCollapsed = btn.classList.contains('collapsed');
                if (isCollapsed) {
                    // Expand: restore flex first so the card can grow
                    card.style.flex          = origFlex;
                    body.style.paddingTop    = '';
                    body.style.paddingBottom = '';
                    body.style.maxHeight     = body.scrollHeight + 'px';
                    btn.classList.remove('collapsed');
                    sessionStorage.removeItem(storageKey);
                    // After animation, remove max-height so dynamic content can grow
                    body.addEventListener('transitionend', function onEnd() {
                        if (!btn.classList.contains('collapsed')) {
                            body.style.maxHeight = '';
                            requestAnimationFrame(() => this.syncPanelHeight());
                        }
                        body.removeEventListener('transitionend', onEnd);
                    }.bind(this));
                } else {
                    // Collapse: pin current height first, then animate to 0
                    body.style.maxHeight = body.scrollHeight + 'px';
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        body.style.maxHeight     = '0';
                        body.style.paddingTop    = '0';
                        body.style.paddingBottom = '0';
                    }));
                    btn.classList.add('collapsed');
                    sessionStorage.setItem(storageKey, '1');
                    // After animation: shrink card and re-sync panel height
                    body.addEventListener('transitionend', function onEnd() {
                        if (btn.classList.contains('collapsed')) {
                            card.style.flex = '0 0 auto';
                            // Let layout reflow, then recalculate min-height
                            requestAnimationFrame(() => this.syncPanelHeight());
                        }
                        body.removeEventListener('transitionend', onEnd);
                    }.bind(this));
                }
            };

            btn.addEventListener('click', e => { e.stopPropagation(); toggle(); });

            // Clicking the header itself (not interactive child elements) also toggles
            header.addEventListener('click', e => {
                if (e.target.closest('button:not(.card-collapse-btn), input, select, a, label')) return;
                toggle();
            });

            // ── restore persisted collapsed state ─────────────────────────
            // Do this AFTER wiring events so the card is fully set up.
            // If the card is inside a hidden container (scrollHeight === 0),
            // skip setting maxHeight – it will be initialised on first interaction.
            if (sessionStorage.getItem(storageKey) === '1') {
                body.style.maxHeight     = '0';
                body.style.paddingTop    = '0';
                body.style.paddingBottom = '0';
                btn.classList.add('collapsed');
                // Immediately shrink the card so siblings are not pushed down
                card.style.flex = '0 0 auto';
            }
        });
    },

});


