/**
 * OmicVerse Single Cell Analysis — Analysis Tools, Parameters & Trajectory
 */

Object.assign(SingleCellAnalysis.prototype, {

    selectAnalysisCategory(category, { silent = false } = {}) {
        this.currentCategory = category;
        const parameterContent = document.getElementById('parameter-content');
        
        // Clear previous content
        parameterContent.innerHTML = '';
        
        // Generate category-specific tools
        const categoryTools = {
            'preprocessing': [
                { id: 'normalize', nameKey: 'tools.normalize', icon: 'fas fa-balance-scale', descKey: 'tools.normalizeDesc' },
                { id: 'log1p', nameKey: 'tools.log1p', icon: 'fas fa-calculator', descKey: 'tools.log1pDesc' },
                { id: 'scale', nameKey: 'tools.scale', icon: 'fas fa-expand-arrows-alt', descKey: 'tools.scaleDesc' }
            ],
            'qc': [
                { id: 'filter_cells', nameKey: 'tools.filterCells', icon: 'fas fa-filter', descKey: 'tools.filterCellsDesc' },
                { id: 'filter_genes', nameKey: 'tools.filterGenes', icon: 'fas fa-tasks', descKey: 'tools.filterGenesDesc' },
                { id: 'filter_outliers', nameKey: 'tools.filterOutliers', icon: 'fas fa-exclamation-triangle', descKey: 'tools.filterOutliersDesc' },
                { id: 'doublets', nameKey: 'tools.doublets', icon: 'fas fa-user-times', descKey: 'tools.doubletsDesc' }
            ],
            'feature': [
                { id: 'hvg', nameKey: 'tools.hvg', icon: 'fas fa-dna', descKey: 'tools.hvgDesc' }
            ],
            'dimreduction': [
                { id: 'pca', nameKey: 'tools.pca', icon: 'fas fa-chart-line', descKey: 'tools.pcaDesc' },
                { id: 'neighbors', nameKey: 'tools.neighbors', icon: 'fas fa-network-wired', descKey: 'tools.neighborsDesc' },
                { id: 'umap', nameKey: 'tools.umap', icon: 'fas fa-map', descKey: 'tools.umapDesc' },
                { id: 'tsne', nameKey: 'tools.tsne', icon: 'fas fa-dot-circle', descKey: 'tools.tsneDesc' }
            ],
            'clustering': [
                { id: 'leiden', nameKey: 'tools.leiden', icon: 'fas fa-object-group', descKey: 'tools.leidenDesc' },
                { id: 'louvain', nameKey: 'tools.louvain', icon: 'fas fa-layer-group', descKey: 'tools.louvainDesc' }
            ],
            'omicverse': [
                { id: 'coming_soon', nameKey: 'tools.cellAnnotation', icon: 'fas fa-tag', descKey: 'tools.cellAnnotationDesc' },
                { id: 'coming_soon', nameKey: 'tools.trajectory', icon: 'fas fa-route', descKey: 'tools.trajectoryDesc' },
                { id: 'coming_soon', nameKey: 'tools.diff', icon: 'fas fa-not-equal', descKey: 'tools.diffDesc' },
                { id: 'coming_soon', nameKey: 'tools.enrichment', icon: 'fas fa-sitemap', descKey: 'tools.enrichmentDesc' }
            ],
            'cell_annotation': [
                { id: 'celltypist',   nameKey: 'tools.celltypist',   icon: 'fas fa-tag',    descKey: 'tools.celltypistDesc' },
                { id: 'gpt4celltype', nameKey: 'tools.gpt4celltype', icon: 'fas fa-robot',  descKey: 'tools.gpt4celltypeDesc' },
                { id: 'scsa',         nameKey: 'tools.scsa',         icon: 'fas fa-star',   descKey: 'tools.scsaDesc' }
            ],
            'trajectory': [
                { id: 'diffusion_map', nameKey: 'tools.diffusionmap', icon: 'fas fa-project-diagram', descKey: 'tools.diffusionmapDesc' },
                { id: 'slingshot',     nameKey: 'tools.slingshot',    icon: 'fas fa-route',           descKey: 'tools.slingshotDesc' },
                { id: 'palantir',      nameKey: 'tools.palantir',     icon: 'fas fa-compass',         descKey: 'tools.palantirDesc' },
                { id: 'sctour',        nameKey: 'tools.sctour',       icon: 'fas fa-brain',           descKey: 'tools.sctourDesc' }
            ],
            'deg_expression': [],
            'dct': []
        };
        
        const tools = categoryTools[category] || [];
        
        tools.forEach(tool => {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'mb-3 p-3 border rounded fade-in c-pointer';

            // In preview mode we overlay a lock badge on each tool
            const previewBadge = this.isPreviewMode
                ? `<span class="badge bg-warning text-dark ms-auto" style="font-size:0.65rem;"><i class="fas fa-lock me-1"></i>${this.t('preview.modeBadge') || '预览模式'}</span>`
                : '';

            toolDiv.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <i class="${tool.icon} me-2 ${this.isPreviewMode ? 'text-secondary' : 'text-primary'}"></i>
                    <strong>${this.t(tool.nameKey)}</strong>
                    ${previewBadge}
                </div>
                <p class="text-muted small mb-0">${this.t(tool.descKey)}</p>`;

            if (this.isPreviewMode) {
                // Disable all tools in preview mode — show tooltip on click
                toolDiv.style.opacity = 0.65;
                toolDiv.style.cursor = 'not-allowed';
                toolDiv.title = this.t('preview.toolDisabledTip') || '预览模式下无法进行分析，如需分析请切换成"分析读取"模式';
                toolDiv.onclick = () => {
                    this.showPreviewModeAlert();
                };
            } else if (tool.id === 'coming_soon') {
                toolDiv.onclick = () => this.showComingSoon();
            } else if (this.currentData) {
                toolDiv.onclick = () => this.renderParameterForm(tool.id, tool.nameKey, tool.descKey, category);
            } else {
                toolDiv.style.opacity = 0.6;
                toolDiv.title = this.t('status.uploadFirst');
            }
            parameterContent.appendChild(toolDiv);
        });
        
        // Show traj-viz-panel only for trajectory category, hide for all others
        const trajPanel = document.getElementById('traj-viz-panel');
        if (trajPanel) {
            if (category === 'trajectory') {
                trajPanel.style.display = '';
                this.updateTrajVizSelects();
            } else {
                trajPanel.style.display = 'none';
            }
        }

        // Show deg-viz-panel only for deg_expression category
        const degPanel = document.getElementById('deg-viz-panel');
        if (degPanel) {
            if (category === 'deg_expression') {
                degPanel.style.display = '';
                this.updateDegVizSelects();
                // Render parameter form in the left settings panel
                this.renderDegParamForm(parameterContent);
            } else {
                degPanel.style.display = 'none';
            }
        }

        // Show dct-viz-panel only for dct category
        const dctPanel = document.getElementById('dct-viz-panel');
        if (dctPanel) {
            if (category === 'dct') {
                dctPanel.style.display = '';
                // Render parameter form in the left settings panel
                this.renderDctParamForm(parameterContent);
            } else {
                dctPanel.style.display = 'none';
            }
        }

        if (!silent) this.addToLog(this.t('panel.categorySelected') + ` ${this.getCategoryName(category)}`);
    },

    getCategoryName(category) {
        const names = {
            'preprocessing': this.t('nav.preprocessing'),
            'dimreduction': this.t('nav.dimReductionSub'),
            'clustering': this.t('nav.clusteringSub'),
            'omicverse': this.t('nav.omicverse'),
            'cell_annotation': this.t('nav.cellAnnotation'),
            'trajectory': this.t('nav.trajectory'),
            'deg_expression': this.t('nav.deg'),
            'dct': this.t('nav.dct'),
        };
        return names[category] || category;
    },

    showParameterDialog(tool) { this.renderParameterForm(tool); },

    _restoreAndTrackParams(tool, container) {
        const cache = this.paramCache[tool] || {};
        container.querySelectorAll('input, select').forEach(el => {
            if (!el.id) return;
            // Restore cached value
            if (el.id in cache) {
                if (el.type === 'checkbox') el.checked = !!cache[el.id];
                else el.value = cache[el.id];
            }
            // Track future changes
            const save = () => {
                if (!this.paramCache[tool]) this.paramCache[tool] = {};
                this.paramCache[tool][el.id] = el.type === 'checkbox' ? el.checked : el.value;
            };
            el.addEventListener('input', save);
            el.addEventListener('change', save);
        });
    },

    renderParameterForm(tool, toolName = '', toolDesc = '') {
        // Annotation tools use a custom multi-step renderer
        if (tool === 'celltypist' || tool === 'gpt4celltype' || tool === 'scsa') {
            return this.renderAnnotationForm(tool, toolName, toolDesc);
        }
        this.currentTool = tool;
        const resolvedName = toolName && toolName.startsWith('tools.') ? this.t(toolName) : toolName;
        const resolvedDesc = toolDesc && toolDesc.startsWith('tools.') ? this.t(toolDesc) : toolDesc;
        this.currentToolLabelKey = toolName && toolName.startsWith('tools.') ? toolName : '';
        this.currentToolDescKey = toolDesc && toolDesc.startsWith('tools.') ? toolDesc : '';
        this.currentToolLabel = resolvedName;
        this.currentToolDesc = resolvedDesc;
        const parameterContent = document.getElementById('parameter-content');
        const toolNames = {
            'normalize': this.t('tools.normalize'),
            'scale': this.t('tools.scale'),
            'hvg': this.t('tools.hvg'),
            'pca': this.t('tools.pca'),
            'umap': this.t('tools.umap'),
            'tsne': this.t('tools.tsne'),
            'neighbors': this.t('tools.neighbors'),
            'leiden': this.t('tools.leiden'),
            'louvain': this.t('tools.louvain'),
            'log1p': this.t('tools.log1p')
        };
        const title = resolvedName || toolNames[tool] || this.t('panel.parameters');
        const desc = resolvedDesc || '';

        const formHTML = `
            <div class="mb-3">
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div>
                        <h6 class="mb-1"><i class="fas fa-sliders-h me-2 text-primary"></i>${title}</h6>
                        ${desc ? `<small class="text-muted">${desc}</small>` : ''}
                    </div>
                    <div class="d-flex gap-1">
                        <button class="btn btn-sm btn-outline-info" onclick="singleCellApp.showCodeModal('${tool}')"><i class="fas fa-code me-1"></i>${this.t('tools.codeRef')}</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.selectAnalysisCategory('${singleCellApp.currentCategory || 'preprocessing'}')">${this.t('tools.backToList')}</button>
                    </div>
                </div>
                <div class="border rounded p-3">
                    ${this.getParameterHTML(tool)}
                    <div class="d-grid mt-3">
                        <button class="btn btn-primary" id="inlineRunBtn">运行</button>
                    </div>
                </div>
            </div>`;
        parameterContent.innerHTML = this.translateFormHtml(formHTML);
        this._restoreAndTrackParams(tool, parameterContent);

        const runBtn = document.getElementById('inlineRunBtn');
        if (runBtn) {
            runBtn.onclick = () => {
                const params = {};
                const inputs = parameterContent.querySelectorAll('input, select');
                inputs.forEach(input => {
                    if (input.type === 'number') {
                        if (input.value !== '') params[input.id] = parseFloat(input.value);
                    } else if (input.type === 'checkbox') {
                        params[input.id] = input.checked;
                    } else {
                        if (input.value !== '') params[input.id] = input.value;
                    }
                });
                this.runTool(tool, params);
            };
        }

        // Auto-detect mt prefixes for filter_outliers
        if (tool === 'filter_outliers') {
            const mtInput = document.getElementById('mt_prefixes');
            if (mtInput) {
                fetch('/api/qc_prefixes')
                    .then(r => r.json())
                    .then(d => {
                        if (d && d.mt_prefixes && !mtInput.value) {
                            mtInput.value = d.mt_prefixes.join(',');
                        }
                    })
                    .catch(() => {});
            }
        }
    },

    renderAnnotationForm(tool, toolNameKey = '', toolDescKey = '') {
        this.currentTool = tool;
        this.currentToolLabelKey = toolNameKey;
        this.currentToolDescKey = toolDescKey;
        this.currentToolLabel = toolNameKey.startsWith('tools.') ? this.t(toolNameKey) : toolNameKey;
        this.currentToolDesc  = toolDescKey.startsWith('tools.')  ? this.t(toolDescKey)  : toolDescKey;

        const paramEl = document.getElementById('parameter-content');
        if (!paramEl) return;

        const backCat = this.currentCategory || 'cell_annotation';
        const obsOpts = (this.currentData?.obs_columns || [])
            .map(c => `<option value="${c}">${c}</option>`).join('');

        const header = `
            <div class="d-flex align-items-center justify-content-between mb-2">
                <div>
                    <h6 class="mb-1"><i class="fas fa-sliders-h me-2 text-primary"></i>${this.currentToolLabel}</h6>
                    <small class="text-muted">${this.currentToolDesc}</small>
                </div>
                <div class="d-flex gap-1">
                    <button class="btn btn-sm btn-outline-info" onclick="singleCellApp.showCodeModal('${tool}')"><i class="fas fa-code me-1"></i>${this.t('tools.codeRef')}</button>
                    <button class="btn btn-sm btn-outline-secondary"
                        onclick="singleCellApp.selectAnalysisCategory('${backCat}')">${this.t('tools.backToList')}</button>
                </div>
            </div>`;

        let body = '';
        if (tool === 'celltypist') {
            body = `
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">模型文件路径 <span class="text-muted">(本地 .pkl)</span></label>
                <input type="text" class="form-control form-control-sm" id="pkl_path"
                    placeholder="下载后自动填入，或手动输入路径">
            </div>
            <div class="mb-2">
                <button class="btn btn-sm btn-outline-secondary w-100" id="fetchModelsBtn">
                    <i class="fas fa-cloud-download-alt me-1"></i>从 CellTypist 获取模型列表
                </button>
                <div id="ct-model-list" class="mt-2" style="display:none">
                    <select class="form-select form-select-sm" id="celltypist_model_select">
                        <option value="">-- 选择模型 --</option>
                    </select>
                    <div id="ct-model-desc" class="text-muted small mt-1"></div>
                    <button class="btn btn-sm btn-primary mt-2 w-100" id="downloadModelBtn" disabled>
                        <i class="fas fa-download me-1"></i>下载选中模型
                    </button>
                    <div id="ct-dl-status" class="text-muted small mt-1"></div>
                </div>
            </div>
            <div class="d-grid mt-3">
                <button class="btn btn-success" id="annoRunBtn">
                    <i class="fas fa-play me-1"></i>运行注释
                </button>
            </div>`;
        } else if (tool === 'gpt4celltype') {
            body = `
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">聚类键</label>
                <select class="form-select form-select-sm" id="cluster_key">
                    ${obsOpts || '<option value="leiden">leiden</option>'}
                </select>
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">组织类型</label>
                <input type="text" class="form-control form-control-sm" id="tissuename"
                    placeholder="e.g. PBMC, Brain, Liver">
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">物种</label>
                <input type="text" class="form-control form-control-sm" id="speciename" value="human">
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">LLM 提供商</label>
                <select class="form-select form-select-sm" id="provider">
                    <option value="qwen">Qwen (通义千问)</option>
                    <option value="openai">OpenAI</option>
                    <option value="kimi">Kimi (Moonshot)</option>
                </select>
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">模型名称</label>
                <input type="text" class="form-control form-control-sm" id="model" value="qwen-plus">
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">API Key</label>
                <input type="password" class="form-control form-control-sm" id="api_key"
                    placeholder="留空则读取 AGI_API_KEY 环境变量">
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">Base URL <span class="text-muted">(可选)</span></label>
                <input type="text" class="form-control form-control-sm" id="base_url"
                    placeholder="自定义 OpenAI 兼容端点">
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">Top marker 基因数</label>
                <input type="number" class="form-control form-control-sm" id="topgenenumber"
                    value="10" min="3" max="50">
            </div>
            <div class="d-grid mt-3">
                <button class="btn btn-success" id="annoRunBtn">
                    <i class="fas fa-play me-1"></i>运行注释
                </button>
            </div>`;
        } else if (tool === 'scsa') {
            body = `
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">聚类键</label>
                <select class="form-select form-select-sm" id="cluster_key">
                    ${obsOpts || '<option value="leiden">leiden</option>'}
                </select>
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">SCSA 数据库路径</label>
                <div class="input-group input-group-sm">
                    <input type="text" class="form-control" id="db_path"
                        placeholder="留空将自动下载">
                    <button class="btn btn-outline-secondary" id="downloadDbBtn" type="button">
                        <i class="fas fa-download"></i> 下载
                    </button>
                </div>
                <div id="scsa-dl-status" class="text-muted small mt-1"></div>
            </div>
            <div class="row g-2 mb-2">
                <div class="col-6">
                    <label class="form-label small fw-semibold">倍数变化阈值</label>
                    <input type="number" class="form-control form-control-sm" id="foldchange"
                        value="1.5" min="0.5" max="10" step="0.5">
                </div>
                <div class="col-6">
                    <label class="form-label small fw-semibold">P 值阈值</label>
                    <input type="number" class="form-control form-control-sm" id="pvalue"
                        value="0.05" min="0.001" max="0.1" step="0.005">
                </div>
            </div>
            <div class="row g-2 mb-2">
                <div class="col-6">
                    <label class="form-label small fw-semibold">细胞类型</label>
                    <select class="form-select form-select-sm" id="celltype">
                        <option value="normal">Normal</option>
                        <option value="cancer">Cancer</option>
                    </select>
                </div>
                <div class="col-6">
                    <label class="form-label small fw-semibold">参考数据库</label>
                    <select class="form-select form-select-sm" id="target">
                        <option value="cellmarker">CellMarker</option>
                        <option value="panglaoDB">PanglaoDB</option>
                        <option value="cancersea">CancerSEA</option>
                    </select>
                </div>
            </div>
            <div class="parameter-input mb-2">
                <label class="form-label small fw-semibold">组织 <span class="text-muted">(留空=All)</span></label>
                <input type="text" class="form-control form-control-sm" id="tissue" value="All">
            </div>
            <div class="d-grid mt-3">
                <button class="btn btn-success" id="annoRunBtn">
                    <i class="fas fa-play me-1"></i>运行注释
                </button>
            </div>`;
        }

        paramEl.innerHTML = this.translateFormHtml(`<div class="mb-3">${header}<div class="border rounded p-3">${body}</div></div>`);

        // ── Attach events ──────────────────────────────────────────────────
        if (tool === 'celltypist') {
            const fetchBtn    = document.getElementById('fetchModelsBtn');
            const modelList   = document.getElementById('ct-model-list');
            const modelSelect = document.getElementById('celltypist_model_select');
            const dlBtn       = document.getElementById('downloadModelBtn');
            const dlStatus    = document.getElementById('ct-dl-status');
            const modelDesc   = document.getElementById('ct-model-desc');
            const pklInput    = document.getElementById('pkl_path');

            fetchBtn.addEventListener('click', () => {
                fetchBtn.disabled = true;
                fetchBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>获取中...';
                fetch('/api/annotation/celltypist_models')
                    .then(r => r.json())
                    .then(d => {
                        if (d.error) throw new Error(d.error);
                        modelSelect.innerHTML = '<option value="">-- 选择模型 --</option>' +
                            d.models.map(m => {
                                const label = `${m.model}  (${m.No_celltypes || '?'} 类型)`;
                                return `<option value="${m.model}" data-desc="${(m.description||'').replace(/"/g,"'")}">${label}</option>`;
                            }).join('');
                        modelList.style.display = 'block';
                        fetchBtn.innerHTML = '<i class="fas fa-check me-1"></i>模型列表已加载';
                    })
                    .catch(err => {
                        fetchBtn.disabled = false;
                        fetchBtn.innerHTML = '<i class="fas fa-cloud-download-alt me-1"></i>从 CellTypist 获取模型列表';
                        alert('获取模型列表失败: ' + err.message);
                    });
            });

            modelSelect.addEventListener('change', () => {
                const opt = modelSelect.options[modelSelect.selectedIndex];
                modelDesc.textContent = opt?.dataset.desc || '';
                dlBtn.disabled = !modelSelect.value;
                // Auto-fill path if already downloaded
                const modelName = modelSelect.value;
                if (modelName) {
                    fetch(`/api/annotation/celltypist_model_path?model_name=${encodeURIComponent(modelName)}`)
                        .then(r => r.json())
                        .then(d => {
                            if (d.exists) {
                                pklInput.value = d.path;
                                dlStatus.textContent = '✓ 已找到本地模型: ' + d.path;
                                dlBtn.innerHTML = '<i class="fas fa-check me-1"></i>已下载';
                            } else {
                                pklInput.value = '';
                                dlStatus.textContent = '';
                                dlBtn.disabled = false;
                                dlBtn.innerHTML = '<i class="fas fa-download me-1"></i>下载选中模型';
                            }
                        })
                        .catch(() => {});
                } else {
                    pklInput.value = '';
                    dlStatus.textContent = '';
                }
            });

            dlBtn.addEventListener('click', () => {
                const modelName = modelSelect.value;
                if (!modelName) return;
                dlBtn.disabled = true;
                dlBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>下载中...';
                dlStatus.textContent = '正在下载，请稍候...';
                fetch('/api/annotation/download_celltypist_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_name: modelName})
                })
                .then(r => r.json())
                .then(d => {
                    if (d.error) throw new Error(d.error);
                    pklInput.value = d.path;
                    dlStatus.textContent = '✓ 下载完成: ' + d.path;
                    dlBtn.innerHTML = '<i class="fas fa-check me-1"></i>已下载';
                })
                .catch(err => {
                    dlBtn.disabled = false;
                    dlBtn.innerHTML = '<i class="fas fa-download me-1"></i>下载选中模型';
                    dlStatus.textContent = '下载失败: ' + err.message;
                });
            });
        }

        if (tool === 'scsa') {
            const dlBtn    = document.getElementById('downloadDbBtn');
            const dlStatus = document.getElementById('scsa-dl-status');
            const dbInput  = document.getElementById('db_path');

            dlBtn.addEventListener('click', () => {
                dlBtn.disabled = true;
                dlBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                dlStatus.textContent = '正在下载 SCSA 数据库，请稍候（约 15 MB）...';
                fetch('/api/annotation/download_scsa_db', {method: 'POST'})
                    .then(r => r.json())
                    .then(d => {
                        if (d.error) throw new Error(d.error);
                        dbInput.value = d.path;
                        dlStatus.textContent = '✓ 下载完成: ' + d.path;
                        dlBtn.innerHTML = '<i class="fas fa-check"></i>';
                    })
                    .catch(err => {
                        dlBtn.disabled = false;
                        dlBtn.innerHTML = '<i class="fas fa-download"></i> 下载';
                        dlStatus.textContent = '下载失败: ' + err.message;
                    });
            });
        }

        // Common run button
        const runBtn = document.getElementById('annoRunBtn');
        if (runBtn) {
            runBtn.addEventListener('click', () => {
                const params = {};
                paramEl.querySelectorAll('input, select').forEach(el => {
                    if (!el.id) return;
                    if (el.type === 'number') {
                        if (el.value !== '') params[el.id] = parseFloat(el.value);
                    } else if (el.type === 'checkbox') {
                        params[el.id] = el.checked;
                    } else {
                        if (el.value !== '') params[el.id] = el.value;
                    }
                });
                this.runTool(tool, params);
            });
        }
    },

    getParameterHTML(tool) {
        const parameters = {
            'filter_cells': `
                <div class="row g-2">
                    <div class="col-12">
                        <label class="form-label">最小UMI数 (min_counts)</label>
                        <input type="number" class="form-control" id="min_counts" value="500" min="0" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最小基因数 (min_genes)</label>
                        <input type="number" class="form-control" id="min_genes" value="200" min="0" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最大UMI数 (max_counts)</label>
                        <input type="number" class="form-control" id="max_counts" placeholder="可留空" min="0" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最大基因数 (max_genes)</label>
                        <input type="number" class="form-control" id="max_genes" placeholder="可留空" min="0" step="1">
                    </div>
                </div>
                <small class="text-muted d-block mt-2">留空表示不限制。仅填写需要的阈值即可。</small>
            `,
            'filter_genes': `
                <div class="row g-2">
                    <div class="col-12">
                        <label class="form-label">最少表达细胞数 (min_cells)</label>
                        <input type="number" class="form-control" id="min_cells" value="3" min="0" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最小UMI数 (min_counts)</label>
                        <input type="number" class="form-control" id="g_min_counts" placeholder="可留空" min="0" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最多表达细胞数 (max_cells)</label>
                        <input type="number" class="form-control" id="max_cells" placeholder="可留空" min="0" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最大UMI数 (max_counts)</label>
                        <input type="number" class="form-control" id="g_max_counts" placeholder="可留空" min="0" step="1">
                    </div>
                </div>
                <small class="text-muted d-block mt-2">可选：设置上下限阈值，未填表示不限制。</small>
            `,
            'filter_outliers': `
                <div class="mb-2"><small class="text-muted">将先计算线粒体/核糖体/血红蛋白等QC指标，再按阈值过滤。</small></div>
                <div class="row g-2">
                    <div class="col-12">
                        <label class="form-label">最大线粒体比例 (百分比)</label>
                        <input type="number" class="form-control" id="max_mt_percent" value="20" min="0" max="100" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">线粒体基因前缀 (自动检测)</label>
                        <input type="text" class="form-control" id="mt_prefixes" placeholder="例如: MT-,mt-">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最大核糖体基因比例 (百分比)</label>
                        <input type="number" class="form-control" id="max_ribo_percent" placeholder="可留空" min="0" max="100" step="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">最大血红蛋白基因比例 (百分比)</label>
                        <input type="number" class="form-control" id="max_hb_percent" placeholder="可留空" min="0" max="100" step="1">
                    </div>
                </div>
                <small class="text-muted d-block mt-2">未填写的阈值不生效；线粒体前缀自动检测可编辑。</small>
            `,
            'doublets': (() => {
                const cols = (this.currentData && this.currentData.obs_columns) ? this.currentData.obs_columns : [];
                const opts = ['<option value="">无</option>'].concat(cols.map(c => `<option value="${c}">${c}</option>`)).join('');
                return `
                <div class="row g-2">
                    <div class="col-12">
                        <label class="form-label">批次列 (batch_key)</label>
                        <select class="form-select" id="batch_key">${opts}</select>
                    </div>
                    <div class="col-12">
                        <label class="form-label">模拟双细胞比 (sim_doublet_ratio)</label>
                        <input type="number" class="form-control" id="sim_doublet_ratio" value="2" step="0.1" min="0.1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">期望双细胞率 (expected_doublet_rate)</label>
                        <input type="number" class="form-control" id="expected_doublet_rate" value="0.05" step="0.01" min="0" max="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">双细胞率标准差 (stdev_doublet_rate)</label>
                        <input type="number" class="form-control" id="stdev_doublet_rate" value="0.02" step="0.01" min="0" max="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">UMI子采样 (synthetic_doublet_umi_subsampling)</label>
                        <input type="number" class="form-control" id="synthetic_doublet_umi_subsampling" value="1" step="0.05" min="0" max="1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">KNN距离度量 (knn_dist_metric)</label>
                        <input type="text" class="form-control" id="knn_dist_metric" value="euclidean">
                    </div>
                    <div class="col-12 form-check form-switch mt-2">
                        <input class="form-check-input" type="checkbox" id="normalize_variance" checked>
                        <label class="form-check-label" for="normalize_variance">normalize_variance</label>
                    </div>
                    <div class="col-12 form-check form-switch mt-2">
                        <input class="form-check-input" type="checkbox" id="log_transform">
                        <label class="form-check-label" for="log_transform">log_transform</label>
                    </div>
                    <div class="col-12 form-check form-switch mt-2">
                        <input class="form-check-input" type="checkbox" id="mean_center" checked>
                        <label class="form-check-label" for="mean_center">mean_center</label>
                    </div>
                    <div class="col-12">
                        <label class="form-label">PCA主成分数 (n_prin_comps)</label>
                        <input type="number" class="form-control" id="n_prin_comps" value="30" min="2" max="200">
                    </div>
                </div>`;
            })(),
            
            'normalize': `
                <div class="parameter-input">
                    <label>目标总数</label>
                    <input type="number" class="form-control" id="target_sum" value="10000" min="1000" max="100000">
                </div>
            `,
            'scale': `
                <div class="parameter-input">
                    <label>最大值</label>
                    <input type="number" class="form-control" id="max_value" value="10" min="1" max="100">
                </div>
            `,
            'hvg': `
                <div class="parameter-input">
                    <label>基因数量</label>
                    <input type="number" class="form-control" id="n_genes" value="2000" min="500" max="5000">
                </div>
                <div class="parameter-input">
                    <label>方法</label>
                    <select class="form-control" id="method">
                        <option value="seurat">Seurat</option>
                        <option value="cell_ranger">Cell Ranger</option>
                        <option value="seurat_v3">Seurat v3</option>
                    </select>
                </div>
            `,
            'pca': `
                <div class="parameter-input">
                    <label>主成分数量</label>
                    <input type="number" class="form-control" id="n_comps" value="50" min="10" max="100">
                </div>
            `,
            'umap': `
                <div class="parameter-input">
                    <label>邻居数量</label>
                    <input type="number" class="form-control" id="n_neighbors" value="15" min="5" max="50">
                </div>
                <div class="parameter-input">
                    <label>最小距离</label>
                    <input type="number" class="form-control" id="min_dist" value="0.5" min="0.1" max="1.0" step="0.1">
                </div>
            `,
            'tsne': `
                <div class="parameter-input">
                    <label>困惑度</label>
                    <input type="number" class="form-control" id="perplexity" value="30" min="5" max="100">
                </div>
            `,
            'neighbors': `
                <div class="parameter-input">
                    <label>邻居数量</label>
                    <input type="number" class="form-control" id="n_neighbors" value="15" min="5" max="50">
                </div>
            `,
            'leiden': `
                <div class="parameter-input">
                    <label>分辨率</label>
                    <input type="number" class="form-control" id="resolution" value="1.0" min="0.1" max="3.0" step="0.1">
                </div>
            `,
            'louvain': `
                <div class="parameter-input">
                    <label>分辨率</label>
                    <input type="number" class="form-control" id="resolution" value="1.0" min="0.1" max="3.0" step="0.1">
                </div>
            `,
        };

        // ── Trajectory tools (built dynamically from current obs columns) ──────
        const obsColumns = (this.currentData && this.currentData.obs_columns) ? this.currentData.obs_columns : [];
        const embeddingKeys = (this.currentData && this.currentData.embeddings) ? this.currentData.embeddings : [];
        const clusterOpts = ['<option value="">-- 选择列 --</option>']
            .concat(obsColumns.map(c => `<option value="${c}">${c}</option>`)).join('');
        const embeddingOpts = ['<option value="">-- 自动检测 --</option>',
            '<option value="X_pca">X_pca</option>']
            .concat(embeddingKeys.filter(k => k !== 'pca').map(k => `<option value="X_${k}">X_${k}</option>`))
            .join('');
        const pseudotimeOpts = ['<option value="">-- 不设置 --</option>']
            .concat(obsColumns.filter(c => c.includes('pseudotime') || c.includes('dpt') || c.includes('paga'))
                .map(c => `<option value="${c}">${c}</option>`)).join('');

        const trajCommonFields = `
            <div class="row g-2">
                <div class="col-12">
                    <label class="form-label">聚类列 (groupby)</label>
                    <select class="form-control" id="groupby">${clusterOpts}</select>
                </div>
                <div class="col-12">
                    <label class="form-label">低维表示 (use_rep)</label>
                    <select class="form-control" id="use_rep">${embeddingOpts}</select>
                </div>
                <div class="col-12">
                    <label class="form-label">主成分数 (n_comps)</label>
                    <input type="number" class="form-control" id="n_comps" value="50" min="10" max="200">
                </div>
                <div class="col-12">
                    <label class="form-label">起始细胞类型 (origin_cells)</label>
                    <input type="text" class="form-control" id="origin_cells" placeholder="例如: Ductal">
                </div>`;

        const trajParameterMap = {
            'diffusion_map': trajCommonFields + `
            </div>
            <small class="text-muted d-block mt-2">将在 obs 中添加 dpt_pseudotime 列，可用于可视化着色。</small>`,

            'slingshot': trajCommonFields + `
                <div class="col-12">
                    <label class="form-label">终止细胞类型 (terminal_cells，逗号分隔)</label>
                    <input type="text" class="form-control" id="terminal_cells" placeholder="例如: Alpha,Beta">
                </div>
                <div class="col-12">
                    <label class="form-label">训练轮数 (num_epochs)</label>
                    <input type="number" class="form-control" id="num_epochs" value="1" min="1" max="100">
                </div>
            </div>
            <small class="text-muted d-block mt-2">将在 obs 中添加 slingshot_pseudotime 列。</small>`,

            'palantir': trajCommonFields + `
                <div class="col-12">
                    <label class="form-label">终止细胞类型 (terminal_cells，逗号分隔)</label>
                    <input type="text" class="form-control" id="terminal_cells" placeholder="例如: Alpha,Beta,Delta">
                </div>
                <div class="col-12">
                    <label class="form-label">路标点数 (num_waypoints)</label>
                    <input type="number" class="form-control" id="num_waypoints" value="500" min="100" max="2000">
                </div>
            </div>
            <small class="text-muted d-block mt-2">将在 obs 中添加 palantir_pseudotime 和 palantir_entropy 列。</small>`,

            'paga': (() => {
                const basisOpts = ['<option value="umap">umap</option>',
                    '<option value="tsne">tsne</option>',
                    '<option value="draw_graph_fa">draw_graph_fa</option>']
                    .concat(embeddingKeys.filter(k => !['umap','tsne','draw_graph_fa'].includes(k))
                        .map(k => `<option value="${k}">${k}</option>`)).join('');
                return `
                <div class="row g-2">
                    <div class="col-12">
                        <label class="form-label">聚类列 (groups)</label>
                        <select class="form-control" id="groups">${clusterOpts}</select>
                    </div>
                    <div class="col-12">
                        <label class="form-label">拟时序列 (use_time_prior)</label>
                        <select class="form-control" id="use_time_prior">
                            <option value="">-- 不设置 --</option>
                            ${obsColumns.filter(c => c.includes('pseudotime') || c.includes('dpt') || c.includes('entropy'))
                                .map(c => `<option value="${c}">${c}</option>`).join('')}
                        </select>
                    </div>
                    <div class="col-12">
                        <label class="form-label">低维表示 (use_rep，用于重算邻居)</label>
                        <select class="form-control" id="use_rep">${embeddingOpts}</select>
                    </div>
                    <div class="col-12">
                        <label class="form-label">可视化嵌入 (basis)</label>
                        <select class="form-control" id="basis">${basisOpts}</select>
                    </div>
                </div>
                <small class="text-muted d-block mt-2">将生成 PAGA 图并在结果区域显示。</small>`;
            })(),

            'sctour': `
                <div class="row g-2">
                    <div class="col-12">
                        <label class="form-label">聚类列 (groupby)</label>
                        <select class="form-control" id="groupby">${clusterOpts}</select>
                    </div>
                    <div class="col-12">
                        <label class="form-label">低维表示 (use_rep)</label>
                        <select class="form-control" id="use_rep">${embeddingOpts}</select>
                    </div>
                    <div class="col-12">
                        <label class="form-label">主成分数 (n_comps)</label>
                        <input type="number" class="form-control" id="n_comps" value="50" min="10" max="200">
                    </div>
                    <div class="col-12">
                        <label class="form-label">lec 重建权重 (alpha_recon_lec)</label>
                        <input type="number" class="form-control" id="alpha_recon_lec" value="0.5" min="0" max="1" step="0.1">
                    </div>
                    <div class="col-12">
                        <label class="form-label">lode 重建权重 (alpha_recon_lode)</label>
                        <input type="number" class="form-control" id="alpha_recon_lode" value="0.5" min="0" max="1" step="0.1">
                    </div>
                </div>
                <small class="text-muted d-block mt-2">需要原始 count 数据在 adata.X。将在 obs 中添加 sctour_pseudotime 列。</small>`
        };

        if (tool in trajParameterMap) return trajParameterMap[tool];

        return parameters[tool] || `<p>${this.t('parameter.none')}</p>`;
    },

    formatToolMessage(toolName, suffix, error) {
        const spacer = this.currentLang === 'zh' ? '' : ' ';
        const base = `${toolName}${spacer}${suffix}`;
        return error ? `${base}: ${error}` : base;
    },

    runTool(tool, params = {}) {
        if (!this.currentData) {
            alert(this.t('status.uploadFirst'));
            return;
        }

        const toolNames = {
            'normalize': this.t('tools.normalize'),
            'log1p': this.t('tools.log1p'),
            'scale': this.t('tools.scale'),
            'hvg': this.t('tools.hvg'),
            'pca': this.t('tools.pca'),
            'umap': this.t('tools.umap'),
            'tsne': this.t('tools.tsne'),
            'neighbors': this.t('tools.neighbors'),
            'leiden': this.t('tools.leiden'),
            'louvain': this.t('tools.louvain'),
            'filter_cells':  this.t('tools.filterCells'),
            'filter_genes':  this.t('tools.filterGenes'),
            'filter_outliers': this.t('tools.filterOutliers'),
            'doublets':      this.t('tools.doublets'),
            'celltypist':    this.t('tools.celltypist'),
            'gpt4celltype':  this.t('tools.gpt4celltype'),
            'scsa':          this.t('tools.scsa'),
            'diffusion_map': this.t('tools.diffusionmap'),
            'slingshot':     this.t('tools.slingshot'),
            'palantir':      this.t('tools.palantir'),
            'paga':          this.t('tools.paga'),
            'sctour':        this.t('tools.sctour'),
        };
        const toolName = toolNames[tool] || tool;
        const runningText = this.currentLang === 'zh'
            ? `${this.t('tool.running')}${toolName}...`
            : `${this.t('tool.running')} ${toolName}...`;
        this.showStatus(runningText, true);
        this.addToLog(`${this.t('tool.start')}: ${toolName}`);

        fetch(`/api/tools/${tool}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog(this.formatToolMessage(toolName, this.t('tool.failed'), data.error), 'error');
                this.showStatus(this.formatToolMessage(toolName, this.t('tool.execFailed'), data.error), false);
                alert(this.formatToolMessage(toolName, this.t('tool.execFailed'), data.error));
            } else {
                this.currentData = data;
                // Use refreshDataFromKernel so the current embedding/color selection
                // is preserved instead of being reset to the first option.
                this.refreshDataFromKernel(data);
                this.addToLog(this.formatToolMessage(toolName, this.t('tool.completed')));
                this.updateAdataStatus(data, data.diff || null);
                requestAnimationFrame(() => this.syncPanelHeight());
                this.showStatus(this.formatToolMessage(toolName, this.t('tool.completed')), false);

                // Auto-color by predicted annotation column if returned
                if (data.predicted_col) {
                    const colorSelect = document.getElementById('color-select');
                    if (colorSelect) {
                        colorSelect.value = 'obs:' + data.predicted_col;
                        this.updatePlot();
                    }
                }

                // Auto-color by pseudotime column if returned by trajectory tools
                if (data.pseudotime_col) {
                    const colorSelect = document.getElementById('color-select');
                    if (colorSelect) {
                        colorSelect.value = 'obs:' + data.pseudotime_col;
                        this.updatePlot();
                    }
                    // Show trajectory visualization panel
                    this.showTrajViz();
                }

                // Display figures returned by tools (e.g., PAGA graph)
                if (data.figures && data.figures.length > 0) {
                    this.showToolFigures(data.figures, toolName);
                }
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(this.formatToolMessage(toolName, this.t('tool.failed'), error.message), 'error');
            this.showStatus(this.formatToolMessage(toolName, this.t('tool.execFailed'), error.message), false);
            alert(this.formatToolMessage(toolName, this.t('tool.execFailed'), error.message));
        });
    },

    showToolFigures(figures, toolName) {
        // Show tool-generated figures (e.g., PAGA graph) in a modal overlay
        const modalId = 'tool-figures-modal';
        let modal = document.getElementById(modalId);
        if (!modal) {
            modal = document.createElement('div');
            modal.id = modalId;
            modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:9999;display:flex;align-items:center;justify-content:center;';
            modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
            document.body.appendChild(modal);
        }
        modal.innerHTML = '';
        const box = document.createElement('div');
        box.style.cssText = 'background:#1e1e2e;border-radius:12px;padding:20px;max-width:90vw;max-height:90vh;overflow:auto;position:relative;';
        const header = document.createElement('div');
        header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;';
        header.innerHTML = `<h6 style="color:#cdd6f4;margin:0">${toolName}</h6>
            <button style="background:none;border:none;color:#cdd6f4;font-size:1.2rem;cursor:pointer;" onclick="document.getElementById('${modalId}').remove()">✕</button>`;
        box.appendChild(header);
        figures.forEach(fig => {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${fig}`;
            img.style.cssText = 'max-width:100%;border-radius:8px;display:block;margin-bottom:8px;';
            box.appendChild(img);
        });
        modal.appendChild(box);
    },

    showTrajViz() {
        const panel = document.getElementById('traj-viz-panel');
        if (panel) {
            panel.style.display = '';
            this.updateTrajVizSelects();
        }
    },

    toggleTrajViz() {
        const body = document.getElementById('traj-viz-body');
        const icon = document.getElementById('traj-viz-toggle-icon');
        if (!body) return;
        const collapsed = body.style.display === 'none';
        body.style.display = collapsed ? '' : 'none';
        if (icon) {
            icon.className = collapsed ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
        }
    },

    togglePagaOptions(checked) {
        const opts = document.getElementById('traj-paga-options');
        if (opts) opts.style.display = checked ? '' : 'none';
    },

    updateTrajVizSelects() {
        if (!this.currentData) return;
        const obs      = this.currentData.obs_columns  || [];
        const embeddings = this.currentData.embeddings || [];
        const layers   = this.currentData.layers       || [];

        // Pseudotime columns (for both embedding and heatmap)
        const ptCols = obs.filter(c =>
            c.includes('pseudotime') || c.includes('dpt_') || c === 'dpt_pseudotime'
        );
        const ptOpts = '<option value="">' + this.t('traj.autoDetect') + '</option>' +
            ptCols.map(c => `<option value="${c}">${c}</option>`).join('');
        for (const id of ['traj-pseudotime-col', 'traj-heatmap-pseudotime']) {
            const el = document.getElementById(id);
            if (el) el.innerHTML = ptOpts;
        }

        // Embedding bases
        const basisSel = document.getElementById('traj-basis');
        if (basisSel) {
            basisSel.innerHTML = embeddings
                .map(e => `<option value="X_${e}">${e.toUpperCase()}</option>`)
                .join('');
        }

        // PAGA groups (obs columns, prefer categorical)
        const grpSel = document.getElementById('traj-paga-groups');
        if (grpSel) {
            grpSel.innerHTML = `<option value="">-- ${this.t('traj.selectCol')} --</option>` +
                obs.map(c => `<option value="${c}">${c}</option>`).join('');
        }

        // Layer selector
        const layerSel = document.getElementById('traj-heatmap-layer');
        if (layerSel) {
            layerSel.innerHTML = '<option value="">X (default)</option>' +
                layers.map(l => `<option value="${l}">${l}</option>`).join('');
        }
    },

    generateTrajEmbedding() {
        const imgDiv = document.getElementById('traj-embedding-img');
        if (!imgDiv) return;
        imgDiv.innerHTML = '<div class="spinner-border spinner-border-sm text-primary"></div>';

        const params = {
            pseudotime_col:      (document.getElementById('traj-pseudotime-col')   || {}).value || '',
            basis:               (document.getElementById('traj-basis')             || {}).value || 'X_umap',
            cmap:                (document.getElementById('traj-cmap')              || {}).value || 'Reds',
            point_size:          (document.getElementById('traj-point-size')        || {}).value || 3,
            paga_overlay:        (document.getElementById('traj-paga-overlay')      || {}).checked || false,
            paga_groups:         (document.getElementById('traj-paga-groups')       || {}).value || '',
            paga_min_edge_width: (document.getElementById('traj-paga-min-edge')     || {}).value || 2,
            paga_node_size_scale:(document.getElementById('traj-paga-node-scale')   || {}).value || 1.5,
        };

        fetch('/api/trajectory/plot_embedding', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                imgDiv.innerHTML = `<span class="text-danger small p-2">${data.error}</span>`;
            } else {
                imgDiv.innerHTML =
                    `<img src="data:image/png;base64,${data.figure}"
                          style="max-width:100%;border-radius:6px;cursor:zoom-in;"
                          onclick="singleCellApp.showToolFigures(['${data.figure}'], 'Pseudotime Embedding')">`;
            }
        })
        .catch(e => {
            imgDiv.innerHTML = `<span class="text-danger small p-2">${e.message}</span>`;
        });
    },

    generateTrajHeatmap() {
        const imgDiv = document.getElementById('traj-heatmap-img');
        if (!imgDiv) return;
        imgDiv.innerHTML = '<div class="spinner-border spinner-border-sm text-warning"></div>';

        const params = {
            genes:          (document.getElementById('traj-heatmap-genes')      || {}).value || '',
            pseudotime_col: (document.getElementById('traj-heatmap-pseudotime') || {}).value || '',
            layer:          (document.getElementById('traj-heatmap-layer')      || {}).value || '',
            n_bins:         (document.getElementById('traj-heatmap-bins')       || {}).value || 50,
            cmap:           (document.getElementById('traj-heatmap-cmap')       || {}).value || 'RdBu_r',
        };

        fetch('/api/trajectory/plot_heatmap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                imgDiv.innerHTML = `<span class="text-danger small p-2">${data.error}</span>`;
            } else {
                imgDiv.innerHTML =
                    `<img src="data:image/png;base64,${data.figure}"
                          style="max-width:100%;border-radius:6px;cursor:zoom-in;"
                          onclick="singleCellApp.showToolFigures(['${data.figure}'], 'Gene Trend Heatmap')">`;
            }
        })
        .catch(e => {
            imgDiv.innerHTML = `<span class="text-danger small p-2">${e.message}</span>`;
        });
    },

    // -----------------------------------------------------------------------
    // DEG Analysis helpers
    // -----------------------------------------------------------------------

    /** Render DEG parameter form into the parameter-content panel. */
    renderDegParamForm(container) {
        // First level: method selection cards (like Leiden / Louvain)
        const methods = [
            { id: 'wilcoxon', label: 'Wilcoxon', icon: 'fas fa-chart-bar', descKey: 'deg.methodWilcoxonDesc' },
            { id: 't-test',   label: 'T-test',   icon: 'fas fa-flask',     descKey: 'deg.methodTtestDesc' },
        ];
        container.innerHTML = '';
        methods.forEach(m => {
            const card = document.createElement('div');
            card.className = 'mb-3 p-3 border rounded fade-in c-pointer';
            card.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <i class="${m.icon} me-2 text-primary"></i>
                    <strong>${m.label}</strong>
                </div>
                <p class="text-muted small mb-0">${this.t(m.descKey)}</p>`;
            if (this.currentData) {
                card.onclick = () => this.renderDegMethodForm(m.id, m.label, container);
            } else {
                card.style.opacity = 0.6;
                card.title = this.t('status.uploadFirst');
            }
            container.appendChild(card);
        });
    },

    renderDegMethodForm(method, methodLabel, container) {
        const obs     = this.currentData ? (this.currentData.obs_columns || []) : [];
        const colOpts = `<option value="">${this.t('deg.selectCol')}</option>` +
            obs.map(c => `<option value="${c}">${c}</option>`).join('');

        container.innerHTML = `
        <div class="d-flex align-items-center justify-content-between mb-2">
            <h6 class="mb-0 small fw-semibold">
                <i class="fas fa-not-equal me-1 text-primary"></i>
                DEG — ${methodLabel}
            </h6>
            <div class="d-flex gap-1">
                <button class="btn btn-sm btn-outline-info"
                        onclick="singleCellApp.showCodeModal('deg_expression')">
                    <i class="fas fa-code me-1"></i>${this.t('tools.codeRef')}
                </button>
                <button class="btn btn-sm btn-outline-secondary"
                        onclick="singleCellApp.renderDegParamForm(document.getElementById('parameter-content'))">
                    ${this.t('tools.backToList')}
                </button>
            </div>
        </div>
        <div class="row g-2">
            <input type="hidden" id="deg-method" value="${method}">
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('deg.conditionCol')}</label>
                <select class="form-select form-select-sm" id="deg-condition-col"
                        onchange="singleCellApp.onDegConditionChange(this.value)">
                    ${colOpts}
                </select>
            </div>
            <div class="col-6">
                <label class="form-label form-label-sm">${this.t('deg.ctrlGroup')}</label>
                <select class="form-select form-select-sm" id="deg-ctrl-group">
                    <option value="">${this.t('deg.selectCol')}</option>
                </select>
            </div>
            <div class="col-6">
                <label class="form-label form-label-sm">${this.t('deg.testGroup')}</label>
                <select class="form-select form-select-sm" id="deg-test-group">
                    <option value="">${this.t('deg.selectCol')}</option>
                </select>
            </div>
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('deg.celltypeKey')}</label>
                <select class="form-select form-select-sm" id="deg-celltype-key"
                        onchange="singleCellApp.onDegCelltypeChange(this.value)">
                    ${colOpts}
                </select>
            </div>
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('deg.celltypeGroup')}</label>
                <select class="form-select form-select-sm" id="deg-celltype-group"
                        multiple size="3" style="height:auto;">
                    <option value="" selected>${this.t('deg.allTypes')}</option>
                </select>
                <small class="text-muted d-block mt-1">${this.t('deg.celltypeHint')}</small>
            </div>
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('deg.maxCells')}</label>
                <input type="number" class="form-control form-control-sm"
                       id="deg-max-cells" value="100000" min="100" max="1000000" step="1000">
            </div>
            <div class="col-12 mt-1">
                <button class="btn btn-success btn-sm w-100" id="deg-run-btn"
                        onclick="singleCellApp.runDegAnalysis()">
                    <i class="fas fa-play me-1"></i>${this.t('deg.run')}
                </button>
            </div>
            <div id="deg-summary" class="col-12" style="display:none;">
                <div class="alert alert-success py-1 px-2 mb-0 small mt-1">
                    ${this.t('deg.total')} <strong id="deg-n-total">-</strong>
                    &nbsp;|&nbsp; <span class="text-danger">${this.t('deg.sigUp')}</span>
                    <strong id="deg-n-up">-</strong>
                    &nbsp;|&nbsp; <span class="text-primary">${this.t('deg.sigDown')}</span>
                    <strong id="deg-n-down">-</strong>
                </div>
            </div>
        </div>`;
        this._restoreAndTrackParams('deg_expression', container);
    },

    showDegViz() {
        const panel = document.getElementById('deg-viz-panel');
        if (panel) {
            panel.style.display = '';
            this.updateDegVizSelects();
        }
    },

    toggleDegViz() {
        const body = document.getElementById('deg-viz-body');
        const icon = document.getElementById('deg-viz-toggle-icon');
        if (!body) return;
        const collapsed = body.style.display === 'none';
        body.style.display = collapsed ? '' : 'none';
        if (icon) {
            icon.className = collapsed ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
        }
    },

    updateDegVizSelects() {
        if (!this.currentData) return;
        const obs    = this.currentData.obs_columns || [];
        const layers = this.currentData.layers      || [];
        const colOpts = `<option value="">${this.t('deg.selectCol')}</option>` +
            obs.map(c => `<option value="${c}">${c}</option>`).join('');

        // Violin groupby selector
        const groupbySel = document.getElementById('deg-violin-groupby');
        if (groupbySel) groupbySel.innerHTML = colOpts;

        // Violin layer selector
        const layerSel = document.getElementById('deg-violin-layer');
        if (layerSel) {
            layerSel.innerHTML = '<option value="">X (default)</option>' +
                layers.map(l => `<option value="${l}">${l}</option>`).join('');
        }
    },

    onDegConditionChange(col) {
        if (!col) return;
        fetch(`/api/deg/get_groups?col=${encodeURIComponent(col)}`)
        .then(r => r.json())
        .then(data => {
            const groups = data.groups || [];
            const opts = `<option value="">${this.t('deg.selectCol')}</option>` +
                groups.map(g => `<option value="${g}">${g}</option>`).join('');
            const ctrl = document.getElementById('deg-ctrl-group');
            const test = document.getElementById('deg-test-group');
            if (ctrl) ctrl.innerHTML = opts;
            if (test) test.innerHTML = opts;
            if (ctrl && groups.length >= 1) ctrl.value = groups[0];
            if (test && groups.length >= 2) test.value = groups[1];
            // Pre-fill violin groupby with condition column
            const groupbySel = document.getElementById('deg-violin-groupby');
            if (groupbySel) groupbySel.value = col;
        })
        .catch(() => {});
    },

    onDegCelltypeChange(col) {
        const selEl = document.getElementById('deg-celltype-group');
        if (!selEl) return;
        if (!col) {
            selEl.innerHTML = `<option value="" selected>${this.t('deg.allTypes')}</option>`;
            return;
        }
        fetch(`/api/deg/get_groups?col=${encodeURIComponent(col)}`)
        .then(r => r.json())
        .then(data => {
            const groups = data.groups || [];
            selEl.innerHTML =
                `<option value="" selected>${this.t('deg.allTypes')}</option>` +
                groups.map(g => `<option value="${g}">${g}</option>`).join('');
        })
        .catch(() => {});
    },

    runDegAnalysis() {
        const btn = document.getElementById('deg-run-btn');
        if (btn) btn.disabled = true;

        const ctSel = document.getElementById('deg-celltype-group');
        const celltypeGroup = ctSel
            ? Array.from(ctSel.selectedOptions).map(o => o.value).filter(v => v !== '')
            : [];

        const params = {
            condition:      (document.getElementById('deg-condition-col') || {}).value || '',
            ctrl_group:     (document.getElementById('deg-ctrl-group')    || {}).value || '',
            test_group:     (document.getElementById('deg-test-group')    || {}).value || '',
            celltype_key:   (document.getElementById('deg-celltype-key')  || {}).value || '',
            celltype_group: celltypeGroup,
            method:         (document.getElementById('deg-method')        || {}).value || 'wilcoxon',
            max_cells:      parseInt((document.getElementById('deg-max-cells') || {}).value || '100000'),
        };

        this.showStatus(this.t('deg.running'), true);
        this.addToLog(this.t('deg.running'));

        fetch('/api/deg/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        })
        .then(r => r.json())
        .then(data => {
            if (btn) btn.disabled = false;
            this.hideStatus();
            if (data.error) {
                this.addToLog(data.error, 'error');
                alert(data.error);
                return;
            }
            this.addToLog(this.t('deg.done'));
            this.showStatus(this.t('deg.done'), false);

            // Update summary in parameter panel
            const summaryDiv = document.getElementById('deg-summary');
            if (summaryDiv) {
                summaryDiv.style.display = '';
                const nTotal = document.getElementById('deg-n-total');
                const nUp    = document.getElementById('deg-n-up');
                const nDown  = document.getElementById('deg-n-down');
                if (nTotal) nTotal.textContent = data.n_total    || 0;
                if (nUp)    nUp.textContent    = data.n_sig_up   || 0;
                if (nDown)  nDown.textContent  = data.n_sig_down || 0;
            }

            // Pre-fill violin groupby with returned condition column
            if (data.condition) {
                const groupbySel = document.getElementById('deg-violin-groupby');
                if (groupbySel) groupbySel.value = data.condition;
            }

            // Store all results for client-side filtering
            this.degAllResults = data.all_results || [];

            // Reset filter sliders and render table
            this._resetDegFilters();
            this.filterDegResults();

            // Auto-generate volcano
            this.generateDegVolcano();
        })
        .catch(error => {
            if (btn) btn.disabled = false;
            this.hideStatus();
            this.addToLog(error.message, 'error');
            alert(error.message);
        });
    },

    _resetDegFilters() {
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.value = val;
        };
        const setTxt = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };
        setVal('deg-filter-fc',   0);   setTxt('deg-filter-fc-val',   '0.0');
        setVal('deg-filter-padj', 1);   setTxt('deg-filter-padj-val', '1.00');
        setVal('deg-filter-pct',  0);   setTxt('deg-filter-pct-val',  '0');
        const geneInput = document.getElementById('deg-filter-gene');
        if (geneInput) geneInput.value = '';
        this.setDegDir('all');
    },

    filterDegResults() {
        if (!this.degAllResults || !this.degAllResults.length) return;

        const fcMin   = parseFloat((document.getElementById('deg-filter-fc')   || {}).value || 0);
        const padjMax = parseFloat((document.getElementById('deg-filter-padj') || {}).value || 1);
        const pctMin  = parseFloat((document.getElementById('deg-filter-pct')  || {}).value || 0);
        const search  = ((document.getElementById('deg-filter-gene') || {}).value || '').toLowerCase();
        const activeBtn = document.querySelector('#deg-dir-group .active');
        const dir = activeBtn ? (activeBtn.dataset.dir || 'all') : 'all';

        const filtered = this.degAllResults.filter(row => {
            const fc      = parseFloat(row.log2FC   || 0);
            const padj    = parseFloat(row.padj     || 1);
            const pctCtrl = parseFloat(row.pct_ctrl || 0);
            const pctTest = parseFloat(row.pct_test || 0);
            const gene    = (row.gene || '').toLowerCase();

            if (Math.abs(fc) < fcMin) return false;
            if (padj > padjMax)       return false;
            if (Math.max(pctCtrl, pctTest) < pctMin) return false;
            if (dir === 'up'   && fc <= 0)  return false;
            if (dir === 'down' && fc >= 0)  return false;
            if (search && !gene.includes(search)) return false;
            return true;
        });

        const counter = document.getElementById('deg-filter-count');
        if (counter) counter.textContent = `${filtered.length} / ${this.degAllResults.length}`;

        this.renderDegTable(filtered.slice(0, 300));
    },

    renderDegTable(rows) {
        const tbody = document.getElementById('deg-results-tbody');
        if (!tbody) return;
        if (!rows.length) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted py-2">${this.t('deg.noResults')}</td></tr>`;
            return;
        }
        tbody.innerHTML = rows.map(row => {
            const fc    = parseFloat(row.log2FC   || 0);
            const padj  = parseFloat(row.padj     || 1);
            const pctC  = parseFloat(row.pct_ctrl || 0).toFixed(1);
            const pctT  = parseFloat(row.pct_test || 0).toFixed(1);
            const gene  = row.gene || '';
            const color = fc > 0 ? '#e06c75' : '#5ba4cf';
            const padjStr = padj < 0.001 ? padj.toExponential(2) : padj.toFixed(4);
            return `<tr style="cursor:pointer;" title="Click to add to violin plot"
                        onclick="singleCellApp.addGeneToViolin('${gene.replace(/'/g,"\\'")}')">
                <td style="color:${color};font-weight:600;">${gene}</td>
                <td style="color:${color};">${fc.toFixed(3)}</td>
                <td>${padjStr}</td>
                <td>${pctC}%</td>
                <td>${pctT}%</td>
            </tr>`;
        }).join('');
    },

    setDegDir(dir) {
        document.querySelectorAll('#deg-dir-group .btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.dir === dir);
        });
        this.filterDegResults();
    },

    addGeneToViolin(gene) {
        const el = document.getElementById('deg-violin-genes');
        if (!el) return;
        const existing = el.value.split(',').map(g => g.trim()).filter(Boolean);
        if (!existing.includes(gene)) {
            existing.push(gene);
            el.value = existing.join(', ');
        }
    },

    generateDegVolcano() {
        const imgDiv = document.getElementById('deg-volcano-img');
        if (!imgDiv) return;
        imgDiv.innerHTML = '<div class="spinner-border spinner-border-sm text-danger"></div>';

        const params = {
            fc_thresh:   parseFloat((document.getElementById('deg-fc-thresh')   || {}).value || '1'),
            padj_thresh: parseFloat((document.getElementById('deg-padj-thresh') || {}).value || '0.05'),
            point_size:  parseFloat((document.getElementById('deg-point-size')  || {}).value || '15'),
            label_top:   parseInt(  (document.getElementById('deg-label-top')   || {}).value || '10'),
        };

        fetch('/api/deg/plot_volcano', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                imgDiv.innerHTML = `<span class="text-danger small p-2">${data.error}</span>`;
            } else {
                imgDiv.innerHTML =
                    `<img src="data:image/png;base64,${data.figure}"
                          style="max-width:100%;border-radius:6px;cursor:zoom-in;"
                          onclick="singleCellApp.showToolFigures(['${data.figure}'], 'Volcano Plot')">`;
            }
        })
        .catch(e => {
            imgDiv.innerHTML = `<span class="text-danger small p-2">${e.message}</span>`;
        });
    },

    generateDegViolin() {
        const imgDiv = document.getElementById('deg-violin-img');
        if (!imgDiv) return;
        imgDiv.innerHTML = '<div class="spinner-border spinner-border-sm" style="color:#a57ded;"></div>';

        const params = {
            genes:   (document.getElementById('deg-violin-genes')   || {}).value || '',
            groupby: (document.getElementById('deg-violin-groupby') || {}).value || '',
            layer:   (document.getElementById('deg-violin-layer')   || {}).value || '',
        };

        fetch('/api/deg/plot_violin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                imgDiv.innerHTML = `<span class="text-danger small p-2">${data.error}</span>`;
            } else {
                imgDiv.innerHTML =
                    `<img src="data:image/png;base64,${data.figure}"
                          style="max-width:100%;border-radius:6px;cursor:zoom-in;"
                          onclick="singleCellApp.showToolFigures(['${data.figure}'], 'Violin Plot')">`;
            }
        })
        .catch(e => {
            imgDiv.innerHTML = `<span class="text-danger small p-2">${e.message}</span>`;
        });
    },

    // -----------------------------------------------------------------------
    // DCT (Differential Cell Type) Analysis helpers
    // -----------------------------------------------------------------------

    /** Render DCT method selection cards (first level). */
    renderDctParamForm(container) {
        const methods = [
            { id: 'sccoda', label: 'scCODA', icon: 'fas fa-wave-square', descKey: 'dct.methodScocodaDesc' },
            { id: 'milopy', label: 'milopy', icon: 'fas fa-broom',       descKey: 'dct.methodMilopyDesc' },
        ];
        container.innerHTML = '';
        methods.forEach(m => {
            const card = document.createElement('div');
            card.className = 'mb-3 p-3 border rounded fade-in c-pointer';
            card.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <i class="${m.icon} me-2 text-primary"></i>
                    <strong>${m.label}</strong>
                </div>
                <p class="text-muted small mb-0">${this.t(m.descKey)}</p>`;
            if (this.currentData) {
                card.onclick = () => this.renderDctMethodForm(m.id, m.label, container);
            } else {
                card.style.opacity = 0.6;
                card.title = this.t('status.uploadFirst');
            }
            container.appendChild(card);
        });
    },

    /** Render DCT parameter form for a specific method (second level). */
    renderDctMethodForm(method, methodLabel, container) {
        const obs        = this.currentData ? (this.currentData.obs_columns || []) : [];
        const embeddings = this.currentData ? (this.currentData.embeddings  || []) : [];
        const colOpts    = `<option value="">${this.t('dct.selectCol')}</option>` +
            obs.map(c => `<option value="${c}">${c}</option>`).join('');
        const repOpts    = embeddings.map(e => `<option value="${e}">${e.replace(/^X_/, '')}</option>`).join('') ||
                           `<option value="X_pca">PCA</option>`;

        const isMilopy = method === 'milopy';
        const samplePlaceholder = isMilopy ? this.t('dct.sampleKeySelect') : this.t('dct.sampleKeyOptional');
        const sampleHint        = isMilopy ? this.t('dct.sampleKeyHintMilopy') : this.t('dct.sampleKeyHintSccoda');
        const sampleRequired    = isMilopy ? `<span class="text-danger ms-1" title="required">*</span>` : '';

        // Update DCT viz panel titles when method changes
        const effectTitle = document.getElementById('dct-effects-title');
        const effectIcon  = document.getElementById('dct-effects-icon');
        if (effectTitle) effectTitle.textContent = isMilopy ? this.t('dct.beeswarmTitle') : this.t('dct.effectsTitle');
        if (effectIcon)  effectIcon.className    = isMilopy ? 'fas fa-broom me-1 text-warning' : 'fas fa-wave-square me-1 text-primary';

        container.innerHTML = `
        <div class="d-flex align-items-center justify-content-between mb-2">
            <h6 class="mb-0 small fw-semibold">
                <i class="fas fa-chart-pie me-1 text-primary"></i>
                DCT — ${methodLabel}
            </h6>
            <div class="d-flex gap-1">
                <button class="btn btn-sm btn-outline-info"
                        onclick="singleCellApp.showCodeModal('dct')">
                    <i class="fas fa-code me-1"></i>${this.t('tools.codeRef')}
                </button>
                <button class="btn btn-sm btn-outline-secondary"
                        onclick="singleCellApp.renderDctParamForm(document.getElementById('parameter-content'))">
                    ${this.t('tools.backToList')}
                </button>
            </div>
        </div>
        <div class="row g-2">
            <input type="hidden" id="dct-method" value="${method}">
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('dct.conditionCol')}</label>
                <select class="form-select form-select-sm" id="dct-condition-col"
                        onchange="singleCellApp.onDctConditionChange(this.value)">
                    ${colOpts}
                </select>
            </div>
            <div class="col-6">
                <label class="form-label form-label-sm">${this.t('dct.ctrlGroup')}</label>
                <select class="form-select form-select-sm" id="dct-ctrl-group">
                    <option value="">${this.t('dct.selectCol')}</option>
                </select>
            </div>
            <div class="col-6">
                <label class="form-label form-label-sm">${this.t('dct.testGroup')}</label>
                <select class="form-select form-select-sm" id="dct-test-group">
                    <option value="">${this.t('dct.selectCol')}</option>
                </select>
            </div>
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('dct.celltypeKey')}</label>
                <select class="form-select form-select-sm" id="dct-celltype-key">
                    ${colOpts}
                </select>
            </div>
            <div class="col-12">
                <label class="form-label form-label-sm">
                    ${this.t('dct.sampleKey')}${sampleRequired}
                </label>
                <select class="form-select form-select-sm" id="dct-sample-key">
                    <option value="">${samplePlaceholder}</option>
                    ${obs.map(c => `<option value="${c}">${c}</option>`).join('')}
                </select>
                <small class="text-muted d-block mt-1">${sampleHint}</small>
            </div>
            ${isMilopy ? `
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('dct.useRep')}</label>
                <select class="form-select form-select-sm" id="dct-use-rep">
                    ${repOpts}
                </select>
            </div>` : `
            <div class="col-12">
                <label class="form-label form-label-sm">${this.t('dct.estFdr')} <strong id="dct-est-fdr-val">0.20</strong></label>
                <input type="range" class="form-range" id="dct-est-fdr"
                       min="0.05" max="0.5" step="0.05" value="0.2"
                       oninput="document.getElementById('dct-est-fdr-val').textContent=parseFloat(this.value).toFixed(2)">
            </div>`}
            <div class="col-12 mt-1">
                <button class="btn btn-success btn-sm w-100" id="dct-run-btn"
                        onclick="singleCellApp.runDctAnalysis()">
                    <i class="fas fa-play me-1"></i>${this.t('dct.run')}
                </button>
            </div>
            <div id="dct-summary" class="col-12" style="display:none;">
                <div class="alert alert-success py-1 px-2 mb-0 small mt-1" id="dct-summary-text"></div>
            </div>
        </div>`;
        this._restoreAndTrackParams('dct', container);
    },

    onDctMethodChange(method) {
        // Kept for compatibility — no longer drives the form (each method has its own form now)
        // Still updates the DCT viz panel titles if called externally
        const effectTitle = document.getElementById('dct-effects-title');
        const effectIcon  = document.getElementById('dct-effects-icon');
        if (method === 'milopy') {
            if (effectTitle) effectTitle.textContent = this.t('dct.beeswarmTitle');
            if (effectIcon)  effectIcon.className    = 'fas fa-broom me-1 text-warning';
        } else {
            if (effectTitle) effectTitle.textContent = this.t('dct.effectsTitle');
            if (effectIcon)  effectIcon.className    = 'fas fa-wave-square me-1 text-primary';
        }
    },

    onDctConditionChange(col) {
        if (!col) return;
        fetch(`/api/deg/get_groups?col=${encodeURIComponent(col)}`)   // reuse same endpoint
        .then(r => r.json())
        .then(data => {
            const groups = data.groups || [];
            const opts   = `<option value="">${this.t('dct.selectCol')}</option>` +
                groups.map(g => `<option value="${g}">${g}</option>`).join('');
            const ctrl = document.getElementById('dct-ctrl-group');
            const test = document.getElementById('dct-test-group');
            if (ctrl) ctrl.innerHTML = opts;
            if (test) test.innerHTML = opts;
            if (ctrl && groups.length >= 1) ctrl.value = groups[0];
            if (test && groups.length >= 2) test.value = groups[1];
        })
        .catch(() => {});
    },

    toggleDctViz() {
        const body = document.getElementById('dct-viz-body');
        const icon = document.getElementById('dct-viz-toggle-icon');
        if (!body) return;
        const collapsed = body.style.display === 'none';
        body.style.display = collapsed ? '' : 'none';
        if (icon) icon.className = collapsed ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
    },

    runDctAnalysis() {
        const btn = document.getElementById('dct-run-btn');

        const params = {
            method:        (document.getElementById('dct-method')        || {}).value || 'sccoda',
            condition:     (document.getElementById('dct-condition-col') || {}).value || '',
            ctrl_group:    (document.getElementById('dct-ctrl-group')    || {}).value || '',
            test_group:    (document.getElementById('dct-test-group')    || {}).value || '',
            cell_type_key: (document.getElementById('dct-celltype-key')  || {}).value || '',
            sample_key:    (document.getElementById('dct-sample-key')    || {}).value || '',
            use_rep:       (document.getElementById('dct-use-rep')       || {}).value || 'X_pca',
            est_fdr:       parseFloat((document.getElementById('dct-est-fdr') || {}).value || '0.2'),
        };

        // Frontend guard: milopy requires sample_key
        if (params.method === 'milopy' && !params.sample_key) {
            const msg = this.t('dct.sampleKeyHintMilopy') || 'milopy 方法需要提供样本列 (sample_key)，请选择一个 obs 列';
            alert(msg);
            return;
        }

        if (btn) btn.disabled = true;
        this.showStatus(this.t('dct.running'), true);
        this.addToLog(this.t('dct.running'));

        fetch('/api/dct/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        })
        .then(r => r.json())
        .then(data => {
            if (btn) btn.disabled = false;
            this.hideStatus();
            if (data.error) { this.addToLog(data.error, 'error'); alert(data.error); return; }
            this.addToLog(this.t('dct.done'));
            this.showStatus(this.t('dct.done'), false);

            // Summary
            const summaryDiv  = document.getElementById('dct-summary');
            const summaryText = document.getElementById('dct-summary-text');
            if (summaryDiv && summaryText) {
                summaryDiv.style.display = '';
                const method = data.method || 'sccoda';
                if (method === 'sccoda') {
                    summaryText.innerHTML = `${this.t('dct.totalCellTypes')} <strong>${data.n_total || 0}</strong>
                        &nbsp;|&nbsp; ${this.t('dct.credible')} <strong class="text-danger">${data.n_credible || 0}</strong>`;
                } else {
                    summaryText.innerHTML = `${this.t('dct.totalNhoods')} <strong>${data.n_total || 0}</strong>
                        &nbsp;|&nbsp; ${this.t('dct.sigUp')} <strong class="text-danger">${data.n_sig_up || 0}</strong>
                        &nbsp;|&nbsp; ${this.t('dct.sigDown')} <strong class="text-primary">${data.n_sig_down || 0}</strong>`;
                }
            }

            // Update count badge
            const countBadge = document.getElementById('dct-results-count');
            if (countBadge) countBadge.textContent = data.n_total || 0;

            // Render table
            this.dctAllResults = data.all_results || [];
            this._renderDctTable(data.method || 'sccoda', this.dctAllResults);

            // Auto-generate composition plot
            this.generateDctComposition();
        })
        .catch(error => {
            if (btn) btn.disabled = false;
            this.hideStatus();
            this.addToLog(error.message, 'error');
            alert(error.message);
        });
    },

    _renderDctTable(method, rows) {
        const tbody  = document.getElementById('dct-results-tbody');
        const thead  = document.getElementById('dct-results-thead');
        if (!tbody) return;
        if (!rows || !rows.length) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted py-3">${this.t('dct.tableEmpty')}</td></tr>`;
            return;
        }
        if (method === 'sccoda') {
            if (thead) thead.innerHTML = `
                <tr>
                    <th>${this.t('dct.col.celltype')}</th>
                    <th>${this.t('dct.col.effect')}</th>
                    <th>${this.t('dct.col.log2fc')}</th>
                    <th>${this.t('dct.col.inclProb')}</th>
                    <th>${this.t('dct.col.sig')}</th>
                </tr>`;
            tbody.innerHTML = rows.map(row => {
                const credible = row.is_credible;
                const color    = credible ? (parseFloat(row.log2fc || 0) > 0 ? '#e06c75' : '#5ba4cf') : '#aaaaaa';
                return `<tr>
                    <td style="font-weight:600;color:${color};">${row.cell_type || row['Cell Type'] || '-'}</td>
                    <td>${parseFloat(row.effect || row['Effect'] || 0).toFixed(3)}</td>
                    <td>${parseFloat(row.log2fc || row['log2-fold change'] || 0).toFixed(3)}</td>
                    <td>${parseFloat(row.inclusion_prob || row['Inclusion probability'] || 0).toFixed(3)}</td>
                    <td>${credible ? '<span class="badge bg-danger">✓</span>' : '<span class="badge bg-secondary">-</span>'}</td>
                </tr>`;
            }).join('');
        } else {
            // milopy
            if (thead) thead.innerHTML = `
                <tr>
                    <th>${this.t('dct.col.celltype')}</th>
                    <th>${this.t('dct.col.logfc')}</th>
                    <th>PValue</th>
                    <th>SpatialFDR</th>
                    <th>${this.t('dct.col.nhoodSize')}</th>
                </tr>`;
            tbody.innerHTML = rows.slice(0, 300).map(row => {
                const fdr   = parseFloat(row.SpatialFDR || 1);
                const lfc   = parseFloat(row.logFC || 0);
                const color = fdr < 0.1 ? (lfc > 0 ? '#e06c75' : '#5ba4cf') : '#aaaaaa';
                return `<tr>
                    <td style="font-weight:600;color:${color};">${row.nhood_annotation || '-'}</td>
                    <td style="color:${color};">${lfc.toFixed(3)}</td>
                    <td>${parseFloat(row.PValue || 1).toExponential(2)}</td>
                    <td>${fdr < 0.001 ? fdr.toExponential(2) : fdr.toFixed(4)}</td>
                    <td>${row.Nhood_size || '-'}</td>
                </tr>`;
            }).join('');
        }
    },

    generateDctComposition() {
        const imgDiv = document.getElementById('dct-composition-img');
        if (!imgDiv) return;
        imgDiv.innerHTML = '<div class="spinner-border spinner-border-sm text-success"></div>';
        fetch('/api/dct/plot_composition', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                imgDiv.innerHTML = `<span class="text-danger small p-2">${data.error}</span>`;
            } else {
                imgDiv.innerHTML = `<img src="data:image/png;base64,${data.figure}"
                    style="max-width:100%;border-radius:6px;cursor:zoom-in;"
                    onclick="singleCellApp.showToolFigures(['${data.figure}'], 'Cell Type Composition')">`;
            }
        })
        .catch(e => { imgDiv.innerHTML = `<span class="text-danger small p-2">${e.message}</span>`; });
    },

    generateDctEffects() {
        const imgDiv = document.getElementById('dct-effects-img');
        if (!imgDiv) return;
        imgDiv.innerHTML = '<div class="spinner-border spinner-border-sm text-primary"></div>';
        fetch('/api/dct/plot_effects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                imgDiv.innerHTML = `<span class="text-danger small p-2">${data.error}</span>`;
            } else {
                imgDiv.innerHTML = `<img src="data:image/png;base64,${data.figure}"
                    style="max-width:100%;border-radius:6px;cursor:zoom-in;"
                    onclick="singleCellApp.showToolFigures(['${data.figure}'], '${data.method === 'milopy' ? 'DA Beeswarm' : 'scCODA Effects'}')">`;
            }
        })
        .catch(e => { imgDiv.innerHTML = `<span class="text-danger small p-2">${e.message}</span>`; });
    },

    saveData() {
        if (!this.currentData) return;

        const suggestedName = (this.currentData.filename || 'data')
            .replace(/\.h5ad$/i, '') + '.h5ad';

        this.showStatus(this.t('status.downloadingData'), true);
        this.addToLog(this.t('status.downloadStart'));

        fetch('/api/save', { method: 'POST' })
        .then(response => {
            if (response.ok) return response.blob();
            throw new Error(this.t('status.saveFailed'));
        })
        .then(async blob => {
            // Use native Save-As dialog when available (Chrome/Edge on HTTPS or localhost)
            if (window.showSaveFilePicker) {
                try {
                    const handle = await window.showSaveFilePicker({
                        suggestedName,
                        types: [{ description: 'AnnData H5AD', accept: { 'application/x-hdf5': ['.h5ad'] } }]
                    });
                    const writable = await handle.createWritable();
                    await writable.write(blob);
                    await writable.close();
                    this.hideStatus();
                    this.addToLog(this.t('status.dataSaved'));
                    this.showStatus(this.t('status.dataSaved'), false);
                    return;
                } catch (e) {
                    // User cancelled the dialog — don't show error
                    if (e.name === 'AbortError') { this.hideStatus(); return; }
                    // Any other error: fall through to blob download
                }
            }
            // Fallback: trigger browser download (browser will ask save location if configured to)
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = suggestedName;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            this.hideStatus();
            this.addToLog(this.t('status.dataSaved'));
            this.showStatus(this.t('status.dataSaved'), false);
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(this.t('status.saveFailed') + ': ' + error.message, 'error');
            this.showStatus(this.t('status.saveFailed') + ': ' + error.message, false);
        });
    },

    resetData() {
        if (confirm(this.t('status.resetConfirm'))) {
            // Clear all session-persisted state (inputs, renderer, view prefs)
            if (this.clearPersistedSession) this.clearPersistedSession();
            this.currentData = null;
            // Reset preview mode on data reset
            this.isPreviewMode = false;
            this.updatePreviewModeBanner(false);
            document.getElementById('upload-section').style.display = 'block';
            document.getElementById('data-status').classList.add('d-none');
            document.getElementById('viz-controls').style.display = 'none';
            document.getElementById('viz-panel').style.display = 'none';
            const _lp2 = document.getElementById('viz-legend-panel');
            if (_lp2) _lp2.style.display = 'none';
            document.getElementById('analysis-log').innerHTML = `<div class="text-muted">${this.t('panel.waitingUpload')}</div>`;
            document.getElementById('fileInput').value = '';
            const previewInput = document.getElementById('fileInputPreview');
            if (previewInput) previewInput.value = '';

            // Reset adata status panel to placeholder
            this.updateAdataStatus(null);
            // Reset parameter panel to placeholder
            this.showParameterPlaceholder();

            // Clear gene input
            const geneInput = document.getElementById('gene-input');
            if (geneInput) geneInput.value = '';

            // Reset palettes to default
            const paletteSelect = document.getElementById('palette-select');
            if (paletteSelect) paletteSelect.value = 'default';

            const categoryPaletteSelect = document.getElementById('category-palette-select');
            if (categoryPaletteSelect) categoryPaletteSelect.value = 'default';

            // Clear vmin/vmax inputs
            const vminInput = document.getElementById('vmin-input');
            const vmaxInput = document.getElementById('vmax-input');
            if (vminInput) vminInput.value = '';
            if (vmaxInput) vmaxInput.value = '';

            // Hide category palette row and vmin/vmax row
            const categoryPaletteRow = document.getElementById('category-palette-row');
            if (categoryPaletteRow) categoryPaletteRow.style.display = 'none';

            const vminmaxRow = document.getElementById('vminmax-row');
            if (vminmaxRow) vminmaxRow.style.display = 'none';

            // ── deck.gl WebGL renderer cleanup ────────────────────────────────
            // Must be done here (not in sc-plot.js monkey-patch) because
            // Object.assign overwrites any previously patched prototype method.
            if (this._deckglRenderer) {
                this._deckglRenderer.destroy();
                this._deckglRenderer         = null;
                this._deckglCurrentEmbedding = null;
                this._deckglCurrentColorBy   = null;
            }
            this._forceRenderer = null;
            // Remove the deck.gl wrapper element from the DOM
            const deckWrap = document.getElementById('deckgl-wrap');
            if (deckWrap) deckWrap.remove();
            // Reset renderer toggle buttons to "auto"
            this._syncRendererButtons('auto');
            // Ensure Plotly div is visible again for the next upload
            const plotlyDiv = document.getElementById('plotly-div');
            if (plotlyDiv) {
                plotlyDiv.style.display = '';
                // Purge Plotly data so hasExistingPlot check resets cleanly
                if (typeof Plotly !== 'undefined') {
                    try { Plotly.purge('plotly-div'); } catch (_) {}
                }
            }
            // Reset embedding/color selects
            const embSel = document.getElementById('embedding-select');
            if (embSel) embSel.innerHTML = '<option value="">Select embedding</option>';
            const colSel = document.getElementById('color-select');
            if (colSel) colSel.innerHTML = '<option value="">None</option>';
        }
    },

    // ── Code Reference Modal ──────────────────────────────────────────────────

    showCodeModal(tool) {
        // Collect current parameter values from the form
        const params = {};
        const container = document.getElementById('parameter-content');
        if (container) {
            container.querySelectorAll('input, select').forEach(el => {
                if (!el.id) return;
                if (el.type === 'checkbox') params[el.id] = el.checked;
                else if (el.value !== '') params[el.id] = el.value;
            });
        }

        const specialLabels = {
            'deg_expression': this.t('deg.title'),
            'dct': this.t('dct.title'),
        };
        const toolLabel = specialLabels[tool] || this.currentToolLabel || tool;
        const code = this.getToolCodeTemplate(tool, params);
        const paramDocs = this.getToolParamDocs(tool);

        // Populate modal
        const titleEl = document.getElementById('codeRefModalTitle');
        const subtitleEl = document.getElementById('codeRefModalSubtitle');
        const preEl = document.getElementById('codeRefPre');
        const tbody = document.querySelector('#codeRefParamTable tbody');

        if (titleEl) titleEl.textContent = `${this.t('tools.codeRef')} — ${toolLabel}`;
        if (subtitleEl) subtitleEl.textContent = this.t('tools.codeRefSubtitle') || 'adata is the current AnnData object';
        if (preEl) preEl.textContent = code;

        // Update modal UI labels from i18n
        const codeTabEl = document.getElementById('codeTab');
        const paramsTabEl = document.getElementById('paramsTab');
        if (codeTabEl) codeTabEl.innerHTML = `<i class="fas fa-file-code me-1"></i>${this.t('tools.codeTabLabel')}`;
        if (paramsTabEl) paramsTabEl.innerHTML = `<i class="fas fa-list me-1"></i>${this.t('tools.paramsTabLabel')}`;

        const copyBtn = document.getElementById('copyCodeBtn');
        if (copyBtn) copyBtn.innerHTML = `<i class="fas fa-copy me-1"></i>${this.t('tools.copyCode')}`;

        const closeBtnText = document.getElementById('closeCodeModalBtnText');
        if (closeBtnText) closeBtnText.textContent = this.t('common.close');

        // Update param table headers
        const paramThead = document.querySelector('#codeRefParamTable thead tr');
        if (paramThead) {
            paramThead.innerHTML = `
                <th style="width:18%">${this.t('tools.paramName')}</th>
                <th style="width:12%">${this.t('tools.paramType')}</th>
                <th style="width:15%">${this.t('tools.paramDefault')}</th>
                <th>${this.t('tools.paramDesc')}</th>`;
        }

        if (tbody) {
            tbody.innerHTML = paramDocs.map(p => `
                <tr>
                    <td><code class="text-danger">${p.name}</code></td>
                    <td><span class="badge bg-secondary">${p.type}</span></td>
                    <td><code>${p.default}</code></td>
                    <td class="small">${p.desc}</td>
                </tr>`).join('');
        }

        // Store code for copy
        this._codeRefCurrentCode = code;

        // Reset to code tab
        this._switchCodeRefTab('code');

        // Show modal
        const el = document.getElementById('codeRefModal');
        if (el) {
            const modal = bootstrap.Modal.getOrCreateInstance(el);
            modal.show();
        }
    },

    _switchCodeRefTab(tab) {
        const codeDiv  = document.getElementById('codeRefTabCode');
        const paramsDiv = document.getElementById('codeRefTabParams');
        const codeLink  = document.getElementById('codeTab');
        const paramsLink = document.getElementById('paramsTab');
        if (tab === 'code') {
            if (codeDiv) codeDiv.style.display = '';
            if (paramsDiv) paramsDiv.style.display = 'none';
            if (codeLink) codeLink.classList.add('active');
            if (paramsLink) paramsLink.classList.remove('active');
        } else {
            if (codeDiv) codeDiv.style.display = 'none';
            if (paramsDiv) paramsDiv.style.display = '';
            if (codeLink) codeLink.classList.remove('active');
            if (paramsLink) paramsLink.classList.add('active');
        }
    },

    _copyCodeRef() {
        const code = this._codeRefCurrentCode || '';
        navigator.clipboard.writeText(code).then(() => {
            const btn = document.getElementById('copyCodeBtn');
            if (btn) {
                const orig = btn.innerHTML;
                btn.innerHTML = `<i class="fas fa-check me-1"></i>${this.t('tools.codeCopied')}`;
                btn.classList.replace('btn-outline-secondary', 'btn-success');
                setTimeout(() => {
                    btn.innerHTML = orig;
                    btn.classList.replace('btn-success', 'btn-outline-secondary');
                }, 1500);
            }
        }).catch(() => {
            // Fallback
            const ta = document.createElement('textarea');
            ta.value = code;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        });
    },

    getToolCodeTemplate(tool, params = {}) {
        const p = params;
        const templates = {
            'normalize': () => {
                const t = p.target_sum || 1e4;
                return `import scanpy as sc

# Normalize each cell to a fixed total count
sc.pp.normalize_total(adata, target_sum=${t})`;
            },
            'log1p': () => `import scanpy as sc

# Apply log(x + 1) transformation to reduce the influence of extreme values
sc.pp.log1p(adata)`,

            'scale': () => {
                const mv = p.max_value || 10;
                return `import scanpy as sc

# Z-score scale gene expression and clip values exceeding max_value
sc.pp.scale(adata, max_value=${mv})`;
            },

            'hvg': () => {
                const n = p.n_genes || 2000;
                const flavor = p.method || 'seurat';
                return `import scanpy as sc

# Select highly variable genes (HVGs) — retains the most informative genes
sc.pp.highly_variable_genes(
    adata,
    flavor='${flavor}',   # algorithm: seurat / cell_ranger / seurat_v3
    n_top_genes=${n}       # number of HVGs to retain
)

# Keep only HVGs (optional — reduces memory for downstream steps)
adata = adata[:, adata.var.highly_variable].copy()`;
            },

            'filter_cells': () => {
                const minC = p.min_counts || 500;
                const minG = p.min_genes || 200;
                const maxC = p.max_counts ? `\nsc.pp.filter_cells(adata, max_counts=${p.max_counts})` : '';
                const maxG = p.max_genes ? `\nsc.pp.filter_cells(adata, max_genes=${p.max_genes})` : '';
                return `import scanpy as sc

# Filter low-quality cells
sc.pp.filter_cells(adata, min_counts=${minC})  # minimum total UMI count
sc.pp.filter_cells(adata, min_genes=${minG})   # minimum number of detected genes${maxC}${maxG}`;
            },

            'filter_genes': () => {
                const minCells = p.min_cells || 3;
                const minCounts = p.g_min_counts ? `\nsc.pp.filter_genes(adata, min_counts=${p.g_min_counts})` : '';
                return `import scanpy as sc

# Filter lowly-expressed genes
sc.pp.filter_genes(adata, min_cells=${minCells})  # must be expressed in at least N cells${minCounts}`;
            },

            'filter_outliers': () => {
                const maxMt = p.max_mt_percent || 20;
                const maxRibo = p.max_ribo_percent ? `\nadata = adata[adata.obs['pct_counts_ribo'] <= ${p.max_ribo_percent}].copy()` : '';
                const maxHb = p.max_hb_percent ? `\nadata = adata[adata.obs['pct_counts_hb'] <= ${p.max_hb_percent}].copy()` : '';
                return `import scanpy as sc

# Calculate QC metrics (mitochondrial / ribosomal / haemoglobin gene fractions)
adata.var['mt']   = adata.var_names.str.startswith('MT-')
adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
adata.var['hb']   = adata.var_names.str.contains('^HB[^P]', regex=True)

sc.pp.calculate_qc_metrics(
    adata, qc_vars=['mt', 'ribo', 'hb'], inplace=True, log1p=True
)

# Filter outlier cells by threshold
adata = adata[adata.obs['pct_counts_mt'] <= ${maxMt}].copy()${maxRibo}${maxHb}`;
            },

            'doublets': () => {
                const bk = p.batch_key ? `, batch_key='${p.batch_key}'` : '';
                const edr = p.expected_doublet_rate || 0.05;
                const sdr = p.sim_doublet_ratio || 2.0;
                return `import scanpy as sc

# Detect and remove doublets using Scrublet
sc.pp.scrublet(
    adata${bk},
    expected_doublet_rate=${edr},  # expected fraction of doublets
    sim_doublet_ratio=${sdr},      # simulated doublets relative to observed cells
)

# Remove predicted doublets
adata = adata[~adata.obs['predicted_doublet']].copy()`;
            },

            'pca': () => {
                const n = p.n_comps || 50;
                return `import scanpy as sc

# Principal component analysis (PCA) dimensionality reduction
sc.pp.pca(adata, n_comps=${n})

# Results stored in adata.obsm['X_pca'] and adata.uns['pca']`;
            },

            'neighbors': () => {
                const nn = p.n_neighbors || 15;
                return `import scanpy as sc

# Build a KNN graph (required for UMAP / clustering)
sc.pp.neighbors(
    adata,
    n_neighbors=${nn},  # number of neighbours
    n_pcs=50            # number of PCA components to use
)`;
            },

            'umap': () => {
                const md = p.min_dist || 0.5;
                return `import scanpy as sc

# UMAP dimensionality reduction for visualization
# Requires sc.pp.neighbors() to be run first
sc.tl.umap(
    adata,
    min_dist=${md}  # minimum distance between embedded points; smaller = tighter clusters
)

# Result stored in adata.obsm['X_umap']`;
            },

            'tsne': () => {
                const perp = p.perplexity || 30;
                return `import scanpy as sc

# t-SNE dimensionality reduction for visualization
sc.tl.tsne(
    adata,
    perplexity=${perp}  # effective number of local neighbours; typically 5–50
)

# Result stored in adata.obsm['X_tsne']`;
            },

            'leiden': () => {
                const res = p.resolution || 1.0;
                return `import scanpy as sc

# Leiden clustering (requires sc.pp.neighbors() first)
sc.tl.leiden(
    adata,
    resolution=${res}  # higher = more clusters
)

# Result stored in adata.obs['leiden']`;
            },

            'louvain': () => {
                const res = p.resolution || 1.0;
                return `import scanpy as sc

# Louvain clustering (requires sc.pp.neighbors() first)
sc.tl.louvain(
    adata,
    resolution=${res}  # higher = more clusters
)

# Result stored in adata.obs['louvain']`;
            },

            'celltypist': () => {
                const pkl = p.pkl_path || '/path/to/model.pkl';
                return `import omicverse as ov

# Annotate cell types using CellTypist
obj = ov.single.Annotation(adata)
obj.add_reference_pkl('${pkl}')   # load pre-downloaded model file
obj.annotate(method='celltypist')

# Annotation result stored in adata.obs['celltypist_prediction']
adata = obj.adata`;
            },

            'gpt4celltype': () => {
                const ck = p.cluster_key || 'leiden';
                const tissue = p.tissuename || 'PBMC';
                const species = p.speciename || 'human';
                const provider = p.provider || 'qwen';
                const model = p.model || 'qwen-plus';
                const topn = p.topgenenumber || 10;
                return `import omicverse as ov
import os

# Annotate cell types using a large language model (LLM)
os.environ['AGI_API_KEY'] = 'your_api_key'  # or set the env variable in advance

obj = ov.single.Annotation(adata)
obj.annotate(
    method='gpt4celltype',
    cluster_key='${ck}',     # existing clustering column
    tissuename='${tissue}',  # tissue of origin (e.g. PBMC, Brain, Liver)
    speciename='${species}', # species (human / mouse)
    provider='${provider}',  # LLM provider: qwen / openai / kimi
    model='${model}',        # model name
    topgenenumber=${topn},   # top marker genes per cluster passed to LLM
)

# Annotation result stored in adata.obs['gpt4celltype_prediction']
adata = obj.adata`;
            },

            'scsa': () => {
                const ck = p.cluster_key || 'leiden';
                const fc = p.foldchange || 1.5;
                const pv = p.pvalue || 0.05;
                const ct = p.celltype || 'normal';
                const tgt = p.target || 'cellmarker';
                const tissue = p.tissue || 'All';
                return `import omicverse as ov

# Annotate cell types using the SCSA database
obj = ov.single.Annotation(adata)
# obj.add_reference_scsa_db('/path/to/scsa.db')  # optional: use a local database

obj.annotate(
    method='scsa',
    cluster_key='${ck}',  # existing clustering column
    foldchange=${fc},      # fold-change threshold for marker gene selection
    pvalue=${pv},          # significance threshold
    celltype='${ct}',      # cell-type mode: normal / cancer
    target='${tgt}',       # reference database: cellmarker / panglaoDB / cancersea
    tissue='${tissue}',    # restrict to tissue ('All' = no restriction)
)

# Annotation result stored in adata.obs['scsa_prediction']
adata = obj.adata`;
            },

            'diffusion_map': () => {
                const gb = p.groupby || 'leiden';
                const rep = p.use_rep || 'X_pca';
                const nc = p.n_comps || 50;
                const orig = p.origin_cells || 'YourStartCellType';
                return `import omicverse as ov

# Infer cell trajectory using Diffusion Map
Traj = ov.single.TrajInfer(
    adata,
    basis='X_umap',    # embedding for visualization
    groupby='${gb}',   # grouping column (usually clustering results)
    use_rep='${rep}',  # low-dimensional representation
    n_comps=${nc}      # number of components to use
)

Traj.set_origin_cells('${orig}')  # set the starting cell type
Traj.inference(method='diffusion_map')

# Pseudotime result stored in adata.obs['dpt_pseudotime']`;
            },

            'slingshot': () => {
                const gb = p.groupby || 'leiden';
                const rep = p.use_rep || 'X_pca';
                const nc = p.n_comps || 50;
                const orig = p.origin_cells || 'YourStartCellType';
                const term = p.terminal_cells || 'TerminalCellType';
                const epochs = p.num_epochs || 1;
                return `import omicverse as ov

# Infer cell trajectory using Slingshot
Traj = ov.single.TrajInfer(
    adata,
    basis='X_umap',
    groupby='${gb}',
    use_rep='${rep}',
    n_comps=${nc}
)

Traj.set_origin_cells('${orig}')
Traj.set_terminal_cells(['${term}'])  # comma-separate multiple terminal states

Traj.inference(method='slingshot', num_epochs=${epochs})

# Pseudotime result stored in adata.obs['slingshot_pseudotime']`;
            },

            'palantir': () => {
                const gb = p.groupby || 'leiden';
                const rep = p.use_rep || 'X_pca';
                const nc = p.n_comps || 50;
                const orig = p.origin_cells || 'YourStartCellType';
                const term = p.terminal_cells || 'TerminalCellType';
                const nwp = p.num_waypoints || 500;
                return `import omicverse as ov

# Infer cell trajectory and branch probabilities using Palantir
Traj = ov.single.TrajInfer(
    adata,
    basis='X_umap',
    groupby='${gb}',
    use_rep='${rep}',
    n_comps=${nc}
)

Traj.set_origin_cells('${orig}')
Traj.set_terminal_cells(['${term}'])

Traj.inference(method='palantir', num_waypoints=${nwp})

# Results stored in:
# adata.obs['palantir_pseudotime']  — pseudotime
# adata.obs['palantir_entropy']     — cell fate uncertainty`;
            },

            'paga': () => {
                const grps = p.groups || 'leiden';
                const rep = p.use_rep || 'X_pca';
                const basis = p.basis || 'umap';
                const timePrior = p.use_time_prior || '';
                const pagaKwExtra = timePrior
                    ? `\nov.utils.cal_paga(adata, vkey='paga', groups='${grps}', use_time_prior='${timePrior}')`
                    : `\nov.utils.cal_paga(adata, vkey='paga', groups='${grps}')`;
                return `import omicverse as ov
import scanpy as sc

# PAGA trajectory analysis (graph-based pseudotime)
sc.pp.neighbors(adata, use_rep='${rep}')
${pagaKwExtra}

# Visualize PAGA graph using the omicverse plotting interface
ov.utils.plot_paga(
    adata,
    basis='${basis}',      # visualization embedding (umap / tsne / etc.)
    size=50,
    alpha=0.1,
    min_edge_width=2,
    node_size_scale=1.5,
    show=True,
    legend_loc='on data',
)`;
            },

            'sctour': () => {
                const gb = p.groupby || 'leiden';
                const rep = p.use_rep || 'X_pca';
                const nc = p.n_comps || 50;
                const lec = p.alpha_recon_lec || 0.5;
                const lode = p.alpha_recon_lode || 0.5;
                return `import omicverse as ov

# Infer cell trajectory using scTour (requires raw count data in adata.X)
Traj = ov.single.TrajInfer(
    adata,
    basis='X_umap',
    groupby='${gb}',
    use_rep='${rep}',
    n_comps=${nc}
)

Traj.inference(
    method='sctour',
    alpha_recon_lec=${lec},   # lec reconstruction weight
    alpha_recon_lode=${lode}  # lode reconstruction weight
)

# Pseudotime result stored in adata.obs['sctour_pseudotime']`;
            },

            'deg_expression': () => {
                const condition = p['deg-condition-col'] || 'condition';
                const ctrl = p['deg-ctrl-group'] || 'control';
                const test = p['deg-test-group'] || 'treatment';
                const ctKey = p['deg-celltype-key'] || 'leiden';
                const method = p['deg-method'] || 'wilcoxon';
                const maxCells = p['deg-max-cells'] || 100000;
                return `import omicverse as ov

# Differential expression analysis (DEG)
# Compares gene expression between two groups of cells
deg_obj = ov.single.DEG(
    adata,
    condition='${condition}',    # obs column that defines the groups
    ctrl_group='${ctrl}',        # name of the control group
    test_group='${test}',        # name of the treatment/test group
    method='${method}',          # statistical test: wilcoxon / t-test
)

deg_obj.run(
    celltype_key='${ctKey}',     # cell-type column (DEG is run per cell type)
    celltype_group=None,         # None = all cell types; or pass a list to subset
    max_cells=${maxCells},       # max cells per group (prevents memory overflow)
)

# Retrieve results DataFrame (columns: log2FC, pvalue, padj, sig)
results = deg_obj.get_results()
print(results.head())

# Volcano plot
deg_obj.plot_volcano(
    fc_threshold=1.0,    # log2 fold-change threshold
    padj_threshold=0.05  # adjusted p-value threshold
)`;
            },

            'dct': () => {
                const condition = p['dct-condition-col'] || 'condition';
                const ctrl = p['dct-ctrl-group'] || 'control';
                const test = p['dct-test-group'] || 'treatment';
                const ctKey = p['dct-celltype-key'] || 'leiden';
                const method = p['dct-method'] || 'sccoda';
                const sampleKey = p['dct-sample-key'] || '';
                const estFdr = p['dct-est-fdr'] || 0.2;
                const useRep = p['dct-use-rep'] || 'X_pca';

                if (method === 'milopy') {
                    const sk = sampleKey || 'sample';
                    return `import omicverse as ov
import scanpy as sc
from omicverse.single._milo_dev import Milo

# Differential cell-type composition — milopy (Milo, wrapped in omicverse)
# Subset to the two groups being compared
adata_sub = adata[adata.obs['${condition}'].isin(['${ctrl}', '${test}'])].copy()

# 1. Build Milo object
milo = Milo()
mdata = milo.load(adata_sub)

# 2. Build neighbour graph on mdata['rna']
sc.pp.neighbors(mdata['rna'], use_rep='${useRep}', n_neighbors=150)

# 3. Assign neighbourhoods (prop controls sampling fraction)
milo.make_nhoods(mdata['rna'], prop=0.1)

# 4. Count cells per sample in each neighbourhood
mdata = milo.count_nhoods(mdata, sample_col='${sk}')

# 5. Differential abundance test
milo.da_nhoods(
    mdata,
    design='~${condition}',
    model_contrasts='${condition}[${test}]-${condition}[${ctrl}]',
    solver='edger',
)

# 6. Build neighbourhood graph for visualization & annotate cell types
milo.build_nhood_graph(mdata, basis='${useRep}')
milo.annotate_nhoods(mdata, anno_col='${ctKey}')

# Results stored in mdata['milo'].var`;
                } else {
                    const sk = sampleKey || 'sample';
                    return `import pertpy as pt

# Differential cell-type composition — scCODA
# Bayesian hierarchical model for detecting cell-type proportion changes
# Subset to the two groups being compared
adata_sub = adata[adata.obs['${condition}'].isin(['${ctrl}', '${test}'])].copy()

# 1. Load data into scCODA model
model = pt.tl.Sccoda()
sccoda_data = model.load(
    adata_sub,
    type='cell_level',
    cell_type_identifier='${ctKey}',   # cell-type column
    sample_identifier='${sk}',         # sample column (one row per sample in counts)
    covariate_obs=['${condition}'],     # condition column
)

# 2. Prepare model formula
sccoda_data = model.prepare(
    sccoda_data,
    modality_key='coda',
    formula='${condition}',            # condition column name passed directly
)

# 3. Run NUTS Bayesian sampling
model.run_nuts(
    sccoda_data,
    modality_key='coda',
    num_samples=5000,    # number of samples
    num_warmup=500,      # warm-up steps
)

# 4. Set FDR threshold and identify credible effects
model.credible_effects(sccoda_data, modality_key='coda')
model.set_fdr(sccoda_data, modality_key='coda', est_fdr=${estFdr})

# 5. Retrieve results DataFrame
results = model.get_effect_df(sccoda_data, modality_key='coda')
print(results)`;
                }
            },
        };

        const fn = templates[tool];
        if (fn) return fn();
        return `# No code template available for: ${tool}`;
    },

    getToolParamDocs(tool) {
        const docs = {
            'normalize': [
                { name: 'target_sum', type: 'float', default: '1e4', desc: 'Target total count per cell after normalization. Common values: 1e4 (10 000), 1e6 (TPM).' },
            ],
            'log1p': [],
            'scale': [
                { name: 'max_value', type: 'float', default: '10', desc: 'Clip threshold — z-score values exceeding this are clipped to prevent outlier dominance.' },
            ],
            'hvg': [
                { name: 'n_top_genes', type: 'int', default: '2000', desc: 'Number of highly variable genes to retain; typically 1 000–5 000.' },
                { name: 'flavor', type: 'str', default: '"seurat"', desc: 'Algorithm: seurat (recommended), cell_ranger, or seurat_v3.' },
            ],
            'filter_cells': [
                { name: 'min_counts', type: 'int', default: '500', desc: 'Minimum total UMI count per cell; cells below this are removed.' },
                { name: 'min_genes', type: 'int', default: '200', desc: 'Minimum number of detected genes per cell; removes empty droplets.' },
                { name: 'max_counts', type: 'int', default: 'None', desc: 'Maximum UMI count per cell; can filter potential doublets.' },
                { name: 'max_genes', type: 'int', default: 'None', desc: 'Maximum number of detected genes per cell.' },
            ],
            'filter_genes': [
                { name: 'min_cells', type: 'int', default: '3', desc: 'Minimum number of cells in which a gene must be expressed; genes below are removed.' },
                { name: 'min_counts', type: 'int', default: 'None', desc: 'Minimum total UMI count across cells for a gene (optional).' },
            ],
            'filter_outliers': [
                { name: 'max_mt_percent', type: 'float', default: '20', desc: 'Upper bound on mitochondrial gene percentage (%). High values indicate poor cell quality.' },
                { name: 'mt_prefixes', type: 'str', default: '"MT-"', desc: 'Comma-separated mitochondrial gene prefixes, e.g. "MT-,mt-".' },
                { name: 'max_ribo_percent', type: 'float', default: 'None', desc: 'Upper bound on ribosomal gene percentage (optional).' },
                { name: 'max_hb_percent', type: 'float', default: 'None', desc: 'Upper bound on haemoglobin gene percentage (optional).' },
            ],
            'doublets': [
                { name: 'expected_doublet_rate', type: 'float', default: '0.05', desc: 'Expected doublet fraction (0–1). Typically 0.05–0.08 for 10x Genomics.' },
                { name: 'sim_doublet_ratio', type: 'float', default: '2.0', desc: 'Number of simulated doublets relative to observed cells.' },
                { name: 'batch_key', type: 'str', default: 'None', desc: 'obs column for batch; set when multiple batches should be scored separately.' },
                { name: 'n_prin_comps', type: 'int', default: '30', desc: 'Number of PCA components used internally by Scrublet.' },
            ],
            'pca': [
                { name: 'n_comps', type: 'int', default: '50', desc: 'Number of principal components to compute; typically 30–100 depending on dataset complexity.' },
            ],
            'neighbors': [
                { name: 'n_neighbors', type: 'int', default: '15', desc: 'Number of neighbours per cell. Larger = smoother graph, coarser clusters; smaller = finer detail.' },
                { name: 'n_pcs', type: 'int', default: '50', desc: 'Number of PCA components used to compute neighbours; should match the PCA step.' },
            ],
            'umap': [
                { name: 'min_dist', type: 'float', default: '0.5', desc: 'Minimum distance between embedded points (0.0–1.0). Smaller = tighter clusters; larger = more uniform spread.' },
            ],
            'tsne': [
                { name: 'perplexity', type: 'float', default: '30', desc: 'Perplexity ~ effective number of local neighbours; typically 5–50. Increase for large datasets.' },
            ],
            'leiden': [
                { name: 'resolution', type: 'float', default: '1.0', desc: 'Clustering resolution. Higher = more, finer clusters; lower = fewer, coarser clusters.' },
            ],
            'louvain': [
                { name: 'resolution', type: 'float', default: '1.0', desc: 'Clustering resolution. Higher = more clusters; lower = fewer clusters.' },
            ],
            'celltypist': [
                { name: 'pkl_path', type: 'str', default: '""', desc: 'Path to a CellTypist model file (.pkl). Download from the CellTypist website.' },
            ],
            'gpt4celltype': [
                { name: 'cluster_key', type: 'str', default: '"leiden"', desc: 'obs column name of existing clustering results, e.g. leiden or louvain.' },
                { name: 'tissuename', type: 'str', default: '""', desc: 'Tissue of origin (e.g. PBMC, Brain, Liver) — helps the LLM narrow its predictions.' },
                { name: 'speciename', type: 'str', default: '"human"', desc: 'Species (human / mouse) — affects gene-name interpretation.' },
                { name: 'provider', type: 'str', default: '"qwen"', desc: 'LLM provider: qwen, openai, or kimi.' },
                { name: 'model', type: 'str', default: '"qwen-plus"', desc: 'Model name matching the provider, e.g. gpt-4o or qwen-plus.' },
                { name: 'topgenenumber', type: 'int', default: '10', desc: 'Number of top marker genes per cluster passed to the LLM for annotation.' },
            ],
            'scsa': [
                { name: 'cluster_key', type: 'str', default: '"leiden"', desc: 'obs column name of existing clustering results.' },
                { name: 'foldchange', type: 'float', default: '1.5', desc: 'Fold-change threshold for filtering marker genes.' },
                { name: 'pvalue', type: 'float', default: '0.05', desc: 'Statistical significance threshold (p-value).' },
                { name: 'celltype', type: 'str', default: '"normal"', desc: 'Cell-type mode: normal (healthy cells) or cancer (tumour cells).' },
                { name: 'target', type: 'str', default: '"cellmarker"', desc: 'Reference database: cellmarker, panglaoDB, or cancersea.' },
                { name: 'tissue', type: 'str', default: '"All"', desc: 'Restrict query to a specific tissue. "All" means no restriction.' },
            ],
            'diffusion_map': [
                { name: 'groupby', type: 'str', default: '"leiden"', desc: 'obs column to group cells (usually clustering results).' },
                { name: 'use_rep', type: 'str', default: '"X_pca"', desc: 'obsm key for the low-dimensional representation, e.g. X_pca or X_scVI.' },
                { name: 'n_comps', type: 'int', default: '50', desc: 'Number of components to use.' },
                { name: 'origin_cells', type: 'str', default: '""', desc: 'Name of the starting cell type (a value in groupby) — sets pseudotime direction.' },
            ],
            'slingshot': [
                { name: 'groupby', type: 'str', default: '"leiden"', desc: 'obs column for cell grouping (clustering results).' },
                { name: 'use_rep', type: 'str', default: '"X_pca"', desc: 'obsm key for the low-dimensional representation.' },
                { name: 'origin_cells', type: 'str', default: '""', desc: 'Starting cell type for pseudotime.' },
                { name: 'terminal_cells', type: 'str', default: '""', desc: 'Terminal cell type(s); separate multiple values with commas.' },
                { name: 'num_epochs', type: 'int', default: '1', desc: 'Number of training epochs for Slingshot.' },
            ],
            'palantir': [
                { name: 'groupby', type: 'str', default: '"leiden"', desc: 'obs column for cell grouping.' },
                { name: 'use_rep', type: 'str', default: '"X_pca"', desc: 'obsm key for the low-dimensional representation.' },
                { name: 'origin_cells', type: 'str', default: '""', desc: 'Starting cell type for pseudotime.' },
                { name: 'terminal_cells', type: 'str', default: '""', desc: 'Terminal cell type(s); separate multiple values with commas.' },
                { name: 'num_waypoints', type: 'int', default: '500', desc: 'Number of waypoints — more = more accurate trajectory but slower computation.' },
            ],
            'paga': [
                { name: 'groups', type: 'str', default: '"leiden"', desc: 'obs column used as PAGA graph nodes (clustering results).' },
                { name: 'use_rep', type: 'str', default: '"X_pca"', desc: 'Low-dimensional representation used when recomputing neighbours.' },
                { name: 'use_time_prior', type: 'str', default: 'None', desc: 'obs column with a prior pseudotime (e.g. dpt_pseudotime) to improve PAGA directionality.' },
                { name: 'basis', type: 'str', default: '"umap"', desc: 'Embedding for visualization (umap / tsne / draw_graph_fa).' },
            ],
            'sctour': [
                { name: 'groupby', type: 'str', default: '"leiden"', desc: 'obs column for cell grouping (clustering results).' },
                { name: 'use_rep', type: 'str', default: '"X_pca"', desc: 'obsm key for the low-dimensional representation.' },
                { name: 'n_comps', type: 'int', default: '50', desc: 'Number of components to use.' },
                { name: 'alpha_recon_lec', type: 'float', default: '0.5', desc: 'Weight for the lec reconstruction term (0–1).' },
                { name: 'alpha_recon_lode', type: 'float', default: '0.5', desc: 'Weight for the lode reconstruction term (0–1).' },
            ],
            'deg_expression': [
                { name: 'condition', type: 'str', default: '""', desc: 'obs column name used to split cells into groups (control vs. treatment).' },
                { name: 'ctrl_group', type: 'str', default: '""', desc: 'Name of the control group (a value in the condition column).' },
                { name: 'test_group', type: 'str', default: '""', desc: 'Name of the test/treatment group (a value in the condition column).' },
                { name: 'method', type: 'str', default: '"wilcoxon"', desc: 'Statistical test: wilcoxon (Wilcoxon rank-sum, recommended) or t-test.' },
                { name: 'celltype_key', type: 'str', default: '"leiden"', desc: 'obs column for cell-type labels — DEG is run separately for each cell type.' },
                { name: 'celltype_group', type: 'list|None', default: 'None', desc: 'List of cell types to analyse; None means all cell types.' },
                { name: 'max_cells', type: 'int', default: '100000', desc: 'Maximum cells per condition group — prevents memory overflow on large datasets.' },
            ],
            'dct': [
                { name: 'condition', type: 'str', default: '""', desc: 'obs column name used to split cells into groups (control vs. treatment).' },
                { name: 'ctrl_group', type: 'str', default: '""', desc: 'Name of the control group (a value in the condition column).' },
                { name: 'test_group', type: 'str', default: '""', desc: 'Name of the test/treatment group (a value in the condition column).' },
                { name: 'cell_type_key', type: 'str', default: '"leiden"', desc: 'obs column for cell-type annotation.' },
                { name: 'sample_key', type: 'str', default: 'None', desc: 'obs column identifying biological samples. Recommended for scCODA; required for milopy.' },
                { name: 'est_fdr (scCODA)', type: 'float', default: '0.2', desc: 'Estimated FDR threshold for scCODA (0.05–0.5); passed to model.set_fdr().' },
                { name: 'use_rep (milopy)', type: 'str', default: '"X_pca"', desc: 'obsm key for the low-dimensional embedding used by milopy to build the neighbour graph.' },
                { name: 'prop (milopy)', type: 'float', default: '0.1', desc: 'Neighbourhood sampling fraction in milopy — controls the number of neighbourhoods.' },
                { name: 'n_neighbors (milopy)', type: 'int', default: '150', desc: 'Number of neighbours for the milopy KNN graph — affects neighbourhood size.' },
            ],
        };
        return docs[tool] || [];
    }

});
