/**
 * OmicVerse Single Cell Analysis Platform - JavaScript Module
 */

class SingleCellAnalysis {
    constructor() {
        this.currentData = null;
        this.currentTool = null;
        this.currentTheme = 'light';
        this.currentView = 'visualization';
        this.codeCells = [];
        this.cellCounter = 0;

        // Initialize high-performance components
        this.dataManager = new DataManager();
        this.webglScatterplot = null;

        this.init();
    }

    init() {
        this.setupFileUpload();
        this.setupNavigation();
        this.setupThemeToggle();
        this.setupGeneAutocomplete();
        this.setupBeforeUnloadWarning();
        this.checkStatus();
        this.selectAnalysisCategory('preprocessing');
    }

    setupFileUpload() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        if (!dropZone || !fileInput) return;

        // Drag and drop events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Click anywhere in the dropZone to upload
        dropZone.addEventListener('click', (e) => {
            fileInput.click();
        });
    }

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
        const menuMiniButton = document.getElementById('menu-mini-button');
        
        if (mobileToggle) {
            mobileToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
        }

        if (menuMiniButton) {
            menuMiniButton.addEventListener('click', () => {
                this.toggleMiniMenu();
            });
        }
    }

    setupThemeToggle() {
        // Setup click handlers for existing theme toggle buttons
        const darkButton = document.getElementById('dark-button');
        const lightButton = document.getElementById('light-button');

        if (darkButton) {
            darkButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleTheme();
            });
        }

        if (lightButton) {
            lightButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleTheme();
            });
        }
    }

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
    }

    setupBeforeUnloadWarning() {
        // Warn user before leaving/refreshing if data is loaded
        window.addEventListener('beforeunload', (e) => {
            if (this.currentData) {
                // Modern browsers require returnValue to be set
                e.preventDefault();
                // Chrome requires returnValue to be set
                e.returnValue = '';
                // Some browsers show a custom message (though most modern browsers ignore it)
                return '您有未保存的数据，刷新页面将丢失所有分析结果。确定要离开吗？';
            }
        });
    }

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
    }

    toggleMobileMenu() {
        const navigation = document.querySelector('.nxl-navigation');
        navigation.classList.toggle('open');
    }

    toggleMiniMenu() {
        const navigation = document.querySelector('.nxl-navigation');
        navigation.classList.toggle('mini');
    }

    toggleTheme() {
        const html = document.documentElement;
        const darkButton = document.getElementById('dark-button');
        const lightButton = document.getElementById('light-button');
        
        if (html.classList.contains('app-skin-dark')) {
            // Switch to light mode
            html.classList.remove('app-skin-dark');
            darkButton.style.display = 'none';
            lightButton.style.display = 'block';
            localStorage.setItem('app-skin-dark', 'app-skin-light');
            this.currentTheme = 'light';
        } else {
            // Switch to dark mode
            html.classList.add('app-skin-dark');
            darkButton.style.display = 'block';
            lightButton.style.display = 'none';
            localStorage.setItem('app-skin-dark', 'app-skin-dark');
            this.currentTheme = 'dark';
        }
        
        // Update Plotly theme and status bar theme
        this.updatePlotlyTheme();
        this.updateStatusBarTheme();
    }

    handleFileUpload(file) {
        if (!file.name.endsWith('.h5ad')) {
            alert('请上传.h5ad格式的文件');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        this.showStatus('正在上传文件...', true);
        this.addToLog('开始上传文件: ' + file.name);

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const contentType = response.headers.get('content-type') || '';
            if (!response.ok) {
                let text = '';
                try { text = await response.text(); } catch (e) {}
                // Try JSON
                try {
                    const js = JSON.parse(text);
                    throw new Error(js.error || text || `HTTP ${response.status}`);
                } catch (e) {
                    if (text && text.trim().startsWith('<!DOCTYPE')) {
                        throw new Error(`服务器返回HTML错误页面 (HTTP ${response.status})`);
                    }
                    throw new Error(text || `HTTP ${response.status}`);
                }
            }
            if (contentType.includes('application/json')) {
                return response.json();
            }
            // Fallback: try parse text as JSON
            const text = await response.text();
            try { return JSON.parse(text); } catch (e) { throw new Error('服务器返回非JSON'); }
        })
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog('错误: ' + data.error, 'error');
                this.showStatus('上传失败: ' + data.error, false);
                alert('上传失败: ' + data.error);
            } else {
                this.currentData = data;
                this.updateUI(data);
                this.addToLog('文件上传成功: ' + data.n_cells + ' 细胞, ' + data.n_genes + ' 基因');
                this.showStatus('文件上传成功', false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog('上传失败: ' + error.message, 'error');
            this.showStatus('上传失败: ' + error.message, false);
            alert('上传失败: ' + error.message);
        });
    }

    updateUI(data) {
        // Hide upload section
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

        // Update embedding options
        const embeddingSelect = document.getElementById('embedding-select');
        embeddingSelect.innerHTML = '<option value="">选择降维方法</option>';
        data.embeddings.forEach(emb => {
            const option = document.createElement('option');
            option.value = emb;
            option.textContent = emb.toUpperCase();
            embeddingSelect.appendChild(option);
        });

        // Update color options
        const colorSelect = document.getElementById('color-select');
        colorSelect.innerHTML = '<option value="">无着色</option>';
        data.obs_columns.forEach(col => {
            const option = document.createElement('option');
            option.value = 'obs:' + col;
            option.textContent = col;
            colorSelect.appendChild(option);
        });

        // Update parameter panel to enable buttons
        this.updateParameterPanel();

        // Fetch gene list for autocomplete
        if (this.fetchGeneList) {
            this.fetchGeneList();
        }

        // Auto-select first embedding and update plot
        if (data.embeddings.length > 0) {
            embeddingSelect.value = data.embeddings[0];
            this.updatePlot();
        }
    }

    updateParameterPanel() {
        // Re-enable all parameter buttons now that data is loaded
        const buttons = document.querySelectorAll('#parameter-content button');
        buttons.forEach(button => {
            if (!button.onclick.toString().includes('showComingSoon')) {
                button.disabled = false;
            }
        });
    }

    updatePlot() {
        const embedding = document.getElementById('embedding-select').value;
        const colorBy = document.getElementById('color-select').value;

        if (!embedding) return;

        // Update palette visibility based on color type
        this.updatePaletteVisibility(colorBy);

        // 检查是否已经有图表存在
        const plotDiv = document.getElementById('plotly-div');
        const hasExistingPlot = plotDiv && plotDiv.data && plotDiv.data.length > 0;

        if (hasExistingPlot) {
            // 如果有现有图表，使用动画过渡
            this.updatePlotWithAnimation(embedding, colorBy);
        } else {
            // 如果没有现有图表，直接创建新图表
            this.createNewPlot(embedding, colorBy);
        }
    }

    updatePaletteVisibility(colorBy) {
        const categoryPaletteRow = document.getElementById('category-palette-row');
        const vminmaxRow = document.getElementById('vminmax-row');
        const paletteLabel = document.getElementById('palette-label');

        if (!colorBy || colorBy.startsWith('gene:')) {
            // Continuous data - show continuous palette and vmin/vmax, hide category palette
            if (categoryPaletteRow) categoryPaletteRow.style.display = 'none';
            if (vminmaxRow) vminmaxRow.style.display = 'flex';
            if (paletteLabel) paletteLabel.textContent = '调色板（连续）';
        } else if (colorBy.startsWith('obs:')) {
            // Check if it's categorical by trying to detect from obs columns
            // We'll let the backend determine this, but show category palette for now
            if (categoryPaletteRow) categoryPaletteRow.style.display = 'flex';
            // Also show vmin/vmax for now - backend will determine if it's continuous
            if (vminmaxRow) vminmaxRow.style.display = 'flex';
            if (paletteLabel) paletteLabel.textContent = '调色板（连续）';
        } else {
            // No coloring
            if (categoryPaletteRow) categoryPaletteRow.style.display = 'none';
            if (vminmaxRow) vminmaxRow.style.display = 'none';
        }
    }

    applyVMinMax() {
        // Just trigger an update
        this.updatePlot();
    }

    createNewPlot(embedding, colorBy) {
        this.currentEmbedding = this.currentEmbedding;
        this.showStatus('正在生成图表...', true);

        // Get selected palettes
        const paletteSelect = document.getElementById('palette-select');
        const categoryPaletteSelect = document.getElementById('category-palette-select');
        const palette = paletteSelect && paletteSelect.value !== 'default' ? paletteSelect.value : null;
        const categoryPalette = categoryPaletteSelect && categoryPaletteSelect.value !== 'default' ? categoryPaletteSelect.value : null;

        // Get vmin/vmax values
        const vminInput = document.getElementById('vmin-input');
        const vmaxInput = document.getElementById('vmax-input');
        const vmin = vminInput && vminInput.value ? parseFloat(vminInput.value) : null;
        const vmax = vmaxInput && vmaxInput.value ? parseFloat(vmaxInput.value) : null;

        fetch('/api/plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                embedding: embedding,
                color_by: colorBy,
                palette: palette,
                category_palette: categoryPalette,
                vmin: vmin,
                vmax: vmax
            })
        })
        .then(response => response.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog('绘图错误: ' + data.error, 'error');
                this.showStatus('绘图失败: ' + data.error, false);
            } else {
                this.plotData(data);
                this.currentEmbedding = embedding;
                this.showStatus('图表生成完成', false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog('绘图失败: ' + error.message, 'error');
            this.showStatus('绘图失败: ' + error.message, false);
        });
    }

    updatePlotWithAnimation(embedding, colorBy) {
        const isEmbeddingChange = (this.currentEmbedding !== embedding);
        this.showStatus(isEmbeddingChange ? '正在切换降维方法...' : '正在更新着色...', true);

        // Get selected palettes
        const paletteSelect = document.getElementById('palette-select');
        const categoryPaletteSelect = document.getElementById('category-palette-select');
        const palette = paletteSelect && paletteSelect.value !== 'default' ? paletteSelect.value : null;
        const categoryPalette = categoryPaletteSelect && categoryPaletteSelect.value !== 'default' ? categoryPaletteSelect.value : null;

        // Get vmin/vmax values
        const vminInput = document.getElementById('vmin-input');
        const vmaxInput = document.getElementById('vmax-input');
        const vmin = vminInput && vminInput.value ? parseFloat(vminInput.value) : null;
        const vmax = vmaxInput && vmaxInput.value ? parseFloat(vmaxInput.value) : null;

        fetch('/api/plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                embedding: embedding,
                color_by: colorBy,
                palette: palette,
                category_palette: categoryPalette,
                vmin: vmin,
                vmax: vmax
            })
        })
        .then(response => response.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog('绘图错误: ' + data.error, 'error');
                this.showStatus('绘图失败: ' + data.error, false);
            } else {
                if (!isEmbeddingChange) {
                    // 仅着色变化：不做位置动画
                    if (data.category_labels && data.category_codes) {
                        // 分类型：重绘为多trace，显示legend
                        this.plotData(data);
                    } else {
                        // 数值型：若当前是多trace（分类残留），需要重绘为单trace；否则仅restyle
                        const plotDiv = document.getElementById('plotly-div');
                        const isMulti = plotDiv && plotDiv.data && plotDiv.data.length > 1;
                        if (isMulti) {
                            this.plotData(data);
                        } else {
                            this.updateColorsOnly(data);
                        }
                    }
                    this.showStatus('着色更新完成', false);
                } else {
                    const plotDiv = document.getElementById('plotly-div');
                    let curX = [], curY = [];
                    if (plotDiv && plotDiv.data && plotDiv.data.length > 0) {
                        if (plotDiv.data.length > 1) {
                            // 合并多trace为单数组
                            for (let tr of plotDiv.data) {
                                if (tr && tr.x && tr.y) {
                                    curX = curX.concat(tr.x);
                                    curY = curY.concat(tr.y);
                                }
                            }
                        } else {
                            curX = (plotDiv.data[0].x || []).slice();
                            curY = (plotDiv.data[0].y || []).slice();
                        }
                    }
                    const minLen = Math.min(curX.length, (data.x||[]).length);
                    this.animatePositionTransitionForAnyData(curX, curY, data, minLen);
                    this.currentEmbedding = embedding;
                    this.showStatus('降维方法切换完成', false);
                }
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog('绘图失败: ' + error.message, 'error');
            this.showStatus('绘图失败: ' + error.message, false);
        });
    }

    animatePlotTransition(data) {
        const plotDiv = document.getElementById('plotly-div');
        
        // 检查是否需要动画过渡
        if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) {
            this.plotData(data);
            return;
        }
        
        // 检查当前图表类型和新数据类型
        const currentIsMultiTrace = plotDiv.data.length > 1;
        const newIsMultiTrace = data.category_labels && data.category_codes;
        
        // 获取所有当前数据点的坐标（合并多个trace）
        let currentX = [];
        let currentY = [];
        
        if (currentIsMultiTrace) {
            // 合并多个trace的数据
            for (let trace of plotDiv.data) {
                if (trace.x && trace.y) {
                    currentX = currentX.concat(trace.x);
                    currentY = currentY.concat(trace.y);
                }
            }
        } else {
            // 单个trace
            const currentData = plotDiv.data[0];
            if (!currentData || !currentData.x || !currentData.y) {
                this.plotData(data);
                return;
            }
            currentX = currentData.x;
            currentY = currentData.y;
        }
        
        const newX = data.x;
        const newY = data.y;
        
        // 确保数据长度一致
        const minLength = Math.min(currentX.length, newX.length);
        if (minLength === 0) {
            this.plotData(data);
            return;
        }
        
        // 检查坐标是否相同（只是着色变化）
        const coordsChanged = this.checkCoordsChanged(currentX, currentY, newX, newY, minLength);
        
        if (!coordsChanged) {
            // 坐标没变，只是着色变化
            if (newIsMultiTrace || currentIsMultiTrace) {
                // 如果涉及分类数据，直接重绘保证legend正确
                this.plotData(data);
            } else {
                // 数值数据使用快速颜色更新
                this.updateColorsOnly(data);
            }
            return;
        }
        
        // 坐标变化了，使用位置动画（无论是否分类数据）
        this.animatePositionTransitionForAnyData(currentX, currentY, data, minLength);
    }
    
    checkCoordsChanged(currentX, currentY, newX, newY, length) {
        const tolerance = 1e-10; // 浮点数比较容差
        for (let i = 0; i < Math.min(length, 100); i++) { // 只检查前100个点以提高性能
            if (Math.abs(currentX[i] - newX[i]) > tolerance || 
                Math.abs(currentY[i] - newY[i]) > tolerance) {
                return true;
            }
        }
        return false;
    }
    
    updateColorsOnly(data) {
        // 只更新颜色，保持位置不变
        let markerConfig = {
            size: data.size || 3,
            opacity: 0.7
        };
        
        if (data.colors) {
            markerConfig.color = data.colors;
            markerConfig.colorscale = data.colorscale || 'Viridis';
            markerConfig.showscale = true;
            markerConfig.colorbar = data.color_label ? {title: data.color_label} : undefined;
        } else {
            markerConfig.color = 'blue';
            markerConfig.showscale = false;
        }
        
        // 使用平滑的颜色过渡
        const duration = 300;
        const steps = 15;
        let currentStep = 0;
        
        const animateColors = () => {
            const progress = currentStep / steps;
            const easedProgress = this.easeInOutCubic(progress);
            
            // 更新marker配置，逐步改变透明度营造过渡效果
            const currentMarkerConfig = {
                ...markerConfig,
                opacity: 0.3 + (0.4 * easedProgress) // 从0.3过渡到0.7
            };
            
            const update = {
                marker: [currentMarkerConfig],
                text: [data.hover_text]
            };
            
            Plotly.restyle('plotly-div', update, [0]);
            
            currentStep++;
            
            if (currentStep <= steps) {
                setTimeout(animateColors, duration / steps);
            } else {
                // 最终确保状态正确
                const finalUpdate = {
                    marker: [markerConfig],
                    text: [data.hover_text]
                };
                Plotly.restyle('plotly-div', finalUpdate, [0]);
            }
        };
        
        animateColors();
    }
    
    animatePositionTransition(currentData, data, minLength) {
        const currentX = currentData.x;
        const currentY = currentData.y;
        const newX = data.x;
        const newY = data.y;
        
        // 创建位置过渡动画
        const duration = 500; // 动画持续时间
        const steps = 20; // 动画步数
        let currentStep = 0;
        
        const animate = () => {
            const progress = currentStep / steps;
            const easedProgress = this.easeInOutCubic(progress);
            
            // 计算当前帧的坐标
            const frameX = [];
            const frameY = [];
            
            for (let i = 0; i < minLength; i++) {
                frameX[i] = currentX[i] + (newX[i] - currentX[i]) * easedProgress;
                frameY[i] = currentY[i] + (newY[i] - currentY[i]) * easedProgress;
            }
            
            // 准备marker配置
            let markerConfig = {
                size: data.size || 3,
                opacity: 0.7
            };
            
            if (data.colors) {
                markerConfig.color = data.colors;
                markerConfig.colorscale = data.colorscale || 'Viridis';
                markerConfig.showscale = true;
                markerConfig.colorbar = data.color_label ? {title: data.color_label} : undefined;
            } else {
                markerConfig.color = currentData.marker.color || 'blue';
            }
            
            // 更新图表
            const update = {
                x: [frameX],
                y: [frameY],
                marker: [markerConfig],
                text: [data.hover_text]
            };
            
            Plotly.restyle('plotly-div', update, [0]);
            
            currentStep++;
            
            if (currentStep <= steps) {
                setTimeout(animate, duration / steps);
            }
        };
        
        // 开始动画
        animate();
    }
    
    animatePositionTransitionForAnyData(currentX, currentY, data, minLength) {
        // 逐点位置过渡 + 轴范围插值（仅嵌入切换使用）
        const plotDiv = document.getElementById('plotly-div');
        const duration = 600;
        const steps = 24;
        let currentStep = 0;

        const newX = data.x;
        const newY = data.y;

        // Ensure we animate a single trace during transition
        const isMulti = plotDiv && plotDiv.data && plotDiv.data.length > 1;

        // Capture current layout and ranges
        let layout = plotDiv && plotDiv.layout ? JSON.parse(JSON.stringify(plotDiv.layout)) : this.getPlotlyLayout();
        // Determine start ranges
        const startXRange = (layout && layout.xaxis && layout.xaxis.range) ? layout.xaxis.range.slice() : [Math.min(...currentX), Math.max(...currentX)];
        const startYRange = (layout && layout.yaxis && layout.yaxis.range) ? layout.yaxis.range.slice() : [Math.min(...currentY), Math.max(...currentY)];
        // Determine final ranges
        const endXRange = [Math.min(...newX), Math.max(...newX)];
        const endYRange = [Math.min(...newY), Math.max(...newY)];

        // If multiple traces, replace with a single anim trace once (preserve ranges)
        if (isMulti || !plotDiv || !plotDiv.data || plotDiv.data.length === 0) {
            // Build NaN arrays to hide unsampled points initially
            const initX = new Array(minLength).fill(NaN);
            const initY = new Array(minLength).fill(NaN);
            // We don't have sampleIdx yet here; we will fill in first animation frame
            const animTrace = {
                x: initX,
                y: initY,
                mode: 'markers',
                type: 'scattergl',
                marker: {
                    size: data.size || 3,
                    opacity: 0.7
                },
                showlegend: false
            };
            layout = this.getPlotlyLayout();
            layout.xaxis = layout.xaxis || {};
            layout.yaxis = layout.yaxis || {};
            layout.xaxis.autorange = false;
            layout.yaxis.autorange = false;
            layout.xaxis.range = startXRange;
            layout.yaxis.range = startYRange;
            Plotly.react('plotly-div', [animTrace], layout, {responsive: true});
        }

        // Precompute a representative sample of indices for animation across全体
        const sampleCount = Math.min(8000, minLength);
        const sampleIdx = [];
        if (minLength > 0) {
            const step = Math.max(1, Math.floor(minLength / sampleCount));
            for (let i = 0; i < minLength; i += step) sampleIdx.push(i);
            // ensure last index included
            if (sampleIdx[sampleIdx.length - 1] !== minLength - 1) sampleIdx.push(minLength - 1);
        }

        const animate = () => {
            const t = currentStep / steps;
            const eased = this.easeInOutCubic(t);
            // Interpolate positions (only for sampled indices; others NaN -> hidden)
            const frameX = new Array(minLength).fill(NaN);
            const frameY = new Array(minLength).fill(NaN);
            for (let k = 0; k < sampleIdx.length; k++) {
                const i = sampleIdx[k];
                frameX[i] = currentX[i] + (newX[i] - currentX[i]) * eased;
                frameY[i] = currentY[i] + (newY[i] - currentY[i]) * eased;
            }
            // Interpolate axis ranges
            const xr0 = startXRange[0] + (endXRange[0] - startXRange[0]) * eased;
            const xr1 = startXRange[1] + (endXRange[1] - startXRange[1]) * eased;
            const yr0 = startYRange[0] + (endYRange[0] - startYRange[0]) * eased;
            const yr1 = startYRange[1] + (endYRange[1] - startYRange[1]) * eased;

            Plotly.restyle('plotly-div', { x: [frameX], y: [frameY] }, [0]);
            Plotly.relayout('plotly-div', { 'xaxis.autorange': false, 'yaxis.autorange': false, 'xaxis.range': [xr0, xr1], 'yaxis.range': [yr0, yr1] });

            currentStep++;
            if (currentStep <= steps) {
                setTimeout(animate, duration / steps);
            } else {
                // Finalize
                this.plotData(data);
            }
        };

        animate();
    }
    
    // 缓动函数：三次贝塞尔曲线
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
    
    // 使用requestAnimationFrame实现流畅动画
    smoothAnimate(frames) {
        const startTime = performance.now();
        const duration = 800; // 总动画时长800ms
        let animationId;
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easedProgress = this.easeInOutCubic(progress);
            
            // 计算当前应该显示的帧
            const frameIndex = Math.floor(easedProgress * (frames.length - 1));
            const currentFrame = frames[frameIndex];
            
            // 更新图表，保持当前主题和legend
            const currentLayout = this.getPlotlyLayout();
            // 保持原有的annotations（如果有的话）
            if (this.currentAnnotations) {
                currentLayout.annotations = this.currentAnnotations;
            }
            Plotly.react('plotly-div', currentFrame.data, currentLayout, {
                staticPlot: false,
                responsive: true
            });
            
            if (progress < 1) {
                animationId = requestAnimationFrame(animate);
            } else {
                // 动画完成后，确保legend正确显示
                const finalLayout = this.getPlotlyLayout();
                if (this.currentAnnotations) {
                    finalLayout.annotations = this.currentAnnotations;
                }
                Plotly.relayout('plotly-div', finalLayout);
            }
        };
        
        // 开始动画
        animationId = requestAnimationFrame(animate);
        
        // 返回取消函数
        return () => {
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        };
    }

    plotData(data) {
        // 处理颜色配置
        let markerConfig = {
            size: data.size || 3,
            opacity: 0.7
        };

        let traces = [];

        if (data.colors) {
            // 处理分类数据的颜色条
            if (data.category_labels && data.category_codes) {
                // 分类数据：为每个类别创建单独的trace，使用plotly默认legend
                const uniqueCategories = data.category_labels;
                const uniqueColors = data.discrete_colors;

                for (let i = 0; i < uniqueCategories.length; i++) {
                    const category = uniqueCategories[i];
                    const color = uniqueColors[i];

                    // 找到属于当前类别的点
                    const categoryIndices = [];
                    const categoryX = [];
                    const categoryY = [];
                    const categoryText = [];

                    for (let j = 0; j < data.category_codes.length; j++) {
                        if (data.category_codes[j] === i) {
                            categoryIndices.push(j);
                            categoryX.push(data.x[j]);
                            categoryY.push(data.y[j]);
                            categoryText.push(data.hover_text[j]);
                        }
                    }

                    // 只有当该类别有数据点时才创建trace
                    if (categoryX.length > 0) {
                        const trace = {
                            x: categoryX,
                            y: categoryY,
                            mode: 'markers',
                            type: 'scattergl',
                            name: category, // 设置trace名称，这将显示在legend中
                            marker: {
                                color: color,
                                size: data.size || 3,
                                opacity: 0.7
                            },
                            text: categoryText,
                            hovertemplate: '%{text}<extra></extra>',
                            showlegend: true // 启用legend显示
                        };

                        traces.push(trace);
                    }
                }
            } else {
                // 数值数据：使用连续颜色映射
                markerConfig.color = data.colors;
                markerConfig.colorscale = data.colorscale || 'Viridis';
                markerConfig.showscale = true;
                markerConfig.colorbar = data.color_label ? {title: data.color_label} : undefined;

                // Apply cmin/cmax if specified
                if (data.cmin !== undefined) {
                    markerConfig.cmin = data.cmin;
                }
                if (data.cmax !== undefined) {
                    markerConfig.cmax = data.cmax;
                }

                const trace = {
                    x: data.x,
                    y: data.y,
                    mode: 'markers',
                    type: 'scattergl',
                    marker: markerConfig,
                    text: data.hover_text,
                    hovertemplate: '%{text}<extra></extra>',
                    showlegend: false // 数值数据不显示legend
                };

                traces.push(trace);
            }
        } else {
            // 没有颜色数据时使用默认颜色
            markerConfig.color = 'blue';
            markerConfig.showscale = false;

            const trace = {
                x: data.x,
                y: data.y,
                mode: 'markers',
                type: 'scattergl',
                marker: markerConfig,
                text: data.hover_text,
                hovertemplate: '%{text}<extra></extra>',
                showlegend: false
            };

            traces.push(trace);
        }

        const layout = this.getPlotlyLayout();
        
        // 清除自定义annotations，使用plotly默认legend
        layout.annotations = [];
        this.currentAnnotations = null;
        
        const config = {responsive: true};

        // 确保至少有一个trace
        if (traces.length === 0) {
            console.warn('No traces to plot, creating default trace');
            const defaultTrace = {
                x: data.x || [],
                y: data.y || [],
                mode: 'markers',
                type: 'scattergl',
                marker: {
                    color: 'blue',
                    size: data.size || 3,
                    opacity: 0.7
                },
                text: data.hover_text || [],
                hovertemplate: '%{text}<extra></extra>',
                showlegend: false
            };
            traces.push(defaultTrace);
        }
        
        console.log('Plotting traces:', traces.length, 'traces');
        console.log('First trace sample:', traces[0] ? {
            x_length: traces[0].x ? traces[0].x.length : 0,
            y_length: traces[0].y ? traces[0].y.length : 0,
            marker: traces[0].marker
        } : 'no traces');

        // 直接使用 Plotly.react 来确保legend正确更新
        Plotly.react('plotly-div', traces, layout, config);
    }

    getPlotlyLayout() {
        const isDark = document.documentElement.classList.contains('app-skin-dark');
        
        const baseLayout = {
            title: {
                text: document.getElementById('embedding-select').value.toUpperCase() + ' Plot',
                font: {color: isDark ? '#e5e7eb' : '#283c50'}
            },
            xaxis: {
                title: 'Dimension 1',
                color: isDark ? '#e5e7eb' : '#283c50',
                gridcolor: isDark ? '#374151' : '#e5e7eb',
                linecolor: isDark ? '#4b5563' : '#d1d5db'
            },
            yaxis: {
                title: 'Dimension 2',
                color: isDark ? '#e5e7eb' : '#283c50',
                gridcolor: isDark ? '#374151' : '#e5e7eb',
                linecolor: isDark ? '#4b5563' : '#d1d5db'
            },
            hovermode: 'closest',
            showlegend: true, // 启用plotly默认legend
            legend: {
                x: 1.02, // 将legend放在图表右侧
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: isDark ? 'rgba(31,41,55,0.8)' : 'rgba(255,255,255,0.8)',
                bordercolor: isDark ? 'rgba(75,85,99,0.3)' : 'rgba(0,0,0,0.1)',
                borderwidth: 1,
                font: {
                    color: isDark ? '#e5e7eb' : '#283c50',
                    size: 12
                }
            },
            margin: {l: 50, r: 150, t: 50, b: 50} // 为legend留出空间
        };

        if (isDark) {
            baseLayout.paper_bgcolor = '#1f2937';
            baseLayout.plot_bgcolor = '#1f2937';
        } else {
            baseLayout.paper_bgcolor = '#ffffff';
            baseLayout.plot_bgcolor = '#ffffff';
        }

        return baseLayout;
    }

    updatePlotlyTheme() {
        // If there's an existing plot, update it with new theme
        const plotDiv = document.getElementById('plotly-div');
        if (plotDiv && plotDiv.data) {
            const layout = this.getPlotlyLayout();
            Plotly.relayout(plotDiv, layout);
        }
    }

    colorByGene() {
        const gene = document.getElementById('gene-input').value.trim();
        if (!gene) return;

        const embedding = document.getElementById('embedding-select').value;
        if (!embedding) {
            alert('请先选择降维方法');
            return;
        }

        // Update palette visibility for gene expression (continuous)
        this.updatePaletteVisibility('gene:' + gene);

        // 检查是否已经有图表存在
        const plotDiv = document.getElementById('plotly-div');
        const hasExistingPlot = plotDiv && plotDiv.data && plotDiv.data.length > 0;

        if (hasExistingPlot) {
            this.showStatus('正在更新基因表达...', true);
        } else {
            this.showStatus('正在加载基因表达...', true);
        }

        // Get selected palette (only continuous for gene expression)
        const paletteSelect = document.getElementById('palette-select');
        const palette = paletteSelect && paletteSelect.value !== 'default' ? paletteSelect.value : null;

        // Get vmin/vmax values
        const vminInput = document.getElementById('vmin-input');
        const vmaxInput = document.getElementById('vmax-input');
        const vmin = vminInput && vminInput.value ? parseFloat(vminInput.value) : null;
        const vmax = vmaxInput && vmaxInput.value ? parseFloat(vmaxInput.value) : null;

        fetch('/api/plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                embedding: embedding,
                color_by: 'gene:' + gene,
                palette: palette,
                category_palette: null,
                vmin: vmin,
                vmax: vmax
            })
        })
        .then(response => response.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog('基因表达错误: ' + data.error, 'error');
                this.showStatus('基因未找到: ' + gene, false);
                alert('基因未找到: ' + gene);
            } else {
                this.plotData(data);
                this.addToLog('显示基因表达: ' + gene);
                this.showStatus('基因表达加载完成', false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog('基因表达加载失败: ' + error.message, 'error');
            this.showStatus('基因表达加载失败: ' + error.message, false);
        });
    }

    selectAnalysisCategory(category) {
        this.currentCategory = category;
        const parameterContent = document.getElementById('parameter-content');
        
        // Clear previous content
        parameterContent.innerHTML = '';
        
        // Generate category-specific tools
        const categoryTools = {
            // 预处理：归一化、对数化、缩放
            'preprocessing': [
                { id: 'normalize', name: '归一化', icon: 'fas fa-balance-scale', desc: 'Total-count 归一化' },
                { id: 'log1p', name: '对数转换', icon: 'fas fa-calculator', desc: '自然对数 log1p' },
                { id: 'scale', name: '数据缩放', icon: 'fas fa-expand-arrows-alt', desc: 'Z-score 标准化' }
            ],
            // 质量控制：过滤细胞、过滤基因、过滤异常细胞、去除双细胞
            'qc': [
                { id: 'filter_cells', name: '过滤细胞', icon: 'fas fa-filter', desc: '按UMI/基因数上下限过滤' },
                { id: 'filter_genes', name: '过滤基因', icon: 'fas fa-tasks', desc: '按表达细胞数/UMI上下限过滤' },
                { id: 'filter_outliers', name: '过滤异常细胞', icon: 'fas fa-exclamation-triangle', desc: '先计算QC，再按线粒体比例等过滤' },
                { id: 'doublets', name: '去除双细胞', icon: 'fas fa-user-times', desc: '识别并去除潜在双细胞（Scrublet）' }
            ],
            // 特征选择：高变基因
            'feature': [
                { id: 'hvg', name: '高变基因', icon: 'fas fa-dna', desc: '选择高变基因' }
            ],
            // 降维
            'dimreduction': [
                { id: 'pca', name: 'PCA分析', icon: 'fas fa-chart-line', desc: '主成分分析' },
                { id: 'umap', name: 'UMAP降维', icon: 'fas fa-map', desc: '统一流形近似投影' },
                { id: 'tsne', name: 't-SNE降维', icon: 'fas fa-dot-circle', desc: 't-分布随机邻域嵌入' }
            ],
            // 聚类
            'clustering': [
                { id: 'neighbors', name: '邻域计算', icon: 'fas fa-network-wired', desc: 'K近邻图构建' },
                { id: 'leiden', name: 'Leiden聚类', icon: 'fas fa-object-group', desc: '高质量社区检测' },
                { id: 'louvain', name: 'Louvain聚类', icon: 'fas fa-layer-group', desc: '经典社区检测' }
            ],
            'omicverse': [
                { id: 'coming_soon', name: '细胞注释', icon: 'fas fa-tag', desc: '自动细胞类型注释' },
                { id: 'coming_soon', name: '轨迹分析', icon: 'fas fa-route', desc: '细胞发育轨迹' },
                { id: 'coming_soon', name: '差异分析', icon: 'fas fa-not-equal', desc: '差异表达基因' },
                { id: 'coming_soon', name: '功能富集', icon: 'fas fa-sitemap', desc: 'GO/KEGG富集分析' }
            ]
        };
        
        const tools = categoryTools[category] || [];
        
        tools.forEach(tool => {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'mb-3 p-3 border rounded fade-in c-pointer';
            toolDiv.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <i class="${tool.icon} me-2 text-primary"></i>
                    <strong>${tool.name}</strong>
                </div>
                <p class="text-muted small mb-0">${tool.desc}</p>`;
            if (tool.id === 'coming_soon') {
                toolDiv.onclick = () => this.showComingSoon();
            } else if (this.currentData) {
                toolDiv.onclick = () => this.renderParameterForm(tool.id, tool.name, tool.desc, category);
            } else {
                toolDiv.style.opacity = 0.6;
                toolDiv.title = '请先上传数据';
            }
            parameterContent.appendChild(toolDiv);
        });
        
        this.addToLog(`选择分析类别: ${this.getCategoryName(category)}`);
    }

    getCategoryName(category) {
        const names = {
            'preprocessing': '数据预处理',
            'dimreduction': '降维分析',
            'clustering': '聚类分析',
            'omicverse': 'OmicVerse工具'
        };
        return names[category] || category;
    }

    showParameterDialog(tool) { this.renderParameterForm(tool); }

    renderParameterForm(tool, toolName = '', toolDesc = '') {
        this.currentTool = tool;
        const parameterContent = document.getElementById('parameter-content');
        const toolNames = {
            'normalize': '标准化',
            'scale': '数据缩放',
            'hvg': '高变基因选择',
            'pca': 'PCA分析',
            'umap': 'UMAP降维',
            'tsne': 't-SNE降维',
            'neighbors': '邻域计算',
            'leiden': 'Leiden聚类',
            'louvain': 'Louvain聚类',
            'log1p': '对数转换'
        };
        const title = toolName || toolNames[tool] || '参数设置';
        const desc = toolDesc || '';

        const formHTML = `
            <div class="mb-3">
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div>
                        <h6 class="mb-1"><i class="fas fa-sliders-h me-2 text-primary"></i>${title}</h6>
                        ${desc ? `<small class="text-muted">${desc}</small>` : ''}
                    </div>
                    <button class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.selectAnalysisCategory('${singleCellApp.currentCategory || 'preprocessing'}')">返回工具列表</button>
                </div>
                <div class="border rounded p-3">
                    ${this.getParameterHTML(tool)}
                    <div class="d-grid mt-3">
                        <button class="btn btn-primary" id="inlineRunBtn">运行</button>
                    </div>
                </div>
            </div>`;
        parameterContent.innerHTML = formHTML;

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
    }

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
            `
        };

        return parameters[tool] || '<p>该工具无需参数设置</p>';
    }

    runTool(tool, params = {}) {
        if (!this.currentData) {
            alert('请先上传数据');
            return;
        }

        const toolNames = {
            'normalize': '标准化',
            'log1p': '对数转换',
            'scale': '数据缩放',
            'hvg': '高变基因选择',
            'pca': 'PCA分析',
            'umap': 'UMAP降维',
            'tsne': 't-SNE降维',
            'neighbors': '邻域计算',
            'leiden': 'Leiden聚类',
            'louvain': 'Louvain聚类',
            'filter_cells': '过滤细胞',
            'filter_genes': '过滤基因',
            'filter_outliers': '过滤异常细胞',
            'doublets': '去除双细胞'
        };
        const toolName = toolNames[tool] || tool;
        this.showStatus(`正在执行${toolName}...`, true);
        this.addToLog(`开始执行: ${toolName}`);

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
                this.addToLog(`${toolName}失败: ${data.error}`, 'error');
                this.showStatus(`${toolName}执行失败: ${data.error}`, false);
                alert(`执行失败: ${data.error}`);
            } else {
                this.currentData = data;
                this.updateUI(data);
                this.addToLog(`${toolName}完成`);
                this.showStatus(`${toolName}执行完成`, false);
                
                // Auto-update plot if embedding is available
                const embeddingSelect = document.getElementById('embedding-select');
                if (embeddingSelect.value) {
                    this.updatePlot();
                }
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(`${toolName}失败: ${error.message}`, 'error');
            this.showStatus(`${toolName}执行失败: ${error.message}`, false);
            alert(`执行失败: ${error.message}`);
        });
    }

    saveData() {
        if (!this.currentData) return;
        
        this.showStatus('正在下载处理后的数据...', true);
        this.addToLog('开始下载处理后的数据...');
        
        fetch('/api/save', {
            method: 'POST'
        })
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                throw new Error('保存失败');
            }
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `processed_${this.currentData.filename}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            this.hideStatus();
            this.addToLog('数据保存成功');
            this.showStatus('数据保存成功', false);
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog('保存失败: ' + error.message, 'error');
            this.showStatus('保存失败: ' + error.message, false);
        });
    }

    resetData() {
        if (confirm('确定要重置所有数据吗？所有未保存的分析结果将丢失。')) {
            this.currentData = null;
            document.getElementById('upload-section').style.display = 'block';
            document.getElementById('data-status').classList.add('d-none');
            document.getElementById('viz-controls').style.display = 'none';
            document.getElementById('viz-panel').style.display = 'none';
            document.getElementById('analysis-log').innerHTML = '<div class="text-muted">等待上传数据...</div>';
            document.getElementById('fileInput').value = '';

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
        }
    }

    checkStatus() {
        fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data.loaded) {
                // If data is already loaded, update UI accordingly
                this.currentData = {
                    filename: data.filename,
                    n_cells: data.cells,
                    n_genes: data.genes
                };
                // Note: This is a simplified status, you might want to fetch full data
            }
        })
        .catch(error => {
            console.log('Status check failed:', error);
        });
    }

    showLoading(text = '正在处理...') {
        const loadingText = document.getElementById('loading-text');
        const loadingOverlay = document.getElementById('loading-overlay');
        
        if (loadingText) loadingText.textContent = text;
        if (loadingOverlay) loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) loadingOverlay.style.display = 'none';
    }

    addToLog(message, type = 'info') {
        const log = document.getElementById('analysis-log');
        if (!log) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const className = type === 'error' ? 'text-danger' : 'text-dark';
        
        const logEntry = document.createElement('div');
        logEntry.className = `mb-1 ${className}`;
        logEntry.innerHTML = `<small class="text-muted">[${timestamp}]</small> ${message}`;
        
        log.appendChild(logEntry);
        log.scrollTop = log.scrollHeight;
    }

    // 状态栏管理方法
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
    }

    hideStatus() {
        const statusBar = document.getElementById('status-bar');
        if (statusBar) {
            statusBar.style.display = 'none';
        }
    }

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
    }

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
    }

    // 获取分类数据的调色板
    getCategoricalColorscale(nCategories) {
        // 根据类别数量选择合适的调色板
        if (nCategories <= 3) {
            return 'Set1';
        } else if (nCategories <= 5) {
            return 'Set1';
        } else if (nCategories <= 8) {
            return 'Set2';
        } else if (nCategories <= 10) {
            return 'Set3';
        } else if (nCategories <= 12) {
            return 'Paired';
        } else if (nCategories <= 20) {
            return 'Set3';
        } else {
            return 'Plotly3';
        }
    }

    showComingSoon() {
        alert('该功能正在开发中，敬请期待！');
    }

    // View switching
    switchView(view) {
        this.currentView = view;

        const vizView = document.getElementById('visualization-view');
        const codeView = document.getElementById('code-editor-view');
        const vizBtn = document.getElementById('view-viz-btn');
        const codeBtn = document.getElementById('view-code-btn');

        if (view === 'visualization') {
            vizView.style.display = 'block';
            codeView.style.display = 'none';
            vizBtn.classList.remove('btn-outline-primary');
            vizBtn.classList.add('btn-primary');
            codeBtn.classList.remove('btn-primary');
            codeBtn.classList.add('btn-outline-primary');
        } else if (view === 'code') {
            vizView.style.display = 'none';
            codeView.style.display = 'block';
            vizBtn.classList.remove('btn-primary');
            vizBtn.classList.add('btn-outline-primary');
            codeBtn.classList.remove('btn-outline-primary');
            codeBtn.classList.add('btn-primary');

            // Add a default cell if none exists
            if (this.codeCells.length === 0) {
                this.addCodeCell();
            }
        }
    }

    // Code cell management
    addCodeCell(code = '') {
        this.cellCounter++;
        const cellId = `cell-${this.cellCounter}`;

        const cellHtml = `
            <div class="code-cell" id="${cellId}">
                <div class="code-cell-header">
                    <span class="cell-number">In [${this.cellCounter}]:</span>
                    <div>
                        <button type="button" class="btn btn-sm btn-success me-1" onclick="singleCellApp.runCodeCell('${cellId}')">
                            <i class="feather-play"></i> 运行
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-danger" onclick="singleCellApp.deleteCodeCell('${cellId}')">
                            <i class="feather-trash-2"></i>
                        </button>
                    </div>
                </div>
                <div class="code-cell-input">
                    <textarea class="code-input" placeholder="# 输入Python代码...
# 可用变量:
#   adata - 当前AnnData对象
#
# 示例:
#   print(adata)
#   print(adata.obs.columns)
#   print(adata.var_names[:10])">${code}</textarea>
                </div>
                <div class="code-cell-output" id="${cellId}-output"></div>
            </div>
        `;

        const container = document.getElementById('code-cells-container');
        container.insertAdjacentHTML('beforeend', cellHtml);

        this.codeCells.push(cellId);

        // Add keyboard shortcut (Shift+Enter to run)
        const textarea = document.querySelector(`#${cellId} .code-input`);
        textarea.addEventListener('keydown', (e) => {
            if (e.shiftKey && e.key === 'Enter') {
                e.preventDefault();
                this.runCodeCell(cellId);
            }
        });
    }

    runCodeCell(cellId) {
        const cell = document.getElementById(cellId);
        const textarea = cell.querySelector('.code-input');
        const outputDiv = cell.querySelector('.code-cell-output');
        const code = textarea.value.trim();

        if (!code) {
            return;
        }

        // Show loading
        outputDiv.className = 'code-cell-output has-content';
        outputDiv.textContent = '执行中...';

        // Execute code on backend
        fetch('/api/execute_code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                code: code
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                outputDiv.className = 'code-cell-output has-content error';
                outputDiv.textContent = data.error;
            } else {
                outputDiv.className = 'code-cell-output has-content success';
                let output = '';
                if (data.output) {
                    output += data.output;
                }
                if (data.result !== null && data.result !== undefined) {
                    if (output) output += '\n';
                    output += `Out: ${data.result}`;
                }
                if (data.data_updated) {
                    if (output) output += '\n';
                    output += '\n✓ AnnData对象已更新';
                    // Refresh data info
                    this.checkStatus();
                }
                outputDiv.textContent = output || '(执行成功，无输出)';
            }
        })
        .catch(error => {
            outputDiv.className = 'code-cell-output has-content error';
            outputDiv.textContent = `错误: ${error.message}`;
        });
    }

    deleteCodeCell(cellId) {
        if (confirm('确定要删除这个代码单元吗？')) {
            const cell = document.getElementById(cellId);
            cell.remove();
            this.codeCells = this.codeCells.filter(id => id !== cellId);
        }
    }

    clearAllCells() {
        if (confirm('确定要清空所有代码单元吗？')) {
            const container = document.getElementById('code-cells-container');
            container.innerHTML = '';
            this.codeCells = [];
            this.cellCounter = 0;
            // Add one empty cell
            this.addCodeCell();
        }
    }

    insertTemplate() {
        const templates = {
            'basic_info': `# 查看基本信息
print(adata)
print(f"细胞数: {adata.n_obs}")
print(f"基因数: {adata.n_vars}")`,
            'obs_info': `# 查看观测值列
print(adata.obs.columns)
print(adata.obs.head())`,
            'filter': `# 过滤细胞
# 保留基因数在200-5000之间的细胞
import scanpy as sc
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, max_genes=5000)
print(f"过滤后细胞数: {adata.n_obs}")`,
            'normalize': `# 标准化
import scanpy as sc
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("标准化完成")`,
            'hvg': `# 高变基因选择
import scanpy as sc
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
print(f"高变基因数: {adata.var.highly_variable.sum()}")`,
        };

        const templateKeys = Object.keys(templates);
        let options = templateKeys.map((key, idx) =>
            `${idx + 1}. ${key.replace('_', ' ')}`
        ).join('\n');

        const choice = prompt(`选择模板:\n${options}\n\n输入编号 (1-${templateKeys.length}):`);
        if (choice) {
            const idx = parseInt(choice) - 1;
            if (idx >= 0 && idx < templateKeys.length) {
                const templateKey = templateKeys[idx];
                this.addCodeCell(templates[templateKey]);
            }
        }
    }

    // 测试legend显示的方法
    testLegend() {
        console.log('开始测试legend显示...');
        
        // 创建测试数据
        const testData = {
            category_labels: ['T细胞', 'B细胞', 'NK细胞'],
            discrete_colors: ['#e41a1c', '#377eb8', '#4daf4a'],
            x: [1, 2, 3, 4, 5],
            y: [1, 2, 3, 4, 5],
            colors: ['#e41a1c', '#377eb8', '#4daf4a', '#e41a1c', '#377eb8'],
            hover_text: ['T细胞', 'B细胞', 'NK细胞', 'T细胞', 'B细胞']
        };
        
        console.log('测试数据:', testData);
        
        // 直接调用plotData测试
        this.plotData(testData);
        
        console.log('测试完成，请检查图表右上角是否有legend');
    }

    // 强制显示legend的方法
    forceShowLegend() {
        console.log('强制显示legend...');
        
        const plotDiv = document.getElementById('plotly-div');
        if (!plotDiv) {
            console.error('找不到plotly-div元素');
            return;
        }
        
        // 创建简单的legend
        const legendItems = [
            {
                x: 0.98,
                y: 0.95,
                xref: 'paper',
                yref: 'paper',
                text: '● T细胞',
                showarrow: false,
                font: {size: 14, color: '#e41a1c'},
                align: 'right',
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: 'rgba(0,0,0,0.1)',
                borderwidth: 1,
                borderpad: 6,
                width: 120,
                height: 30
            },
            {
                x: 0.98,
                y: 0.87,
                xref: 'paper',
                yref: 'paper',
                text: '● B细胞',
                showarrow: false,
                font: {size: 14, color: '#377eb8'},
                align: 'right',
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: 'rgba(0,0,0,0.1)',
                borderwidth: 1,
                borderpad: 6,
                width: 120,
                height: 30
            }
        ];
        
        // 直接更新layout
        Plotly.relayout('plotly-div', {
            annotations: legendItems,
            margin: {l: 50, r: 200, t: 50, b: 50}
        });
        
        console.log('强制显示legend完成');
    }

    // 最简单的legend测试
    simpleLegendTest() {
        console.log('开始最简单的legend测试...');
        
        // 创建最简单的图表
        const data = [{
            x: [1, 2, 3, 4, 5],
            y: [1, 2, 3, 4, 5],
            mode: 'markers',
            type: 'scattergl',
            marker: {color: 'red', size: 10}
        }];
        
        const layout = {
            title: 'Legend Test',
            annotations: [{
                x: 0.5,
                y: 0.5,
                xref: 'paper',
                yref: 'paper',
                text: '● Test Legend',
                showarrow: false,
                font: {size: 16, color: 'red'},
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: 'rgba(0,0,0,0.1)',
                borderwidth: 1
            }],
            margin: {l: 50, r: 50, t: 50, b: 50}
        };
        
        Plotly.newPlot('plotly-div', data, layout);
        console.log('简单legend测试完成，应该能看到图表中央的legend');
    }

    // 立即显示legend的方法
    showLegendNow() {
        console.log('立即显示legend...');
        
        const plotDiv = document.getElementById('plotly-div');
        if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) {
            console.error('没有找到图表或图表为空');
            return;
        }
        
        // 检查当前数据是否有颜色信息
        const currentData = plotDiv.data[0];
        console.log('当前图表数据:', currentData);
        
        // 创建简单的legend
        const legendItems = [
            {
                x: 0.98,
                y: 0.95,
                xref: 'paper',
                yref: 'paper',
                text: '● Cluster 0',
                showarrow: false,
                font: {size: 14, color: '#ff7f0e'},
                align: 'right',
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: 'rgba(0,0,0,0.2)',
                borderwidth: 1,
                borderpad: 6,
                width: 120,
                height: 30
            },
            {
                x: 0.98,
                y: 0.87,
                xref: 'paper',
                yref: 'paper',
                text: '● Cluster 1',
                showarrow: false,
                font: {size: 14, color: '#2ca02c'},
                align: 'right',
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: 'rgba(0,0,0,0.2)',
                borderwidth: 1,
                borderpad: 6,
                width: 120,
                height: 30
            },
            {
                x: 0.98,
                y: 0.79,
                xref: 'paper',
                yref: 'paper',
                text: '● Cluster 2',
                showarrow: false,
                font: {size: 14, color: '#d62728'},
                align: 'right',
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: 'rgba(0,0,0,0.2)',
                borderwidth: 1,
                borderpad: 6,
                width: 120,
                height: 30
            }
        ];
        
        // 直接更新layout
        Plotly.relayout('plotly-div', {
            annotations: legendItems,
            margin: {l: 50, r: 200, t: 50, b: 50}
        });
        
        console.log('Legend已添加到图表右上角');
    }
}

// Global functions for backward compatibility
let singleCellApp;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    singleCellApp = new SingleCellAnalysis();
    
    // Setup parameter dialog
    document.getElementById('runToolBtn').addEventListener('click', function() {
        const params = {};
        const inputs = document.querySelectorAll('#parameterModal input, #parameterModal select');
        
        inputs.forEach(input => {
            if (input.type === 'number') {
                params[input.id] = parseFloat(input.value);
            } else {
                params[input.id] = input.value;
            }
        });

        bootstrap.Modal.getInstance(document.getElementById('parameterModal')).hide();
        singleCellApp.runTool(singleCellApp.currentTool, params);
    });
    
    // Load saved theme - based on moban7855 template
    const savedTheme = localStorage.getItem('app-skin-dark') || 'app-skin-light';
    const html = document.documentElement;
    const darkButton = document.getElementById('dark-button');
    const lightButton = document.getElementById('light-button');
    
    if (savedTheme === 'app-skin-dark') {
        html.classList.add('app-skin-dark');
        if (darkButton) darkButton.style.display = 'block';
        if (lightButton) lightButton.style.display = 'none';
        singleCellApp.currentTheme = 'dark';
    } else {
        html.classList.remove('app-skin-dark');
        if (darkButton) darkButton.style.display = 'none';
        if (lightButton) lightButton.style.display = 'block';
        singleCellApp.currentTheme = 'light';
    }
});

// Global functions for HTML onclick handlers
function selectAnalysisCategory(category) {
        this.currentCategory = category;
    singleCellApp.selectAnalysisCategory(category);
}

function updatePlot() {
    singleCellApp.updatePlot();
}

function colorByGene() {
    singleCellApp.colorByGene();
}

function showParameterDialog(tool) {
    singleCellApp.showParameterDialog(tool);
}

function runTool(tool, params) {
    singleCellApp.runTool(tool, params);
}

function saveData() {
    singleCellApp.saveData();
}

function resetData() {
    singleCellApp.resetData();
}

function showComingSoon() {
    singleCellApp.showComingSoon();
}
