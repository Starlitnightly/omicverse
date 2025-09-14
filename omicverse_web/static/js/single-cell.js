/**
 * OmicVerse Single Cell Analysis Platform - JavaScript Module
 */

class SingleCellAnalysis {
    constructor() {
        this.currentData = null;
        this.currentTool = null;
        this.currentTheme = 'light';
        this.init();
    }

    init() {
        this.setupFileUpload();
        this.setupNavigation();
        this.setupThemeToggle();
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

        // Click to upload - prevent event bubbling
        dropZone.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
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
        .then(response => response.json())
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

    createNewPlot(embedding, colorBy) {
        this.showStatus('正在生成图表...', true);

        fetch('/api/plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                embedding: embedding,
                color_by: colorBy
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
        // 显示状态栏而不是加载圆圈
        this.showStatus('正在切换降维方法...', true);

        fetch('/api/plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                embedding: embedding,
                color_by: colorBy
            })
        })
        .then(response => response.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog('绘图错误: ' + data.error, 'error');
                this.showStatus('绘图失败: ' + data.error, false);
            } else {
                this.animatePlotTransition(data); // 使用动画过渡
                this.showStatus('降维方法切换完成', false);
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
        // 创建位置过渡动画，适用于任何数据类型
        const duration = 500; // 动画持续时间
        const steps = 20; // 动画步数
        let currentStep = 0;
        
        const newX = data.x;
        const newY = data.y;
        
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
            
            // 创建临时的单一trace进行动画
            const tempTrace = {
                x: frameX,
                y: frameY,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: 'rgba(31, 119, 180, 0.6)', // 临时颜色
                    size: data.size || 3,
                    opacity: 0.6
                },
                showlegend: false
            };
            
            const layout = this.getPlotlyLayout();
            layout.annotations = []; // 清除annotations
            
            // 使用react更新图表
            Plotly.react('plotly-div', [tempTrace], layout, {responsive: true});
            
            currentStep++;
            
            if (currentStep <= steps) {
                setTimeout(animate, duration / steps);
            } else {
                // 动画完成，显示最终的正确图表
                this.plotData(data);
            }
        };
        
        // 开始动画
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
                            type: 'scatter',
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
                
                const trace = {
                    x: data.x,
                    y: data.y,
                    mode: 'markers',
                    type: 'scatter',
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
                type: 'scatter',
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
                type: 'scatter',
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

        // 检查是否已经有图表存在
        const plotDiv = document.getElementById('plotly-div');
        const hasExistingPlot = plotDiv && plotDiv.data && plotDiv.data.length > 0;

        if (hasExistingPlot) {
            this.showStatus('正在更新基因表达...', true);
        } else {
            this.showStatus('正在加载基因表达...', true);
        }

        fetch('/api/plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                embedding: embedding,
                color_by: 'gene:' + gene
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
        const parameterContent = document.getElementById('parameter-content');
        
        // Clear previous content
        parameterContent.innerHTML = '';
        
        // Generate category-specific tools
        const categoryTools = {
            'preprocessing': [
                { id: 'normalize', name: '标准化', icon: 'fas fa-balance-scale', desc: '细胞总数标准化' },
                { id: 'log1p', name: '对数转换', icon: 'fas fa-calculator', desc: 'Log1p转换' },
                { id: 'scale', name: '数据缩放', icon: 'fas fa-expand-arrows-alt', desc: 'Z-score标准化' },
                { id: 'hvg', name: '高变基因', icon: 'fas fa-dna', desc: '选择高变基因' }
            ],
            'dimreduction': [
                { id: 'pca', name: 'PCA分析', icon: 'fas fa-chart-line', desc: '主成分分析' },
                { id: 'umap', name: 'UMAP降维', icon: 'fas fa-map', desc: '统一流形近似投影' },
                { id: 'tsne', name: 't-SNE降维', icon: 'fas fa-dot-circle', desc: 't-分布随机邻域嵌入' }
            ],
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
            toolDiv.className = 'mb-3 p-3 border rounded fade-in';
            toolDiv.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <i class="${tool.icon} me-2 text-primary"></i>
                    <strong>${tool.name}</strong>
                </div>
                <p class="text-muted small mb-2">${tool.desc}</p>
                <button class="btn btn-sm btn-primary w-100" onclick="${tool.id === 'coming_soon' ? 'singleCellApp.showComingSoon()' : (tool.id === 'log1p' ? `singleCellApp.runTool('${tool.id}')` : `singleCellApp.showParameterDialog('${tool.id}')`)}" ${!this.currentData && tool.id !== 'coming_soon' ? 'disabled' : ''}>
                    ${tool.id === 'coming_soon' ? '敬请期待' : '设置参数'}
                </button>
            `;
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

    showParameterDialog(tool) {
        this.currentTool = tool;
        const modal = new bootstrap.Modal(document.getElementById('parameterModal'));
        const title = document.getElementById('parameterModalTitle');
        const body = document.getElementById('parameterModalBody');

        const toolNames = {
            'normalize': '标准化',
            'scale': '数据缩放',
            'hvg': '高变基因选择',
            'pca': 'PCA分析',
            'umap': 'UMAP降维',
            'tsne': 't-SNE降维',
            'neighbors': '邻域计算',
            'leiden': 'Leiden聚类',
            'louvain': 'Louvain聚类'
        };

        title.textContent = toolNames[tool] + ' - 参数设置';
        body.innerHTML = this.getParameterHTML(tool);
        modal.show();
    }

    getParameterHTML(tool) {
        const parameters = {
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
            'louvain': 'Louvain聚类'
        };

        this.showStatus(`正在执行${toolNames[tool]}...`, true);
        this.addToLog(`开始执行: ${toolNames[tool]}`);

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
                this.addToLog(`${toolNames[tool]}失败: ${data.error}`, 'error');
                this.showStatus(`${toolNames[tool]}执行失败: ${data.error}`, false);
                alert(`执行失败: ${data.error}`);
            } else {
                this.currentData = data;
                this.updateUI(data);
                this.addToLog(`${toolNames[tool]}完成`);
                this.showStatus(`${toolNames[tool]}执行完成`, false);
                
                // Auto-update plot if embedding is available
                const embeddingSelect = document.getElementById('embedding-select');
                if (embeddingSelect.value) {
                    this.updatePlot();
                }
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(`${toolNames[tool]}失败: ${error.message}`, 'error');
            this.showStatus(`${toolNames[tool]}执行失败: ${error.message}`, false);
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
        if (confirm('确定要重置所有数据吗？')) {
            this.currentData = null;
            document.getElementById('upload-section').style.display = 'block';
            document.getElementById('data-status').classList.add('d-none');
            document.getElementById('viz-controls').style.display = 'none';
            document.getElementById('viz-panel').style.display = 'none';
            document.getElementById('analysis-log').innerHTML = '<div class="text-muted">等待上传数据...</div>';
            document.getElementById('fileInput').value = '';
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
            type: 'scatter',
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
