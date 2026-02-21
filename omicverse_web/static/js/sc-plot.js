/**
 * OmicVerse Single Cell Analysis — Visualization & Plotting (Plotly)
 */

Object.assign(SingleCellAnalysis.prototype, {

    onColorSelectChange() {
        const geneInput = document.getElementById('gene-input');
        if (geneInput) geneInput.value = '';
        this.updatePlot();
    },

    updatePlot() {
        const embedding = document.getElementById('embedding-select').value;
        if (!embedding) return;

        // Gene expression takes priority: if gene-input has a value use it,
        // regardless of what color-select says. This ensures palette/style
        // changes don't accidentally revert to the obs categorical variable.
        const geneInput = document.getElementById('gene-input');
        const geneValue = geneInput ? geneInput.value.trim() : '';
        const colorBy = geneValue
            ? 'gene:' + geneValue
            : document.getElementById('color-select').value;

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
    },

    updatePaletteVisibility(colorBy) {
        const categoryPaletteRow = document.getElementById('category-palette-row');
        const vminmaxRow = document.getElementById('vminmax-row');
        const paletteLabel = document.getElementById('palette-label');

        if (!colorBy || colorBy.startsWith('gene:')) {
            // Continuous data - show continuous palette and vmin/vmax, hide category palette
            if (categoryPaletteRow) categoryPaletteRow.style.display = 'none';
            if (vminmaxRow) vminmaxRow.style.display = 'flex';
            if (paletteLabel) paletteLabel.textContent = this.t('controls.paletteContinuous');
        } else if (colorBy.startsWith('obs:')) {
            // Check if it's categorical by trying to detect from obs columns
            // We'll let the backend determine this, but show category palette for now
            if (categoryPaletteRow) categoryPaletteRow.style.display = 'flex';
            // Also show vmin/vmax for now - backend will determine if it's continuous
            if (vminmaxRow) vminmaxRow.style.display = 'flex';
            if (paletteLabel) paletteLabel.textContent = this.t('controls.paletteContinuous');
        } else {
            // No coloring
            if (categoryPaletteRow) categoryPaletteRow.style.display = 'none';
            if (vminmaxRow) vminmaxRow.style.display = 'none';
        }
    },

    applyVMinMax() {
        // Just trigger an update
        this.updatePlot();
    },

    createNewPlot(embedding, colorBy) {
        this.currentEmbedding = this.currentEmbedding;
        this.showStatus(this.t('plot.generating'), true);

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
                this.addToLog(`${this.t('plot.errorPrefix')}: ${data.error}`, 'error');
                this.showStatus(`${this.t('plot.failedPrefix')}: ${data.error}`, false);
            } else {
                this.plotData(data);
                this.currentEmbedding = embedding;
                this.showStatus(this.t('plot.done'), false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(`${this.t('plot.failedPrefix')}: ${error.message}`, 'error');
            this.showStatus(`${this.t('plot.failedPrefix')}: ${error.message}`, false);
        });
    },

    updatePlotWithAnimation(embedding, colorBy) {
        const isEmbeddingChange = (this.currentEmbedding !== embedding);
        this.showStatus(
            isEmbeddingChange ? this.t('plot.switchEmbedding') : this.t('plot.updateColor'),
            true
        );

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
                this.addToLog(`${this.t('plot.errorPrefix')}: ${data.error}`, 'error');
                this.showStatus(`${this.t('plot.failedPrefix')}: ${data.error}`, false);
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
                    this.showStatus(this.t('plot.colorUpdated'), false);
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
                    this.showStatus(this.t('plot.embeddingSwitched'), false);
                }
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(`${this.t('plot.failedPrefix')}: ${error.message}`, 'error');
            this.showStatus(`${this.t('plot.failedPrefix')}: ${error.message}`, false);
        });
    },

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
    },

    checkCoordsChanged(currentX, currentY, newX, newY, length) {
        const tolerance = 1e-10; // 浮点数比较容差
        for (let i = 0; i < Math.min(length, 100); i++) { // 只检查前100个点以提高性能
            if (Math.abs(currentX[i] - newX[i]) > tolerance || 
                Math.abs(currentY[i] - newY[i]) > tolerance) {
                return true;
            }
        }
        return false;
    },

    updateColorsOnly(data) {
        // 只更新颜色，保持位置不变
        let markerConfig = {
            size: this.getMarkerSize(),
            opacity: this.getMarkerOpacity()
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
    },

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
                size: this.getMarkerSize(),
                opacity: this.getMarkerOpacity()
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
    },

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
                    size: this.getMarkerSize(),
                    opacity: this.getMarkerOpacity()
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
    },

    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    },

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
    },

    computeDefaultPointSize() {
        const nCells = parseInt(document.getElementById('cell-count')?.textContent || '0', 10);
        if (!nCells || nCells <= 0) return 4;
        // Python: s = 120000 / n  (matplotlib area units)
        // Plotly size ≈ diameter in px ≈ sqrt(s)
        return Math.max(1, Math.round(Math.sqrt(120000 / nCells) * 10) / 10);
    },

    getMarkerSize() {
        const slider = document.getElementById('point-size-slider');
        if (!slider || slider.dataset.auto === 'true') {
            return this.computeDefaultPointSize();
        }
        return parseFloat(slider.value);
    },

    getMarkerOpacity() {
        const slider = document.getElementById('opacity-slider');
        return slider ? parseFloat(slider.value) : 0.7;
    },

    onPointSizeChange(value) {
        const slider = document.getElementById('point-size-slider');
        const label  = document.getElementById('point-size-value');
        if (slider) slider.dataset.auto = 'false';
        if (label)  label.textContent = parseFloat(value).toFixed(1);
        this.applyPointStyleLive();
    },

    onOpacityChange(value) {
        const label = document.getElementById('opacity-value');
        if (label) label.textContent = parseFloat(value).toFixed(2);
        this.applyPointStyleLive();
    },

    resetPointStyle() {
        const sizeSlider    = document.getElementById('point-size-slider');
        const opacitySlider = document.getElementById('opacity-slider');
        const sizeLabel     = document.getElementById('point-size-value');
        const opacityLabel  = document.getElementById('opacity-value');

        if (sizeSlider) {
            sizeSlider.dataset.auto = 'true';
            const def = this.computeDefaultPointSize();
            sizeSlider.value = def;
            if (sizeLabel) sizeLabel.textContent = 'Auto';
        }
        if (opacitySlider) {
            opacitySlider.value = 0.7;
            if (opacityLabel) opacityLabel.textContent = '0.70';
        }
        this.applyPointStyleLive();
    },

    initPointSizeSlider() {
        const slider = document.getElementById('point-size-slider');
        const label  = document.getElementById('point-size-value');
        if (!slider) return;
        const def = this.computeDefaultPointSize();
        slider.value = def;
        slider.dataset.auto = 'true';
        if (label) label.textContent = 'Auto';
    },

    applyPointStyleLive() {
        const plotDiv = document.getElementById('plotly-div');
        if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) return;
        const size    = this.getMarkerSize();
        const opacity = this.getMarkerOpacity();
        const traceIndices = plotDiv.data.map((_, i) => i);
        Plotly.restyle('plotly-div', { 'marker.size': size, 'marker.opacity': opacity }, traceIndices);
    },

    plotData(data) {
        // 处理颜色配置
        let markerConfig = {
            size: this.getMarkerSize(),
            opacity: this.getMarkerOpacity()
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
                                size: this.getMarkerSize(),
                                opacity: this.getMarkerOpacity()
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
                    size: this.getMarkerSize(),
                    opacity: this.getMarkerOpacity()
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
    },

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
    },

    colorByGene() {
        const gene = document.getElementById('gene-input').value.trim();
        if (!gene) return;

        const embedding = document.getElementById('embedding-select').value;
        if (!embedding) {
            alert(this.t('controls.embeddingPlaceholder'));
            return;
        }

        // Update palette visibility for gene expression (continuous)
        this.updatePaletteVisibility('gene:' + gene);

        // 检查是否已经有图表存在
        const plotDiv = document.getElementById('plotly-div');
        const hasExistingPlot = plotDiv && plotDiv.data && plotDiv.data.length > 0;

        if (hasExistingPlot) {
            this.showStatus(this.t('gene.updating'), true);
        } else {
            this.showStatus(this.t('gene.loading'), true);
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
                this.addToLog(this.t('gene.error') + ': ' + data.error, 'error');
                this.showStatus(this.t('gene.notFound') + ': ' + gene, false);
                alert(this.t('gene.notFound') + ': ' + gene);
            } else {
                this.plotData(data);
                this.addToLog(this.t('gene.showing') + ': ' + gene);
                this.showStatus(this.t('gene.loaded'), false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(this.t('gene.loadFailed') + ': ' + error.message, 'error');
            this.showStatus(this.t('gene.loadFailed') + ': ' + error.message, false);
        });
    },

    updateAdataStatus(data, diff = null) {
        const content = document.getElementById('adata-status-content');
        if (!content) return;

        // No data loaded yet — show placeholder
        if (!data) {
            content.innerHTML = `
                <div class="d-flex flex-column align-items-center justify-content-center text-center text-muted h-100" style="min-height:160px">
                    <i class="fas fa-database fa-2x mb-3 opacity-25"></i>
                    <p class="small mb-0">${this.currentLang === 'zh'
                        ? '请在右方上传 h5ad 文件<br>或从代码编辑器加载 adata'
                        : 'Upload an h5ad file on the right<br>or load adata from the code editor'}</p>
                </div>`;
            return;
        }

        // Small bordered chip for each key name
        const chip = k => `<span class="adata-key-chip">${k}</span>`;

        const labelStyle = 'font-size:0.72rem;min-width:40px;flex-shrink:0;color:#6c757d;font-weight:500';
        const row = (label, valueHtml) =>
            `<div class="d-flex align-items-start gap-2 mb-2">
                <span style="${labelStyle}">${label}</span>
                <span style="line-height:1.8">${valueHtml}</span>
            </div>`;

        // ── Full current structure ───────────────────────────────────────────
        const cells = (data.n_cells || 0).toLocaleString();
        const genes = (data.n_genes || 0).toLocaleString();
        let html = row('shape', `<span class="adata-key-chip" style="font-weight:600">${cells} × ${genes}</span>`);

        // ── Data state (preprocessing status) — directly below shape ─────────
        const ds = data.data_state;
        if (ds && Object.keys(ds).length > 0) {
            const badge = (label, color, title='') =>
                `<span class="badge rounded-pill text-bg-${color} me-1" style="font-size:0.65rem" ${title ? `title="${title}"` : ''}>${label}</span>`;

            const dtype = ds.is_int ? this.t('status.dsInt') : this.t('status.dsFloat');
            const maxVal = ds.x_max !== undefined ? ds.x_max.toLocaleString(undefined, {maximumFractionDigits: 3}) : '?';
            html += row('X', `<span class="adata-key-chip">${this.t('status.dsMax')}: <b>${maxVal}</b></span> <span class="adata-key-chip">${dtype}</span>`);

            let stateBadges = '';
            if (ds.is_normalized === true)
                stateBadges += badge(this.t('status.dsNorm'), 'success', this.t('status.dsNormTip'));
            else if (ds.is_normalized === false)
                stateBadges += badge(this.t('status.dsRaw'), 'secondary', this.t('status.dsRawTip'));
            if (ds.is_log1p === true)
                stateBadges += badge('log1p', 'info', this.t('status.dsLog1pTip'));
            if (ds.is_scaled === true)
                stateBadges += badge(this.t('status.dsScaled'), 'warning', this.t('status.dsScaledTip'));
            if (ds.estimated_target_sum)
                stateBadges += badge(`target_sum ≈ ${ds.estimated_target_sum.toLocaleString()}`, 'primary', this.t('status.dsTargetTip'));
            if (stateBadges) html += row('state', stateBadges);
        }

        const obs = data.obs_columns || [];
        if (obs.length) html += row('obs', obs.map(chip).join(''));

        const vr = data.var_columns || [];
        if (vr.length) html += row('var', vr.map(chip).join(''));

        const uns = data.uns_keys || [];
        if (uns.length) html += row('uns', uns.map(chip).join(''));

        const obsm = (data.embeddings || []).map(e => `X_${e}`);
        if (obsm.length) html += row('obsm', obsm.map(chip).join(''));

        const lyr = data.layers || [];
        if (lyr.length) html += row('layers', lyr.map(chip).join(''));

        // ── Last operation diff (reset each time) ───────────────────────────
        if (diff) {
            const [cb, gb] = diff.shape_before;
            const [ca, ga] = diff.shape_after;
            const slotColor = { obs:'primary', var:'success', uns:'warning',
                                obsm:'info', obsp:'secondary', layers:'danger' };
            const changes = diff.changes || {};
            let diffHtml = '';

            if (cb !== ca || gb !== ga) {
                diffHtml += `<span class="adata-key-chip text-muted">${cb.toLocaleString()}×${gb.toLocaleString()}</span>`
                    + `<i class="fas fa-arrow-right mx-1 text-muted" style="font-size:0.55rem;vertical-align:middle"></i>`
                    + `<span class="adata-key-chip text-success fw-semibold">${ca.toLocaleString()}×${ga.toLocaleString()}</span> `;
            }
            for (const [slot, ch] of Object.entries(changes)) {
                const col = slotColor[slot] || 'secondary';
                const badge = `<span class="badge rounded-pill text-bg-${col} me-1" style="font-size:0.6rem;vertical-align:middle">${slot}</span>`;
                if (ch.added)
                    diffHtml += badge + `<span class="text-success me-1">+</span>`
                        + ch.added.map(chip).join('') + ' ';
                if (ch.removed)
                    diffHtml += badge + `<span class="text-danger me-1">−</span>`
                        + ch.removed.map(chip).join('') + ' ';
            }
            if (!diffHtml) diffHtml = `<span class="text-muted fst-italic" style="font-size:0.72rem">no changes</span>`;
            diffHtml += `<span class="text-muted ms-1" style="font-size:0.7rem">${diff.duration}s</span>`;

            html += `<hr class="my-2">` + row('diff', diffHtml);
        }

        content.innerHTML = html;
    },

    showParameterPlaceholder() {
        const parameterContent = document.getElementById('parameter-content');
        if (!parameterContent) return;
        parameterContent.innerHTML = `
            <div class="d-flex flex-column align-items-center justify-content-center text-center text-muted py-5" style="min-height:160px">
                <i class="fas fa-hand-pointer fa-2x mb-3 opacity-50"></i>
                <p class="mb-0 small" data-i18n="panel.selectAnalysis">${this.t('panel.selectAnalysis')}</p>
            </div>`;
    },

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
    },

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
    },

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
    },

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
    },

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

});
