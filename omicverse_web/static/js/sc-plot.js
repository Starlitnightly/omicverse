/**
 * OmicVerse Single Cell Analysis — Visualization & Plotting (Plotly)
 */

Object.assign(SingleCellAnalysis.prototype, {

    _computeAxisRangesFromData(xArr, yArr, padFraction = 0.04) {
        const toNums = (arr) => (arr || [])
            .map(v => Number(v))
            .filter(v => Number.isFinite(v));

        const x = toNums(xArr);
        const y = toNums(yArr);
        if (!x.length || !y.length) return null;

        let xMin = Math.min(...x), xMax = Math.max(...x);
        let yMin = Math.min(...y), yMax = Math.max(...y);

        if (xMin === xMax) {
            const d = Math.abs(xMin || 1) * 0.02 || 1;
            xMin -= d; xMax += d;
        }
        if (yMin === yMax) {
            const d = Math.abs(yMin || 1) * 0.02 || 1;
            yMin -= d; yMax += d;
        }

        const dx = xMax - xMin;
        const dy = yMax - yMin;
        const px = Math.max(dx * padFraction, 1e-9);
        const py = Math.max(dy * padFraction, 1e-9);

        return {
            xRange: [xMin - px, xMax + px],
            yRange: [yMin - py, yMax + py]
        };
    },

    onColorSelectChange() {
        const geneInput = document.getElementById('gene-input');
        if (geneInput) {
            geneInput.value = '';
            // Clear gene-input from localStorage too (programmatic clear doesn't fire events)
            try { localStorage.removeItem('ov:s:gene-input'); } catch (_) {}
        }
        // Save the new color-select value explicitly (belt + suspenders)
        const colSel = document.getElementById('color-select');
        if (colSel && this._persistSaveEl) this._persistSaveEl(colSel);
        this._syncDensityControlStateBySelection();
        this.updatePlot();
    },

    updatePlot() {
        const embedding = document.getElementById('embedding-select').value;
        // Allow plotting even without a preset embedding if custom axes are active
        const xyAxes = this.getXYAxes ? this.getXYAxes() : null;
        if (!embedding && !xyAxes) return;

        // Gene expression takes priority: if gene-input has a value use it,
        // regardless of what color-select says.
        const geneInput = document.getElementById('gene-input');
        const geneValue = geneInput ? geneInput.value.trim() : '';
        const colorBy = geneValue
            ? 'gene:' + geneValue
            : document.getElementById('color-select').value;
        this._syncDensityControlStateBySelection();

        // Update palette visibility based on color type
        this.updatePaletteVisibility(colorBy);

        // ── deck.gl path ─────────────────────────────────────────────────────
        // Check _forceRenderer FIRST so switching controls always works in WebGL mode.
        const nCells  = this.currentData ? (this.currentData.n_cells || 0) : 0;
        const useDeck = this._forceRenderer === 'deckgl'
            || (this._forceRenderer !== 'plotly' && nCells > this._rasterThreshold);

        if (useDeck) {
            this.createDeckGLPlot(embedding, colorBy, xyAxes);
            return;
        }

        // ── Plotly path ───────────────────────────────────────────────────────
        const plotDiv = document.getElementById('plotly-div');
        const hasExistingPlot = plotDiv && plotDiv.data && plotDiv.data.length > 0;

        if (hasExistingPlot) {
            this.updatePlotWithAnimation(embedding, colorBy, xyAxes);
        } else {
            this.createNewPlot(embedding, colorBy, xyAxes);
        }
    },

    /**
     * Build the base payload for /api/plot.
     * Injects x_axis/y_axis when custom axes are active; otherwise uses embedding.
     */
    _buildPlotPayload(embedding, colorBy, xyAxes) {
        const paletteSelect = document.getElementById('palette-select');
        const categoryPaletteSelect = document.getElementById('category-palette-select');
        const palette = paletteSelect && paletteSelect.value !== 'default' ? paletteSelect.value : null;
        const categoryPalette = categoryPaletteSelect && categoryPaletteSelect.value !== 'default' ? categoryPaletteSelect.value : null;
        const vminInput = document.getElementById('vmin-input');
        const vmaxInput = document.getElementById('vmax-input');
        const vmin = vminInput && vminInput.value ? parseFloat(vminInput.value) : null;
        const vmax = vmaxInput && vmaxInput.value ? parseFloat(vmaxInput.value) : null;

        const payload = {
            color_by: colorBy,
            palette: palette,
            category_palette: categoryPalette,
            vmin: vmin,
            vmax: vmax,
            density_adjust: this.getDensityAdjust(),
            density_active: this.isDensityActive(),
        };

        if (xyAxes) {
            // Custom axes override the embedding
            payload.x_axis = xyAxes.x_axis;
            payload.y_axis = xyAxes.y_axis;
            // Also pass embedding for backward compat / axis label
            payload.embedding = embedding;
        } else {
            payload.embedding = embedding;
        }

        return payload;
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

    createNewPlot(embedding, colorBy, xyAxes) {
        this.currentEmbedding = this.currentEmbedding;
        this.showStatus(this.t('plot.generating'), true);

        fetch('/api/plot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(this._buildPlotPayload(embedding, colorBy, xyAxes))
        })
        .then(response => response.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog(`${this.t('plot.errorPrefix')}: ${data.error}`, 'error');
                this.showStatus(`${this.t('plot.failedPrefix')}: ${data.error}`, false);
            } else {
                this._setDensityControlState(
                    data.density_enabled !== false,
                    data.density_message || ''
                );
                this.plotData(data);
                this.currentEmbedding = embedding;
                this._currentAxesKey = xyAxes ? `${xyAxes.x_axis}|${xyAxes.y_axis}` : embedding;
                this.showStatus(this.t('plot.done'), false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(`${this.t('plot.failedPrefix')}: ${error.message}`, 'error');
            this.showStatus(`${this.t('plot.failedPrefix')}: ${error.message}`, false);
        });
    },

    updatePlotWithAnimation(embedding, colorBy, xyAxes) {
        // Route to deck.gl when forced or when dataset is large
        const nCells  = this.currentData ? (this.currentData.n_cells || 0) : 0;
        const useDeck = this._forceRenderer === 'deckgl'
            || (this._forceRenderer !== 'plotly' && nCells > this._rasterThreshold);
        if (useDeck) {
            this.createDeckGLPlot(embedding, colorBy, xyAxes);
            return;
        }

        // Build a full coordinate identity key that includes custom axes.
        // This ensures switching obs/gene axes is treated as a position change,
        // even when the embedding-select value hasn't changed.
        const axesKey = xyAxes ? `${xyAxes.x_axis}|${xyAxes.y_axis}` : embedding;
        const isEmbeddingChange = (this._currentAxesKey !== undefined)
            ? (this._currentAxesKey !== axesKey)
            : (this.currentEmbedding !== embedding);

        this.showStatus(
            isEmbeddingChange ? this.t('plot.switchEmbedding') : this.t('plot.updateColor'),
            true
        );

        fetch('/api/plot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(this._buildPlotPayload(embedding, colorBy, xyAxes))
        })
        .then(response => response.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog(`${this.t('plot.errorPrefix')}: ${data.error}`, 'error');
                this.showStatus(`${this.t('plot.failedPrefix')}: ${data.error}`, false);
            } else {
                this._setDensityControlState(
                    data.density_enabled !== false,
                    data.density_message || ''
                );
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
                            // Ensure axis labels are updated even in color-only mode
                            this._applyAxisLabels(data);
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
                    this._currentAxesKey = axesKey;
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

    /**
     * Update Plotly axis titles from data.axis_labels without a full redraw.
     * Called after color-only updates to keep axis titles in sync.
     */
    _applyAxisLabels(data) {
        if (!data || !data.axis_labels) return;
        this._currentAxisLabels = data.axis_labels;
        const axLbls = data.axis_labels;
        try {
            Plotly.relayout('plotly-div', {
                'xaxis.title': axLbls.x || 'Dimension 1',
                'yaxis.title': axLbls.y || 'Dimension 2',
            });
        } catch (e) { /* ignore if no plot yet */ }
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
            markerConfig.showscale = false; // colorbar shown in custom panel below
        } else {
            markerConfig.color = 'blue';
            markerConfig.showscale = false;
        }
        // Update the custom legend panel
        this._renderLegendPanel(data, 'plotly');
        
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
        const startRanges = this._computeAxisRangesFromData(currentX, currentY) || {
            xRange: [Math.min(...currentX), Math.max(...currentX)],
            yRange: [Math.min(...currentY), Math.max(...currentY)]
        };
        const endRanges = this._computeAxisRangesFromData(newX, newY) || {
            xRange: [Math.min(...newX), Math.max(...newX)],
            yRange: [Math.min(...newY), Math.max(...newY)]
        };
        // Determine start ranges
        const startXRange = (layout && layout.xaxis && layout.xaxis.range) ? layout.xaxis.range.slice() : startRanges.xRange;
        const startYRange = (layout && layout.yaxis && layout.yaxis.range) ? layout.yaxis.range.slice() : startRanges.yRange;
        // Determine final ranges
        const endXRange = endRanges.xRange;
        const endYRange = endRanges.yRange;

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

    getDensityAdjust() {
        const slider = document.getElementById('density-adjust-slider');
        if (!slider) return 1.0;
        return parseFloat(slider.value || '1');
    },

    isDensityActive() {
        const toggle = document.getElementById('density-enable-toggle');
        return !!(toggle && toggle.checked);
    },

    onPointSizeChange(value) {
        const slider = document.getElementById('point-size-slider');
        const label  = document.getElementById('point-size-value');
        if (slider) slider.dataset.auto = 'false';
        if (label)  label.textContent = parseFloat(value).toFixed(1);
        // Persist auto=false flag so restore knows user set a manual size
        try { localStorage.setItem('ov:s:__point-size-auto', 'false'); } catch(_){}
        this.applyPointStyleLive();
    },

    onOpacityChange(value) {
        const label = document.getElementById('opacity-value');
        if (label) label.textContent = parseFloat(value).toFixed(2);
        this.applyPointStyleLive();
    },

    onDensityAdjustInput(value) {
        const label = document.getElementById('density-adjust-value');
        if (label) label.textContent = parseFloat(value).toFixed(2);
        this._scheduleDensityUpdate(220);
    },

    onDensityAdjustCommit(value) {
        const label = document.getElementById('density-adjust-value');
        if (label) label.textContent = parseFloat(value).toFixed(2);
        this._scheduleDensityUpdate(0);
    },

    _scheduleDensityUpdate(delayMs = 220) {
        const slider = document.getElementById('density-adjust-slider');
        if (!slider || slider.disabled || !this.isDensityActive()) return;
        if (this._densityUpdateTimer) clearTimeout(this._densityUpdateTimer);
        this._densityUpdateTimer = setTimeout(() => {
            this._densityUpdateTimer = null;
            this.updatePlot();
        }, Math.max(0, delayMs || 0));
    },

    onDensityToggleChange(enabled) {
        try { localStorage.setItem('ov:s:density-enable-toggle', enabled ? 'true' : 'false'); } catch (_) {}
        this._updateDensityControlState();
        this.updatePlot();
    },

    _setDensityControlState(enabled, reason = '') {
        this._densityCapable = !!enabled;
        this._densityReason = reason || '';
        this._updateDensityControlState();
    },

    _updateDensityControlState() {
        const slider = document.getElementById('density-adjust-slider');
        const hint = document.getElementById('density-adjust-hint');
        const toggle = document.getElementById('density-enable-toggle');
        if (!slider) return;

        const active = !!(toggle && toggle.checked);
        const capable = !!this._densityCapable;
        const enabled = active && capable;
        slider.disabled = !enabled;
        if (enabled) {
            slider.classList.remove('disabled');
        } else {
            slider.classList.add('disabled');
        }
        let msg = '';
        if (!active) {
            msg = this.t ? this.t('controls.densityDisabledByToggle') : 'Density adjust is off';
        } else if (!capable) {
            msg = this._densityReason || (this.t ? this.t('controls.densityDisabledNoFeature') : 'Select a numeric feature');
        }
        if (hint) hint.textContent = msg;
        slider.title = enabled
            ? 'Adjust density smoothing for numeric features'
            : (msg || 'Density adjustment is available only for numeric features');
    },

    _syncDensityControlStateBySelection() {
        const geneInput = document.getElementById('gene-input');
        const geneValue = geneInput ? geneInput.value.trim() : '';
        const colorBy = geneValue
            ? 'gene:' + geneValue
            : (document.getElementById('color-select') || {}).value || '';
        if (!colorBy) {
            this._setDensityControlState(false, this.t ? this.t('controls.densityDisabledNoFeature') : 'Select a feature to enable density adjustment');
            return;
        }
        if (colorBy.startsWith('gene:')) {
            this._setDensityControlState(true, '');
            return;
        }
        // obs column type (numeric/categorical) is confirmed by backend response.
        this._setDensityControlState(true, this.t ? this.t('controls.densityPendingType') : 'Type checking...');
    },

    resetPointStyle() {
        const sizeSlider    = document.getElementById('point-size-slider');
        const opacitySlider = document.getElementById('opacity-slider');
        const densitySlider = document.getElementById('density-adjust-slider');
        const densityToggle = document.getElementById('density-enable-toggle');
        const sizeLabel     = document.getElementById('point-size-value');
        const opacityLabel  = document.getElementById('opacity-value');
        const densityLabel  = document.getElementById('density-adjust-value');

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
        if (densitySlider) {
            densitySlider.value = 1;
            if (densityLabel) densityLabel.textContent = '1.00';
        }
        if (densityToggle) densityToggle.checked = false;
        // Persist the reset state
        try {
            localStorage.setItem('ov:s:__point-size-auto', 'true');
            localStorage.removeItem('ov:s:point-size-slider'); // forget manual size
            localStorage.setItem('ov:s:opacity-slider', '0.7');
            localStorage.setItem('ov:s:density-adjust-slider', '1');
            localStorage.setItem('ov:s:density-enable-toggle', 'false');
        } catch(_) {}
        this.applyPointStyleLive();
        this._updateDensityControlState();
        this.updatePlot();
    },

    initPointSizeSlider() {
        const slider = document.getElementById('point-size-slider');
        const label  = document.getElementById('point-size-value');
        if (!slider) return;
        const def = this.computeDefaultPointSize();
        slider.value = def;
        slider.dataset.auto = 'true';
        if (label) label.textContent = 'Auto';
        this._syncDensityControlStateBySelection();
    },

    applyPointStyleLive() {
        const size    = this.getMarkerSize();
        const opacity = this.getMarkerOpacity();

        // ── deck.gl path ──────────────────────────────────────────────────────
        if (this._deckglRenderer && this._deckglRenderer._positions) {
            this._deckglRenderer.updateStyle(size, opacity);
            return;
        }

        // ── Plotly path ───────────────────────────────────────────────────────
        const plotDiv = document.getElementById('plotly-div');
        if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) return;
        const traceIndices = plotDiv.data.map((_, i) => i);
        Plotly.restyle('plotly-div', { 'marker.size': size, 'marker.opacity': opacity }, traceIndices);
    },

    plotData(data) {
        // Stash axis labels so getPlotlyLayout() can use them
        this._currentAxisLabels = data.axis_labels || null;

        // Show decimation notice when backend returned a subset
        if (data.decimated) {
            const shown = data.n_shown ? Math.round(data.n_shown / 1000) : '?';
            const total = data.n_total ? Math.round(data.n_total / 1000) : '?';
            this.addToLog(
                `⚠️ ${this.t('plot.decimatedNotice') || `Showing ${shown}K / ${total}K cells (spatially sampled)`}`,
                'warning'
            );
        }

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
                            name: category,
                            marker: {
                                color: color,
                                size: this.getMarkerSize(),
                                opacity: this.getMarkerOpacity()
                            },
                            text: categoryText,
                            hovertemplate: '%{text}<extra></extra>',
                            showlegend: false // Legend rendered in custom panel below chart
                        };

                        traces.push(trace);
                    }
                }
            } else {
                // 数值数据：使用连续颜色映射 (colorbar rendered in custom panel below)
                markerConfig.color = data.colors;
                markerConfig.colorscale = data.colorscale || 'Viridis';
                markerConfig.showscale = false;

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
        const dataRanges = this._computeAxisRangesFromData(data.x, data.y);
        if (dataRanges) {
            layout.xaxis = layout.xaxis || {};
            layout.yaxis = layout.yaxis || {};
            layout.xaxis.autorange = false;
            layout.yaxis.autorange = false;
            layout.xaxis.range = dataRanges.xRange;
            layout.yaxis.range = dataRanges.yRange;
        }
        
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
        
        // 直接使用 Plotly.react 来确保正确更新
        Plotly.react('plotly-div', traces, layout, config);

        // Render custom legend panel below the chart
        this._renderLegendPanel(data, 'plotly');
    },

    // ═══════════════════════════════════════════════════════════════════════════
    //  Custom Legend Panel  (below the visualization canvas)
    // ═══════════════════════════════════════════════════════════════════════════

    /** OmicVerse default categorical palette (matches backend). */
    _OV_SC_COLOR: [
        '#1F577B','#A56BA7','#E0A7C8','#E069A6','#941456',
        '#FCBC10','#EF7B77','#279AD7','#F0EEF0','#EAEFC5',
        '#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED',
        '#866017','#9F987F','#E0DFED','#01A0A7','#75C8CC',
        '#F0D7BC','#D5B26C','#D5DA48','#B6B812','#9DC3C3',
        '#A89C92','#FEE00C','#FEF2A1',
    ],

    /**
     * Render the custom HTML legend below the visualization.
     * @param {object} data   – Plotly data object OR deck.gl meta object
     * @param {string} renderer – 'plotly' | 'deckgl'
     */
    _renderLegendPanel(data, renderer) {
        const panel   = document.getElementById('viz-legend-panel');
        const content = document.getElementById('viz-legend-content');
        if (!panel || !content) return;

        // Normalise field names between Plotly (data.*) and deck.gl (meta.*)
        const isCategorical  = !!(data.category_labels && data.category_labels.length);
        const isContinuous   = !isCategorical && !!(data.colorscale || data.colors);
        const colorLabel     = data.color_label || '';
        const catLabels      = data.category_labels  || [];
        const catColors      = data.category_colors  || data.discrete_colors || null;
        const colorscaleName = data.colorscale || 'viridis';
        const vmin           = (data.vmin !== undefined && data.vmin !== null) ? data.vmin.toFixed(3) : '';
        const vmax           = (data.vmax !== undefined && data.vmax !== null) ? data.vmax.toFixed(3) : '';

        // Reset selection state when new data arrives
        this._legendRenderer  = renderer;
        this._legendData      = data;
        this._legendSelected  = new Set();

        content.innerHTML = '';

        if (!isCategorical && !isContinuous) {
            panel.style.display = 'none';
            return;
        }
        panel.style.display = '';

        // Optional label prefix
        if (colorLabel) {
            const lbl = document.createElement('span');
            lbl.className = 'fw-semibold small text-muted me-2 flex-shrink-0';
            lbl.textContent = colorLabel + ':';
            content.appendChild(lbl);
        }

        if (isCategorical) {
            catLabels.forEach((label, idx) => {
                const color = (catColors && catColors[idx])
                    || this._OV_SC_COLOR[idx % this._OV_SC_COLOR.length];
                const chip = document.createElement('div');
                chip.className = 'legend-chip';
                chip.dataset.legendIdx = idx;
                chip.innerHTML =
                    `<span class="legend-dot" style="background:${color};border-color:rgba(0,0,0,.15);"></span>` +
                    `<span>${label}</span>`;
                chip.addEventListener('click', () => this._onLegendItemClick(idx));
                content.appendChild(chip);
            });
        } else {
            // Horizontal continuous colorbar
            const grad = this._colormapGradientH(colorscaleName);
            const wrap = document.createElement('div');
            wrap.className = 'legend-colorbar-wrap';
            wrap.innerHTML =
                `<span class="legend-colorbar-val text-end">${vmin}</span>` +
                `<div class="legend-colorbar-bar" style="background:${grad};"></div>` +
                `<span class="legend-colorbar-val">${vmax}</span>`;
            content.appendChild(wrap);
        }
    },

    /**
     * Handle a legend chip click – toggle selection, update visuals, filter plot.
     */
    _onLegendItemClick(idx) {
        if (!this._legendSelected) this._legendSelected = new Set();
        if (this._legendSelected.has(idx)) {
            this._legendSelected.delete(idx);
        } else {
            this._legendSelected.add(idx);
        }
        this._updateLegendChipStates();
        this._applyLegendFilter();
    },

    /** Sync the visual state (dim/selected) of legend chips to _legendSelected. */
    _updateLegendChipStates() {
        const chips        = document.querySelectorAll('#viz-legend-content .legend-chip');
        const hasSelection = this._legendSelected && this._legendSelected.size > 0;
        chips.forEach(chip => {
            const idx        = parseInt(chip.dataset.legendIdx);
            const isSelected = !hasSelection || this._legendSelected.has(idx);
            chip.classList.toggle('legend-chip--dim',      !isSelected);
            chip.classList.toggle('legend-chip--selected', isSelected && hasSelection);
        });
    },

    /** Apply the current legend selection as a dim filter to Plotly or deck.gl. */
    _applyLegendFilter() {
        if (this._legendRenderer === 'plotly') {
            this._applyPlotlyLegendFilter();
        } else if (this._legendRenderer === 'deckgl') {
            this._applyDeckGLLegendFilter();
        }
    },

    _applyPlotlyLegendFilter() {
        const plotDiv = document.getElementById('plotly-div');
        if (!plotDiv || !plotDiv.data) return;
        const hasSelection = this._legendSelected && this._legendSelected.size > 0;
        const baseOpacity  = this.getMarkerOpacity() || 0.8;
        const dimOpacity   = 0.05;

        // Map trace index → category index.
        // Because plotData creates traces in category order, traceIdx === categoryIdx
        // for the traces that were created (skipped empty categories shift things).
        // We stored category info in _legendData — use the catLabels order.
        const catLabels = (this._legendData && this._legendData.category_labels) || [];

        // Build a lookup: trace.name → category index
        let traceOpacities = [];
        let catIdx = 0;
        for (let ti = 0; ti < plotDiv.data.length; ti++) {
            const traceName  = plotDiv.data[ti].name || '';
            const foundIdx   = catLabels.indexOf(traceName);
            const selected   = !hasSelection || (foundIdx >= 0 && this._legendSelected.has(foundIdx));
            traceOpacities.push(selected ? baseOpacity : dimOpacity);
        }

        traceOpacities.forEach((op, ti) => {
            Plotly.restyle('plotly-div', { 'marker.opacity': op }, [ti]);
        });
    },

    _applyDeckGLLegendFilter() {
        const dr = this._deckglRenderer;
        if (!dr || !dr._hoverValues || !dr._colors) return;
        const hov          = dr._hoverValues;
        const origColors   = dr._colors;   // _colors is never modified by updateColors → always original
        const n            = dr._n;
        const hasSelection = this._legendSelected && this._legendSelected.size > 0;

        if (!hasSelection) {
            // Restore original colors
            dr.updateColors(origColors);
            return;
        }

        const filteredColors = new Uint8Array(origColors); // copy
        for (let i = 0; i < n; i++) {
            const code = Math.round(hov[i]);
            if (!this._legendSelected.has(code)) {
                filteredColors[i * 4]     = 180;
                filteredColors[i * 4 + 1] = 180;
                filteredColors[i * 4 + 2] = 180;
                filteredColors[i * 4 + 3] = 35;
            }
        }
        dr.updateColors(filteredColors);
    },

    /**
     * Return a horizontal (left→right) CSS linear-gradient for a named colormap.
     * Low values on the left, high on the right.
     */
    _colormapGradientH(name) {
        const maps = {
            viridis:  '#440154,#482878,#3e4989,#31688e,#26828e,#1f9e89,#35b779,#6ece58,#b5de2b,#fde725',
            plasma:   '#0d0887,#46039f,#7201a8,#9c179e,#bd3786,#d8576b,#ed7953,#fb9f3a,#fdcf18,#f0f921',
            magma:    '#000004,#1b1044,#3b0f70,#641a80,#8c2981,#b5367a,#de4968,#f7705c,#fe9f6d,#fcfdbf',
            inferno:  '#000004,#1f0c48,#550f6d,#88226a,#ac5765,#cc7b5c,#e69c5b,#f4c86a,#f8e88e,#fcffa4',
            cividis:  '#002051,#0b307f,#35499d,#5762ad,#767ab5,#9592bb,#b4adc6,#d3cade,#ede0c6,#fee8a0',
            coolwarm: '#3b4cc0,#6688ee,#98b4fa,#c9d8ef,#eddbc7,#f7a789,#e36a53,#b40426',
            RdYlBu:   '#d73027,#f46d43,#fdae61,#fee090,#ffffbf,#e0f3f8,#abd9e9,#74add1,#4575b4',
            RdBu:     '#ca0020,#f4a582,#f7f7f7,#92c5de,#0571b0',
            bwr:      '#0000ff,#ffffff,#ff0000',
            hot:      '#000000,#ff0000,#ffff00,#ffffff',
            Blues:    '#f7fbff,#deebf7,#c6dbef,#9ecae1,#6baed6,#4292c6,#2171b5,#08519c,#08306b',
            Reds:     '#fff5f0,#fee0d2,#fcbba1,#fc9272,#fb6a4a,#ef3b2c,#cb181d,#a50f15,#67000d',
            Greens:   '#f7fcf5,#e5f5e0,#c7e9c0,#a1d99b,#74c476,#41ab5d,#238b45,#006d2c,#00441b',
            YlOrRd:   '#ffffcc,#ffeda0,#fed976,#feb24c,#fd8d3c,#fc4e2a,#e31a1c,#bd0026,#800026',
            Spectral: '#9e0142,#d53e4f,#f46d43,#fdae61,#fee08b,#ffffbf,#e6f598,#abdda4,#66c2a5,#3288bd,#5e4fa2',
            tab10:    '#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf',
        };
        const key   = (name || 'viridis').toLowerCase().replace(/_/g, '');
        const stops = maps[key] || maps[name] || maps.viridis;
        return `linear-gradient(to right, ${stops})`;
    },

    getPlotlyLayout() {
        const isDark = document.documentElement.classList.contains('app-skin-dark');
        
        const axLbls = this._currentAxisLabels;
        const embVal = (document.getElementById('embedding-select') || {}).value || '';
        const titleText = axLbls
            ? `${axLbls.x} vs ${axLbls.y}`
            : (embVal ? embVal.toUpperCase() + ' Plot' : 'Embedding Plot');
        const xTitle = axLbls ? axLbls.x : 'Dimension 1';
        const yTitle = axLbls ? axLbls.y : 'Dimension 2';
        
        const baseLayout = {
            title: {
                text: titleText,
                font: {color: isDark ? '#e5e7eb' : '#283c50'}
            },
            xaxis: {
                title: xTitle,
                color: isDark ? '#e5e7eb' : '#283c50',
                gridcolor: isDark ? '#374151' : '#e5e7eb',
                linecolor: isDark ? '#4b5563' : '#d1d5db',
                constrain: 'domain'
            },
            yaxis: {
                title: yTitle,
                color: isDark ? '#e5e7eb' : '#283c50',
                gridcolor: isDark ? '#374151' : '#e5e7eb',
                linecolor: isDark ? '#4b5563' : '#d1d5db',
                constrain: 'domain',
                scaleanchor: 'x',
                scaleratio: 1
            },
            hovermode: 'closest',
            showlegend: false, // Legend is rendered in the custom panel below the chart
            margin: {l: 50, r: 30, t: 50, b: 50}
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
        // Delegate entirely to updatePlot() which already:
        //   - reads gene-input value and uses it as color_by
        //   - handles both Plotly and WebGL (deck.gl) render paths
        //   - correctly passes custom xyAxes when active
        //   - calls updatePaletteVisibility internally
        const gene = document.getElementById('gene-input').value.trim();
        if (!gene) return;
        this.updatePlot();
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
    },

    // =========================================================================
    // deck.gl WebGL renderer (replaces the old Datashader raster approach)
    // =========================================================================

    /** Threshold above which we switch from Plotly to the deck.gl WebGL renderer. */
    _rasterThreshold: 200_000,

    /** deck.gl renderer instance (created on first use). */
    _deckglRenderer: null,

    /** Embedding / colorBy currently displayed by deck.gl. */
    _deckglCurrentEmbedding: null,
    _deckglCurrentColorBy:   null,

    /**
     * Main entry point for large datasets.  Fetches binary data from
     * /api/plot_gpu and hands it to DeckGLRenderer.
     * Falls back to the old raster PNG path if deck.gl is unavailable.
     */
    createDeckGLPlot(embedding, colorBy, xyAxes) {
        this.showStatus(this.t('plot.generating'), true);

        // ── ensure deck.gl container exists ──────────────────────────────────
        const plotlyDiv = document.getElementById('plotly-div');
        const parentEl  = plotlyDiv ? plotlyDiv.parentElement : null;
        if (!parentEl) { this.showStatus('viz container not found', false); return; }

        let wrap = document.getElementById('deckgl-wrap');
        if (!wrap) {
            wrap = document.createElement('div');
            wrap.id = 'deckgl-wrap';
            // Do NOT put background here – let CSS (including dark-mode.css) handle it
            wrap.style.cssText = 'position:relative; width:100%; min-height:480px; flex:1 1 auto;';
            parentEl.insertBefore(wrap, plotlyDiv);
        }
        if (plotlyDiv) plotlyDiv.style.display = 'none';
        wrap.style.display = 'block';

        // ── init renderer once ────────────────────────────────────────────────
        if (!this._deckglRenderer) {
            if (typeof DeckGLRenderer === 'undefined') {
                // deck.gl not loaded → fall back to raster
                wrap.style.display = 'none';
                if (plotlyDiv) plotlyDiv.style.display = '';
                this.createRasterPlot(embedding, colorBy);
                return;
            }
            this._deckglRenderer = new DeckGLRenderer(wrap);
            if (!this._deckglRenderer.init()) {
                this._deckglRenderer = null;
                wrap.style.display = 'none';
                if (plotlyDiv) plotlyDiv.style.display = '';
                this.createRasterPlot(embedding, colorBy);
                return;
            }
            this._buildDeckGLOverlays(wrap);
        }

        // ── collect palette / range controls ─────────────────────────────────
        const palSel    = document.getElementById('palette-select');
        const catPalSel = document.getElementById('category-palette-select');
        const vminEl    = document.getElementById('vmin-input');
        const vmaxEl    = document.getElementById('vmax-input');
        const body = {
            embedding,
            color_by:         colorBy,
            palette:          (palSel    && palSel.value    !== 'default') ? palSel.value    : 'viridis',
            category_palette: (catPalSel && catPalSel.value !== 'default') ? catPalSel.value : null,
            vmin: (vminEl && vminEl.value) ? parseFloat(vminEl.value) : null,
            vmax: (vmaxEl && vmaxEl.value) ? parseFloat(vmaxEl.value) : null,
            density_adjust: this.getDensityAdjust(),
            density_active: this.isDensityActive(),
        };
        // Inject custom axes if active
        if (xyAxes) {
            body.x_axis = xyAxes.x_axis;
            body.y_axis = xyAxes.y_axis;
        }

        // Use a combined key so that switching custom axes also triggers a full refetch
        const axesKey = xyAxes ? `${xyAxes.x_axis}|${xyAxes.y_axis}` : embedding;
        const isEmbeddingChange = this._deckglCurrentEmbedding !== axesKey;
        const hasExisting       = this._deckglRenderer._positions !== null;

        // ── color-only update: skip position fetch/encode (3× faster) ────────
        if (!isEmbeddingChange && hasExisting) {
            const colorBody = {
                embedding:        body.embedding,
                x_axis:           body.x_axis || '',
                y_axis:           body.y_axis || '',
                color_by:         body.color_by,
                palette:          body.palette,
                category_palette: body.category_palette,
                vmin:             body.vmin,
                vmax:             body.vmax,
                density_adjust:   body.density_adjust,
                density_active:   body.density_active,
                n_cells:          this._deckglRenderer._n,
            };
            fetch('/api/plot_gpu_colors', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(colorBody),
            })
            .then(r => {
                if (!r.ok) return r.json().then(e => Promise.reject(e.error || 'server error'));
                return r.arrayBuffer();
            })
            .then(buf => {
                this.hideStatus();
                const { n, meta, colors, hoverValues } = parseColorOnlyBuffer(buf);
                this._setDensityControlState(
                    meta.density_enabled !== false,
                    meta.density_message || ''
                );
                this._deckglRenderer.animateToColors(colors, meta, hoverValues);
                this._updateDeckGLLegend(wrap, meta);
                this._deckglCurrentColorBy = colorBy;
                this.showStatus(`${this.t('plot.colorUpdated')} · WebGL · ${(n / 1000).toFixed(0)}K cells`, false);
            })
            .catch(err => {
                this.hideStatus();
                this.showStatus(`${this.t('plot.failedPrefix')}: ${err}`, false);
            });
            return;
        }

        // ── full fetch: embedding changed or first render ─────────────────────
        fetch('/api/plot_gpu', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(body),
        })
        .then(r => {
            if (!r.ok) return r.json().then(e => Promise.reject(e.error || 'server error'));
            return r.arrayBuffer();
        })
        .then(buf => {
            this.hideStatus();
            const { n, meta, positions, colors, hoverValues } = parsePlotGPUBuffer(buf);
            this._setDensityControlState(
                meta.density_enabled !== false,
                meta.density_message || ''
            );

            if (isEmbeddingChange && hasExisting) {
                this._deckglRenderer.animateToPositions(positions, colors, meta, hoverValues);
            } else {
                this._deckglRenderer.setData(positions, colors, meta, hoverValues);
            }

            this._updateDeckGLLegend(wrap, meta);
            this._deckglCurrentEmbedding = axesKey;
            this._deckglCurrentColorBy   = colorBy;

            const action = isEmbeddingChange ? this.t('plot.embeddingSwitched') : this.t('plot.done');
            this.showStatus(`${action} · WebGL · ${(n / 1000).toFixed(0)}K cells`, false);
        })
        .catch(err => {
            this.hideStatus();
            this.showStatus(`${this.t('plot.failedPrefix')}: ${err}`, false);
        });
    },

    /** Build badge + control-bar overlays inside the deck.gl wrapper.
     *  Legend is now rendered in the shared #viz-legend-panel below the chart. */
    _buildDeckGLOverlays(wrap) {
        const self = this;

        // Info badge (top-left)
        const badge = document.createElement('div');
        badge.id = 'deckgl-badge';
        badge.style.cssText = [
            'position:absolute; top:6px; left:8px; z-index:10;',
            'font-size:0.68rem; padding:2px 8px; border-radius:4px;',
            'background:rgba(0,0,0,0.5); color:#fff; pointer-events:none;',
        ].join('');
        badge.textContent = 'WebGL · deck.gl';
        wrap.appendChild(badge);

        // Control bar (bottom-right): switch to Plotly
        const mkBtn = (label, title, onClick) => {
            const b = document.createElement('button');
            b.className = 'btn btn-sm btn-light border';
            b.style.cssText = 'width:28px; height:28px; padding:0; font-size:14px; line-height:1;';
            b.textContent = label;
            b.title = title;
            b.addEventListener('click', onClick);
            return b;
        };
        const bar = document.createElement('div');
        bar.style.cssText = 'position:absolute; bottom:8px; right:8px; z-index:10; display:flex; gap:4px;';
        bar.appendChild(mkBtn('⊞', 'Switch to Plotly (slower for large datasets)', () => {
            self._switchDeckGLToPlotly();
        }));
        wrap.appendChild(bar);
    },

    /** Update the legend panel (now shared below the chart, not in the canvas overlay). */
    _updateDeckGLLegend(wrap, meta) {
        const badge = document.getElementById('deckgl-badge');
        const n = meta.n_total || 0;
        if (badge) badge.textContent = `WebGL · ${(n / 1000).toFixed(0)}K cells`;

        // Render legend in the shared panel below the visualization
        this._renderLegendPanel(meta, 'deckgl');
    },

    /** User clicked "switch to Plotly" in the deck.gl panel. */
    _switchDeckGLToPlotly() {
        const wrap = document.getElementById('deckgl-wrap');
        if (wrap) wrap.style.display = 'none';
        const plotlyDiv = document.getElementById('plotly-div');
        if (plotlyDiv) plotlyDiv.style.display = '';
        // Force a fresh Plotly render (large dataset — warn user)
        if (this._deckglCurrentEmbedding) {
            const _origThreshold = this._rasterThreshold;
            this._rasterThreshold = Infinity; // temporarily disable auto-routing
            const xyAxes = this.getXYAxes ? this.getXYAxes() : null;
            this.createNewPlot(this._deckglCurrentEmbedding, this._deckglCurrentColorBy, xyAxes);
            this._rasterThreshold = _origThreshold;
        }
    },

    // ── Legacy raster path (kept as fallback when deck.gl is unavailable) ────

    /** Threshold alias (same value, kept for back-compat). */
    _rasterViewport: null,

    /**
     * Raster (Datashader / matplotlib PNG) fallback.
     * Only called when deck.gl is not available.
     */
    createRasterPlot(embedding, colorBy) {
        this.showStatus(this.t('plot.generating'), true);

        const paletteSelect = document.getElementById('palette-select');
        const catPaletteSelect = document.getElementById('category-palette-select');
        const palette = paletteSelect && paletteSelect.value !== 'default' ? paletteSelect.value : 'Viridis';
        const catPalette = catPaletteSelect && catPaletteSelect.value !== 'default' ? catPaletteSelect.value : null;

        const plotDiv = document.getElementById('plotly-div');
        const panelEl = plotDiv ? plotDiv.parentElement : null;
        const w = panelEl ? panelEl.clientWidth : 800;
        const h = panelEl ? panelEl.clientHeight : 600;

        const body = {
            embedding, color_by: colorBy,
            width: Math.max(400, w),
            height: Math.max(300, h),
            palette, category_palette: catPalette,
            x_range: this._rasterViewport ? this._rasterViewport.x_range : null,
            y_range: this._rasterViewport ? this._rasterViewport.y_range : null,
        };

        fetch('/api/plot_raster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        })
        .then(r => r.json())
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.showStatus(`${this.t('plot.failedPrefix')}: ${data.error}`, false);
                return;
            }
            this._rasterViewport = { x_range: data.x_range, y_range: data.y_range };
            this._renderRasterImage(data, embedding, colorBy);
        })
        .catch(err => {
            this.hideStatus();
            this.showStatus(`${this.t('plot.failedPrefix')}: ${err.message}`, false);
        });
    },

    /** Render the returned PNG into the raster overlay panel. */
    _renderRasterImage(data, embedding, colorBy) {
        // Ensure raster panel exists
        let panel = document.getElementById('raster-plot-panel');
        if (!panel) {
            panel = this._buildRasterPanel();
        }

        // Show raster, hide Plotly
        const plotlyDiv = document.getElementById('plotly-div');
        if (plotlyDiv) plotlyDiv.style.display = 'none';
        panel.style.display = 'flex';

        // Update image
        const img = panel.querySelector('#raster-img');
        if (img) img.src = `data:image/png;base64,${data.image}`;

        // Update badge
        const badge = panel.querySelector('#raster-badge');
        if (badge) {
            const engine = data.engine === 'datashader' ? 'Datashader' : 'Matplotlib';
            badge.textContent = `${engine} · ${(data.n_total / 1000).toFixed(0)}K cells`;
            badge.title = `Rendered with ${engine}. Zoom/pan triggers re-render.`;
        }

        // Update legend
        this._renderRasterLegend(panel, data);

        // Save current embedding/colorBy for re-render on zoom
        this._currentRasterEmbedding = embedding;
        this._currentRasterColorBy   = colorBy;

        this.showStatus(`${this.t('plot.done')} (raster, ${(data.n_total/1000).toFixed(0)}K cells)`, false);
    },

    /** Build the raster panel DOM once. */
    _buildRasterPanel() {
        const container = document.getElementById('plotly-div')?.parentElement;
        if (!container) return document.createElement('div');

        const panel = document.createElement('div');
        panel.id = 'raster-plot-panel';
        panel.style.cssText = 'display:none; flex-direction:column; width:100%; height:100%; position:relative; overflow:hidden; background:var(--bs-body-bg);';

        // Image (fills panel, preserves aspect ratio)
        const img = document.createElement('img');
        img.id = 'raster-img';
        img.style.cssText = 'width:100%; height:100%; object-fit:contain; cursor:crosshair;';
        img.alt = 'Raster scatter plot';
        panel.appendChild(img);

        // Engine/count badge
        const badge = document.createElement('div');
        badge.id = 'raster-badge';
        badge.style.cssText = [
            'position:absolute; top:6px; left:8px;',
            'font-size:0.68rem; padding:2px 7px; border-radius:4px;',
            'background:rgba(0,0,0,0.55); color:#fff; pointer-events:none;',
        ].join('');
        panel.appendChild(badge);

        // Legend overlay (top-right)
        const legend = document.createElement('div');
        legend.id = 'raster-legend';
        legend.style.cssText = [
            'position:absolute; top:6px; right:8px; max-height:70%; overflow-y:auto;',
            'font-size:0.7rem; background:rgba(var(--bs-body-bg-rgb,255,255,255),0.88);',
            'border:1px solid var(--bs-border-color,#dee2e6); border-radius:5px;',
            'padding:4px 8px; display:none;',
        ].join('');
        panel.appendChild(legend);

        // Zoom controls
        const zoomBar = document.createElement('div');
        zoomBar.style.cssText = 'position:absolute; bottom:8px; right:8px; display:flex; gap:4px;';
        const mkBtn = (label, title, onClick) => {
            const b = document.createElement('button');
            b.className = 'btn btn-sm btn-light border';
            b.style.cssText = 'width:28px; height:28px; padding:0; font-size:14px; line-height:1;';
            b.textContent = label;
            b.title = title;
            b.addEventListener('click', onClick);
            return b;
        };
        zoomBar.appendChild(mkBtn('⟳', 'Reset zoom', () => {
            this._rasterViewport = null;
            this.createRasterPlot(this._currentRasterEmbedding, this._currentRasterColorBy);
        }));
        zoomBar.appendChild(mkBtn('⊞', 'Switch to Plotly (may be slow for large data)', () => {
            this._switchToPlotly();
        }));
        panel.appendChild(zoomBar);

        // Drag-to-zoom: draw a selection rectangle on the image
        this._attachRasterZoom(img, panel);

        container.appendChild(panel);
        return panel;
    },

    /** Drag-rectangle zoom on the raster image. */
    _attachRasterZoom(img, panel) {
        let dragging = false, startX, startY;
        let rect = null;

        const getImgCoords = (clientX, clientY) => {
            const r = img.getBoundingClientRect();
            const fx = (clientX - r.left)  / r.width;
            const fy = (clientY - r.top)   / r.height;
            return { fx: Math.max(0, Math.min(1, fx)), fy: Math.max(0, Math.min(1, fy)) };
        };

        img.addEventListener('mousedown', e => {
            if (e.button !== 0) return;
            dragging = true;
            startX = e.clientX; startY = e.clientY;
            // selection box
            rect = document.createElement('div');
            rect.style.cssText = [
                'position:absolute; border:2px dashed rgba(59,130,246,0.9);',
                'background:rgba(59,130,246,0.08); pointer-events:none;',
            ].join('');
            panel.appendChild(rect);
            e.preventDefault();
        });

        window.addEventListener('mousemove', e => {
            if (!dragging || !rect) return;
            const ir = img.getBoundingClientRect();
            const pr = panel.getBoundingClientRect();
            const x1 = Math.min(startX, e.clientX) - pr.left;
            const y1 = Math.min(startY, e.clientY) - pr.top;
            const w  = Math.abs(e.clientX - startX);
            const h  = Math.abs(e.clientY - startY);
            rect.style.left   = x1 + 'px';
            rect.style.top    = y1 + 'px';
            rect.style.width  = w  + 'px';
            rect.style.height = h  + 'px';
        });

        window.addEventListener('mouseup', e => {
            if (!dragging) return;
            dragging = false;
            if (rect) { rect.remove(); rect = null; }

            const dx = Math.abs(e.clientX - startX);
            const dy = Math.abs(e.clientY - startY);
            if (dx < 8 || dy < 8) return; // too small — ignore

            const p1 = getImgCoords(startX, startY);
            const p2 = getImgCoords(e.clientX, e.clientY);
            const vp = this._rasterViewport;
            if (!vp) return;

            const [xlo, xhi] = vp.x_range;
            const [ylo, yhi] = vp.y_range;
            const xspan = xhi - xlo;
            const yspan = yhi - ylo;

            // image y-axis is flipped (top=ymax)
            const newX = [xlo + Math.min(p1.fx, p2.fx) * xspan,
                          xlo + Math.max(p1.fx, p2.fx) * xspan];
            const newY = [yhi - Math.max(p1.fy, p2.fy) * yspan,
                          yhi - Math.min(p1.fy, p2.fy) * yspan];

            this._rasterViewport = { x_range: newX, y_range: newY };
            this.createRasterPlot(this._currentRasterEmbedding, this._currentRasterColorBy);
        });
    },

    /** Render categorical legend into the raster panel. */
    _renderRasterLegend(panel, data) {
        const legend = panel.querySelector('#raster-legend');
        if (!legend) return;
        if (!data.legend || data.legend.length === 0) { legend.style.display = 'none'; return; }
        legend.style.display = 'block';
        legend.innerHTML = data.color_label
            ? `<div style="font-weight:600;margin-bottom:3px;font-size:0.72rem;">${data.color_label}</div>`
            : '';
        data.legend.forEach(({ label, color }) => {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:5px;margin:1px 0;white-space:nowrap;';
            row.innerHTML = `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${color};flex-shrink:0;"></span>${label}`;
            legend.appendChild(row);
        });
    },

    /** Switch back from raster panel to Plotly (user override). */
    _switchToPlotly() {
        const panel = document.getElementById('raster-plot-panel');
        if (panel) panel.style.display = 'none';
        const plotlyDiv = document.getElementById('plotly-div');
        if (plotlyDiv) plotlyDiv.style.display = '';
        this._rasterViewport = null;
        const xyAxes = this.getXYAxes ? this.getXYAxes() : null;
        this.createNewPlot(this._currentRasterEmbedding, this._currentRasterColorBy, xyAxes);
    },

});

// ============================================================================
// Renderer selection helpers
// ============================================================================

/**
 * _forceRenderer: null | 'deckgl' | 'plotly'
 *   null  → auto (deck.gl when n > threshold, Plotly otherwise)
 *   'deckgl' → always deck.gl regardless of cell count
 *   'plotly' → always Plotly regardless of cell count
 */
SingleCellAnalysis.prototype._forceRenderer = null;

/** Update the renderer toggle buttons to reflect the active mode. */
SingleCellAnalysis.prototype._syncRendererButtons = function (active /* 'deckgl'|'plotly'|'auto' */) {
    const btnDeck   = document.getElementById('renderer-btn-deckgl');
    const btnPlotly = document.getElementById('renderer-btn-plotly');
    const btnAuto   = document.getElementById('renderer-btn-auto');
    [btnDeck, btnPlotly, btnAuto].forEach(b => b && b.classList.remove('active'));

    if (active === 'deckgl')       { if (btnDeck)   btnDeck.classList.add('active'); }
    else if (active === 'plotly')  { if (btnPlotly) btnPlotly.classList.add('active'); }
    else                           { if (btnAuto)   btnAuto.classList.add('active'); }
};

/**
 * Public: called by the toggle buttons.
 * mode: 'deckgl' | 'plotly' | 'auto'
 */
SingleCellAnalysis.prototype.setRenderer = function (mode) {
    this._forceRenderer = (mode === 'auto') ? null : mode;
    if (this.persistRenderer) this.persistRenderer(mode); // cache for page refresh

    const emb     = document.getElementById('embedding-select') && document.getElementById('embedding-select').value;
    // Get custom axes if active (same logic as updatePlot)
    const xyAxes = this.getXYAxes ? this.getXYAxes() : null;
    // Allow rendering even without preset embedding if custom axes are active
    if (!emb && !xyAxes) return; // nothing to render yet

    const colorBy = (() => {
        const geneEl  = document.getElementById('gene-input');
        const geneVal = geneEl ? geneEl.value.trim() : '';
        return geneVal ? 'gene:' + geneVal : (document.getElementById('color-select') || {}).value || '';
    })();

    if (mode === 'deckgl') {
        this._syncRendererButtons('deckgl');
        const plotlyDiv = document.getElementById('plotly-div');
        if (plotlyDiv) plotlyDiv.style.display = 'none';
        this.createDeckGLPlot(emb, colorBy, xyAxes);
    } else if (mode === 'plotly') {
        this._syncRendererButtons('plotly');
        const wrap = document.getElementById('deckgl-wrap');
        if (wrap) wrap.style.display = 'none';
        const plotlyDiv = document.getElementById('plotly-div');
        if (plotlyDiv) plotlyDiv.style.display = '';
        const savedThreshold = this._rasterThreshold;
        this._rasterThreshold = Infinity;
        _origCreateNewPlot.call(this, emb, colorBy, xyAxes);
        this._rasterThreshold = savedThreshold;
    } else {
        // auto: re-render with default routing
        this._syncRendererButtons('auto');
        this.updatePlot();
    }
};

// ============================================================================
// Auto-routing patch: respects _forceRenderer, then falls back to threshold
// ============================================================================
const _origCreateNewPlot = SingleCellAnalysis.prototype.createNewPlot;
SingleCellAnalysis.prototype.createNewPlot = function (embedding, colorBy, xyAxes) {
    const nCells   = this.currentData ? (this.currentData.n_cells || 0) : 0;
    const useDeck  = this._forceRenderer === 'deckgl'
        || (this._forceRenderer !== 'plotly' && nCells > this._rasterThreshold);

    if (useDeck) {
        if (this._forceRenderer !== 'deckgl') this._syncRendererButtons('auto');
        else this._syncRendererButtons('deckgl');
        this.createDeckGLPlot(embedding, colorBy, xyAxes);
    } else {
        if (this._forceRenderer !== 'plotly') this._syncRendererButtons('auto');
        else this._syncRendererButtons('plotly');
        // Hide deck.gl wrap and old raster panel
        const wrap = document.getElementById('deckgl-wrap');
        if (wrap) wrap.style.display = 'none';
        const rasterPanel = document.getElementById('raster-plot-panel');
        if (rasterPanel) rasterPanel.style.display = 'none';
        // Show Plotly
        const plotlyDiv = document.getElementById('plotly-div');
        if (plotlyDiv) plotlyDiv.style.display = '';
        _origCreateNewPlot.call(this, embedding, colorBy, xyAxes);
    }
};

// Note: deck.gl renderer cleanup is handled inside resetData() in sc-tools.js,
// because Object.assign there would overwrite any monkey-patch set here.
