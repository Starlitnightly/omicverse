/**
 * OmicVerse Single Cell Analysis — Notebook Editor & Code Cells
 */

Object.assign(SingleCellAnalysis.prototype, {

    _ensureNotebookUndoStack() {
        if (!Array.isArray(this.notebookUndoStack)) {
            this.notebookUndoStack = [];
        }
        if (typeof this._suppressNotebookUndo !== 'boolean') {
            this._suppressNotebookUndo = false;
        }
    },

    _pushNotebookUndoSnapshot() {
        this._ensureNotebookUndoStack();
        if (this._suppressNotebookUndo) return;
        const snapshot = this.buildNotebookCellsFromUI();
        if (!snapshot || !snapshot.length) return;
        this.notebookUndoStack.push(snapshot);
        if (this.notebookUndoStack.length > 30) {
            this.notebookUndoStack.shift();
        }
    },

    undoLastNotebookAction() {
        this._ensureNotebookUndoStack();
        if (this.notebookUndoStack.length === 0) return;
        const snapshot = this.notebookUndoStack.pop();
        this._suppressNotebookUndo = true;
        try {
            this.loadNotebookCells(snapshot);
        } finally {
            this._suppressNotebookUndo = false;
        }
    },

    addCodeCell(code = '', outputs = [], cellType = 'code', afterCellId = null) {
        this.cellCounter++;
        const cellId = `cell-${this.cellCounter}`;

        const cellHtml = `
            <div class="code-cell" id="${cellId}" data-cell-counter="${this.cellCounter}">
                <div class="code-cell-header">
                    <span class="cell-number">In [ ]:</span>
                    <div class="cell-toolbar">
                        <select class="form-select form-select-sm" onchange="singleCellApp.changeCellType('${cellId}', this.value)">
                            <option value="code" data-i18n="cell.typeCode">Code</option>
                            <option value="markdown" data-i18n="cell.typeMarkdown">Markdown</option>
                            <option value="raw" data-i18n="cell.typeRaw">Raw</option>
                        </select>
                        <button type="button" class="btn btn-sm btn-success" onclick="singleCellApp.runCodeCell('${cellId}')" title="运行 (Shift+Enter)" data-i18n-title="cell.run">
                            <i class="feather-play"></i>
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.toggleCellOutput('${cellId}')" title="折叠输出" data-i18n-title="cell.toggleOutput">
                            <i class="fas fa-compress-alt"></i>
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.toggleCellOutputFull('${cellId}')" title="隐藏输出" data-i18n-title="cell.hideOutput">
                            <i class="fas fa-eye-slash"></i>
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.clearCellOutput('${cellId}')" title="清空输出" data-i18n-title="cell.clearOutput">
                            <i class="fas fa-eraser"></i>
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-danger" onclick="singleCellApp.deleteCodeCell('${cellId}')" title="Delete" data-i18n-title="cell.delete">
                            <i class="feather-trash-2"></i>
                        </button>
                    </div>
                </div>
                <div class="code-cell-input">
                    <pre class="code-highlight language-python"><code class="language-python"></code></pre>
                    <div class="markdown-render" id="${cellId}-markdown"></div>
                    <textarea class="code-input" placeholder="# Enter Python code (variables: adata, sc, pd, np)
# Shift+Enter to run" data-i18n-placeholder="code.placeholder">${code}</textarea>
                </div>
                <div class="code-cell-output" id="${cellId}-output"></div>
                <div class="output-hidden-note" id="${cellId}-output-hidden" data-i18n="cell.outputHidden">Output hidden</div>
            </div>
        `;

        const container = document.getElementById('code-cells-container');
        let insertedAfterId = null;
        if (afterCellId) {
            const anchorCell = document.getElementById(afterCellId);
            if (anchorCell) {
                anchorCell.insertAdjacentHTML('afterend', cellHtml);
                insertedAfterId = afterCellId;
            }
        }
        if (!insertedAfterId) {
            container.insertAdjacentHTML('beforeend', cellHtml);
        }
        this.applyLanguage(this.currentLang);

        if (insertedAfterId) {
            const anchorIndex = this.codeCells.indexOf(insertedAfterId);
            if (anchorIndex >= 0) {
                this.codeCells.splice(anchorIndex + 1, 0, cellId);
            } else {
                this.codeCells.push(cellId);
            }
        } else {
            this.codeCells.push(cellId);
        }
        this.setCellType(cellId, cellType);

        // Add keyboard shortcut (Shift+Enter to run)
        const textarea = document.querySelector(`#${cellId} .code-input`);
        const highlight = document.querySelector(`#${cellId} .code-highlight code`);
        const markdownRender = document.getElementById(`${cellId}-markdown`);
        const inputContainer = document.querySelector(`#${cellId} .code-cell-input`);
        const cellRoot = document.getElementById(cellId);

        // Auto-resize textarea based on content
        const autoResize = () => {
            textarea.style.height = 'auto';
            textarea.style.height = Math.max(60, textarea.scrollHeight) + 'px';
            const highlightContainer = document.querySelector(`#${cellId} .code-highlight`);
            if (highlightContainer) {
                highlightContainer.style.height = textarea.style.height;
            }
            if (cellRoot && cellRoot.dataset.cellType === 'markdown') {
                this.resizeMarkdownEditor(textarea);
            }
        };

        textarea.addEventListener('input', autoResize);
        textarea.addEventListener('keydown', (e) => {
            // Shift+Enter: Run cell and create new cell below (JupyterLab-like behavior)
            if (e.shiftKey && e.key === 'Enter') {
                e.preventDefault();
                this.runCodeCell(cellId);

                // After running, focus next cell or create new one
                setTimeout(() => {
                    const currentIndex = this.codeCells.indexOf(cellId);
                    if (currentIndex === -1) return;

                    if (currentIndex === this.codeCells.length - 1) {
                        // Last cell - create new empty cell below
                        this.addCodeCell();
                        // Focus the newly created cell
                        const newCellId = this.codeCells[this.codeCells.length - 1];
                        const newCell = document.getElementById(newCellId);
                        if (newCell) {
                            const newTextarea = newCell.querySelector('.code-input');
                            if (newTextarea) {
                                newTextarea.focus();
                            }
                        }
                    } else {
                        // Not last cell - focus next cell
                        const nextCellId = this.codeCells[currentIndex + 1];
                        const nextCell = document.getElementById(nextCellId);
                        if (nextCell) {
                            const nextTextarea = nextCell.querySelector('.code-input');
                            if (nextTextarea) {
                                nextTextarea.focus();
                            }
                        }
                    }
                }, 100); // Small delay to ensure cell execution started
                return;
            }

            // Tab: Indent
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = textarea.selectionStart;
                const end = textarea.selectionEnd;
                const value = textarea.value;

                if (start === end) {
                    // No selection - insert 4 spaces at cursor
                    if (!e.shiftKey) {
                        // Tab: Insert spaces
                        const indent = '    ';
                        textarea.value = value.substring(0, start) + indent + value.substring(end);
                        textarea.selectionStart = textarea.selectionEnd = start + indent.length;
                    } else {
                        // Shift+Tab: Remove up to 4 spaces before cursor
                        const lineStart = value.lastIndexOf('\n', start - 1) + 1;
                        const beforeCursor = value.substring(lineStart, start);
                        const spacesToRemove = Math.min(4, beforeCursor.match(/^ */)[0].length);
                        if (spacesToRemove > 0) {
                            textarea.value = value.substring(0, lineStart) +
                                           value.substring(lineStart + spacesToRemove);
                            textarea.selectionStart = textarea.selectionEnd = start - spacesToRemove;
                        }
                    }
                } else {
                    // Has selection - indent/dedent all selected lines
                    const lineStart = value.lastIndexOf('\n', start - 1) + 1;
                    const lineEnd = value.indexOf('\n', end);
                    const selectedLines = value.substring(lineStart, lineEnd === -1 ? value.length : lineEnd);

                    let newLines;
                    if (!e.shiftKey) {
                        // Tab: Add 4 spaces to each line
                        newLines = selectedLines.split('\n').map(line => '    ' + line).join('\n');
                    } else {
                        // Shift+Tab: Remove up to 4 spaces from each line
                        newLines = selectedLines.split('\n').map(line => {
                            const match = line.match(/^( {1,4})/);
                            return match ? line.substring(match[1].length) : line;
                        }).join('\n');
                    }

                    textarea.value = value.substring(0, lineStart) + newLines +
                                   value.substring(lineEnd === -1 ? value.length : lineEnd);

                    // Restore selection
                    textarea.selectionStart = lineStart;
                    textarea.selectionEnd = lineStart + newLines.length;
                }

                // Trigger input event to update syntax highlighting
                textarea.dispatchEvent(new Event('input'));
                return;
            }
        });
        // Track last focused cell for toolbar run button
        textarea.addEventListener('focus', () => {
            this.lastFocusedCellId = cellId;
        });
        textarea.addEventListener('input', () => this.updateCodeHighlight(textarea, highlight));
        textarea.addEventListener('scroll', () => {
            const highlightContainer = document.querySelector(`#${cellId} .code-highlight`);
            if (highlightContainer) {
                highlightContainer.scrollTop = textarea.scrollTop;
                highlightContainer.scrollLeft = textarea.scrollLeft;
            }
        });
        const cellElement = document.getElementById(cellId);
        const openMarkdownEditor = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            if (!cellElement || cellElement.dataset.cellType !== 'markdown') return;
            if (markdownRender) markdownRender.style.display = 'none';
            if (textarea) {
                textarea.style.display = 'block';
                textarea.focus();
                this.resizeMarkdownEditor(textarea);
            }
        };
        if (markdownRender) {
            markdownRender.addEventListener('dblclick', openMarkdownEditor, true);
        }
        if (inputContainer) {
            inputContainer.addEventListener('dblclick', (e) => {
                openMarkdownEditor(e);
            }, true);
        }
        if (cellRoot) {
            cellRoot.addEventListener('dblclick', (e) => {
                openMarkdownEditor(e);
            }, true);
        }

        // Initial resize
        setTimeout(autoResize, 0);
        this.updateCodeHighlight(textarea, highlight);
        this.applyCodeFontSize();

        if (outputs && outputs.length > 0) {
            const outputDiv = document.getElementById(`${cellId}-output`);
            this.renderNotebookOutputs(outputDiv, outputs);
        }

        if (cellType === 'markdown') {
            this.renderMarkdownCell(cellId);
        }
    },

    addCodeCellAtCurrent() {
        const anchor = this.lastFocusedCellId && document.getElementById(this.lastFocusedCellId)
            ? this.lastFocusedCellId
            : null;
        this._pushNotebookUndoSnapshot();
        this.addCodeCell('', [], 'code', anchor);
    },

    updateCellNumber(cellId, status) {
        const cell = document.getElementById(cellId);
        if (!cell) return;

        const cellNumber = cell.querySelector('.cell-number');
        if (!cellNumber) return;

        const cellCounter = cell.dataset.cellCounter || '';

        if (status === 'executing') {
            cellNumber.textContent = 'In [*]:';
        } else if (status === 'complete') {
            cellNumber.textContent = `In [${cellCounter}]:`;
        } else if (status === 'idle') {
            cellNumber.textContent = 'In [ ]:';
        }
    },

    runCodeCell(cellId) {
        return this.runCodeCellPromise(cellId);
    },

    runCodeCellPromise(cellId) {
        const cell = document.getElementById(cellId);
        const textarea = cell.querySelector('.code-input');
        const outputDiv = cell.querySelector('.code-cell-output');
        const code = textarea.value.trim();
        const cellType = cell.dataset.cellType || 'code';

        if (cellType === 'markdown') {
            this.renderMarkdownCell(cellId);
            return Promise.resolve();
        }

        if (cellType === 'raw') {
            this.clearCellOutput(cellId);
            return Promise.resolve();
        }

        if (!code) {
            return Promise.resolve();
        }

        // Create AbortController for this execution
        this.executionAbortController = new AbortController();

        // Update cell number to executing state
        this.updateCellNumber(cellId, 'executing');

        // Show loading
        outputDiv.className = 'code-cell-output has-content';
        outputDiv.textContent = this.t('status.executing');

        // Show interrupt button and start polling status
        this.showInterruptButton();
        this.startExecutionStatusPolling();

        // Execute code on backend with streaming
        const kernelId = this.getActiveKernelId();
        const dfLimits = this.getDataFramePreviewLimits ? this.getDataFramePreviewLimits() : { rows: 50, cols: 20 };

        // Variables to track output across promise chain
        let accumulatedOutput = '';
        let figures = [];

        // Use streaming endpoint for real-time output
        return fetch('/api/execute_code_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                code: code,
                kernel_id: kernelId,
                df_max_rows: dfLimits.rows,
                df_max_cols: dfLimits.cols
            }),
            signal: this.executionAbortController.signal
        })
        .then(async response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            // Clear output div for streaming
            outputDiv.className = 'code-cell-output has-content';
            outputDiv.textContent = '';

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let hasError = false;

            while (true) {
                const {done, value} = await reader.read();

                if (done) break;

                // Decode chunk
                buffer += decoder.decode(value, {stream: true});

                // Process SSE messages
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.trim() || line.startsWith(':')) continue;

                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));
                        // Debug logging (uncomment for debugging)
                        // console.log('SSE event:', data.type, data.text ? data.text.substring(0, 50) : '');

                        if (data.type === 'output' || data.type === 'stderr') {
                            // Append output in real-time
                            accumulatedOutput += data.text;
                            this.renderCodeOutput(outputDiv, {
                                text: accumulatedOutput,
                                isError: data.type === 'stderr',
                                figures: figures
                            });
                        } else if (data.type === 'error') {
                            accumulatedOutput += data.text;
                            hasError = true;
                            this.renderCodeOutput(outputDiv, {
                                text: data.text,
                                isError: true,
                                figures: []
                            });
                        } else if (data.type === 'interrupted') {
                            // Append interrupted message to accumulated output
                            accumulatedOutput += '\n⚠️ ' + data.text;
                            this.renderCodeOutput(outputDiv, {
                                text: accumulatedOutput,
                                isError: false,
                                figures: figures
                            });
                        } else if (data.type === 'complete') {
                            // Final result
                            if (data.figures && data.figures.length > 0) {
                                figures = data.figures;
                            }

                            if (data.error) {
                                this.renderCodeOutput(outputDiv, {
                                    text: data.error,
                                    isError: true,
                                    figures: figures
                                });
                            } else {
                                let finalOutput = data.output || accumulatedOutput || '';
                                const hasStructuredTable = data.result_kind === 'dataframe' && !!data.result_table;

                                // Add result if present (like Jupyter's auto-display of last expression)
                                if (!hasStructuredTable && data.result !== null && data.result !== undefined) {
                                    const resultStr = String(data.result);
                                    if (resultStr && resultStr !== 'None') {
                                        if (finalOutput) {
                                            finalOutput += '\n' + resultStr;
                                        } else {
                                            finalOutput = resultStr;
                                        }
                                    }
                                }

                                // If still no output, show success message
                                if (!finalOutput && figures.length === 0 && !hasStructuredTable) {
                                    finalOutput = this.t('status.noOutput');
                                }

                                this.renderCodeOutput(outputDiv, {
                                    text: finalOutput,
                                    isError: false,
                                    figures: figures,
                                    resultKind: data.result_kind,
                                    table: data.result_table,
                                    dtypes: data.result_dtypes,
                                    shape: data.result_shape
                                });
                            }

                            if (data.data_updated && data.data_info) {
                                this.refreshDataFromKernel(data.data_info);
                            }

                            // Update cell number to complete state
                            this.updateCellNumber(cellId, 'complete');
                        }
                    }
                }
            }

            // Hide interrupt button and stop polling
            this.hideInterruptButton();
            this.stopExecutionStatusPolling();

            // Update cell number to complete state if not already done
            this.updateCellNumber(cellId, 'complete');

            // Auto-refresh variable viewer and kernel stats after execution
            this.refreshKernelInfo();
        })
        .catch(error => {
            // Hide interrupt button and stop polling
            this.hideInterruptButton();
            this.stopExecutionStatusPolling();

            if (error.name === 'AbortError') {
                // Append cancel message to accumulated output instead of overwriting
                if (accumulatedOutput) {
                    accumulatedOutput += '\n⚠️ Request cancelled';
                } else {
                    accumulatedOutput = '⚠️ Request cancelled';
                }
                this.renderCodeOutput(outputDiv, {
                    text: accumulatedOutput,
                    isError: false,
                    figures: figures
                });
            } else {
                this.renderCodeOutput(outputDiv, {
                    text: `${this.t('common.error')}: ${error.message}`,
                    isError: true,
                    figures: []
                });
            }

            // Update cell number to complete state
            this.updateCellNumber(cellId, 'complete');
        });
    },

    runCurrentCell() {
        if (this.codeCells.length === 0) {
            return;
        }

        // Run the last focused cell, or the first cell if no cell was focused
        const cellToRun = this.lastFocusedCellId || this.codeCells[0];
        if (cellToRun && document.getElementById(cellToRun)) {
            this.runCodeCell(cellToRun);
        }
    },

    runAllCells() {
        if (this.codeCells.length === 0) {
            return;
        }
        let chain = Promise.resolve();
        this.codeCells.forEach(cellId => {
            chain = chain.then(() => this.runCodeCellPromise(cellId));
        });
    },

    toggleCellOutput(cellId) {
        const output = document.getElementById(`${cellId}-output`);
        const hiddenNote = document.getElementById(`${cellId}-output-hidden`);
        if (!output) return;
        output.classList.remove('collapsed');
        output.classList.toggle('partial');
        if (hiddenNote) {
            hiddenNote.classList.remove('visible');
        }
    },

    toggleCellOutputFull(cellId) {
        const output = document.getElementById(`${cellId}-output`);
        const hiddenNote = document.getElementById(`${cellId}-output-hidden`);
        if (!output) return;
        output.classList.remove('partial');
        output.classList.toggle('collapsed');
        if (hiddenNote) {
            hiddenNote.classList.toggle('visible', output.classList.contains('collapsed'));
        }
    },

    clearCellOutput(cellId) {
        const output = document.getElementById(`${cellId}-output`);
        const hiddenNote = document.getElementById(`${cellId}-output-hidden`);
        if (!output) return;
        output.innerHTML = '';
        output.className = 'code-cell-output';
        const cell = document.getElementById(cellId);
        if (cell) {
            delete cell.dataset.outputPayload;
        }
        if (hiddenNote) {
            hiddenNote.classList.remove('visible');
        }
    },

    importNotebookFile(file) {
        if (!file.name.endsWith('.ipynb')) {
            alert(this.t('notebook.selectIpynb'));
            return;
        }
        const hasCells = this.codeCells.length > 0;
        if (hasCells) {
            const ok = confirm(this.t('notebook.importConfirm'));
            if (!ok) return;
        }
        const formData = new FormData();
        formData.append('file', file);
        fetch('/api/notebooks/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.importFailed')}: ${data.error}`);
                return;
            }
            this.openNotebookTab({
                name: data.filename || file.name,
                path: data.filename || file.name,
                cells: data.cells || []
            });
            this.fetchFileList(this.currentBrowsePath);
        })
        .catch(error => {
            alert(`${this.t('status.importFailed')}: ${error.message}`);
        });
    },

    loadNotebookCells(cells) {
        const container = document.getElementById('code-cells-container');
        if (!container) return;
        container.innerHTML = '';
        this.codeCells = [];
        this.cellCounter = 0;
        const textView = document.getElementById('text-file-view');
        if (textView) textView.style.display = 'none';
        const varView = document.getElementById('var-detail-view');
        if (varView) varView.style.display = 'none';
        const mdView = document.getElementById('md-file-view');
        if (mdView) mdView.style.display = 'none';
        const imageView = document.getElementById('image-file-view');
        if (imageView) imageView.style.display = 'none';
        container.style.display = 'block';
        if (!cells.length) {
            this.addCodeCell();
            return;
        }
        cells.forEach(cell => {
            let source = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
            this.addCodeCell(source, cell.outputs || [], cell.cell_type || 'code');
        });
    },

    renderNotebookOutputs(outputDiv, outputs) {
        if (!outputDiv) return;
        let textBuffer = '';
        const images = [];
        let hasError = false;
        outputs.forEach(output => {
            if (output.output_type === 'stream') {
                textBuffer += output.text || '';
            } else if (output.output_type === 'error') {
                const traceback = output.traceback ? output.traceback.join('\n') : (output.ename || 'Error');
                textBuffer += traceback + '\n';
                hasError = true;
            } else if (output.output_type === 'execute_result' || output.output_type === 'display_data') {
                const data = output.data || {};
                if (data['text/plain']) {
                    textBuffer += Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
                    textBuffer += '\n';
                }
                if (data['image/png']) {
                    images.push(Array.isArray(data['image/png']) ? data['image/png'].join('') : data['image/png']);
                }
            }
        });
        this.renderCodeOutput(outputDiv, {
            text: textBuffer.trim(),
            isError: hasError,
            figures: images
        });
        this.storeCellOutputPayload(outputDiv, {
            text: textBuffer.trim(),
            isError: hasError,
            figures: images
        });
    },

    renderMarkdown(input) {
        const source = String(input || '');
        const markedLib = window.marked;
        const purifier = window.DOMPurify;

        if (!markedLib || typeof markedLib.parse !== 'function') {
            const escapeHtml = (text) => String(text || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            return `<pre><code>${escapeHtml(source)}</code></pre>`;
        }

        if (!this._markedConfigured) {
            markedLib.setOptions({
                gfm: true,
                breaks: true,
                headerIds: false,
                mangle: false,
                langPrefix: 'language-',
            });
            this._markedConfigured = true;
        }

        const rawHtml = markedLib.parse(source);
        if (purifier && typeof purifier.sanitize === 'function') {
            return purifier.sanitize(rawHtml, {
                USE_PROFILES: { html: true },
            });
        }
        return rawHtml;
    },

    getTextEditorContent() {
        const active = this.getActiveTab();
        if (active && (active.type === 'markdown' || active.type === 'skill')) {
            if (this._mdEditor) {
                return this._mdEditor.getValue();
            }
            const mdEditor = document.getElementById('md-file-editor');
            return mdEditor ? mdEditor.value : '';
        }
        // If CodeMirror is active (used for .py and other code files) read from it
        if (this._cmEditor) {
            return this._cmEditor.getValue();
        }
        const textEditor = document.getElementById('text-file-editor');
        return textEditor ? textEditor.value : '';
    },

    captureNotebookOutputs() {
        const outputs = [];
        this.codeCells.forEach(cellId => {
            const outputDiv = document.getElementById(`${cellId}-output`);
            const hiddenNote = document.getElementById(`${cellId}-output-hidden`);
            if (!outputDiv) {
                outputs.push(null);
                return;
            }
            outputs.push({
                html: outputDiv.innerHTML || '',
                collapsed: outputDiv.classList.contains('collapsed'),
                partial: outputDiv.classList.contains('partial'),
                hidden: hiddenNote ? hiddenNote.classList.contains('visible') : false
            });
        });
        return outputs;
    },

    restoreNotebookOutputs(outputs) {
        if (!outputs || !outputs.length) return;
        this.codeCells.forEach((cellId, idx) => {
            const state = outputs[idx];
            if (!state) return;
            const outputDiv = document.getElementById(`${cellId}-output`);
            const hiddenNote = document.getElementById(`${cellId}-output-hidden`);
            if (!outputDiv) return;
            outputDiv.innerHTML = state.html || '';
            outputDiv.className = 'code-cell-output';
            if (state.html) {
                outputDiv.classList.add('has-content');
            }
            if (state.partial) outputDiv.classList.add('partial');
            if (state.collapsed) outputDiv.classList.add('collapsed');
            if (hiddenNote) {
                hiddenNote.classList.toggle('visible', !!state.hidden);
            }
            // Re-attach AnnData chip click handlers lost when innerHTML is reassigned
            outputDiv.querySelectorAll('.adata-chip-clickable').forEach(chip => {
                chip.addEventListener('click', () => {
                    this.showAdataSlotDetail(chip.dataset.var, chip.dataset.slot, chip.dataset.key);
                });
            });
            // Re-attach DataFrame "Open in Viewer" button handlers
            outputDiv.querySelectorAll('.df-open-viewer-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    if (this.openVarTab) this.openVarTab(btn.dataset.var);
                });
            });
        });
    },

    buildNotebookCellsFromUI() {
        const cells = [];
        this.codeCells.forEach(cellId => {
            const cell = document.getElementById(cellId);
            if (!cell) return;
            const textarea = cell.querySelector('.code-input');
            if (!textarea) return;
            const cellType = cell.dataset.cellType || 'code';
            const outputs = this.getNotebookOutputsFromCell(cell);
            cells.push({
                cell_type: cellType,
                source: textarea.value,
                outputs: outputs
            });
        });
        if (cells.length === 0) {
            cells.push({ cell_type: 'code', source: '', outputs: [] });
        }
        return cells;
    },

    getNotebookOutputsFromCell(cell) {
        if (!cell || cell.dataset.cellType !== 'code') return [];
        if (!cell.dataset.outputPayload) return [];
        let payload = null;
        try {
            payload = JSON.parse(cell.dataset.outputPayload);
        } catch (e) {
            return [];
        }
        if (!payload) return [];
        const outputs = [];
        if (payload.text) {
            if (payload.isError) {
                outputs.push({
                    output_type: 'error',
                    ename: 'Error',
                    evalue: '',
                    traceback: payload.text.split('\n')
                });
            } else {
                outputs.push({
                    output_type: 'stream',
                    name: 'stdout',
                    text: payload.text
                });
            }
        }
        if (payload.figures && payload.figures.length > 0) {
            payload.figures.forEach(fig => {
                outputs.push({
                    output_type: 'display_data',
                    data: {
                        'image/png': fig
                    },
                    metadata: {}
                });
            });
        }
        return outputs;
    },

    saveActiveFile() {
        const activeTab = this.getActiveTab();
        let targetPath = activeTab ? activeTab.path : 'default.ipynb';
        let payload = null;
        if (!activeTab || activeTab.type === 'notebook') {
            payload = {
                path: targetPath,
                type: 'notebook',
                cells: this.buildNotebookCellsFromUI()
            };
        } else if (activeTab.type === 'text' || activeTab.type === 'markdown') {
            payload = {
                path: targetPath,
                type: 'text',
                content: this.getTextEditorContent()
            };
        } else if (activeTab.type === 'skill') {
            payload = {
                path: targetPath,
                content: this.getTextEditorContent()
            };
        }

        if (!payload) {
            alert(this.t('file.noSaveContent'));
            return;
        }

        const saveUrl = activeTab && activeTab.type === 'skill'
            ? '/api/skills/save'
            : '/api/files/save';

        fetch(saveUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.saveFailed')}: ${data.error}`);
                return;
            }
            if (activeTab && activeTab.type === 'notebook') {
                activeTab.cells = payload.cells;
            }
            alert(this.t('status.saveSuccess'));
        })
        .catch(error => {
            alert(`${this.t('status.saveFailed')}: ${error.message}`);
        });
    },

    isAnnDataRepr(text) {
        return /^AnnData object with n_obs\s*[×x]\s*n_vars\s*=\s*\d/.test(text.trim());
    },

    parseAnnDataRepr(text) {
        const lines = text.trim().split('\n');
        const shapeMatch = lines[0].match(/=\s*([\d,]+)\s*[×x]\s*([\d,]+)/);
        const n_obs = shapeMatch ? shapeMatch[1].replace(/,/g, '') : '?';
        const n_vars = shapeMatch ? shapeMatch[2].replace(/,/g, '') : '?';
        const slots = {};
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            const colonIdx = line.indexOf(':');
            if (colonIdx === -1) continue;
            const slot = line.slice(0, colonIdx).trim();
            const keys = line.slice(colonIdx + 1).trim()
                .split(',')
                .map(k => k.trim().replace(/^['"]|['"]$/g, ''))
                .filter(k => k.length > 0);
            if (keys.length > 0) slots[slot] = keys;
        }
        return { n_obs, n_vars, slots };
    },

    isDataFrameRepr(text) {
        return /\[\d[\d,]* rows [x×] \d[\d,]* columns\]/.test(text);
    },

    parseDataFrameRepr(text) {
        const shapeMatch = text.match(/\[(\d[\d,]*) rows [x×] (\d[\d,]*) columns\]/);
        const n_rows = shapeMatch ? parseInt(shapeMatch[1].replace(/,/g, '')) : 0;
        const n_cols = shapeMatch ? parseInt(shapeMatch[2].replace(/,/g, '')) : 0;
        // Parse column names from the first non-empty line (best-effort)
        const cols = [];
        const lines = text.trim().split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            // Header line: split by 2+ whitespace characters
            const parts = trimmed.split(/\s{2,}/);
            if (parts.length >= 1) {
                parts.forEach(p => { if (p.trim()) cols.push(p.trim()); });
            }
            break;
        }
        return { n_rows, n_cols, cols };
    },

    renderDataFrameCard(text, varName) {
        const escapeHtml = (s) => String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
        const { n_rows, n_cols, cols } = this.parseDataFrameRepr(text);
        const maxCols = 8;
        const shown = cols.slice(0, maxCols);
        const extra = cols.length > maxCols ? cols.length - maxCols : 0;
        const colChips = shown.map(c => `<span class="df-col-chip">${escapeHtml(c)}</span>`).join('');
        const moreBadge = extra ? `<span class="df-col-more">+${extra} more</span>` : '';
        const openBtn = varName
            ? `<button class="df-open-btn df-open-viewer-btn" data-var="${escapeHtml(varName)}" title="Open in variable viewer">Open in Viewer ↗</button>`
            : '';
        return `<div class="df-card">
  <span class="df-card-title">DataFrame</span><span class="df-card-shape">${n_rows.toLocaleString()} rows × ${n_cols} cols</span>
  <div class="df-card-cols">${colChips}${moreBadge}</div>
  <div class="df-card-footer">${openBtn}</div>
</div>`;
    },

    renderDataFrameCardFromTable(table, shape, varName) {
        const escapeHtml = (s) => String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
        const n_rows = Array.isArray(shape) ? Number(shape[0] || 0) : (table?.index?.length || 0);
        const n_cols = Array.isArray(shape) ? Number(shape[1] || 0) : (table?.columns?.length || 0);
        const cols = Array.isArray(table?.columns) ? table.columns.map(String) : [];
        const maxCols = 10;
        const shown = cols.slice(0, maxCols);
        const extra = cols.length > maxCols ? cols.length - maxCols : 0;
        const colChips = shown.map(c => `<span class="df-col-chip">${escapeHtml(c)}</span>`).join('');
        const moreBadge = extra ? `<span class="df-col-more">+${extra} more</span>` : '';
        const openBtn = varName
            ? `<button class="df-open-btn df-open-viewer-btn" data-var="${escapeHtml(varName)}" title="Open in variable viewer">Open in Viewer ↗</button>`
            : '';
        return `<div class="df-card">
  <span class="df-card-title">DataFrame</span><span class="df-card-shape">${n_rows.toLocaleString()} rows × ${n_cols} cols</span>
  <div class="df-card-cols">${colChips}${moreBadge}</div>
  <div class="df-card-footer">${openBtn}</div>
</div>`;
    },

    extractLastExpression(cellId) {
        const cell = document.getElementById(cellId);
        if (!cell) return null;
        const textarea = cell.querySelector('.code-input');
        if (!textarea) return null;
        const lines = textarea.value.split('\n');
        for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i].trim();
            if (!line || line.startsWith('#')) continue;
            return line;
        }
        return null;
    },

    _normalizeExpressionVarPath(expr) {
        if (!expr) return null;
        const raw = String(expr).trim();
        if (!raw) return null;
        const wrappers = [
            /\.head\([^)]*\)\s*$/,
            /\.tail\([^)]*\)\s*$/,
            /\.sample\([^)]*\)\s*$/
        ];
        let normalized = raw;
        for (const pattern of wrappers) {
            normalized = normalized.replace(pattern, '');
        }
        if (/^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$/.test(normalized)) {
            return normalized;
        }
        return null;
    },

    extractAdataVarName(cellId) {
        const expr = this.extractLastExpression(cellId);
        const path = this._normalizeExpressionVarPath(expr);
        if (!path) return null;
        return path.split('.')[0];
    },

    extractDataFrameVarPath(cellId) {
        const expr = this.extractLastExpression(cellId);
        return this._normalizeExpressionVarPath(expr);
    },

    renderDataFrameTable(tableData, dtypes) {
        if (!tableData || !Array.isArray(tableData.columns) || !Array.isArray(tableData.data)) {
            return null;
        }

        const wrap = document.createElement('div');
        wrap.className = 'df-viewer-wrap';

        const table = document.createElement('table');
        table.className = 'df-viewer-table';

        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        const corner = document.createElement('th');
        corner.className = 'df-th-index';
        corner.textContent = '#';
        headRow.appendChild(corner);

        tableData.columns.forEach((col, colIdx) => {
            const th = document.createElement('th');
            th.textContent = String(col);
            if (this._applyDfColumnTheme) this._applyDfColumnTheme(th, colIdx, 'header');
            const dtype = dtypes?.[col];
            if (dtype) {
                const badge = document.createElement('span');
                badge.className = `df-dtype-badge ${this._dtypeClass ? this._dtypeClass(dtype) : 'df-dtype-other'}`;
                badge.textContent = dtype;
                th.appendChild(badge);
            }
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        tableData.data.forEach((row, rowIdx) => {
            const tr = document.createElement('tr');
            const idxCell = document.createElement('td');
            idxCell.textContent = String((tableData.index || [])[rowIdx] ?? rowIdx);
            tr.appendChild(idxCell);
            (row || []).forEach((cell, colIdx) => {
                const td = document.createElement('td');
                const val = cell !== null && cell !== undefined ? String(cell) : '';
                td.textContent = val;
                td.title = val;
                if (this._applyDfColumnTheme) this._applyDfColumnTheme(td, colIdx, 'cell');
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        wrap.appendChild(table);
        return wrap;
    },

    renderAnnDataRepr(text, adataVarName) {
        const { n_obs, n_vars, slots } = this.parseAnnDataRepr(text);
        const clickable = adataVarName
            ? (k, slot) => `<span class="adata-key-chip adata-chip-clickable" data-var="${adataVarName}" data-slot="${slot}" data-key="${k}" title="Click to view ${slot}['${k}']" style="cursor:pointer">${k}</span>`
            : (k) => `<span class="adata-key-chip">${k}</span>`;
        const labelStyle = 'font-size:0.72rem;min-width:48px;flex-shrink:0;color:#6c757d;font-weight:600;padding-top:3px;font-family:monospace';
        const row = (label, valueHtml) =>
            `<div class="d-flex align-items-start gap-2 mb-1">
                <span style="${labelStyle}">${label}</span>
                <span style="line-height:1.8">${valueHtml}</span>
            </div>`;

        let html = `<div class="anndata-output">`;
        html += `<div class="d-flex align-items-center gap-2 mb-2">
            <i class="fas fa-database" style="font-size:0.8rem;color:#6c757d"></i>
            <span style="font-size:0.8rem;font-weight:600;color:#283c50">AnnData</span>
            <span class="adata-key-chip" style="font-weight:700;font-size:0.75rem">${parseInt(n_obs).toLocaleString()} × ${parseInt(n_vars).toLocaleString()}</span>
        </div>`;

        const slotOrder = ['obs', 'var', 'uns', 'obsm', 'varm', 'obsp', 'layers'];
        const extra = Object.keys(slots).filter(k => !slotOrder.includes(k));
        for (const slot of [...slotOrder, ...extra]) {
            if (slots[slot]) html += row(slot, slots[slot].map(k => clickable(k, slot)).join(''));
        }
        html += `</div>`;
        return html;
    },

    showAdataSlotDetail(varName, slot, key) {
        if (!varName) return;
        const kernelId = (this.getActiveKernelId ? this.getActiveKernelId() : null) || 'default.ipynb';
        const dfLimits = this.getDataFramePreviewLimits ? this.getDataFramePreviewLimits() : { rows: 50, cols: 20 };
        const params = new URLSearchParams({ var_name: varName, slot, key });
        if (kernelId) params.append('kernel_id', kernelId);
        params.append('df_max_rows', String(dfLimits.rows));
        params.append('df_max_cols', String(dfLimits.cols));
        fetch(`/api/kernel/adata_slot?${params}`)
            .then(r => r.json())
            .then(detail => {
                if (detail.error) { alert(detail.error); return; }
                const tabId = `adata-${varName}-${slot}-${key}`;
                if (this.openTabs !== undefined) {
                    const existing = this.openTabs.find(t => t.id === tabId);
                    if (!existing) {
                        this.openTabs.push({ id: tabId, type: 'var', name: detail.name, path: detail.name, detail });
                    } else {
                        existing.detail = detail;
                    }
                    if (this.setActiveTab) this.setActiveTab(tabId);
                    else if (this.showVarDetail) this.showVarDetail(detail);
                } else {
                    if (this.showVarDetail) this.showVarDetail(detail);
                }
            })
            .catch(err => console.error('adata_slot fetch error:', err));
    },

    renderCodeOutput(outputDiv, payload) {
        outputDiv.className = `code-cell-output has-content ${payload.isError ? 'error' : 'success'}`;
        outputDiv.classList.remove('markdown');
        outputDiv.innerHTML = '';
        const cellId = outputDiv.id.replace('-output', '');

        if (!payload.isError && payload.table) {
            if (payload.text && payload.text.trim()) {
                const pre = document.createElement('pre');
                pre.className = 'code-output-text';
                pre.innerHTML = this.ansiToHtml(payload.text);
                outputDiv.appendChild(pre);
            }
            const dfVar = this.extractDataFrameVarPath(cellId);
            const card = document.createElement('div');
            card.innerHTML = this.renderDataFrameCardFromTable(payload.table, payload.shape, dfVar);
            card.addEventListener('click', (e) => {
                const btn = e.target.closest('.df-open-viewer-btn');
                if (!btn) return;
                if (this.openVarTab) this.openVarTab(btn.dataset.var);
            });
            outputDiv.appendChild(card);
            const tableWrap = this.renderDataFrameTable(payload.table, payload.dtypes || {});
            if (tableWrap) {
                outputDiv.appendChild(tableWrap);
            }
        } else if (payload.text) {
            if (!payload.isError && this.isAnnDataRepr(payload.text)) {
                const adataVar = this.extractAdataVarName(cellId);
                const div = document.createElement('div');
                div.innerHTML = this.renderAnnDataRepr(payload.text, adataVar);
                div.addEventListener('click', (e) => {
                    const chip = e.target.closest('.adata-chip-clickable');
                    if (!chip) return;
                    this.showAdataSlotDetail(chip.dataset.var, chip.dataset.slot, chip.dataset.key);
                });
                outputDiv.appendChild(div);
            } else if (!payload.isError && this.isDataFrameRepr(payload.text)) {
                const dfVar = this.extractDataFrameVarPath(cellId);
                const div = document.createElement('div');
                div.innerHTML = this.renderDataFrameCard(payload.text, dfVar);
                div.addEventListener('click', (e) => {
                    const btn = e.target.closest('.df-open-viewer-btn');
                    if (!btn) return;
                    if (this.openVarTab) this.openVarTab(btn.dataset.var);
                });
                outputDiv.appendChild(div);
            } else {
                const pre = document.createElement('pre');
                pre.className = 'code-output-text';
                pre.innerHTML = this.ansiToHtml(payload.text);
                outputDiv.appendChild(pre);
            }
        }
        if (payload.figures && payload.figures.length > 0) {
            payload.figures.forEach(fig => {
                const img = document.createElement('img');
                img.className = 'code-output-figure';
                img.src = `data:image/png;base64,${fig}`;
                img.alt = 'plot';
                img.onload = () => {
                    if (img.naturalWidth) {
                        img.style.width = `${Math.round(img.naturalWidth * 0.5)}px`;
                    }
                };
                outputDiv.appendChild(img);
            });
        }
        this.storeCellOutputPayload(outputDiv, payload);
        outputDiv.classList.remove('collapsed');
        outputDiv.classList.remove('partial');
        const hiddenNote = document.getElementById(`${outputDiv.id}-hidden`);
        if (hiddenNote) {
            hiddenNote.classList.remove('visible');
        }
    },

    storeCellOutputPayload(outputDiv, payload) {
        if (!outputDiv || !payload) return;
        const cellId = outputDiv.id.replace('-output', '');
        const cell = document.getElementById(cellId);
        if (!cell) return;
        const outputPayload = {
            text: payload.text || '',
            figures: payload.figures || [],
            isError: !!payload.isError,
            table: payload.table || null,
            dtypes: payload.dtypes || null,
            shape: payload.shape || null
        };
        cell.dataset.outputPayload = JSON.stringify(outputPayload);
    },

    ansiToHtml(text) {
        const escapeHtml = (value) => value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        const ansiRegex = /\x1b\[([0-9;]*)m/g;
        const colorMap = {
            30: 'ansi-black',
            31: 'ansi-red',
            32: 'ansi-green',
            33: 'ansi-yellow',
            34: 'ansi-blue',
            35: 'ansi-magenta',
            36: 'ansi-cyan',
            37: 'ansi-white',
            90: 'ansi-bright-black',
            91: 'ansi-bright-red',
            92: 'ansi-bright-green',
            93: 'ansi-bright-yellow',
            94: 'ansi-bright-blue',
            95: 'ansi-bright-magenta',
            96: 'ansi-bright-cyan',
            97: 'ansi-bright-white'
        };

        let result = '';
        let lastIndex = 0;
        let openSpan = null;
        let match;
        while ((match = ansiRegex.exec(text)) !== null) {
            const chunk = text.slice(lastIndex, match.index);
            if (chunk) {
                result += openSpan ? `<span class="${openSpan}">${escapeHtml(chunk)}</span>` : escapeHtml(chunk);
            }
            const codes = match[1].split(';').map(c => parseInt(c, 10)).filter(Number.isFinite);
            if (codes.includes(0)) {
                openSpan = null;
            } else {
                const colorCode = codes.find(code => colorMap[code]);
                if (colorCode !== undefined) {
                    openSpan = colorMap[colorCode];
                }
            }
            lastIndex = ansiRegex.lastIndex;
        }
        const tail = text.slice(lastIndex);
        if (tail) {
            result += openSpan ? `<span class="${openSpan}">${escapeHtml(tail)}</span>` : escapeHtml(tail);
        }
        return result;
    },

    updateCodeHighlight(textarea, highlight) {
        if (!highlight || !textarea) return;
        const code = textarea.value || '';
        if (window.Prism && Prism.languages && Prism.languages.python) {
            highlight.innerHTML = Prism.highlight(code, Prism.languages.python, 'python') + '\n';
        } else {
            highlight.textContent = code;
        }
    },

    changeCellType(cellId, type) {
        this.setCellType(cellId, type);
    },

    setCellType(cellId, type) {
        const cell = document.getElementById(cellId);
        if (!cell) return;
        cell.dataset.cellType = type;
        cell.classList.remove('cell-markdown', 'cell-raw');
        if (type === 'markdown') cell.classList.add('cell-markdown');
        if (type === 'raw') cell.classList.add('cell-raw');
        const select = cell.querySelector('select.form-select');
        if (select) select.value = type;
        const textarea = cell.querySelector('.code-input');
        if (textarea) {
            if (type === 'markdown') {
                textarea.placeholder = this.t('cell.placeholderMarkdown');
            } else if (type === 'raw') {
                textarea.placeholder = this.t('cell.placeholderRaw');
            } else {
                textarea.placeholder = this.t('code.placeholder');
            }
        }
        const highlight = cell.querySelector('.code-highlight');
        if (highlight) {
            highlight.style.display = type === 'code' ? 'block' : 'none';
        }
        if (textarea) {
            textarea.style.color = type === 'code' ? 'transparent' : '';
            textarea.style.webkitTextFillColor = type === 'code' ? 'transparent' : '';
        }
        if (type === 'markdown') {
            this.renderMarkdownCell(cellId);
            if (textarea) this.resizeMarkdownEditor(textarea);
        } else {
            const markdownRender = document.getElementById(`${cellId}-markdown`);
            if (markdownRender) markdownRender.style.display = 'none';
            if (textarea) {
                textarea.style.display = 'block';
                // Recalculate height after textarea becomes visible again
                textarea.style.height = 'auto';
                setTimeout(() => {
                    textarea.style.height = Math.max(60, textarea.scrollHeight) + 'px';
                    const highlightContainer = cell.querySelector('.code-highlight');
                    if (highlightContainer) highlightContainer.style.height = textarea.style.height;
                }, 0);
            }
        }
    },

    renderMarkdownCell(cellId) {
        const cell = document.getElementById(cellId);
        if (!cell) return;
        const textarea = cell.querySelector('.code-input');
        const markdownRender = document.getElementById(`${cellId}-markdown`);
        if (!textarea || !markdownRender) return;
        markdownRender.innerHTML = this.renderMarkdown(textarea.value || '');
        markdownRender.style.display = 'block';
        textarea.style.display = 'none';
        const outputDiv = cell.querySelector('.code-cell-output');
        if (outputDiv) {
            outputDiv.innerHTML = '';
            outputDiv.className = 'code-cell-output';
        }
    },

    resizeMarkdownEditor(textarea) {
        if (!textarea) return;
        textarea.style.height = 'auto';
        textarea.style.height = Math.max(120, textarea.scrollHeight) + 'px';
    },

    deleteCodeCell(cellId) {
        const cell = document.getElementById(cellId);
        if (!cell) return;
        this._pushNotebookUndoSnapshot();
        cell.remove();
        this.codeCells = this.codeCells.filter(id => id !== cellId);
        if (this.lastFocusedCellId === cellId) {
            this.lastFocusedCellId = this.codeCells.length ? this.codeCells[Math.max(0, this.codeCells.length - 1)] : null;
        }
    },

    clearAllCells() {
        this._pushNotebookUndoSnapshot();
        const container = document.getElementById('code-cells-container');
        container.innerHTML = '';
        this.codeCells = [];
        this.cellCounter = 0;
        // Add one empty cell
        this.addCodeCell();
    },

    insertTemplate() {
        const templates = this.currentLang === 'zh'
            ? {
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
            }
            : {
                'basic_info': `# Basic info
print(adata)
print(f"Cells: {adata.n_obs}")
print(f"Genes: {adata.n_vars}")`,
                'obs_info': `# Observation columns
print(adata.obs.columns)
print(adata.obs.head())`,
                'filter': `# Filter cells
# Keep cells with 200-5000 genes
import scanpy as sc
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, max_genes=5000)
print(f"Cells after filter: {adata.n_obs}")`,
                'normalize': `# Normalize
import scanpy as sc
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Normalization done")`,
                'hvg': `# Highly variable genes
import scanpy as sc
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
print(f"HVG count: {adata.var.highly_variable.sum()}")`,
            };

        const templateKeys = Object.keys(templates);
        let options = templateKeys.map((key, idx) =>
            `${idx + 1}. ${key.replace('_', ' ')}`
        ).join('\n');

        const choice = prompt(`${this.t('template.chooseTitle')}\n${options}\n\n${this.t('template.chooseIndex')} (1-${templateKeys.length}):`);
        if (choice) {
            const idx = parseInt(choice) - 1;
            if (idx >= 0 && idx < templateKeys.length) {
                const templateKey = templateKeys[idx];
                this.addCodeCell(templates[templateKey]);
            }
        }
    }

});
