/**
 * OmicVerse Single Cell Analysis — File System, Browser & Tab Management
 */

Object.assign(SingleCellAnalysis.prototype, {

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

        // ── Preview Mode drop zone ────────────────────────────────────────────
        const dropZonePreview = document.getElementById('dropZonePreview');
        const fileInputPreview = document.getElementById('fileInputPreview');

        if (dropZonePreview && fileInputPreview) {
            dropZonePreview.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZonePreview.classList.add('dragover');
            });
            dropZonePreview.addEventListener('dragleave', (e) => {
                e.preventDefault();
                dropZonePreview.classList.remove('dragover');
            });
            dropZonePreview.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZonePreview.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) this.handleFileUploadPreview(files[0]);
            });
            fileInputPreview.addEventListener('change', (e) => {
                if (e.target.files.length > 0) this.handleFileUploadPreview(e.target.files[0]);
            });
            dropZonePreview.addEventListener('click', () => fileInputPreview.click());
        }
    },

    handleFileUploadPreview(file) {
        if (!file.name.endsWith('.h5ad')) {
            alert(this.t('upload.invalidFormat'));
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        this.showStatus(this.t('upload.previewUploading') || '正在以预览模式读取…', true);
        this.addToLog((this.t('upload.previewStart') || '预览读取') + ': ' + file.name);

        fetch('/api/upload_preview', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const contentType = response.headers.get('content-type') || '';
            if (!response.ok) {
                let text = '';
                try { text = await response.text(); } catch (e) {}
                try {
                    const js = JSON.parse(text);
                    throw new Error(js.error || text || `HTTP ${response.status}`);
                } catch (e) {
                    if (text && text.trim().startsWith('<!DOCTYPE')) {
                        throw new Error(this.t('upload.htmlError') + ` (HTTP ${response.status})`);
                    }
                    throw new Error(text || `HTTP ${response.status}`);
                }
            }
            if (contentType.includes('application/json')) return response.json();
            const text = await response.text();
            try { return JSON.parse(text); } catch (e) { throw new Error(this.t('upload.invalidResponse')); }
        })
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog(this.t('common.error') + ': ' + data.error, 'error');
                this.showStatus(this.t('upload.failed') + ': ' + data.error, false);
                alert(this.t('upload.failed') + ': ' + data.error);
            } else {
                // Mark frontend as preview mode
                data.preview_mode = true;
                this.isPreviewMode = true;
                this.currentData = data;
                this.updateUI(data);
                this.updateAdataStatus(data);
                this.updatePreviewModeBanner(true);
                this.addToLog((this.t('upload.previewSuccess') || '预览读取成功') + ': ' + data.n_cells + ' ' + this.t('status.cells') + ', ' + data.n_genes + ' ' + this.t('status.genes'));
                this.showStatus(this.t('upload.previewSuccess') || '预览模式已加载', false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(this.t('upload.failed') + ': ' + error.message, 'error');
            this.showStatus(this.t('upload.failed') + ': ' + error.message, false);
            alert(this.t('upload.failed') + ': ' + error.message);
        });
    },

    /** Show/hide the preview-mode banner and dim the Save button */
    updatePreviewModeBanner(isPreview) {
        const banner    = document.getElementById('preview-mode-banner');
        const switchBtn = document.getElementById('switch-analysis-btn');
        const saveBtn   = document.getElementById('save-btn');
        if (banner)    banner.classList.toggle('d-none', !isPreview);
        if (switchBtn) switchBtn.classList.toggle('d-none', !isPreview);
        if (saveBtn) {
            saveBtn.disabled = isPreview;
            saveBtn.title = isPreview ? (this.t('preview.saveDisabled') || '预览模式下无法保存') : '';
        }
    },

    /** Switch current backed file to full analysis mode by re-uploading from cache */
    switchToAnalysisMode() {
        if (!this.currentData || !this.currentData.filename) return;
        const msg = this.t('preview.switchConfirm') || '切换分析模式将完整加载数据到内存，可能需要一些时间。确定继续？';
        if (!confirm(msg)) return;
        // Reset preview flag immediately so the upload section reappears
        this.isPreviewMode = false;
        this.currentData = null;
        document.getElementById('upload-section').style.display = '';
        document.getElementById('data-status').classList.add('d-none');
        document.getElementById('viz-controls').style.display = 'none';
        document.getElementById('viz-panel').style.display = 'none';
        this.updatePreviewModeBanner(false);
        this.addToLog(this.t('preview.switchHint') || '请在左侧"分析读取模式"区域重新上传文件以完整加载数据。');
    },

    triggerNotebookUpload() {
        const fileInput = document.getElementById('notebook-file-input');
        if (fileInput) fileInput.click();
    },

    fetchFileTree() {
        const tree = document.getElementById('file-tree');
        if (!tree) return;
        tree.oncontextmenu = (e) => {
            e.preventDefault();
            this.openContextMenu(e.clientX, e.clientY, this.currentBrowsePath || '', true);
        };
        tree.innerHTML = `<li class="file-tree-node">${this.t('common.loading')}</li>`;
        this.loadTreeNode('', tree);
    },

    loadTreeNode(path, container) {
        const url = new URL('/api/files/list', window.location.origin);
        if (path) url.searchParams.set('path', path);
        fetch(url.toString())
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<li class="file-tree-node">${data.error}</li>`;
                return;
            }
            this.currentBrowsePath = data.path || '';
            this.renderFileTree(container, data.entries || [], data.path || '');
            const rootLabel = document.getElementById('file-browser-root');
            if (rootLabel) {
                rootLabel.textContent = data.path ? `./${data.path}` : './';
            }
        })
        .catch(error => {
            container.innerHTML = `<li class="file-tree-node">${error.message}</li>`;
        });
    },

    renderFileTree(container, entries, path) {
        container.innerHTML = '';
        if (!entries.length) {
            const empty = document.createElement('li');
            empty.className = 'file-tree-node';
            empty.textContent = this.t('file.empty');
            container.appendChild(empty);
            return;
        }
        entries.forEach(entry => {
            const li = document.createElement('li');
            const node = document.createElement('div');
            node.className = 'file-tree-node';
            const toggle = document.createElement('span');
            toggle.className = 'node-toggle';
            toggle.textContent = entry.type === 'dir' ? '▸' : '';
            const icon = document.createElement('i');
            icon.className = entry.type === 'dir' ? 'feather-folder' : 'feather-file-text';
            node.appendChild(toggle);
            node.appendChild(icon);
            node.appendChild(document.createTextNode(entry.name));
            li.appendChild(node);
            if (entry.type === 'dir') {
                const children = document.createElement('ul');
                children.className = 'file-tree-children';
                li.appendChild(children);
                node.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isOpen = children.classList.contains('open');
                    if (isOpen) {
                        children.classList.remove('open');
                        toggle.textContent = '▸';
                        return;
                    }
                    toggle.textContent = '▾';
                    children.classList.add('open');
                    if (!children.dataset.loaded) {
                        children.innerHTML = `<li class="file-tree-node">${this.t('common.loading')}</li>`;
                        const nextPath = path ? `${path}/${entry.name}` : entry.name;
                        this.loadTreeNode(nextPath, children);
                        children.dataset.loaded = '1';
                    }
                });
                node.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                    const nextPath = path ? `${path}/${entry.name}` : entry.name;
                    this.openContextMenu(e.clientX, e.clientY, nextPath, true);
                });
            } else {
                node.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const filePath = path ? `${path}/${entry.name}` : entry.name;
                    this.openFileFromServer(filePath);
                });
                node.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                    const filePath = path ? `${path}/${entry.name}` : entry.name;
                    this.openContextMenu(e.clientX, e.clientY, filePath, false);
                });
            }
            container.appendChild(li);
        });
    },

    handleFileUpload(file) {
        if (!file.name.endsWith('.h5ad')) {
            alert(this.t('upload.invalidFormat'));
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        this.showStatus(this.t('upload.uploading'), true);
        this.addToLog(this.t('upload.start') + ': ' + file.name);

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
                        throw new Error(this.t('upload.htmlError') + ` (HTTP ${response.status})`);
                    }
                    throw new Error(text || `HTTP ${response.status}`);
                }
            }
            if (contentType.includes('application/json')) {
                return response.json();
            }
            // Fallback: try parse text as JSON
            const text = await response.text();
            try { return JSON.parse(text); } catch (e) { throw new Error(this.t('upload.invalidResponse')); }
        })
        .then(data => {
            this.hideStatus();
            if (data.error) {
                this.addToLog(this.t('common.error') + ': ' + data.error, 'error');
                this.showStatus(this.t('upload.failed') + ': ' + data.error, false);
                alert(this.t('upload.failed') + ': ' + data.error);
            } else {
                // Normal full-load — clear preview mode
                this.isPreviewMode = false;
                data.preview_mode = false;
                this.currentData = data;
                this.updateUI(data);
                this.updateAdataStatus(data);
                this.updatePreviewModeBanner(false);
                this.addToLog(this.t('upload.successDetail') + ': ' + data.n_cells + ' ' + this.t('status.cells') + ', ' + data.n_genes + ' ' + this.t('status.genes'));
                this.showStatus(this.t('upload.success'), false);
            }
        })
        .catch(error => {
            this.hideStatus();
            this.addToLog(this.t('upload.failed') + ': ' + error.message, 'error');
            this.showStatus(this.t('upload.failed') + ': ' + error.message, false);
            alert(this.t('upload.failed') + ': ' + error.message);
        });
    },

    openFileFromServer(path) {
        if (!path) return;
        this.persistActiveTab();
        fetch('/api/files/open', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.openFailed')}: ${data.error}`);
                return;
            }
            if (data.type === 'notebook') {
                this.openNotebookTab({
                    name: data.name,
                    path: data.path,
                    cells: data.cells || []
                });
            } else if (data.type === 'text') {
                if ((data.ext || '').toLowerCase() === '.md') {
                    this.openMarkdownTab({
                        name: data.name,
                        path: data.path,
                        content: data.content || ''
                    });
                } else {
                    this.openTextTab({
                        name: data.name,
                        path: data.path,
                        content: data.content || ''
                    });
                }
            } else if (data.type === 'image') {
                this.openImageTab({
                    name: data.name,
                    path: data.path,
                    content: data.content,
                    mime: data.mime || 'image/png'
                });
            }
        })
        .catch(error => {
            alert(`${this.t('status.openFailed')}: ${error.message}`);
        });
    },

    openNotebookTab(tab) {
        this.persistActiveTab();
        const hasCells = this.codeCells.length > 0;
        const activeTab = this.getActiveTab();
        if (hasCells && this.activeTabId && activeTab && activeTab.type !== 'notebook') {
            const ok = confirm(this.t('notebook.openConfirm'));
            if (!ok) return;
        }
        const existing = this.openTabs.find(t => t.path === tab.path);
        if (existing) {
            this.setActiveTab(existing.id);
            return;
        }
        const id = `tab-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        const kernelId = tab.path || 'default.ipynb';
        const kernelName = tab.kernelName || 'python3';
        this.openTabs.push({
            id,
            name: tab.name,
            path: tab.path,
            type: 'notebook',
            cells: tab.cells,
            kernelId,
            kernelName
        });
        this.setActiveTab(id);
    },

    openTextTab(tab) {
        this.persistActiveTab();
        const existing = this.openTabs.find(t => t.path === tab.path);
        if (existing) {
            this.setActiveTab(existing.id);
            return;
        }
        const id = `tab-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        this.openTabs.push({
            id,
            name: tab.name,
            path: tab.path,
            type: 'text',
            content: tab.content
        });
        this.setActiveTab(id);
    },

    openMarkdownTab(tab) {
        this.persistActiveTab();
        const existing = this.openTabs.find(t => t.path === tab.path);
        if (existing) {
            this.setActiveTab(existing.id);
            return;
        }
        const id = `tab-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        this.openTabs.push({
            id,
            name: tab.name,
            path: tab.path,
            type: 'markdown',
            content: tab.content
        });
        this.setActiveTab(id);
    },

    openImageTab(tab) {
        this.persistActiveTab();
        const existing = this.openTabs.find(t => t.path === tab.path);
        if (existing) {
            this.setActiveTab(existing.id);
            return;
        }
        const id = `tab-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        this.openTabs.push({
            id,
            name: tab.name,
            path: tab.path,
            type: 'image',
            content: tab.content,
            mime: tab.mime
        });
        this.setActiveTab(id);
    },

    setActiveTab(tabId, shouldPersist = true) {
        if (shouldPersist) {
            this.persistActiveTab();
        }
        this.activeTabId = tabId;
        const tab = this.getActiveTab();
        this.renderTabs();
        if (!tab) return;
        if (tab.type === 'notebook') {
            this.loadNotebookCells(tab.cells || []);
            if (tab.outputs) {
                this.restoreNotebookOutputs(tab.outputs);
            }
            this.updateKernelSelectorForTab(tab);
            this.fetchKernelStats(tab.kernelId);
            this.fetchKernelVars(tab.kernelId);
        } else if (tab.type === 'text') {
            this.showTextFile(tab.content || '');
            this.updateKernelSelectorForTab(null);
        } else if (tab.type === 'markdown') {
            this.showMarkdownFile(tab.content || '');
            this.updateKernelSelectorForTab(null);
        } else if (tab.type === 'image') {
            this.showImageFile(tab);
            this.updateKernelSelectorForTab(null);
        } else if (tab.type === 'var') {
            this.showVarDetail(tab.detail || {});
            this.updateKernelSelectorForTab(null);
        }
    },

    getActiveTab() {
        return this.openTabs.find(t => t.id === this.activeTabId);
    },

    getActiveKernelId() {
        const tab = this.getActiveTab();
        if (tab && tab.type === 'notebook') {
            return tab.kernelId || tab.path || 'default.ipynb';
        }
        if (tab && tab.type === 'var') {
            return tab.kernelId || 'default.ipynb';
        }
        return null;
    },

    renderTabs() {
        const tabs = document.getElementById('editor-tabs');
        if (!tabs) return;
        if (this.openTabs.length === 0) {
            tabs.style.display = 'none';
            return;
        }
        tabs.style.display = 'flex';
        tabs.innerHTML = '';
        this.openTabs.forEach(tab => {
            const item = document.createElement('div');
            item.className = `editor-tab ${tab.id === this.activeTabId ? 'active' : ''}`;
            const name = document.createElement('span');
            name.textContent = tab.name;
            const close = document.createElement('span');
            close.className = 'tab-close';
            close.textContent = '×';
            close.addEventListener('click', (e) => {
                e.stopPropagation();
                this.closeTab(tab.id);
            });
            item.appendChild(name);
            item.appendChild(close);
            item.addEventListener('click', () => this.setActiveTab(tab.id));
            tabs.appendChild(item);
        });
    },

    closeTab(tabId) {
        const index = this.openTabs.findIndex(t => t.id === tabId);
        if (index === -1) return;
        if (this.activeTabId === tabId) {
            this.persistActiveTab();
        }
        this.openTabs.splice(index, 1);
        if (this.activeTabId === tabId) {
            const next = this.openTabs[index] || this.openTabs[index - 1];
            this.activeTabId = next ? next.id : null;
            if (next) {
                this.setActiveTab(next.id, false);
            } else {
                this.renderTabs();
                this.showTextFile('');
                const container = document.getElementById('code-cells-container');
                if (container) container.innerHTML = '';
                this.codeCells = [];
                this.cellCounter = 0;
                this.updateKernelSelectorForTab(null);
            }
        } else {
            this.renderTabs();
        }
    },

    showTextFile(content) {
        const container = document.getElementById('code-cells-container');
        const textView = document.getElementById('text-file-view');
        const textEditor = document.getElementById('text-file-editor');
        const varView = document.getElementById('var-detail-view');
        const mdView = document.getElementById('md-file-view');
        const imageView = document.getElementById('image-file-view');
        if (container) container.style.display = 'none';
        if (textView) {
            textView.style.display = 'block';
        }
        if (textEditor) {
            textEditor.value = content;
        }
        if (varView) {
            varView.style.display = 'none';
        }
        if (mdView) mdView.style.display = 'none';
        if (imageView) imageView.style.display = 'none';
    },

    showMarkdownFile(content) {
        const container = document.getElementById('code-cells-container');
        const textView = document.getElementById('text-file-view');
        const varView = document.getElementById('var-detail-view');
        const mdView = document.getElementById('md-file-view');
        const imageView = document.getElementById('image-file-view');
        const editor = document.getElementById('md-file-editor');
        const preview = document.getElementById('md-file-preview');
        if (container) container.style.display = 'none';
        if (textView) textView.style.display = 'none';
        if (varView) varView.style.display = 'none';
        if (imageView) imageView.style.display = 'none';
        if (mdView) mdView.style.display = 'flex';
        if (editor) {
            editor.value = content;
            editor.oninput = () => {
                if (preview) preview.innerHTML = this.renderMarkdown(editor.value);
            };
        }
        if (preview) {
            preview.innerHTML = this.renderMarkdown(content);
        }
    },

    showImageFile(tab) {
        const container = document.getElementById('code-cells-container');
        const textView = document.getElementById('text-file-view');
        const varView = document.getElementById('var-detail-view');
        const mdView = document.getElementById('md-file-view');
        const imageView = document.getElementById('image-file-view');
        const image = document.getElementById('image-file-content');
        if (container) container.style.display = 'none';
        if (textView) textView.style.display = 'none';
        if (varView) varView.style.display = 'none';
        if (mdView) mdView.style.display = 'none';
        if (imageView) imageView.style.display = 'block';
        if (image && tab.content) {
            image.src = `data:${tab.mime};base64,${tab.content}`;
        }
    },

    openContextMenu(x, y, path, isDir) {
        const menu = document.getElementById('file-context-menu');
        if (!menu) return;
        this.contextTargetPath = path || '';
        this.contextTargetIsDir = isDir;
        menu.style.left = `${x}px`;
        menu.style.top = `${y}px`;
        menu.style.display = 'block';
        const deleteItem = menu.querySelector('[data-action="delete"]');
        if (deleteItem) deleteItem.style.display = path ? 'flex' : 'none';
        const renameItem = menu.querySelector('[data-action="rename"]');
        if (renameItem) renameItem.style.display = path ? 'flex' : 'none';
        const copyItem = menu.querySelector('[data-action="copy"]');
        if (copyItem) copyItem.style.display = path ? 'flex' : 'none';
        const moveItem = menu.querySelector('[data-action="move"]');
        if (moveItem) moveItem.style.display = path ? 'flex' : 'none';
        const pasteItem = menu.querySelector('[data-action="paste"]');
        if (pasteItem) pasteItem.style.display = this.contextClipboard ? 'flex' : 'none';
    },

    hideContextMenu() {
        const menu = document.getElementById('file-context-menu');
        if (menu) menu.style.display = 'none';
    },

    contextNewFile() {
        const name = prompt(this.t('prompt.newFile'));
        if (!name) return;
        const base = this.contextTargetIsDir ? this.contextTargetPath : (this.contextTargetPath.split('/').slice(0, -1).join('/'));
        const path = base ? `${base}/${name}` : name;
        fetch('/api/files/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, type: 'file' })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.createFailed')}: ${data.error}`);
                return;
            }
            this.fetchFileTree();
        })
        .catch(err => alert(`${this.t('status.createFailed')}: ${err.message}`));
    },

    contextNewFolder() {
        const name = prompt(this.t('prompt.newFolder'));
        if (!name) return;
        const base = this.contextTargetIsDir ? this.contextTargetPath : (this.contextTargetPath.split('/').slice(0, -1).join('/'));
        const path = base ? `${base}/${name}` : name;
        fetch('/api/files/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, type: 'folder' })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.createFailed')}: ${data.error}`);
                return;
            }
            this.fetchFileTree();
        })
        .catch(err => alert(`${this.t('status.createFailed')}: ${err.message}`));
    },

    contextRefresh() {
        this.fetchFileTree();
    },

    contextDelete() {
        if (!this.contextTargetPath) return;
        const ok = confirm(this.t('prompt.deleteConfirm') + ` ${this.contextTargetPath}?`);
        if (!ok) return;
        fetch('/api/files/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: this.contextTargetPath })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.deleteFailed')}: ${data.error}`);
                return;
            }
            this.fetchFileTree();
        })
        .catch(err => alert(`${this.t('status.deleteFailed')}: ${err.message}`));
    },

    contextRename() {
        if (!this.contextTargetPath) return;
        const base = this.contextTargetPath.split('/').slice(0, -1).join('/');
        const currentName = this.contextTargetPath.split('/').pop();
        const name = prompt(this.t('prompt.renameTo'), currentName);
        if (!name || name === currentName) return;
        const dst = base ? `${base}/${name}` : name;
        fetch('/api/files/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ src: this.contextTargetPath, dst })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.renameFailed')}: ${data.error}`);
                return;
            }
            this.fetchFileTree();
        })
        .catch(err => alert(`${this.t('status.renameFailed')}: ${err.message}`));
    },

    contextCopy() {
        if (!this.contextTargetPath) return;
        this.contextClipboard = { path: this.contextTargetPath, mode: 'copy' };
    },

    contextPaste() {
        if (!this.contextClipboard) return;
        const base = this.contextTargetIsDir ? this.contextTargetPath : (this.contextTargetPath.split('/').slice(0, -1).join('/'));
        const name = this.contextClipboard.path.split('/').pop();
        const dst = base ? `${base}/${name}` : name;
        fetch('/api/files/copy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ src: this.contextClipboard.path, dst })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.pasteFailed')}: ${data.error}`);
                return;
            }
            this.fetchFileTree();
        })
        .catch(err => alert(`${this.t('status.pasteFailed')}: ${err.message}`));
    },

    contextMove() {
        if (!this.contextTargetPath) return;
        const dst = prompt(this.t('prompt.moveTo'), this.contextTargetPath);
        if (!dst || dst === this.contextTargetPath) return;
        fetch('/api/files/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ src: this.contextTargetPath, dst })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(`${this.t('status.moveFailed')}: ${data.error}`);
                return;
            }
            this.fetchFileTree();
        })
        .catch(err => alert(`${this.t('status.moveFailed')}: ${err.message}`));
    },

    persistActiveTab() {
        const active = this.getActiveTab();
        if (!active) return;
        if (active.type === 'notebook') {
            active.cells = this.buildNotebookCellsFromUI();
            active.outputs = this.captureNotebookOutputs();
            return;
        }
        if (active.type === 'markdown' || active.type === 'text') {
            active.content = this.getTextEditorContent();
            return;
        }
    }

});
