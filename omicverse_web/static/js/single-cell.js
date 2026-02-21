/**
 * OmicVerse Single Cell Analysis Platform - JavaScript Module
 */

class SingleCellAnalysis {
    constructor() {
        this.currentData = null;
        this.currentTool = null;
        this.currentTheme = 'light';
        this.currentView = 'visualization';
        this.currentLang = 'en';
        this.paramCache = {};  // persists tool parameter values across re-renders
        this.codeCells = [];
        this.cellCounter = 0;
        this.pendingPlotRefresh = false;
        this.currentBrowsePath = '';
        this.openTabs = [];
        this.activeTabId = null;
        this.codeFontSize = 13;
        this.fileTreeLoaded = false;
        this.contextTargetPath = '';
        this.contextTargetIsDir = true;
        this.contextClipboard = null;

        // Execution state tracking for interrupt support
        this.isExecuting = false;
        this.executionAbortController = null;
        this.executionStatusPollInterval = null;

        // Track last focused cell for toolbar run button
        this.lastFocusedCellId = null;

        // Initialize high-performance components
        this.dataManager = new DataManager();
        this.webglScatterplot = null;

        this.init();
    }

    init() {
        this.setupLanguageToggle();
        this.setupFileUpload();
        this.setupNavigation();
        this.setupThemeToggle();
        this.setupGeneAutocomplete();
        this.setupBeforeUnloadWarning();
        this.setupNotebookManager();
        this.setupAgentConfig();
        this.setupAgentChat();
        this.setupKernelSelector();
        this.setupSidebarResize(); // JupyterLab-like resizable sidebar
        this.checkStatus();
        this.showParameterPlaceholder();
        this.updateAdataStatus(null);
        this.applyCodeFontSize();
        this.fetchKernelVars();
        window.addEventListener('resize', () => this.syncPanelHeight());
    }

    setupLanguageToggle() {
        const stored = localStorage.getItem('omicverse.lang');
        this.currentLang = stored || 'en';
        const toggle = document.getElementById('lang-toggle');
        if (toggle) {
            toggle.addEventListener('click', () => {
                this.currentLang = this.currentLang === 'en' ? 'zh' : 'en';
                localStorage.setItem('omicverse.lang', this.currentLang);
                this.applyLanguage(this.currentLang);
            });
        }
        this.applyLanguage(this.currentLang);
    }

    t(key) {
        const dict = {
            en: {
                'lang.toggle': '中文',
                'nav.singleCell': 'Single-cell Analysis',
                'nav.preprocessing': 'Preprocessing',
                'nav.normalization': 'Normalization',
                'nav.qc': 'Quality Control',
                'nav.featureSelection': 'Feature Selection',
                'nav.dimReduction': 'Visualization',
                'nav.dimReductionSub': 'Dimensionality Reduction',
                'nav.linearDR': 'Linear DR',
                'nav.nonlinearDR': 'Nonlinear DR',
                'nav.visualization': 'Visualization',
                'nav.clustering': 'Basic Analysis',
                'nav.clusteringSub': 'Clustering',
                'nav.communityDetection': 'Community Detection',
                'nav.cellTypeId': 'Cell Type ID',
                'nav.clusterValidation': 'Cluster Validation',
                'nav.omicverse': 'Advanced Analysis',
                'nav.cellAnnotation': 'Cell Annotation',
                'nav.trajectory': 'Trajectory Analysis',
                'nav.envSection': 'Environment',
                'nav.envInstall': 'Python Package Install',
                'nav.envInfo': 'Python Env Info',
                'envInfo.title': 'Python Environment Info',
                'envInfo.loading': 'Loading environment info...',
                'envInfo.system': 'System',
                'envInfo.python': 'Python Environment',
                'envInfo.gpu': 'GPU / Accelerator',
                'envInfo.noGpu': 'No GPU / accelerator detected',
                'envInfo.keyPkgs': 'Key Package Versions',
                'envInfo.pkgName': 'Package',
                'envInfo.pkgVersion': 'Version',
                'envInfo.pkgStatus': 'Status',
                'envInfo.notInstalled': 'Not installed',
                'envInfo.allPkgs': 'All Installed Packages',
                'envInfo.filter': 'Filter packages...',
                'nav.diffAnalysis': 'Differential Analysis',
                'env.title': 'Python Environment Manager',
                'env.pkgPlaceholder': 'e.g. sctour, scvi-tools, harmonypy',
                'env.search': 'Search',
                'env.foundOnPypi': 'Found on PyPI',
                'env.notFound': 'Package not found on PyPI.',
                'env.tabPip': 'uv pip',
                'env.tabConda': 'mamba / conda',
                'env.mirror': 'Mirror:',
                'env.testMirrors': 'Test Speed',
                'env.testingMirrors': 'Testing mirror latency...',
                'env.extraArgs': 'Extra args:',
                'env.channels': 'Channels:',
                'env.customChannel': 'custom-channel',
                'env.addChannel': '+ Add',
                'env.install': 'Install',
                'env.output': 'Output',
                'env.clear': 'Clear',
                'env.consolePlaceholder': 'Ready.',
                'env.installSuccess': '✓ Installation completed successfully.',
                'env.installFailed': '✗ Installation failed',
                'env.enterPkg': 'Please enter a package name first.',
                'nav.enrichment': 'Functional Enrichment',
                'search.placeholder': 'Search genes, cell types...',
                'search.hint': 'Searching...',
                'search.genes': 'Genes',
                'search.cellTypes': 'Cell Types',
                'search.pathways': 'Pathways',
                'search.analysis': 'Analysis',
                'view.visualization': 'Visualization',
                'view.codeEditor': 'Code Editor',
                'breadcrumb.home': 'Home',
                'breadcrumb.title': 'Single-cell Analysis',
                'upload.title': 'Upload H5AD File',
                'upload.subtitle': 'Drag & drop or click to select a file',
                'upload.button': 'Select File',
                'upload.invalidFormat': 'Please upload a .h5ad file',
                'upload.uploading': 'Uploading file...',
                'upload.start': 'Starting upload',
                'upload.htmlError': 'Server returned HTML error page',
                'upload.invalidResponse': 'Server returned non-JSON response',
                'upload.failed': 'Upload failed',
                'upload.success': 'Upload successful',
                'upload.successDetail': 'Upload successful',
                'status.filename': 'Filename',
                'status.cells': 'cells',
                'status.genes': 'genes',
                'status.save': 'Save',
                'status.reset': 'Reset',
                'status.dsMax': 'max',
                'status.dsInt': 'int',
                'status.dsFloat': 'float',
                'status.dsNorm': 'normalized',
                'status.dsNormTip': 'Row sums are uniform — data appears normalized',
                'status.dsRaw': 'raw counts',
                'status.dsRawTip': 'Row sums vary significantly — data appears to be raw counts',
                'status.dsLog1pTip': 'log1p transformation has been applied',
                'status.dsScaled': 'scaled',
                'status.dsScaledTip': 'Data has negative values — sc.pp.scale may have been applied',
                'status.dsTargetTip': 'Estimated normalization target_sum',
                'controls.embedding': 'Embedding',
                'controls.embeddingPlaceholder': 'Select embedding',
                'controls.colorBy': 'Color by',
                'controls.colorNone': 'None',
                'controls.gene': 'Gene Expression',
                'controls.genePlaceholder': 'Enter gene name',
                'controls.paletteContinuous': 'Palette (continuous)',
                'controls.paletteDefault': 'Default',
                'controls.paletteDiverging': 'RdBu (diverging)',
                'controls.paletteSpectral': 'Spectral (diverging)',
                'controls.categoryPalette': 'Categorical palette',
                'controls.ovDefault': 'OmicVerse (28 colors)',
                'controls.ov56': 'OmicVerse 56',
                'controls.ov112': 'OmicVerse 112',
                'controls.vibrant': 'Vibrant (24)',
                'controls.paletteDefaultScanpy': 'Default (Scanpy)',
                'controls.tab10': 'Tab10 (10)',
                'controls.tab20': 'Tab20 (20)',
                'controls.set1': 'Set1 (9)',
                'controls.set2': 'Set2 (8)',
                'controls.set3': 'Set3 (12)',
                'controls.paired': 'Paired (12)',
                'controls.vmin': 'Min (vmin)',
                'controls.vmax': 'Max (vmax)',
                'controls.auto': 'Auto',
                'controls.apply': 'Apply',
                'controls.pointSize': 'Point size',
                'controls.opacity': 'Opacity',
                'controls.resetStyle': 'Reset style',
                'loading.processing': 'Processing data...',
                'panel.parameters': 'Parameters',
                'panel.selectAnalysis': 'Select an analysis type from the left menu',
                'panel.adataStatus': 'adata Status',
                'panel.analysisStatus': 'Analysis Status',
                'panel.waitingUpload': 'Waiting for data upload...',
                'toolbar.kernel': 'Kernel',
                'toolbar.runAll': 'Run all',
                'toolbar.addCell': 'Add cell',
                'toolbar.save': 'Save',
                'toolbar.insertTemplate': 'Insert template',
                'toolbar.clearAll': 'Clear all',
                'toolbar.fontDown': 'Decrease font',
                'toolbar.fontUp': 'Increase font',
                'toolbar.interrupt': 'Interrupt',
                'file.browser': 'File Browser',
                'file.root': 'Working Directory',
                'file.empty': 'Empty folder',
                'kernel.monitor': 'Kernel Monitor',
                'kernel.memory': 'Memory Usage',
                'kernel.topVars': 'Top Variables',
                'kernel.vars': 'Variable Viewer',
                'kernel.varName': 'Variable',
                'kernel.varType': 'Type',
                'kernel.varPreview': 'Preview',
                'kernel.switching': 'Switching kernel to',
                'kernel.switched': 'Kernel switched:',
                'kernel.switchFailed': 'Kernel switch failed:',
                'common.refresh': 'Refresh',
                'common.loading': 'Loading...',
                'common.noData': 'No data',
                'common.collapse': 'Collapse',
                'common.expand': 'Expand',
                'common.close': 'Close',
                'common.cancel': 'Cancel',
                'common.run': 'Run',
                'common.failed': 'Failed',
                'common.error': 'Error',
                'common.unknownError': 'Unknown error',
                'agent.configTitle': 'Agent Configuration',
                'agent.apiBase': 'API Base URL',
                'agent.apiKey': 'API Key',
                'agent.model': 'Model',
                'agent.modelPlaceholder': 'Model name',
                'agent.temperature': 'Temperature',
                'agent.topP': 'Top P',
                'agent.maxTokens': 'Max Tokens',
                'agent.timeout': 'Timeout (s)',
                'agent.systemPrompt': 'System Prompt',
                'agent.systemPromptPlaceholder': 'You are a single-cell analysis assistant...',
                'agent.save': 'Save',
                'agent.reset': 'Reset',
                'agent.localNotice': 'Saved in browser storage',
                'agent.title': 'Single-cell Agent',
                'agent.configButton': 'Configure',
                'agent.greeting': 'Hi, I can help analyze single-cell data. Tell me what you want to do.',
                'agent.inputPlaceholder': 'Ask a question, e.g. suggest cell type annotation',
                'agent.send': 'Send',
                'agent.analyzing': 'Analyzing...',
                'agent.done': 'Analysis completed.',
                'status.ready': 'Ready',
                'status.agentSaved': 'Agent settings saved',
                'status.agentReset': 'Agent settings reset',
                'status.downloadingData': 'Downloading processed data...',
                'status.downloadStart': 'Starting download...',
                'status.saveFailed': 'Save failed',
                'status.dataSaved': 'Data saved',
                'status.resetConfirm': 'Reset all data? Unsaved results will be lost.',
                'status.importFailed': 'Import failed',
                'status.openFailed': 'Open failed',
                'status.saveSuccess': 'Saved successfully',
                'status.uploadFirst': 'Please upload data first',
                'status.executing': 'Running...',
                'status.noOutput': '(Success, no output)',
                'status.beforeLeave': 'You have unsaved data. Leave the page?',
                'status.backendUnavailable': 'Cannot reach backend. Ensure the server is running.',
                'status.createFailed': 'Create failed',
                'status.deleteFailed': 'Delete failed',
                'status.renameFailed': 'Rename failed',
                'status.pasteFailed': 'Paste failed',
                'status.moveFailed': 'Move failed',
                'prompt.newFile': 'New file name',
                'prompt.newFolder': 'New folder name',
                'prompt.deleteConfirm': 'Delete',
                'prompt.renameTo': 'Rename to',
                'prompt.moveTo': 'Move to (relative path)',
                'common.comingSoon': 'Coming soon',
                'panel.categorySelected': 'Selected category:',
                'tools.normalize': 'Normalize',
                'tools.normalizeDesc': 'Total-count normalization',
                'tools.log1p': 'Log transform',
                'tools.log1pDesc': 'Natural log1p',
                'tools.scale': 'Scale',
                'tools.scaleDesc': 'Z-score scaling',
                'tools.filterCells': 'Filter cells',
                'tools.filterCellsDesc': 'Filter by UMI/gene thresholds',
                'tools.filterGenes': 'Filter genes',
                'tools.filterGenesDesc': 'Filter by expressing cells/UMI thresholds',
                'tools.filterOutliers': 'Filter outliers',
                'tools.filterOutliersDesc': 'QC then filter by mitochondrial ratios, etc.',
                'tools.doublets': 'Remove doublets',
                'tools.doubletsDesc': 'Detect potential doublets (Scrublet)',
                'tools.hvg': 'Highly variable genes',
                'tools.hvgDesc': 'Select HVGs',
                'tools.pca': 'PCA',
                'tools.pcaDesc': 'Principal component analysis',
                'tools.umap': 'UMAP',
                'tools.umapDesc': 'Uniform Manifold Approximation',
                'tools.tsne': 't-SNE',
                'tools.tsneDesc': 't-distributed Stochastic Neighbor Embedding',
                'tools.neighbors': 'Neighbors',
                'tools.neighborsDesc': 'KNN graph construction',
                'tools.leiden': 'Leiden',
                'tools.leidenDesc': 'Community detection',
                'tools.louvain': 'Louvain',
                'tools.louvainDesc': 'Community detection',
                'tools.cellAnnotation': 'Cell annotation',
                'tools.cellAnnotationDesc': 'Automatic cell type annotation',
                'tools.trajectory': 'Trajectory',
                'tools.trajectoryDesc': 'Cell developmental trajectory',
                'tools.diff': 'Differential analysis',
                'tools.diffDesc': 'Differential expression genes',
                'tools.enrichment': 'Enrichment',
                'tools.enrichmentDesc': 'GO/KEGG enrichment',
                'tools.celltypist': 'CellTypist',
                'tools.celltypistDesc': 'Logistic regression-based cell type annotation',
                'tools.gpt4celltype': 'GPT4Celltype',
                'tools.gpt4celltypeDesc': 'LLM-based cell type annotation',
                'tools.scsa': 'SCSA',
                'tools.scsaDesc': 'Score-based cell type annotation',
                'tools.diffusionmap': 'Diffusion Map',
                'tools.diffusionmapDesc': 'Diffusion map trajectory + DPT pseudotime',
                'tools.slingshot': 'Slingshot',
                'tools.slingshotDesc': 'Cell lineage and pseudotime inference',
                'tools.palantir': 'Palantir',
                'tools.palantirDesc': 'Cell fate prediction via diffusion',
                'tools.paga': 'PAGA',
                'tools.pagaDesc': 'Partition-based graph abstraction visualization',
                'tools.sctour': 'scTour',
                'tools.sctourDesc': 'Deep learning-based trajectory analysis',
                'gene.error': 'Gene expression error',
                'gene.notFound': 'Gene not found',
                'gene.showing': 'Showing gene expression',
                'gene.loaded': 'Gene expression loaded',
                'gene.loadFailed': 'Failed to load gene expression',
                'gene.updating': 'Updating gene expression...',
                'gene.loading': 'Loading gene expression...',
                'plot.generating': 'Generating plot...',
                'plot.errorPrefix': 'Plot error',
                'plot.failedPrefix': 'Plot failed',
                'plot.done': 'Plot generated',
                'plot.switchEmbedding': 'Switching embedding...',
                'plot.updateColor': 'Updating colors...',
                'plot.colorUpdated': 'Color updated',
                'plot.embeddingSwitched': 'Embedding switched',
                'tool.running': 'Running',
                'tool.start': 'Started',
                'tool.failed': 'failed',
                'tool.completed': 'completed',
                'tool.execFailed': 'failed to run',
                'traj.title': 'Trajectory Visualization',
                'traj.embTitle': 'Pseudotime Embedding',
                'traj.hmTitle': 'Gene Trend Heatmap',
                'traj.generate': 'Generate',
                'traj.clickGenerate': 'Click Generate to render',
                'traj.hmHint': 'Enter genes then click Generate',
                'traj.pseudotimeCol': 'Pseudotime column',
                'traj.autoDetect': 'Auto-detect',
                'traj.basis': 'Embedding',
                'traj.colormap': 'Colormap',
                'traj.pointSize': 'Point size',
                'traj.pagaOverlay': 'Overlay PAGA graph',
                'traj.pagaGroups': 'Cluster column (groups)',
                'traj.selectCol': 'Select',
                'traj.minEdge': 'Min edge width',
                'traj.nodeScale': 'Node size scale',
                'traj.genes': 'Genes (comma or newline separated)',
                'traj.layer': 'Data layer',
                'traj.bins': 'Pseudotime bins',
                'traj.hmColormap': 'Colormap',
                'notebook.selectIpynb': 'Please select a .ipynb file',
                'notebook.importConfirm': 'Importing a notebook will replace current cells. Continue?',
                'notebook.openConfirm': 'Opening a notebook will replace current cells. Continue?',
                'file.noSaveContent': 'Nothing to save',
                'cell.placeholderMarkdown': 'Enter Markdown...',
                'cell.placeholderRaw': 'Enter raw text...',
                'cell.deleteConfirm': 'Delete this code cell?',
                'cell.clearConfirm': 'Clear all code cells?',
                'template.chooseTitle': 'Choose a template:',
                'template.chooseIndex': 'Enter a number',
                'var.preview50': 'Preview 50x50',
                'var.dataframeLabel': 'DataFrame',
                'var.anndataLabel': 'AnnData',
                'var.loadToViz': 'Load to Visualization',
                'parameter.none': 'No parameters required.',
                'view.agentTitle': 'Agent Chat',
                'view.codeTitle': 'Python Code Editor',
                'breadcrumb.agent': 'Agent Chat',
                'breadcrumb.code': 'Python Code Editor',
                'code.placeholder': '# Enter Python code (variables: adata, sc, pd, np)\n# Shift+Enter to run',
                'cell.typeCode': 'Code',
                'cell.typeMarkdown': 'Markdown',
                'cell.typeRaw': 'Raw',
                'cell.run': 'Run (Shift+Enter)',
                'cell.toggleOutput': 'Toggle output',
                'cell.hideOutput': 'Hide output',
                'cell.clearOutput': 'Clear output',
                'cell.delete': 'Delete'
                , 'cell.outputHidden': 'Output hidden'
                , 'context.newFile': 'New File'
                , 'context.newFolder': 'New Folder'
                , 'context.rename': 'Rename'
                , 'context.copy': 'Copy'
                , 'context.paste': 'Paste'
                , 'context.move': 'Move'
                , 'context.delete': 'Delete'
                , 'notify.title': 'Analysis Notifications'
                , 'notify.markRead': 'Mark as read'
                , 'notify.preprocessing': 'Preprocessing'
                , 'notify.preprocessingDone': 'normalization completed'
                , 'notify.timeAgo': '2 minutes ago'
                , 'notify.empty': 'No notifications'
                , 'notify.all': 'All notifications'
                , 'user.menu': 'User Menu'
                , 'user.profile': 'Profile'
                , 'user.settings': 'Settings'
                , 'user.help': 'Help'
                , 'user.logout': 'Sign out'
            },
            zh: {
                'lang.toggle': 'EN',
                'nav.singleCell': '单细胞分析',
                'nav.preprocessing': '数据预处理',
                'nav.normalization': '标准化处理',
                'nav.qc': '质量控制',
                'nav.featureSelection': '特征选择',
                'nav.dimReduction': '可视化',
                'nav.dimReductionSub': '降维',
                'nav.linearDR': '线性降维',
                'nav.nonlinearDR': '非线性降维',
                'nav.visualization': '可视化',
                'nav.clustering': '基础分析',
                'nav.clusteringSub': '聚类',
                'nav.communityDetection': '社区检测',
                'nav.cellTypeId': '细胞类型识别',
                'nav.clusterValidation': '聚类验证',
                'nav.omicverse': '进阶分析',
                'nav.cellAnnotation': '细胞注释',
                'nav.trajectory': '轨迹分析',
                'nav.envSection': '环境管理',
                'nav.envInstall': 'Python 包安装',
                'nav.envInfo': 'Python 环境信息',
                'envInfo.title': 'Python 环境信息',
                'envInfo.loading': '正在加载环境信息...',
                'envInfo.system': '系统信息',
                'envInfo.python': 'Python 环境',
                'envInfo.gpu': 'GPU / 加速器',
                'envInfo.noGpu': '未检测到 GPU / 加速器',
                'envInfo.keyPkgs': '核心包版本',
                'envInfo.pkgName': '包名',
                'envInfo.pkgVersion': '版本',
                'envInfo.pkgStatus': '状态',
                'envInfo.notInstalled': '未安装',
                'envInfo.allPkgs': '所有已安装包',
                'envInfo.filter': '过滤包名...',
                'nav.diffAnalysis': '差异分析',
                'env.title': 'Python 环境管理',
                'env.pkgPlaceholder': '例如 sctour, scvi-tools, harmonypy',
                'env.search': '搜索',
                'env.foundOnPypi': 'PyPI 上有此包',
                'env.notFound': 'PyPI 上未找到该包。',
                'env.tabPip': 'uv pip',
                'env.tabConda': 'mamba / conda',
                'env.mirror': '镜像源:',
                'env.testMirrors': '测速',
                'env.testingMirrors': '正在测试镜像延迟...',
                'env.extraArgs': '附加参数:',
                'env.channels': 'Channel:',
                'env.customChannel': '自定义 channel',
                'env.addChannel': '+ 添加',
                'env.install': '安装',
                'env.output': '输出',
                'env.clear': '清空',
                'env.consolePlaceholder': '就绪。',
                'env.installSuccess': '✓ 安装成功。',
                'env.installFailed': '✗ 安装失败',
                'env.enterPkg': '请先输入包名。',
                'nav.enrichment': '功能富集',
                'search.placeholder': '搜索基因、细胞类型...',
                'search.hint': '搜索内容...',
                'search.genes': '基因',
                'search.cellTypes': '细胞类型',
                'search.pathways': '通路',
                'search.analysis': '分析',
                'view.visualization': '可视化',
                'view.codeEditor': '代码编辑器',
                'breadcrumb.home': '主页',
                'breadcrumb.title': '单细胞分析',
                'upload.title': '上传H5AD文件',
                'upload.subtitle': '拖拽文件到此处或点击选择文件',
                'upload.button': '选择文件',
                'upload.invalidFormat': '请上传.h5ad格式的文件',
                'upload.uploading': '正在上传文件...',
                'upload.start': '开始上传文件',
                'upload.htmlError': '服务器返回HTML错误页面',
                'upload.invalidResponse': '服务器返回非JSON',
                'upload.failed': '上传失败',
                'upload.success': '文件上传成功',
                'upload.successDetail': '文件上传成功',
                'status.filename': '文件名',
                'status.cells': '细胞',
                'status.genes': '基因',
                'status.save': '保存',
                'status.reset': '重置',
                'status.dsMax': '最大值',
                'status.dsInt': '整数',
                'status.dsFloat': '浮点数',
                'status.dsNorm': '已标准化',
                'status.dsNormTip': '各细胞行和均匀，数据已完成标准化',
                'status.dsRaw': '原始计数',
                'status.dsRawTip': '各细胞行和差异显著，数据为原始计数',
                'status.dsLog1pTip': '已执行 log1p 变换',
                'status.dsScaled': '已缩放',
                'status.dsScaledTip': '数据存在负值，可能已执行 sc.pp.scale',
                'status.dsTargetTip': '推测的标准化 target_sum',
                'controls.embedding': '降维方法',
                'controls.embeddingPlaceholder': '选择降维方法',
                'controls.colorBy': '着色方式',
                'controls.colorNone': '无着色',
                'controls.gene': '基因表达',
                'controls.genePlaceholder': '输入基因名',
                'controls.paletteContinuous': '调色板（连续）',
                'controls.paletteDefault': '默认',
                'controls.paletteDiverging': 'RdBu（发散）',
                'controls.paletteSpectral': 'Spectral（发散）',
                'controls.categoryPalette': '分类调色板',
                'controls.ovDefault': 'OmicVerse（28色）',
                'controls.ov56': 'OmicVerse 56',
                'controls.ov112': 'OmicVerse 112',
                'controls.vibrant': 'Vibrant（24色）',
                'controls.paletteDefaultScanpy': '默认（Scanpy）',
                'controls.tab10': 'Tab10（10色）',
                'controls.tab20': 'Tab20（20色）',
                'controls.set1': 'Set1（9色）',
                'controls.set2': 'Set2（8色）',
                'controls.set3': 'Set3（12色）',
                'controls.paired': 'Paired（12色）',
                'controls.vmin': '最小值 (vmin)',
                'controls.vmax': '最大值 (vmax)',
                'controls.auto': '自动',
                'controls.apply': '应用',
                'controls.pointSize': '点大小',
                'controls.opacity': '透明度',
                'controls.resetStyle': '重置样式',
                'loading.processing': '正在处理数据...',
                'panel.parameters': '参数设置',
                'panel.selectAnalysis': '点击左侧菜单栏选取功能',
                'panel.adataStatus': 'adata 状态',
                'panel.analysisStatus': '分析状态',
                'panel.waitingUpload': '等待上传数据...',
                'toolbar.kernel': '内核',
                'toolbar.runAll': '运行全部',
                'toolbar.addCell': '新增单元',
                'toolbar.save': '保存',
                'toolbar.insertTemplate': '插入模板',
                'toolbar.clearAll': '清空所有',
                'toolbar.fontDown': '减小字号',
                'toolbar.fontUp': '增大字号',
                'toolbar.interrupt': '中断',
                'file.browser': '文件浏览器',
                'file.root': '运行目录',
                'file.empty': '空目录',
                'kernel.monitor': 'Kernel 监控',
                'kernel.memory': '内存占用',
                'kernel.topVars': '变量占用 Top 10',
                'kernel.vars': '变量查看器',
                'kernel.varName': '变量',
                'kernel.varType': '类型',
                'kernel.varPreview': '预览',
                'kernel.switching': '正在切换内核到',
                'kernel.switched': '内核已切换:',
                'kernel.switchFailed': '切换内核失败:',
                'common.refresh': '刷新',
                'common.loading': '加载中...',
                'common.noData': '暂无数据',
                'common.collapse': '收起',
                'common.expand': '展开',
                'common.close': '关闭',
                'common.cancel': '取消',
                'common.run': '运行',
                'common.failed': '失败',
                'common.error': '错误',
                'common.unknownError': '未知错误',
                'agent.configTitle': 'Agent 配置',
                'agent.apiBase': 'API Base URL',
                'agent.apiKey': 'API Key',
                'agent.model': '模型',
                'agent.modelPlaceholder': '模型名称',
                'agent.temperature': 'Temperature',
                'agent.topP': 'Top P',
                'agent.maxTokens': 'Max Tokens',
                'agent.timeout': 'Timeout (s)',
                'agent.systemPrompt': 'System Prompt',
                'agent.systemPromptPlaceholder': '你是一个单细胞分析助手...',
                'agent.save': '保存',
                'agent.reset': '重置',
                'agent.localNotice': '配置保存在浏览器本地',
                'agent.title': '单细胞分析 Agent',
                'agent.configButton': '配置模型',
                'agent.greeting': '你好，我可以帮你分析单细胞数据。告诉我你想做什么分析。',
                'agent.inputPlaceholder': '输入你的问题，比如：帮我做细胞类型注释建议',
                'agent.send': '发送',
                'agent.analyzing': '正在分析...',
                'agent.done': '已完成分析。',
                'status.ready': '就绪',
                'status.agentSaved': 'Agent 配置已保存',
                'status.agentReset': 'Agent 配置已重置',
                'status.downloadingData': '正在下载处理后的数据...',
                'status.downloadStart': '开始下载处理后的数据...',
                'status.saveFailed': '保存失败',
                'status.dataSaved': '数据保存成功',
                'status.resetConfirm': '确定要重置所有数据吗？所有未保存的分析结果将丢失。',
                'status.importFailed': '导入失败',
                'status.openFailed': '打开失败',
                'status.saveSuccess': '保存成功',
                'status.uploadFirst': '请先上传数据',
                'status.executing': '执行中...',
                'status.noOutput': '(执行成功，无输出)',
                'status.beforeLeave': '您有未保存的数据，刷新页面将丢失所有分析结果。确定要离开吗？',
                'status.backendUnavailable': '无法连接后端，请确认服务已启动并允许访问。',
                'status.createFailed': '创建失败',
                'status.deleteFailed': '删除失败',
                'status.renameFailed': '重命名失败',
                'status.pasteFailed': '粘贴失败',
                'status.moveFailed': '移动失败',
                'prompt.newFile': '新建文件名',
                'prompt.newFolder': '新建文件夹名',
                'prompt.deleteConfirm': '确认删除',
                'prompt.renameTo': '重命名为',
                'prompt.moveTo': '移动到 (相对路径)',
                'common.comingSoon': '该功能正在开发中，敬请期待！',
                'panel.categorySelected': '选择分析类别:',
                'tools.normalize': '归一化',
                'tools.normalizeDesc': 'Total-count 归一化',
                'tools.log1p': '对数转换',
                'tools.log1pDesc': '自然对数 log1p',
                'tools.scale': '数据缩放',
                'tools.scaleDesc': 'Z-score 标准化',
                'tools.filterCells': '过滤细胞',
                'tools.filterCellsDesc': '按UMI/基因数上下限过滤',
                'tools.filterGenes': '过滤基因',
                'tools.filterGenesDesc': '按表达细胞数/UMI上下限过滤',
                'tools.filterOutliers': '过滤异常细胞',
                'tools.filterOutliersDesc': '先计算QC，再按线粒体比例等过滤',
                'tools.doublets': '去除双细胞',
                'tools.doubletsDesc': '识别并去除潜在双细胞（Scrublet）',
                'tools.hvg': '高变基因',
                'tools.hvgDesc': '选择高变基因',
                'tools.pca': 'PCA分析',
                'tools.pcaDesc': '主成分分析',
                'tools.umap': 'UMAP降维',
                'tools.umapDesc': '统一流形近似投影',
                'tools.tsne': 't-SNE降维',
                'tools.tsneDesc': 't-分布随机邻域嵌入',
                'tools.neighbors': '邻域计算',
                'tools.neighborsDesc': 'K近邻图构建',
                'tools.leiden': 'Leiden聚类',
                'tools.leidenDesc': '高质量社区检测',
                'tools.louvain': 'Louvain聚类',
                'tools.louvainDesc': '经典社区检测',
                'tools.cellAnnotation': '细胞注释',
                'tools.cellAnnotationDesc': '自动细胞类型注释',
                'tools.trajectory': '轨迹分析',
                'tools.trajectoryDesc': '细胞发育轨迹',
                'tools.diff': '差异分析',
                'tools.diffDesc': '差异表达基因',
                'tools.enrichment': '功能富集',
                'tools.enrichmentDesc': 'GO/KEGG富集分析',
                'tools.celltypist': 'CellTypist',
                'tools.celltypistDesc': '基于逻辑回归的细胞类型自动注释',
                'tools.gpt4celltype': 'GPT4Celltype',
                'tools.gpt4celltypeDesc': '基于大语言模型的细胞类型注释',
                'tools.scsa': 'SCSA',
                'tools.scsaDesc': '基于打分系统的细胞类型注释',
                'tools.diffusionmap': 'Diffusion Map',
                'tools.diffusionmapDesc': '扩散映射轨迹分析，计算 DPT 拟时序',
                'tools.slingshot': 'Slingshot',
                'tools.slingshotDesc': '细胞谱系与拟时序推断',
                'tools.palantir': 'Palantir',
                'tools.palantirDesc': '基于扩散过程的细胞命运预测',
                'tools.paga': 'PAGA',
                'tools.pagaDesc': '基于分区的图抽象可视化',
                'tools.sctour': 'scTour',
                'tools.sctourDesc': '基于深度学习的轨迹分析',
                'gene.error': '基因表达错误',
                'gene.notFound': '基因未找到',
                'gene.showing': '显示基因表达',
                'gene.loaded': '基因表达加载完成',
                'gene.loadFailed': '基因表达加载失败',
                'gene.updating': '正在更新基因表达...',
                'gene.loading': '正在加载基因表达...',
                'plot.generating': '正在生成图表...',
                'plot.errorPrefix': '绘图错误',
                'plot.failedPrefix': '绘图失败',
                'plot.done': '图表生成完成',
                'plot.switchEmbedding': '正在切换降维方法...',
                'plot.updateColor': '正在更新着色...',
                'plot.colorUpdated': '着色更新完成',
                'plot.embeddingSwitched': '降维方法切换完成',
                'tool.running': '正在执行',
                'tool.start': '开始执行',
                'tool.failed': '失败',
                'tool.completed': '完成',
                'tool.execFailed': '执行失败',
                'traj.title': '轨迹分析可视化',
                'traj.embTitle': '拟时序嵌入图',
                'traj.hmTitle': '基因拟时序热图',
                'traj.generate': '生成',
                'traj.clickGenerate': '点击「生成」显示嵌入图',
                'traj.hmHint': '选择基因后点击「生成」',
                'traj.pseudotimeCol': '拟时序列',
                'traj.autoDetect': '自动检测',
                'traj.basis': '嵌入方法',
                'traj.colormap': '配色',
                'traj.pointSize': '点大小',
                'traj.pagaOverlay': '叠加 PAGA 图',
                'traj.pagaGroups': '聚类列 (groups)',
                'traj.selectCol': '选择',
                'traj.minEdge': '最小边宽',
                'traj.nodeScale': '节点大小缩放',
                'traj.genes': '基因列表（逗号或换行分隔）',
                'traj.layer': '数据层',
                'traj.bins': '拟时序分箱数',
                'traj.hmColormap': '热图配色',
                'notebook.selectIpynb': '请选择 .ipynb 文件',
                'notebook.importConfirm': '导入笔记本会替换当前代码单元，是否继续？',
                'notebook.openConfirm': '打开笔记本会替换当前代码单元，是否继续？',
                'file.noSaveContent': '没有可保存的内容',
                'cell.placeholderMarkdown': '输入 Markdown...',
                'cell.placeholderRaw': '输入原始文本...',
                'cell.deleteConfirm': '确定要删除这个代码单元吗？',
                'cell.clearConfirm': '确定要清空所有代码单元吗？',
                'template.chooseTitle': '选择模板:',
                'template.chooseIndex': '输入编号',
                'var.preview50': '预览 50x50',
                'var.dataframeLabel': 'DataFrame',
                'var.anndataLabel': 'AnnData',
                'var.loadToViz': '加载到可视化',
                'parameter.none': '该工具无需参数设置',
                'view.agentTitle': 'Agent 对话',
                'view.codeTitle': 'Python 代码编辑器',
                'breadcrumb.agent': 'Agent 对话',
                'breadcrumb.code': 'Python 代码编辑器',
                'code.placeholder': '# 输入Python代码 (可用变量: adata, sc, pd, np)\n# Shift+Enter 运行代码',
                'cell.typeCode': 'Code',
                'cell.typeMarkdown': 'Markdown',
                'cell.typeRaw': 'Raw',
                'cell.run': '运行 (Shift+Enter)',
                'cell.toggleOutput': '折叠输出',
                'cell.hideOutput': '隐藏输出',
                'cell.clearOutput': '清空输出',
                'cell.delete': '删除'
                , 'cell.outputHidden': '输出已隐藏'
                , 'context.newFile': '新建文件'
                , 'context.newFolder': '新建文件夹'
                , 'context.rename': '重命名'
                , 'context.copy': '复制'
                , 'context.paste': '粘贴'
                , 'context.move': '移动'
                , 'context.delete': '删除'
                , 'notify.title': '分析通知'
                , 'notify.markRead': '标记为已读'
                , 'notify.preprocessing': '数据预处理'
                , 'notify.preprocessingDone': '已完成标准化处理'
                , 'notify.timeAgo': '2分钟前'
                , 'notify.empty': '暂无通知'
                , 'notify.all': '所有通知'
                , 'user.menu': '用户菜单'
                , 'user.profile': '个人资料'
                , 'user.settings': '设置'
                , 'user.help': '帮助'
                , 'user.logout': '退出'
            }
        };
        if (!key) return '';
        return (dict[this.currentLang] && dict[this.currentLang][key]) || key;
    }

    applyLanguage(lang) {
        this.currentLang = lang;
        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (key) {
                el.textContent = this.t(key);
            }
        });
        const placeholders = document.querySelectorAll('[data-i18n-placeholder]');
        placeholders.forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            if (key) {
                el.setAttribute('placeholder', this.t(key));
            }
        });
        const titles = document.querySelectorAll('[data-i18n-title]');
        titles.forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            if (key) {
                el.setAttribute('title', this.t(key));
            }
        });
        const langToggle = document.getElementById('lang-toggle');
        if (langToggle) {
            langToggle.textContent = this.t('lang.toggle');
        }
        // Re-render dynamic content in the parameter panel
        if (this.currentTool) {
            // A tool form is open — re-render it (saves & restores values)
            this.refreshParameterFormLanguage();
        } else if (this.currentCategory) {
            // The tool list is showing — re-render so category labels update
            this.selectAnalysisCategory(this.currentCategory, { silent: true });
        }
        this.updateCodeCellPlaceholders();
    }

    translateFormHtml(html) {
        if (this.currentLang !== 'en') {
            return html;
        }
        const replacements = [
            ['最小UMI数', 'Min UMI'],
            ['最小基因数', 'Min genes'],
            ['最大UMI数', 'Max UMI'],
            ['最大基因数', 'Max genes'],
            ['最少表达细胞数', 'Min expressed cells'],
            ['最多表达细胞数', 'Max expressed cells'],
            ['可留空', 'Optional'],
            ['留空表示不限制。仅填写需要的阈值即可。', 'Leave blank for no limit.'],
            ['可选：设置上下限阈值，未填表示不限制。', 'Optional: set min/max thresholds; blank means no limit.'],
            ['将先计算线粒体/核糖体/血红蛋白等QC指标，再按阈值过滤。', 'QC metrics will be computed first, then filtered by thresholds.'],
            ['最大线粒体比例', 'Max mitochondrial'],
            ['线粒体基因前缀 (自动检测)', 'Mito gene prefixes (auto-detect)'],
            ['例如', 'e.g.'],
            ['最大核糖体基因比例', 'Max ribosomal'],
            ['最大血红蛋白基因比例', 'Max hemoglobin'],
            ['(百分比)', '(%)'],
            ['未填写的阈值不生效；线粒体前缀自动检测可编辑。', 'Unfilled thresholds are ignored; prefix can be edited.'],
            ['无', 'None'],
            ['批次列', 'Batch column'],
            ['模拟双细胞比', 'Simulated doublet ratio'],
            ['期望双细胞率', 'Expected doublet rate'],
            ['双细胞率标准差', 'Doublet rate stdev'],
            ['UMI子采样', 'UMI subsampling'],
            ['KNN距离度量', 'KNN distance metric'],
            ['PCA主成分数', 'PCA components'],
            ['目标总数', 'Target sum'],
            ['最大值', 'Max value'],
            ['基因数量', 'Gene count'],
            ['方法', 'Method'],
            ['主成分数量', 'Number of PCs'],
            ['邻居数量', 'Number of neighbors'],
            ['最小距离', 'Min distance'],
            ['困惑度', 'Perplexity'],
            ['分辨率', 'Resolution'],
            ['该工具无需参数设置', 'No parameters required.'],
            ['返回工具列表', 'Back to tools'],
            ['运行', 'Run'],
            // Annotation forms
            ['模型文件路径', 'Model file path'],
            ['本地 .pkl', 'local .pkl'],
            ['下载后自动填入，或手动输入路径', 'Auto-filled after download, or enter path manually'],
            ['从 CellTypist 获取模型列表', 'Fetch CellTypist model list'],
            ['获取中...', 'Fetching...'],
            ['模型列表已加载', 'Model list loaded'],
            ['-- 选择模型 --', '-- Select model --'],
            ['下载选中模型', 'Download selected model'],
            ['下载中...', 'Downloading...'],
            ['✓ 已下载', '✓ Downloaded'],
            ['✓ 已找到本地模型: ', '✓ Found local model: '],
            ['运行注释', 'Run annotation'],
            ['聚类键', 'Cluster key'],
            ['组织类型', 'Tissue type'],
            ['物种', 'Species'],
            ['LLM 提供商', 'LLM Provider'],
            ['通义千问', 'Tongyi Qianwen'],
            ['模型名称', 'Model name'],
            ['留空则读取 AGI_API_KEY 环境变量', 'Leave blank to use AGI_API_KEY env var'],
            ['(可选)', '(optional)'],
            ['自定义 OpenAI 兼容端点', 'Custom OpenAI-compatible endpoint'],
            ['Top marker 基因数', 'Top marker genes'],
            ['SCSA 数据库路径', 'SCSA database path'],
            ['留空将自动下载', 'Leave blank to auto-download'],
            ['下载', 'Download'],
            ['倍数变化阈值', 'Fold change threshold'],
            ['P 值阈值', 'P-value threshold'],
            ['细胞类型', 'Cell type'],
            ['参考数据库', 'Reference database'],
            ['组织', 'Tissue'],
            ['留空=All', 'blank=All'],
            // Trajectory forms
            ['聚类列', 'Cluster column'],
            ['低维表示', 'Low-dim representation'],
            ['主成分数', 'Number of PCs'],
            ['起始细胞类型', 'Origin cell type'],
            ['终止细胞类型', 'Terminal cell types'],
            ['逗号分隔', 'comma-separated'],
            ['训练轮数', 'Training epochs'],
            ['路标点数', 'Waypoints'],
            ['拟时序列', 'Pseudotime column'],
            ['低维表示 (use_rep，用于重算邻居)', 'Low-dim rep (use_rep, for neighbor recompute)'],
            ['可视化嵌入', 'Visualization embedding'],
            ['lec 重建权重', 'lec reconstruction weight'],
            ['lode 重建权重', 'lode reconstruction weight'],
            ['将在 obs 中添加 dpt_pseudotime 列，可用于可视化着色。', 'Adds dpt_pseudotime column to obs for coloring.'],
            ['将在 obs 中添加 slingshot_pseudotime 列。', 'Adds slingshot_pseudotime column to obs.'],
            ['将在 obs 中添加 palantir_pseudotime 和 palantir_entropy 列。', 'Adds palantir_pseudotime and palantir_entropy columns to obs.'],
            ['将生成 PAGA 图并在结果区域显示。', 'Generates a PAGA graph shown in the result overlay.'],
            ['需要原始 count 数据在 adata.X。将在 obs 中添加 sctour_pseudotime 列。', 'Requires raw counts in adata.X. Adds sctour_pseudotime column to obs.'],
            ['-- 选择列 --', '-- Select column --'],
            ['-- 自动检测 --', '-- Auto-detect --'],
            ['-- 不设置 --', '-- Not set --'],
            ['例如: Ductal', 'e.g. Ductal'],
            ['例如: Alpha,Beta', 'e.g. Alpha,Beta'],
            ['例如: Alpha,Beta,Delta', 'e.g. Alpha,Beta,Delta'],
            // Save modal
            ['文件名', 'Filename'],
            ['文件将下载到浏览器默认下载目录', 'File will be saved to the browser default download folder'],
            ['确认下载', 'Confirm download']
        ];
        let output = html;
        replacements.forEach(([from, to]) => {
            output = output.split(from).join(to);
        });
        return output;
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

        if (mobileToggle) {
            mobileToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
        }
    }

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
    }

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
    }

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
    }

    setupAgentConfig() {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        this.loadAgentConfig();
        Object.values(fields).forEach(field => {
            field.addEventListener('change', () => this.saveAgentConfig(true));
        });
    }

    setupAgentChat() {
        const input = document.getElementById('agent-input');
        if (!input) return;
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.sendAgentMessage();
            }
        });
    }

    getAgentConfigFields() {
        const fields = {
            apiBase: document.getElementById('agent-api-base'),
            apiKey: document.getElementById('agent-api-key'),
            model: document.getElementById('agent-model'),
            temperature: document.getElementById('agent-temperature'),
            topP: document.getElementById('agent-top-p'),
            maxTokens: document.getElementById('agent-max-tokens'),
            timeout: document.getElementById('agent-timeout'),
            systemPrompt: document.getElementById('agent-system-prompt')
        };
        const hasAll = Object.values(fields).every(Boolean);
        return hasAll ? fields : null;
    }

    loadAgentConfig() {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        let stored = null;
        try {
            stored = JSON.parse(localStorage.getItem('omicverse.agentConfig') || 'null');
        } catch (e) {
            stored = null;
        }
        if (!stored) {
            fields.apiBase.value = fields.apiBase.value || 'https://api.openai.com/v1';
            fields.model.value = fields.model.value || 'gpt-5';
            return;
        }
        fields.apiBase.value = stored.apiBase || fields.apiBase.value || 'https://api.openai.com/v1';
        fields.apiKey.value = stored.apiKey || '';
        fields.model.value = stored.model || fields.model.value || 'gpt-5';
        fields.temperature.value = stored.temperature ?? fields.temperature.value;
        fields.topP.value = stored.topP ?? fields.topP.value;
        fields.maxTokens.value = stored.maxTokens ?? fields.maxTokens.value;
        fields.timeout.value = stored.timeout ?? fields.timeout.value;
        fields.systemPrompt.value = stored.systemPrompt || '';
    }

    saveAgentConfig(silent = false) {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        const payload = {
            apiBase: fields.apiBase.value.trim(),
            apiKey: fields.apiKey.value.trim(),
            model: fields.model.value.trim(),
            temperature: fields.temperature.value,
            topP: fields.topP.value,
            maxTokens: fields.maxTokens.value,
            timeout: fields.timeout.value,
            systemPrompt: fields.systemPrompt.value.trim()
        };
        localStorage.setItem('omicverse.agentConfig', JSON.stringify(payload));
        if (!silent) {
            this.showStatus(this.t('status.agentSaved'), false);
            setTimeout(() => this.hideStatus(), 1200);
        }
    }

    resetAgentConfig() {
        const fields = this.getAgentConfigFields();
        if (!fields) return;
        localStorage.removeItem('omicverse.agentConfig');
        fields.apiBase.value = 'https://api.openai.com/v1';
        fields.apiKey.value = '';
        fields.model.value = 'gpt-5';
        fields.temperature.value = 0.3;
        fields.topP.value = 1;
        fields.maxTokens.value = 2048;
        fields.timeout.value = 60;
        fields.systemPrompt.value = '';
        this.showStatus(this.t('status.agentReset'), false);
        setTimeout(() => this.hideStatus(), 1200);
    }

    getAgentConfig() {
        let stored = null;
        try {
            stored = JSON.parse(localStorage.getItem('omicverse.agentConfig') || 'null');
        } catch (e) {
            stored = null;
        }
        if (stored) {
            return stored;
        }
        const fields = this.getAgentConfigFields();
        if (!fields) return {};
        return {
            apiBase: fields.apiBase.value.trim(),
            apiKey: fields.apiKey.value.trim(),
            model: fields.model.value.trim(),
            temperature: fields.temperature.value,
            topP: fields.topP.value,
            maxTokens: fields.maxTokens.value,
            timeout: fields.timeout.value,
            systemPrompt: fields.systemPrompt.value.trim()
        };
    }

    appendAgentMessage(text, role = 'assistant', useMarkdown = false) {
        const container = document.getElementById('agent-messages');
        if (!container) return null;
        const item = document.createElement('div');
        item.className = `agent-message ${role}`;
        if (useMarkdown) {
            item.innerHTML = this.renderMarkdown(text);
        } else {
            item.textContent = text;
        }
        container.appendChild(item);
        container.scrollTop = container.scrollHeight;
        return item;
    }

    updateAgentMessageContent(target, text, code) {
        if (!target) return;
        target.innerHTML = this.renderMarkdown(text || '');
        if (code) {
            const pre = document.createElement('pre');
            pre.textContent = code;
            target.appendChild(pre);
        }
    }

    sendAgentMessage() {
        const input = document.getElementById('agent-input');
        if (!input) return;
        const message = input.value.trim();
        if (!message) return;
        input.value = '';
        this.appendAgentMessage(message, 'user');
        const pending = this.appendAgentMessage(this.t('agent.analyzing'), 'assistant');
        fetch('/api/agent/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                config: this.getAgentConfig()
            })
        })
        .then(async response => {
            let data = null;
            try {
                data = await response.json();
            } catch (e) {
                data = null;
            }
            if (!response.ok) {
                const message = (data && data.error) ? data.error : `HTTP ${response.status}`;
                throw new Error(message);
            }
            return data || {};
        })
        .then(data => {
            if (data.error) {
                this.updateAgentMessageContent(pending, `${this.t('common.failed')}: ${data.error}`);
                return;
            }
            this.updateAgentMessageContent(pending, data.reply || this.t('agent.done'), data.code);
            if (data.data_updated) {
                this.refreshDataFromKernel(data.data_info);
            }
        })
        .catch(error => {
            const detail = error && error.message ? error.message : this.t('common.unknownError');
            const message = detail === 'Failed to fetch'
                ? this.t('status.backendUnavailable')
                : detail;
            this.updateAgentMessageContent(pending, `${this.t('common.failed')}: ${message}`);
        });
    }

    showAgentConfig() {
        const panel = document.getElementById('agent-config-nav');
        if (panel) {
            panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

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
    }

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
    }

    updateKernelSelectorForTab(tab) {
        const kernelSelect = document.getElementById('kernel-select');
        if (!kernelSelect) return;
        if (!tab || tab.type !== 'notebook') {
            kernelSelect.disabled = true;
            return;
        }
        kernelSelect.disabled = false;
        this.loadKernelOptions(kernelSelect, tab);
    }

    triggerNotebookUpload() {
        const fileInput = document.getElementById('notebook-file-input');
        if (fileInput) fileInput.click();
    }

    fetchFileTree() {
        const tree = document.getElementById('file-tree');
        if (!tree) return;
        tree.oncontextmenu = (e) => {
            e.preventDefault();
            this.openContextMenu(e.clientX, e.clientY, this.currentBrowsePath || '', true);
        };
        tree.innerHTML = `<li class="file-tree-node">${this.t('common.loading')}</li>`;
        this.loadTreeNode('', tree);
    }

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
    }

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
    }

    setupThemeToggle() {
        // Setup click handlers for existing theme toggle buttons
        const themeToggle = document.getElementById('theme-toggle');

        if (themeToggle) {
            themeToggle.addEventListener('click', (e) => {
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
                return this.t('status.beforeLeave');
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
    }

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
                this.currentData = data;
                this.updateUI(data);
                this.updateAdataStatus(data);
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
    }

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

        // Sync left-panel height to match data-status + viz-panel
        requestAnimationFrame(() => this.syncPanelHeight());

        // Initialise point-size slider to auto default for this dataset
        this.initPointSizeSlider();

        // Update embedding options
        const embeddingSelect = document.getElementById('embedding-select');
        embeddingSelect.innerHTML = `<option value="">${this.t('controls.embeddingPlaceholder')}</option>`;
        data.embeddings.forEach(emb => {
            const option = document.createElement('option');
            option.value = emb;
            option.textContent = emb.toUpperCase();
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
    }

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
                option.textContent = emb.toUpperCase();
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
    }

    updateParameterPanel() {
        // Re-enable all parameter buttons now that data is loaded
        const buttons = document.querySelectorAll('#parameter-content button');
        buttons.forEach(button => {
            if (!button.onclick || !button.onclick.toString().includes('showComingSoon')) {
                button.disabled = false;
            }
        });
    }

    /** Called when the user explicitly picks a new obs column from color-select.
     *  Clears gene-input so gene mode doesn't interfere. */
    onColorSelectChange() {
        const geneInput = document.getElementById('gene-input');
        if (geneInput) geneInput.value = '';
        this.updatePlot();
    }

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
    }

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
    }

    applyVMinMax() {
        // Just trigger an update
        this.updatePlot();
    }

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
    }

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

    // ── Point-style helpers ──────────────────────────────────────────────────

    /** Compute default Plotly marker size from cell count (mirrors Python 120000/n). */
    computeDefaultPointSize() {
        const nCells = parseInt(document.getElementById('cell-count')?.textContent || '0', 10);
        if (!nCells || nCells <= 0) return 4;
        // Python: s = 120000 / n  (matplotlib area units)
        // Plotly size ≈ diameter in px ≈ sqrt(s)
        return Math.max(1, Math.round(Math.sqrt(120000 / nCells) * 10) / 10);
    }

    getMarkerSize() {
        const slider = document.getElementById('point-size-slider');
        if (!slider || slider.dataset.auto === 'true') {
            return this.computeDefaultPointSize();
        }
        return parseFloat(slider.value);
    }

    getMarkerOpacity() {
        const slider = document.getElementById('opacity-slider');
        return slider ? parseFloat(slider.value) : 0.7;
    }

    /** Called when the size slider is moved manually. */
    onPointSizeChange(value) {
        const slider = document.getElementById('point-size-slider');
        const label  = document.getElementById('point-size-value');
        if (slider) slider.dataset.auto = 'false';
        if (label)  label.textContent = parseFloat(value).toFixed(1);
        this.applyPointStyleLive();
    }

    /** Called when the opacity slider is moved. */
    onOpacityChange(value) {
        const label = document.getElementById('opacity-value');
        if (label) label.textContent = parseFloat(value).toFixed(2);
        this.applyPointStyleLive();
    }

    /** Reset sliders back to auto defaults and redraw. */
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
    }

    /** Initialize the size slider to the auto-computed default after data load. */
    initPointSizeSlider() {
        const slider = document.getElementById('point-size-slider');
        const label  = document.getElementById('point-size-value');
        if (!slider) return;
        const def = this.computeDefaultPointSize();
        slider.value = def;
        slider.dataset.auto = 'true';
        if (label) label.textContent = 'Auto';
    }

    /** Restyle the existing Plotly traces in-place (no data re-fetch). */
    applyPointStyleLive() {
        const plotDiv = document.getElementById('plotly-div');
        if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) return;
        const size    = this.getMarkerSize();
        const opacity = this.getMarkerOpacity();
        const traceIndices = plotDiv.data.map((_, i) => i);
        Plotly.restyle('plotly-div', { 'marker.size': size, 'marker.opacity': opacity }, traceIndices);
    }

    // ────────────────────────────────────────────────────────────────────────

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
    }

    syncPanelHeight() {
        const leftMain  = document.getElementById('left-main-panel');
        const dataStatus = document.getElementById('data-status');
        const vizPanel  = document.getElementById('viz-panel');
        if (!leftMain || !vizPanel) return;
        const dsH = (dataStatus && !dataStatus.classList.contains('d-none'))
            ? dataStatus.offsetHeight : 0;
        const vpH = vizPanel.offsetHeight;
        if (dsH + vpH > 0) {
            leftMain.style.minHeight = (dsH + vpH) + 'px';
        }
    }

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
    }

    showParameterPlaceholder() {
        const parameterContent = document.getElementById('parameter-content');
        if (!parameterContent) return;
        parameterContent.innerHTML = `
            <div class="d-flex flex-column align-items-center justify-content-center text-center text-muted py-5" style="min-height:160px">
                <i class="fas fa-hand-pointer fa-2x mb-3 opacity-50"></i>
                <p class="mb-0 small" data-i18n="panel.selectAnalysis">${this.t('panel.selectAnalysis')}</p>
            </div>`;
    }

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
            ]
        };
        
        const tools = categoryTools[category] || [];
        
        tools.forEach(tool => {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'mb-3 p-3 border rounded fade-in c-pointer';
            toolDiv.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <i class="${tool.icon} me-2 text-primary"></i>
                    <strong>${this.t(tool.nameKey)}</strong>
                </div>
                <p class="text-muted small mb-0">${this.t(tool.descKey)}</p>`;
            if (tool.id === 'coming_soon') {
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

        if (!silent) this.addToLog(this.t('panel.categorySelected') + ` ${this.getCategoryName(category)}`);
    }

    getCategoryName(category) {
        const names = {
            'preprocessing': this.t('nav.preprocessing'),
            'dimreduction': this.t('nav.dimReductionSub'),
            'clustering': this.t('nav.clusteringSub'),
            'omicverse': this.t('nav.omicverse'),
            'cell_annotation': this.t('nav.cellAnnotation'),
            'trajectory': this.t('nav.trajectory')
        };
        return names[category] || category;
    }

    showParameterDialog(tool) { this.renderParameterForm(tool); }

    /** Save / restore parameter values across form re-renders. */
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
    }

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
                    <button class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.selectAnalysisCategory('${singleCellApp.currentCategory || 'preprocessing'}')">返回工具列表</button>
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
    }

    refreshParameterFormLanguage() {
        const parameterContent = document.getElementById('parameter-content');
        if (!parameterContent || !this.currentTool) return;
        const inputs = parameterContent.querySelectorAll('input, select');
        const values = {};
        inputs.forEach(input => {
            if (input.type === 'checkbox') {
                values[input.id] = input.checked;
            } else {
                values[input.id] = input.value;
            }
        });
        const nameKeyOrLabel = this.currentToolLabelKey || this.currentToolLabel;
        const descKeyOrLabel = this.currentToolDescKey || this.currentToolDesc;
        this.renderParameterForm(this.currentTool, nameKeyOrLabel, descKeyOrLabel);
        Object.entries(values).forEach(([id, value]) => {
            const target = parameterContent.querySelector(`#${CSS.escape(id)}`);
            if (!target) return;
            if (target.type === 'checkbox') {
                target.checked = !!value;
            } else {
                target.value = value;
            }
        });
    }

    // ── Annotation Tools ──────────────────────────────────────────────────────

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
                <button class="btn btn-sm btn-outline-secondary"
                    onclick="singleCellApp.selectAnalysisCategory('${backCat}')">返回工具列表</button>
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
    }

    updateCodeCellPlaceholders() {
        const cells = document.querySelectorAll('.code-cell');
        cells.forEach(cell => {
            const type = cell.dataset.cellType || 'code';
            const textarea = cell.querySelector('.code-input');
            if (!textarea) return;
            if (type === 'markdown') {
                textarea.placeholder = this.t('cell.placeholderMarkdown');
            } else if (type === 'raw') {
                textarea.placeholder = this.t('cell.placeholderRaw');
            } else {
                textarea.placeholder = this.t('code.placeholder');
            }
        });
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
    }

    formatToolMessage(toolName, suffix, error) {
        const spacer = this.currentLang === 'zh' ? '' : ' ';
        const base = `${toolName}${spacer}${suffix}`;
        return error ? `${base}: ${error}` : base;
    }

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
    }

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
    }

    // ── Trajectory Visualization ───────────────────────────────────────────

    showTrajViz() {
        const panel = document.getElementById('traj-viz-panel');
        if (panel) {
            panel.style.display = '';
            this.updateTrajVizSelects();
        }
    }

    toggleTrajViz() {
        const body = document.getElementById('traj-viz-body');
        const icon = document.getElementById('traj-viz-toggle-icon');
        if (!body) return;
        const collapsed = body.style.display === 'none';
        body.style.display = collapsed ? '' : 'none';
        if (icon) {
            icon.className = collapsed ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
        }
    }

    togglePagaOptions(checked) {
        const opts = document.getElementById('traj-paga-options');
        if (opts) opts.style.display = checked ? '' : 'none';
    }

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
    }

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
    }

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
    }

    // ── end Trajectory Visualization ──────────────────────────────────────

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
    }

    resetData() {
        if (confirm(this.t('status.resetConfirm'))) {
            this.currentData = null;
            document.getElementById('upload-section').style.display = 'block';
            document.getElementById('data-status').classList.add('d-none');
            document.getElementById('viz-controls').style.display = 'none';
            document.getElementById('viz-panel').style.display = 'none';
            document.getElementById('analysis-log').innerHTML = `<div class="text-muted">${this.t('panel.waitingUpload')}</div>`;
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
        .then(r => r.json())
        .then(data => {
            if (!data.loaded) return;
            // Guard: ensure the response has the fields updateUI needs
            if (!Array.isArray(data.embeddings)) {
                console.warn('checkStatus: /api/status response missing embeddings field', data);
                return;
            }
            // Server has adata in memory — restore full UI without re-uploading
            this.currentData = data;
            this.updateUI(data);
            this.updateAdataStatus(data);
            requestAnimationFrame(() => this.syncPanelHeight());
            this.addToLog(
                `${this.t('upload.successDetail')}: ${data.n_cells} ${this.t('status.cells')}, ${data.n_genes} ${this.t('status.genes')}`
            );
        })
        .catch(err => { console.warn('checkStatus error:', err); });
    }

    showLoading(text = null) {
        const loadingText = document.getElementById('loading-text');
        const loadingOverlay = document.getElementById('loading-overlay');

        const resolved = text || this.t('loading.processing');
        if (loadingText) loadingText.textContent = resolved;
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
        alert(this.t('common.comingSoon'));
    }

    // View switching
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
    }

    // Code cell management
    addCodeCell(code = '', outputs = [], cellType = 'code') {
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
                        <button type="button" class="btn btn-sm btn-success" onclick="singleCellApp.runCodeCell('${cellId}')" title="Run (Shift+Enter)" data-i18n-title="cell.run">
                            <i class="feather-play"></i> <span data-i18n="cell.run">Run</span>
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.toggleCellOutput('${cellId}')" title="Toggle output" data-i18n-title="cell.toggleOutput">
                            <span data-i18n="cell.toggleOutput">Toggle output</span>
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.toggleCellOutputFull('${cellId}')" title="Hide output" data-i18n-title="cell.hideOutput">
                            <span data-i18n="cell.hideOutput">Hide output</span>
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary" onclick="singleCellApp.clearCellOutput('${cellId}')" title="Clear output" data-i18n-title="cell.clearOutput">
                            <span data-i18n="cell.clearOutput">Clear output</span>
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
        container.insertAdjacentHTML('beforeend', cellHtml);
        this.applyLanguage(this.currentLang);

        this.codeCells.push(cellId);
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
    }

    // Update cell execution number display
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
    }

    runCodeCell(cellId) {
        return this.runCodeCellPromise(cellId);
    }

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
                kernel_id: kernelId
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

                                // Add result if present (like Jupyter's auto-display of last expression)
                                if (data.result !== null && data.result !== undefined) {
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
                                if (!finalOutput && figures.length === 0) {
                                    finalOutput = this.t('status.noOutput');
                                }

                                this.renderCodeOutput(outputDiv, {
                                    text: finalOutput,
                                    isError: false,
                                    figures: figures
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
    }

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
    }

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
    }

    refreshKernelInfo() {
        // Refresh both variable viewer and kernel stats
        this.fetchKernelVars();
        this.fetchKernelStats();
    }

    showInterruptButton() {
        this.isExecuting = true;
        const btn = document.getElementById('interrupt-btn');
        if (btn) {
            btn.style.display = 'inline-block';
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-stop"></i> <span data-i18n="toolbar.interrupt">Interrupt</span>';
        }
    }

    hideInterruptButton() {
        this.isExecuting = false;
        const btn = document.getElementById('interrupt-btn');
        if (btn) {
            btn.style.display = 'none';
        }
    }

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
    }

    stopExecutionStatusPolling() {
        if (this.executionStatusPollInterval) {
            clearInterval(this.executionStatusPollInterval);
            this.executionStatusPollInterval = null;
        }
    }

    // Run the currently focused cell (or first cell if none focused)
    runCurrentCell() {
        if (this.codeCells.length === 0) {
            return;
        }

        // Run the last focused cell, or the first cell if no cell was focused
        const cellToRun = this.lastFocusedCellId || this.codeCells[0];
        if (cellToRun && document.getElementById(cellToRun)) {
            this.runCodeCell(cellToRun);
        }
    }

    // Run all cells sequentially
    runAllCells() {
        if (this.codeCells.length === 0) {
            return;
        }
        let chain = Promise.resolve();
        this.codeCells.forEach(cellId => {
            chain = chain.then(() => this.runCodeCellPromise(cellId));
        });
    }

    toggleCellOutput(cellId) {
        const output = document.getElementById(`${cellId}-output`);
        const hiddenNote = document.getElementById(`${cellId}-output-hidden`);
        if (!output) return;
        output.classList.remove('collapsed');
        output.classList.toggle('partial');
        if (hiddenNote) {
            hiddenNote.classList.remove('visible');
        }
    }

    toggleCellOutputFull(cellId) {
        const output = document.getElementById(`${cellId}-output`);
        const hiddenNote = document.getElementById(`${cellId}-output-hidden`);
        if (!output) return;
        output.classList.remove('partial');
        output.classList.toggle('collapsed');
        if (hiddenNote) {
            hiddenNote.classList.toggle('visible', output.classList.contains('collapsed'));
        }
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

    getActiveTab() {
        return this.openTabs.find(t => t.id === this.activeTabId);
    }

    getActiveKernelId() {
        const tab = this.getActiveTab();
        if (tab && tab.type === 'notebook') {
            return tab.kernelId || tab.path || 'default.ipynb';
        }
        if (tab && tab.type === 'var') {
            return tab.kernelId || 'default.ipynb';
        }
        return null;
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

    toggleSection(sectionId) {
        const section = document.getElementById(sectionId);
        const toggle = document.getElementById(`${sectionId}-toggle`);
        if (!section || !toggle) return;
        const isHidden = section.style.display === 'none';
        section.style.display = isHidden ? 'block' : 'none';
        toggle.textContent = isHidden ? this.t('common.collapse') : this.t('common.expand');
    }

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
    }

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
    }

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
    }

    hideContextMenu() {
        const menu = document.getElementById('file-context-menu');
        if (menu) menu.style.display = 'none';
    }

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
    }

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
    }

    contextRefresh() {
        this.fetchFileTree();
    }

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
    }

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
    }

    contextCopy() {
        if (!this.contextTargetPath) return;
        this.contextClipboard = { path: this.contextTargetPath, mode: 'copy' };
    }

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
    }

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
    }

    renderMarkdown(input) {
        const escapeHtml = (text) => text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        const formatInline = (text) => {
            let out = escapeHtml(text);
            out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
            out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            out = out.replace(/\*(.+?)\*/g, '<em>$1</em>');
            return out;
        };

        const lines = input.split('\n');
        let html = '';
        let para = [];
        let list = [];
        let inCode = false;
        let codeLines = [];

        const flushPara = () => {
            if (!para.length) return;
            const text = para.join(' ').trim();
            if (text) {
                html += `<p>${formatInline(text)}</p>`;
            }
            para = [];
        };

        const flushList = () => {
            if (!list.length) return;
            html += '<ul>' + list.map(item => `<li>${formatInline(item)}</li>`).join('') + '</ul>';
            list = [];
        };

        lines.forEach(line => {
            if (line.startsWith('```')) {
                if (inCode) {
                    html += `<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`;
                    codeLines = [];
                    inCode = false;
                } else {
                    flushPara();
                    flushList();
                    inCode = true;
                }
                return;
            }

            if (inCode) {
                codeLines.push(line);
                return;
            }

            if (!line.trim()) {
                flushPara();
                flushList();
                return;
            }

            if (line.startsWith('# ')) {
                flushPara();
                flushList();
                html += `<h1>${formatInline(line.slice(2))}</h1>`;
                return;
            }
            if (line.startsWith('## ')) {
                flushPara();
                flushList();
                html += `<h2>${formatInline(line.slice(3))}</h2>`;
                return;
            }
            if (line.startsWith('### ')) {
                flushPara();
                flushList();
                html += `<h3>${formatInline(line.slice(4))}</h3>`;
                return;
            }
            if (line.startsWith('> ')) {
                flushPara();
                flushList();
                html += `<blockquote>${formatInline(line.slice(2))}</blockquote>`;
                return;
            }
            if (line.startsWith('- ')) {
                list.push(line.slice(2));
                return;
            }

            para.push(line);
        });

        if (inCode) {
            html += `<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`;
        }
        flushPara();
        flushList();
        return html;
    }

    getTextEditorContent() {
        const active = this.getActiveTab();
        if (active && active.type === 'markdown') {
            const mdEditor = document.getElementById('md-file-editor');
            return mdEditor ? mdEditor.value : '';
        }
        const textEditor = document.getElementById('text-file-editor');
        return textEditor ? textEditor.value : '';
    }

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
    }

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
        });
    }

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
    }

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
    }

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
        }

        if (!payload) {
            alert(this.t('file.noSaveContent'));
            return;
        }

        fetch('/api/files/save', {
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
    }


    renderCodeOutput(outputDiv, payload) {
        outputDiv.className = `code-cell-output has-content ${payload.isError ? 'error' : 'success'}`;
        outputDiv.classList.remove('markdown');
        outputDiv.innerHTML = '';
        if (payload.text) {
            const pre = document.createElement('pre');
            pre.className = 'code-output-text';
            pre.innerHTML = this.ansiToHtml(payload.text);
            outputDiv.appendChild(pre);
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
    }

    storeCellOutputPayload(outputDiv, payload) {
        if (!outputDiv || !payload) return;
        const cellId = outputDiv.id.replace('-output', '');
        const cell = document.getElementById(cellId);
        if (!cell) return;
        const outputPayload = {
            text: payload.text || '',
            figures: payload.figures || [],
            isError: !!payload.isError
        };
        cell.dataset.outputPayload = JSON.stringify(outputPayload);
    }

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
    }

    updateCodeHighlight(textarea, highlight) {
        if (!highlight || !textarea) return;
        const code = textarea.value || '';
        if (window.Prism && Prism.languages && Prism.languages.python) {
            highlight.innerHTML = Prism.highlight(code, Prism.languages.python, 'python') + '\n';
        } else {
            highlight.textContent = code;
        }
    }

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
    }

    adjustFontSize(delta) {
        const next = Math.min(20, Math.max(10, this.codeFontSize + delta));
        this.codeFontSize = next;
        this.applyCodeFontSize();
    }

    changeCellType(cellId, type) {
        this.setCellType(cellId, type);
    }

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
        } else {
            const markdownRender = document.getElementById(`${cellId}-markdown`);
            if (markdownRender) markdownRender.style.display = 'none';
            if (textarea) textarea.style.display = 'block';
        }
        if (type === 'markdown' && textarea) {
            this.resizeMarkdownEditor(textarea);
        }
    }

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
    }

    resizeMarkdownEditor(textarea) {
        if (!textarea) return;
        textarea.style.height = 'auto';
        textarea.style.height = Math.max(120, textarea.scrollHeight) + 'px';
    }

    deleteCodeCell(cellId) {
        if (confirm(this.t('cell.deleteConfirm'))) {
            const cell = document.getElementById(cellId);
            cell.remove();
            this.codeCells = this.codeCells.filter(id => id !== cellId);
        }
    }

    clearAllCells() {
        if (confirm(this.t('cell.clearConfirm'))) {
            const container = document.getElementById('code-cells-container');
            container.innerHTML = '';
            this.codeCells = [];
            this.cellCounter = 0;
            // Add one empty cell
            this.addCodeCell();
        }
    }

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

    // ── Python Environment Manager ─────────────────────────────────────────

    _envDefaultChannels = [
        { name: 'conda-forge', checked: true },
        { name: 'bioconda',    checked: true },
        { name: 'defaults',    checked: false },
        { name: 'pytorch',     checked: false },
        { name: 'nvidia',      checked: false },
    ];

    showEnvManager() {
        const modal = document.getElementById('env-manager-modal');
        if (!modal) return;
        modal.style.display = '';
        this._envInitChannels();
        this._envUpdatePipPreview();
        this._envUpdateCondaPreview();
        this.applyLanguage(this.currentLang);
    }

    hideEnvManager() {
        const modal = document.getElementById('env-manager-modal');
        if (modal) modal.style.display = 'none';
    }

    switchEnvTab(tab) {
        document.getElementById('env-panel-pip').style.display   = tab === 'pip'   ? '' : 'none';
        document.getElementById('env-panel-conda').style.display = tab === 'conda' ? '' : 'none';
        document.getElementById('env-tab-pip').classList.toggle('active',   tab === 'pip');
        document.getElementById('env-tab-conda').classList.toggle('active', tab === 'conda');
    }

    _envInitChannels() {
        const list = document.getElementById('env-channel-list');
        if (!list || list.dataset.initialized) return;
        list.dataset.initialized = '1';
        this._envDefaultChannels.forEach(ch => {
            const id = `ch-${ch.name}`;
            const wrap = document.createElement('div');
            wrap.className = 'form-check form-check-inline';
            wrap.innerHTML = `<input class="form-check-input" type="checkbox" id="${id}"
                                     value="${ch.name}" ${ch.checked ? 'checked' : ''}
                                     onchange="singleCellApp._envUpdateCondaPreview()">
                              <label class="form-check-label small" for="${id}">${ch.name}</label>`;
            list.appendChild(wrap);
        });
    }

    envAddChannel() {
        const inp = document.getElementById('env-custom-channel');
        const name = inp ? inp.value.trim() : '';
        if (!name) return;
        const list = document.getElementById('env-channel-list');
        const id = `ch-${name}`;
        if (document.getElementById(id)) { inp.value = ''; return; }
        const wrap = document.createElement('div');
        wrap.className = 'form-check form-check-inline';
        wrap.innerHTML = `<input class="form-check-input" type="checkbox" id="${id}"
                                 value="${name}" checked
                                 onchange="singleCellApp._envUpdateCondaPreview()">
                          <label class="form-check-label small" for="${id}">${name}</label>`;
        list.appendChild(wrap);
        inp.value = '';
        this._envUpdateCondaPreview();
    }

    _envGetPkg() {
        return (document.getElementById('env-pkg-input') || {}).value?.trim() || '<package>';
    }

    _envGetChannels() {
        return [...document.querySelectorAll('#env-channel-list input:checked')].map(i => i.value);
    }

    _envUpdatePipPreview() {
        const pkg    = this._envGetPkg();
        const mirror = (document.getElementById('env-mirror-select') || {}).value || '';
        const extra  = (document.getElementById('env-pip-extra') || {}).value?.trim() || '';
        let cmd = `uv pip install ${pkg}`;
        if (mirror) cmd += ` --index-url ${mirror}`;
        if (extra)  cmd += ` ${extra}`;
        const el = document.getElementById('env-pip-preview');
        if (el) el.textContent = cmd;
    }

    _envUpdateCondaPreview() {
        const pkg      = this._envGetPkg();
        const channels = this._envGetChannels();
        const extra    = (document.getElementById('env-conda-extra') || {}).value?.trim() || '';
        const chStr    = channels.map(c => `-c ${c}`).join(' ');
        let cmd = `mamba install -y ${chStr} ${pkg}`;
        if (extra) cmd += ` ${extra}`;
        const el = document.getElementById('env-conda-preview');
        if (el) el.textContent = cmd;
    }

    _envLog(text, type = 'output') {
        const con = document.getElementById('env-console');
        if (!con) return;
        const color = type === 'error' ? '#f38ba8' : type === 'cmd' ? '#89b4fa' : type === 'ok' ? '#a6e3a1' : '#cdd6f4';
        con.innerHTML += `<span style="color:${color}">${this._escHtml(text)}</span>`;
        con.scrollTop = con.scrollHeight;
    }

    _escHtml(s) {
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    async envSearch() {
        const pkg = this._envGetPkg();
        if (pkg === '<package>' || !pkg) return;
        const infoEl    = document.getElementById('env-pkg-info');
        const notFound  = document.getElementById('env-pkg-notfound');
        infoEl.style.display = 'none';
        notFound.style.display = 'none';
        this._envUpdatePipPreview();
        this._envUpdateCondaPreview();
        try {
            const r = await fetch(`/api/env/search_pypi?package=${encodeURIComponent(pkg)}`);
            const d = await r.json();
            if (d.found) {
                document.getElementById('env-pkg-name').textContent    = d.name;
                document.getElementById('env-pkg-version').textContent = d.version || '';
                document.getElementById('env-pkg-pyver').textContent   = d.requires_python ? `Python ${d.requires_python}` : '';
                document.getElementById('env-pkg-summary').textContent = d.summary || '';
                document.getElementById('env-pkg-found-badge').innerHTML =
                    `<span class="badge text-bg-success">${this.t('env.foundOnPypi')}</span>`;
                infoEl.style.display = '';
            } else {
                notFound.style.display = '';
            }
        } catch (e) {
            this._envLog('Search error: ' + e.message, 'error');
        }
    }

    async envTestMirrors() {
        const sel = document.getElementById('env-mirror-select');
        this._envLog(this.t('env.testingMirrors') + '\n', 'cmd');
        try {
            const r = await fetch('/api/env/test_mirrors');
            const d = await r.json();
            sel.innerHTML = '';
            (d.mirrors || []).forEach((m, i) => {
                const opt = document.createElement('option');
                opt.value = m.url;
                const tag = m.ok ? `${m.latency_ms}ms` : 'timeout';
                opt.textContent = `${m.name} — ${tag}`;
                if (i === 0 && m.ok) opt.selected = true;
                sel.appendChild(opt);
                this._envLog(`  ${m.name}: ${tag}\n`, m.ok ? 'output' : 'error');
            });
            this._envUpdatePipPreview();
        } catch (e) {
            this._envLog('Mirror test error: ' + e.message, 'error');
        }
    }

    envInstallPip() {
        const pkg   = this._envGetPkg();
        const mirror = (document.getElementById('env-mirror-select') || {}).value || '';
        const extra  = (document.getElementById('env-pip-extra') || {}).value?.trim() || '';
        if (!pkg || pkg === '<package>') { alert(this.t('env.enterPkg')); return; }

        const con = document.getElementById('env-console');
        if (con) con.textContent = '';
        const es = new EventSource(`/api/env/install_pip?_dummy=${Date.now()}`);
        // Use POST via fetch + ReadableStream instead
        es.close();
        this._envStreamInstall('/api/env/install_pip', { package: pkg, mirror, extra_args: extra });
    }

    envInstallConda() {
        const pkg      = this._envGetPkg();
        const channels = this._envGetChannels();
        const extra    = (document.getElementById('env-conda-extra') || {}).value?.trim() || '';
        if (!pkg || pkg === '<package>') { alert(this.t('env.enterPkg')); return; }

        const con = document.getElementById('env-console');
        if (con) con.textContent = '';
        this._envStreamInstall('/api/env/install_conda', { package: pkg, channels, extra_args: extra });
    }

    async _envStreamInstall(url, body) {
        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const evt = JSON.parse(line.slice(6));
                        if (evt.type === 'cmd')      this._envLog('$ ' + evt.text + '\n', 'cmd');
                        else if (evt.type === 'output') this._envLog(evt.text);
                        else if (evt.type === 'error')  this._envLog(evt.text + '\n', 'error');
                        else if (evt.type === 'complete') {
                            const msg = evt.success
                                ? this.t('env.installSuccess')
                                : `${this.t('env.installFailed')} (code ${evt.returncode})`;
                            this._envLog('\n' + msg + '\n', evt.success ? 'ok' : 'error');
                        }
                    } catch {}
                }
            }
        } catch (e) {
            this._envLog('Stream error: ' + e.message + '\n', 'error');
        }
    }

    // ── Python Environment Info ────────────────────────────────────────────

    _envAllPkgsData = [];   // cache for filter

    showEnvInfo() {
        const modal = document.getElementById('env-info-modal');
        if (!modal) return;
        modal.style.display = '';
        this.applyLanguage(this.currentLang);
        this.loadEnvInfo();
    }

    hideEnvInfo() {
        const modal = document.getElementById('env-info-modal');
        if (modal) modal.style.display = 'none';
    }

    async loadEnvInfo() {
        const loading = document.getElementById('env-info-loading');
        const content = document.getElementById('env-info-content');
        const errEl   = document.getElementById('env-info-error');
        loading.style.display = '';
        content.style.display = 'none';
        errEl.style.display   = 'none';

        try {
            const r = await fetch('/api/env/info');
            const d = await r.json();

            this._renderEnvSystem(d.system || {});
            this._renderEnvPython(d.python || {});
            this._renderEnvGpu(d.gpu || {});
            this._renderEnvKeyPkgs(d.key_pkgs || []);
            this._renderEnvAllPkgs(d.all_pkgs || []);

            loading.style.display = 'none';
            content.style.display = '';
        } catch (e) {
            loading.style.display = 'none';
            errEl.textContent = 'Failed to load environment info: ' + e.message;
            errEl.style.display = '';
        }
    }

    _kv(label, value) {
        if (value === null || value === undefined || value === '') return '';
        return `<div class="d-flex gap-2 mb-1">
            <span class="text-muted small" style="min-width:90px;flex-shrink:0">${this._escHtml(label)}</span>
            <span class="small fw-medium" style="word-break:break-all">${this._escHtml(String(value))}</span>
        </div>`;
    }

    _renderEnvSystem(s) {
        const el = document.getElementById('env-info-system');
        if (!el) return;
        el.innerHTML =
            this._kv('OS', `${s.os} ${s.machine}`) +
            this._kv('Version', s.os_ver) +
            this._kv('Host', s.hostname) +
            (s.cpu_count   ? this._kv('CPU cores', s.cpu_count) : '') +
            (s.ram_total_gb ? this._kv('RAM', `${s.ram_used_gb} / ${s.ram_total_gb} GB`) : '');
    }

    _renderEnvPython(p) {
        const el = document.getElementById('env-info-python');
        if (!el) return;
        const ver = p.version ? p.version.split(' ')[0] : '';
        el.innerHTML =
            this._kv('Python', ver) +
            this._kv('Executable', p.executable) +
            this._kv('Prefix', p.prefix) +
            (p.uv     ? this._kv('uv',     p.uv)     : '') +
            (p.mamba  ? this._kv('mamba',  p.mamba)  : '') +
            (p.conda  ? this._kv('conda',  p.conda)  : '') +
            (p.pip    ? this._kv('pip',    p.pip)     : '');
    }

    _renderEnvGpu(g) {
        const el = document.getElementById('env-info-gpu');
        if (!el) return;
        let html = g.torch_version ? this._kv('PyTorch', g.torch_version) : '';
        if (g.cuda_available) {
            html += this._kv('CUDA', g.cuda_version);
            (g.devices || []).forEach(dev => {
                html += this._kv(`GPU ${dev.index}`, `${dev.name}${dev.mem_gb ? ' — ' + dev.mem_gb + ' GB' : ''}`);
            });
        } else if (g.mps_available) {
            html += `<span class="badge text-bg-success">Apple MPS</span>`;
        } else {
            html += `<span class="text-muted small">${this.t('envInfo.noGpu')}</span>`;
        }
        el.innerHTML = html || `<span class="text-muted small">${this.t('envInfo.noGpu')}</span>`;
    }

    _renderEnvKeyPkgs(pkgs) {
        const tbody = document.getElementById('env-info-pkgs');
        if (!tbody) return;
        tbody.innerHTML = pkgs.map(p => {
            const badge = p.installed
                ? `<span class="badge text-bg-success">✓ ${this._escHtml(p.version)}</span>`
                : `<span class="badge text-bg-secondary">${this.t('envInfo.notInstalled')}</span>`;
            return `<tr>
                <td class="ps-3 fw-medium">${this._escHtml(p.name)}</td>
                <td>${p.installed ? this._escHtml(p.version) : '—'}</td>
                <td>${badge}</td>
            </tr>`;
        }).join('');
    }

    _renderEnvAllPkgs(pkgs) {
        this._envAllPkgsData = pkgs;
        const count = document.getElementById('env-info-pkg-count');
        if (count) count.textContent = pkgs.length;
        this._renderEnvAllPkgsTable(pkgs);
    }

    _renderEnvAllPkgsTable(pkgs) {
        const tbody = document.getElementById('env-allpkgs-table');
        if (!tbody) return;
        tbody.innerHTML = pkgs.map(p =>
            `<tr><td class="ps-3">${this._escHtml(p.name)}</td><td class="text-muted">${this._escHtml(p.version)}</td></tr>`
        ).join('');
    }

    filterEnvPkgs(query) {
        const q = query.toLowerCase();
        const filtered = q
            ? this._envAllPkgsData.filter(p => p.name.toLowerCase().includes(q))
            : this._envAllPkgsData;
        this._renderEnvAllPkgsTable(filtered);
    }

    toggleEnvAllPkgs() {
        const body = document.getElementById('env-allpkgs-body');
        const icon = document.getElementById('env-allpkgs-icon');
        if (!body) return;
        const collapsed = body.style.display === 'none';
        body.style.display = collapsed ? '' : 'none';
        if (icon) icon.className = collapsed ? 'feather-chevron-up' : 'feather-chevron-down';
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
    const toggleIcon = document.getElementById('theme-toggle-icon');
    
    if (savedTheme === 'app-skin-dark') {
        html.classList.add('app-skin-dark');
        if (toggleIcon) {
            toggleIcon.classList.remove('feather-moon');
            toggleIcon.classList.add('feather-sun');
        }
        singleCellApp.currentTheme = 'dark';
    } else {
        html.classList.remove('app-skin-dark');
        if (toggleIcon) {
            toggleIcon.classList.remove('feather-sun');
            toggleIcon.classList.add('feather-moon');
        }
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
