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


        this._envDefaultChannels = [
            { name: 'conda-forge', checked: true },
            { name: 'bioconda',    checked: true },
            { name: 'defaults',    checked: false },
            { name: 'pytorch',     checked: false },
            { name: 'nvidia',      checked: false },
        ];

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
