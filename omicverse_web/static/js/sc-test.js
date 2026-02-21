/**
 * OmicVerse Single Cell Analysis — Module Integrity Test
 *
 * Paste this into the browser console (or load it after all sc-*.js files)
 * to verify every method from the original single-cell.js is present.
 */
(function runModuleTests() {
    'use strict';

    // Complete list of methods that must exist on SingleCellAnalysis.prototype
    // (constructor & init are on the class itself, not prototype)
    const REQUIRED_METHODS = [
        // sc-i18n
        'setupLanguageToggle','t','applyLanguage','translateFormHtml',
        'refreshParameterFormLanguage','updateCodeCellPlaceholders',
        // sc-ui
        'setupNavigation','setupSidebarResize','setupGeneAutocomplete',
        'setupBeforeUnloadWarning','setupThemeToggle','setupNotebookManager',
        'toggleSubmenu','toggleMobileMenu','toggleTheme',
        'updatePlotlyTheme','updateStatusBarTheme',
        'showLoading','hideLoading','addToLog',
        'showStatus','hideStatus','updateStatus',
        'switchView','showComingSoon',
        'updateUI','refreshDataFromKernel','updateParameterPanel','checkStatus',
        'applyCodeFontSize','adjustFontSize','syncPanelHeight',
        // sc-agent
        'setupAgentConfig','setupAgentChat','getAgentConfigFields',
        'loadAgentConfig','saveAgentConfig','resetAgentConfig','getAgentConfig',
        'appendAgentMessage','updateAgentMessageContent','sendAgentMessage','showAgentConfig',
        // sc-kernel
        'setupKernelSelector','loadKernelOptions','changeKernel','updateKernelSelectorForTab',
        'interruptExecution','restartKernel','refreshKernelInfo',
        'showInterruptButton','hideInterruptButton',
        'startExecutionStatusPolling','stopExecutionStatusPolling',
        'fetchKernelVars','fetchKernelStats','loadAnndataToVisualization',
        'openVarTab','showVarDetail','toggleSection',
        // sc-filesystem
        'setupFileUpload','handleFileUpload','triggerNotebookUpload',
        'fetchFileTree','loadTreeNode','renderFileTree',
        'openContextMenu','hideContextMenu',
        'contextNewFile','contextNewFolder','contextRefresh',
        'contextDelete','contextRename','contextCopy','contextPaste','contextMove',
        'openFileFromServer',
        'openNotebookTab','openTextTab','openMarkdownTab','openImageTab',
        'setActiveTab','getActiveTab','getActiveKernelId',
        'renderTabs','closeTab',
        'showTextFile','showMarkdownFile','showImageFile','persistActiveTab',
        // sc-plot
        'onColorSelectChange','updatePlot','updatePaletteVisibility','applyVMinMax',
        'createNewPlot','updatePlotWithAnimation','animatePlotTransition',
        'checkCoordsChanged','updateColorsOnly',
        'animatePositionTransition','animatePositionTransitionForAnyData',
        'easeInOutCubic','smoothAnimate',
        'computeDefaultPointSize','getMarkerSize','getMarkerOpacity',
        'onPointSizeChange','onOpacityChange','resetPointStyle',
        'initPointSizeSlider','applyPointStyleLive',
        'plotData','getPlotlyLayout','colorByGene',
        'updateAdataStatus','showParameterPlaceholder','getCategoricalColorscale',
        'testLegend','forceShowLegend','simpleLegendTest','showLegendNow',
        // sc-tools
        'selectAnalysisCategory','getCategoryName',
        'showParameterDialog','_restoreAndTrackParams',
        'renderParameterForm','renderAnnotationForm','getParameterHTML',
        'formatToolMessage','runTool','showToolFigures',
        'showTrajViz','toggleTrajViz','togglePagaOptions',
        'updateTrajVizSelects','generateTrajEmbedding','generateTrajHeatmap',
        'saveData','resetData',
        // sc-notebook
        'addCodeCell','updateCellNumber',
        'runCodeCell','runCodeCellPromise','runCurrentCell','runAllCells',
        'toggleCellOutput','toggleCellOutputFull','clearCellOutput',
        'importNotebookFile','loadNotebookCells','renderNotebookOutputs',
        'captureNotebookOutputs','restoreNotebookOutputs',
        'buildNotebookCellsFromUI','getNotebookOutputsFromCell',
        'saveActiveFile','renderCodeOutput','storeCellOutputPayload',
        'ansiToHtml','updateCodeHighlight',
        'changeCellType','setCellType','renderMarkdownCell','resizeMarkdownEditor',
        'deleteCodeCell','clearAllCells','insertTemplate',
        'renderMarkdown','getTextEditorContent',
        // sc-environment
        'showEnvManager','hideEnvManager','switchEnvTab',
        '_envInitChannels','envAddChannel','_envGetPkg','_envGetChannels',
        '_envUpdatePipPreview','_envUpdateCondaPreview','_envLog','_escHtml',
        'envInstallPip','envInstallConda','_envStreamInstall',
        'envSearch','envTestMirrors',
        'showEnvInfo','hideEnvInfo','loadEnvInfo',
        '_kv','_renderEnvSystem','_renderEnvPython','_renderEnvGpu',
        '_renderEnvKeyPkgs','_renderEnvAllPkgs','_renderEnvAllPkgsTable',
        'filterEnvPkgs','toggleEnvAllPkgs',
    ];

    const proto = SingleCellAnalysis.prototype;
    const results = { pass: [], fail: [] };

    REQUIRED_METHODS.forEach(name => {
        if (typeof proto[name] === 'function') {
            results.pass.push(name);
        } else {
            results.fail.push(name);
        }
    });

    // Check constructor-level properties via instantiation (light check, no DOM needed)
    const REQUIRED_INSTANCE_FIELDS = [
        'currentData','currentTool','currentTheme','currentLang',
        'paramCache','codeCells','cellCounter','openTabs','activeTabId',
        'isExecuting','lastFocusedCellId','_envDefaultChannels',
    ];

    console.group('OmicVerse Module Integrity Test');
    console.log(`Checking ${REQUIRED_METHODS.length} prototype methods...`);

    if (results.fail.length === 0) {
        console.log(`%c✅ ALL ${results.pass.length} methods present`, 'color:green;font-weight:bold');
    } else {
        console.error(`%c❌ ${results.fail.length} MISSING methods:`, 'color:red;font-weight:bold');
        results.fail.forEach(name => console.error(`  • ${name}`));
        console.log(`%c✅ ${results.pass.length} methods OK`, 'color:green');
    }

    // Check global wrappers
    const GLOBAL_FNS = [
        'selectAnalysisCategory','updatePlot','colorByGene',
        'showParameterDialog','runTool','saveData','resetData','showComingSoon',
    ];
    const missingGlobals = GLOBAL_FNS.filter(fn => typeof window[fn] !== 'function');
    if (missingGlobals.length === 0) {
        console.log('%c✅ All global wrapper functions present', 'color:green');
    } else {
        console.error('%c❌ Missing global functions:', 'color:red', missingGlobals);
    }

    // Summary
    const totalFail = results.fail.length + missingGlobals.length;
    console.log(`\nResult: ${totalFail === 0 ? '✅ PASS' : '❌ FAIL'} (${results.fail.length} missing methods, ${missingGlobals.length} missing globals)`);
    console.groupEnd();

    return { pass: results.pass, fail: results.fail, missingGlobals };
})();
