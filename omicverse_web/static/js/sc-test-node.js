/**
 * Module integrity test — verifies all methods from original single-cell.js
 * are present in the split module files.
 */
const fs   = require('fs');
const path = require('path');

const BASE = '/Users/fernandozeng/Desktop/analysis/omicverse/omicverse_web/static/js';

// Browser stubs so the class body can be evaluated without errors
const PREAMBLE = `
  class DataManager {}
  const Plotly = { newPlot(){}, react(){}, restyle(){}, relayout(){}, animate(){} };
  const document = new Proxy({}, {
    get(t, k) {
      if (k === 'addEventListener') return ()=>{};
      if (k === 'getElementById' || k === 'querySelector' || k === 'createElement') return ()=>el();
      if (k === 'querySelectorAll') return ()=>[];
      return el();
    }
  });
  function el() {
    return new Proxy({}, {
      get(t, k) {
        if (k === 'classList') return {add:()=>{},remove:()=>{},contains:()=>false,toggle:()=>{}};
        if (k === 'style') return {};
        if (k === 'dataset') return {};
        if (k === 'addEventListener') return ()=>{};
        if (k === 'querySelector') return ()=>null;
        if (k === 'querySelectorAll') return ()=>[];
        if (k === 'appendChild' || k === 'removeChild') return ()=>{};
        if (k === 'options') return {add:()=>{},length:0};
        if (k === 'children') return {length:0};
        if (typeof k === 'symbol') return undefined;
        return '';
      },
      set: ()=>true,
    });
  }
  const window = { addEventListener:()=>{}, location:{href:''} };
  const localStorage   = { getItem:()=>null, setItem:()=>{} };
  const sessionStorage = { getItem:()=>null, setItem:()=>{}, removeItem:()=>{} };
  const bootstrap = { Modal: { getInstance:()=>({hide:()=>{}}) } };
  const fetch = async()=>({ok:true,json:async()=>({}),text:async()=>'',body:{getReader:()=>({read:async()=>({done:true})})}});
  const AbortController = class { constructor(){this.signal={};} abort(){} };
  const FormData = class {};
  const requestAnimationFrame = ()=>0;
  const cancelAnimationFrame = ()=>{};
  const alert=()=>{}, confirm=()=>true, prompt=()=>'';
  const setTimeout=()=>0, clearTimeout=()=>{}, setInterval=()=>0, clearInterval=()=>{};
`;

function stripDomContentLoaded(src) {
  // Line-by-line brace counting to remove the DOMContentLoaded block
  const lines = src.split('\n');
  const out = [];
  let inBlock = false;
  let depth = 0;
  for (const line of lines) {
    if (!inBlock && /document\.addEventListener\s*\(\s*['"]DOMContentLoaded/.test(line)) {
      inBlock = true;
      depth = 0;
    }
    if (inBlock) {
      for (const ch of line) {
        if (ch === '{') depth++;
        else if (ch === '}') depth--;
      }
      if (depth <= 0 && line.trim() !== '') {
        inBlock = false;  // consumed closing });
      }
      out.push('// [removed DOMContentLoaded]');
      continue;
    }
    out.push(line);
  }
  return out.join('\n');
}

const files = [
  'sc-core.js','sc-i18n.js','sc-ui.js','sc-agent.js','sc-kernel.js',
  'sc-filesystem.js','sc-plot.js','sc-tools.js','sc-notebook.js','sc-environment.js'
];

// Build combined source
let combined = PREAMBLE + '\n';
for (const f of files) {
  let src = fs.readFileSync(path.join(BASE, f), 'utf8');
  src = stripDomContentLoaded(src);
  combined += '\n// ── ' + f + '\n' + src;
}
combined += '\nreturn SingleCellAnalysis;';

// Evaluate
let Cls;
try {
  Cls = new Function(combined)();
} catch(e) {
  console.error('❌ Evaluation error:', e.message);
  // Diagnose: find which file caused it by loading progressively
  let cum = PREAMBLE;
  for (const f of files) {
    let src = fs.readFileSync(path.join(BASE, f), 'utf8');
    src = stripDomContentLoaded(src);
    cum += '\n' + src;
    try {
      new Function(cum + '\nreturn typeof SingleCellAnalysis;')();
      process.stdout.write('  ✅ ' + f + '\n');
    } catch(e2) {
      process.stdout.write('  ❌ ' + f + ': ' + e2.message + '\n');
      break;
    }
  }
  process.exit(1);
}

if (typeof Cls !== 'function') {
  console.error('❌ SingleCellAnalysis is not a function:', typeof Cls);
  process.exit(1);
}
console.log('✅ SingleCellAnalysis class loaded\n');

// ── Method verification ────────────────────────────────────────────────────
const REQUIRED = [
  // i18n
  'setupLanguageToggle','t','applyLanguage','translateFormHtml',
  'refreshParameterFormLanguage','updateCodeCellPlaceholders',
  // ui
  'setupNavigation','setupSidebarResize','setupGeneAutocomplete',
  'setupBeforeUnloadWarning','setupThemeToggle','setupNotebookManager',
  'toggleSubmenu','toggleMobileMenu','toggleTheme',
  'updatePlotlyTheme','updateStatusBarTheme',
  'showLoading','hideLoading','addToLog','showStatus','hideStatus','updateStatus',
  'switchView','showComingSoon','updateUI','refreshDataFromKernel',
  'updateParameterPanel','checkStatus','applyCodeFontSize','adjustFontSize','syncPanelHeight',
  // agent
  'setupAgentConfig','setupAgentChat','getAgentConfigFields',
  'loadAgentConfig','saveAgentConfig','resetAgentConfig','getAgentConfig',
  'appendAgentMessage','updateAgentMessageContent','sendAgentMessage','showAgentConfig',
  // kernel
  'setupKernelSelector','loadKernelOptions','changeKernel','updateKernelSelectorForTab',
  'interruptExecution','restartKernel','refreshKernelInfo',
  'showInterruptButton','hideInterruptButton',
  'startExecutionStatusPolling','stopExecutionStatusPolling',
  'fetchKernelVars','fetchKernelStats','loadAnndataToVisualization',
  'openVarTab','showVarDetail','toggleSection',
  // filesystem
  'setupFileUpload','handleFileUpload','triggerNotebookUpload',
  'fetchFileTree','loadTreeNode','renderFileTree',
  'openContextMenu','hideContextMenu',
  'contextNewFile','contextNewFolder','contextRefresh',
  'contextDelete','contextRename','contextCopy','contextPaste','contextMove',
  'openFileFromServer','openNotebookTab','openTextTab','openMarkdownTab','openImageTab',
  'setActiveTab','getActiveTab','getActiveKernelId','renderTabs','closeTab',
  'showTextFile','showMarkdownFile','showImageFile','persistActiveTab',
  // plot
  'onColorSelectChange','updatePlot','updatePaletteVisibility','applyVMinMax',
  'createNewPlot','updatePlotWithAnimation','animatePlotTransition',
  'checkCoordsChanged','updateColorsOnly',
  'animatePositionTransition','animatePositionTransitionForAnyData',
  'easeInOutCubic','smoothAnimate',
  'computeDefaultPointSize','getMarkerSize','getMarkerOpacity',
  'onPointSizeChange','onOpacityChange','resetPointStyle','initPointSizeSlider','applyPointStyleLive',
  'plotData','getPlotlyLayout','colorByGene',
  'updateAdataStatus','showParameterPlaceholder','getCategoricalColorscale',
  'testLegend','forceShowLegend','simpleLegendTest','showLegendNow',
  // tools
  'selectAnalysisCategory','getCategoryName','showParameterDialog','_restoreAndTrackParams',
  'renderParameterForm','renderAnnotationForm','getParameterHTML',
  'formatToolMessage','runTool','showToolFigures',
  'showTrajViz','toggleTrajViz','togglePagaOptions',
  'updateTrajVizSelects','generateTrajEmbedding','generateTrajHeatmap','saveData','resetData',
  // notebook
  'addCodeCell','updateCellNumber','runCodeCell','runCodeCellPromise',
  'runCurrentCell','runAllCells','toggleCellOutput','toggleCellOutputFull','clearCellOutput',
  'importNotebookFile','loadNotebookCells','renderNotebookOutputs',
  'captureNotebookOutputs','restoreNotebookOutputs','buildNotebookCellsFromUI','getNotebookOutputsFromCell',
  'saveActiveFile','renderCodeOutput','storeCellOutputPayload','ansiToHtml','updateCodeHighlight',
  'changeCellType','setCellType','renderMarkdownCell','resizeMarkdownEditor',
  'deleteCodeCell','clearAllCells','insertTemplate','renderMarkdown','getTextEditorContent',
  // environment
  'showEnvManager','hideEnvManager','switchEnvTab',
  '_envInitChannels','envAddChannel','_envGetPkg','_envGetChannels',
  '_envUpdatePipPreview','_envUpdateCondaPreview','_envLog','_escHtml',
  'envInstallPip','envInstallConda','_envStreamInstall',
  'envSearch','envTestMirrors','showEnvInfo','hideEnvInfo','loadEnvInfo',
  '_kv','_renderEnvSystem','_renderEnvPython','_renderEnvGpu',
  '_renderEnvKeyPkgs','_renderEnvAllPkgs','_renderEnvAllPkgsTable',
  'filterEnvPkgs','toggleEnvAllPkgs',
];

const proto   = Cls.prototype;
const missing = REQUIRED.filter(n => typeof proto[n] !== 'function');
const ok      = REQUIRED.filter(n => typeof proto[n] === 'function');

// Per-module summary
const modules = {
  'sc-i18n':       ['setupLanguageToggle','t','applyLanguage','translateFormHtml','refreshParameterFormLanguage','updateCodeCellPlaceholders'],
  'sc-ui':         ['setupNavigation','setupSidebarResize','setupGeneAutocomplete','setupBeforeUnloadWarning','setupThemeToggle','setupNotebookManager','toggleSubmenu','toggleMobileMenu','toggleTheme','updatePlotlyTheme','updateStatusBarTheme','showLoading','hideLoading','addToLog','showStatus','hideStatus','updateStatus','switchView','showComingSoon','updateUI','refreshDataFromKernel','updateParameterPanel','checkStatus','applyCodeFontSize','adjustFontSize','syncPanelHeight'],
  'sc-agent':      ['setupAgentConfig','setupAgentChat','getAgentConfigFields','loadAgentConfig','saveAgentConfig','resetAgentConfig','getAgentConfig','appendAgentMessage','updateAgentMessageContent','sendAgentMessage','showAgentConfig'],
  'sc-kernel':     ['setupKernelSelector','loadKernelOptions','changeKernel','updateKernelSelectorForTab','interruptExecution','restartKernel','refreshKernelInfo','showInterruptButton','hideInterruptButton','startExecutionStatusPolling','stopExecutionStatusPolling','fetchKernelVars','fetchKernelStats','loadAnndataToVisualization','openVarTab','showVarDetail','toggleSection'],
  'sc-filesystem': ['setupFileUpload','handleFileUpload','triggerNotebookUpload','fetchFileTree','loadTreeNode','renderFileTree','openContextMenu','hideContextMenu','contextNewFile','contextNewFolder','contextRefresh','contextDelete','contextRename','contextCopy','contextPaste','contextMove','openFileFromServer','openNotebookTab','openTextTab','openMarkdownTab','openImageTab','setActiveTab','getActiveTab','getActiveKernelId','renderTabs','closeTab','showTextFile','showMarkdownFile','showImageFile','persistActiveTab'],
  'sc-plot':       ['onColorSelectChange','updatePlot','updatePaletteVisibility','applyVMinMax','createNewPlot','updatePlotWithAnimation','animatePlotTransition','checkCoordsChanged','updateColorsOnly','animatePositionTransition','animatePositionTransitionForAnyData','easeInOutCubic','smoothAnimate','computeDefaultPointSize','getMarkerSize','getMarkerOpacity','onPointSizeChange','onOpacityChange','resetPointStyle','initPointSizeSlider','applyPointStyleLive','plotData','getPlotlyLayout','colorByGene','updateAdataStatus','showParameterPlaceholder','getCategoricalColorscale','testLegend','forceShowLegend','simpleLegendTest','showLegendNow'],
  'sc-tools':      ['selectAnalysisCategory','getCategoryName','showParameterDialog','_restoreAndTrackParams','renderParameterForm','renderAnnotationForm','getParameterHTML','formatToolMessage','runTool','showToolFigures','showTrajViz','toggleTrajViz','togglePagaOptions','updateTrajVizSelects','generateTrajEmbedding','generateTrajHeatmap','saveData','resetData'],
  'sc-notebook':   ['addCodeCell','updateCellNumber','runCodeCell','runCodeCellPromise','runCurrentCell','runAllCells','toggleCellOutput','toggleCellOutputFull','clearCellOutput','importNotebookFile','loadNotebookCells','renderNotebookOutputs','captureNotebookOutputs','restoreNotebookOutputs','buildNotebookCellsFromUI','getNotebookOutputsFromCell','saveActiveFile','renderCodeOutput','storeCellOutputPayload','ansiToHtml','updateCodeHighlight','changeCellType','setCellType','renderMarkdownCell','resizeMarkdownEditor','deleteCodeCell','clearAllCells','insertTemplate','renderMarkdown','getTextEditorContent'],
  'sc-environment':['showEnvManager','hideEnvManager','switchEnvTab','_envInitChannels','envAddChannel','_envGetPkg','_envGetChannels','_envUpdatePipPreview','_envUpdateCondaPreview','_envLog','_escHtml','envInstallPip','envInstallConda','_envStreamInstall','envSearch','envTestMirrors','showEnvInfo','hideEnvInfo','loadEnvInfo','_kv','_renderEnvSystem','_renderEnvPython','_renderEnvGpu','_renderEnvKeyPkgs','_renderEnvAllPkgs','_renderEnvAllPkgsTable','filterEnvPkgs','toggleEnvAllPkgs'],
};

console.log('── Per-module results ─────────────────────────────────────────────────');
for (const [mod, methods] of Object.entries(modules)) {
  const modMissing = methods.filter(n => typeof proto[n] !== 'function');
  const status = modMissing.length === 0 ? '✅' : '❌';
  console.log(`  ${status} ${mod.padEnd(16)} ${methods.length - modMissing.length}/${methods.length} methods`);
  if (modMissing.length) console.log('     missing: ' + modMissing.join(', '));
}

console.log('\n── Summary ────────────────────────────────────────────────────────────');
if (missing.length === 0) {
  console.log('✅ ALL ' + REQUIRED.length + ' methods present on SingleCellAnalysis.prototype');
} else {
  console.error('❌ MISSING ' + missing.length + ' methods: ' + missing.join(', '));
}
console.log('Result: ' + ok.length + '/' + REQUIRED.length + '  →  ' + (missing.length === 0 ? '✅ PASS' : '❌ FAIL'));
