/**
 * High-performance scatterplot component using WebGL
 * Optimized for large-scale single-cell data visualization
 */

import createPointRenderer, { createContinuousPointRenderer, createSelectionRenderer } from '../../util/webgl/drawPointsRegl.js';
import Camera from '../../util/webgl/camera.js';
import { createColorBuffer, createPositionBuffer, createFlagBuffer, createPerformanceMonitor, generateDiscreteColors } from '../../util/webgl/glHelpers.js';
import { createCrossfilter } from '../../util/typedCrossfilter/index.js';

export default class WebGLScatterplot {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            width: options.width || 800,
            height: options.height || 600,
            backgroundColor: options.backgroundColor || [0.1, 0.1, 0.1, 1.0],
            pointSize: options.pointSize || 'auto',
            maxPoints: options.maxPoints || 1000000,
            enableSelection: options.enableSelection !== false,
            ...options
        };

        // Data
        this.data = null;
        this.positions = null;
        this.colors = null;
        this.flags = null;
        this.crossfilter = null;

        // Rendering
        this.canvas = null;
        this.regl = null;
        this.pointRenderer = null;
        this.continuousRenderer = null;
        this.selectionRenderer = null;
        this.camera = null;
        this.performanceMonitor = createPerformanceMonitor();

        // State
        this.needsRender = true;
        this.renderMode = 'discrete'; // 'discrete', 'continuous'
        this.currentColorBy = null;
        this.selectedIndices = new Set();

        // Event handlers
        this.onSelectionChange = null;
        this.onHover = null;

        this.init();
    }

    init() {
        this.createCanvas();
        this.initWebGL();
        this.initCamera();
        this.setupEventHandlers();
        this.startRenderLoop();
    }

    createCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.options.width;
        this.canvas.height = this.options.height;
        this.canvas.style.width = this.options.width + 'px';
        this.canvas.style.height = this.options.height + 'px';
        this.canvas.style.display = 'block';
        this.canvas.style.cursor = 'grab';

        this.container.appendChild(this.canvas);
    }

    async initWebGL() {
        // Dynamic import for regl since it's installed via npm
        const createREGL = await import('/node_modules/regl/dist/regl.js');

        this.regl = createREGL.default({
            canvas: this.canvas,
            attributes: {
                antialias: true,
                alpha: true,
                depth: true,
                stencil: false
            },
            extensions: ['OES_texture_float', 'ANGLE_instanced_arrays']
        });

        // Create renderers
        this.pointRenderer = createPointRenderer(this.regl);
        this.continuousRenderer = createContinuousPointRenderer(this.regl);
        this.selectionRenderer = createSelectionRenderer(this.regl);

        console.log('WebGL initialized successfully');
    }

    initCamera() {
        this.camera = new Camera({
            canvas: this.canvas,
            width: this.options.width,
            height: this.options.height
        });

        this.camera.onChange = () => {
            this.needsRender = true;
        };
    }

    setupEventHandlers() {
        // Selection handling
        if (this.options.enableSelection) {
            this.canvas.addEventListener('click', this.onCanvasClick.bind(this));
        }

        // Hover handling
        this.canvas.addEventListener('mousemove', this.onCanvasHover.bind(this));

        // Resize handling
        window.addEventListener('resize', this.onResize.bind(this));
    }

    // Public API
    setData(data) {
        if (!data || !data.positions) {
            console.error('Invalid data format. Expected { positions: [[x,y], ...], ... }');
            return;
        }

        this.data = data;
        this.positions = data.positions;

        // Create crossfilter for data management
        const records = this.positions.map((pos, i) => ({
            index: i,
            x: pos[0],
            y: pos[1],
            ...data.metadata?.[i]
        }));

        this.crossfilter = createCrossfilter(records);

        // Set camera bounds based on data
        this.updateDataBounds();

        // Initialize colors and flags
        this.initializeRenderData();

        this.needsRender = true;
        console.log(`Loaded ${this.positions.length} points`);
    }

    updateDataBounds() {
        if (!this.positions || this.positions.length === 0) return;

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        for (let pos of this.positions) {
            minX = Math.min(minX, pos[0]);
            maxX = Math.max(maxX, pos[0]);
            minY = Math.min(minY, pos[1]);
            maxY = Math.max(maxY, pos[1]);
        }

        this.camera.setDataBounds({ minX, maxX, minY, maxY });
    }

    initializeRenderData() {
        const nPoints = this.positions.length;

        // Initialize colors (default gray)
        this.colors = new Array(nPoints).fill([0.7, 0.7, 0.7]);

        // Initialize flags (all normal points)
        this.flags = new Array(nPoints).fill(0);

        // Create WebGL buffers
        this.updateBuffers();
    }

    updateBuffers() {
        if (!this.regl || !this.positions) return;

        this.positionBuffer = createPositionBuffer(this.regl, this.positions);
        this.colorBuffer = createColorBuffer(this.regl, this.colors, this.positions.length);
        this.flagBuffer = createFlagBuffer(this.regl, this.flags);
    }

    // Color mapping
    colorBy(attribute, options = {}) {
        if (!this.data || !this.crossfilter) {
            console.warn('No data available for coloring');
            return;
        }

        this.currentColorBy = attribute;

        if (attribute === 'default') {
            // Reset to default colors
            this.colors = new Array(this.positions.length).fill([0.7, 0.7, 0.7]);
            this.renderMode = 'discrete';
        } else if (this.data.categorical?.[attribute]) {
            // Categorical coloring
            this.applyCategoricalColoring(attribute, options);
            this.renderMode = 'discrete';
        } else if (this.data.continuous?.[attribute]) {
            // Continuous coloring
            this.applyContinuousColoring(attribute, options);
            this.renderMode = 'continuous';
        } else {
            console.warn(`Unknown attribute: ${attribute}`);
            return;
        }

        this.updateBuffers();
        this.needsRender = true;
    }

    applyCategoricalColoring(attribute, options) {
        const values = this.data.categorical[attribute];
        const uniqueValues = [...new Set(values)];
        const colorPalette = options.colors || generateDiscreteColors(uniqueValues.length);
        const colorMap = new Map();

        uniqueValues.forEach((value, i) => {
            colorMap.set(value, colorPalette[i]);
        });

        this.colors = values.map(value => {
            const color = colorMap.get(value);
            return color ? [color[0]/255, color[1]/255, color[2]/255] : [0.7, 0.7, 0.7];
        });
    }

    applyContinuousColoring(attribute, options) {
        const values = this.data.continuous[attribute];
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);

        const lowColor = options.lowColor || [0, 0, 0.5];
        const highColor = options.highColor || [1, 0.8, 0];

        this.continuousData = {
            values: values,
            minValue: minValue,
            maxValue: maxValue,
            lowColor: lowColor,
            highColor: highColor
        };
    }

    // Selection
    selectPoints(indices) {
        this.selectedIndices = new Set(indices);
        this.updateFlags();
        this.needsRender = true;

        if (this.onSelectionChange) {
            this.onSelectionChange(Array.from(this.selectedIndices));
        }
    }

    updateFlags() {
        for (let i = 0; i < this.flags.length; i++) {
            let flag = 0;
            if (this.selectedIndices.has(i)) {
                flag |= 2; // Selected flag
            }
            this.flags[i] = flag;
        }

        if (this.regl) {
            this.flagBuffer = createFlagBuffer(this.regl, this.flags);
        }
    }

    // Event handlers
    onCanvasClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const worldPos = this.camera.screenToWorld(x, y);
        const clickedIndex = this.findNearestPoint(worldPos[0], worldPos[1]);

        if (clickedIndex >= 0) {
            if (event.ctrlKey || event.metaKey) {
                // Toggle selection
                if (this.selectedIndices.has(clickedIndex)) {
                    this.selectedIndices.delete(clickedIndex);
                } else {
                    this.selectedIndices.add(clickedIndex);
                }
            } else {
                // Single selection
                this.selectedIndices.clear();
                this.selectedIndices.add(clickedIndex);
            }

            this.updateFlags();
            this.needsRender = true;

            if (this.onSelectionChange) {
                this.onSelectionChange(Array.from(this.selectedIndices));
            }
        }
    }

    onCanvasHover(event) {
        if (!this.onHover) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const worldPos = this.camera.screenToWorld(x, y);
        const hoveredIndex = this.findNearestPoint(worldPos[0], worldPos[1], 0.02); // 2% threshold

        this.onHover(hoveredIndex, { x, y });
    }

    findNearestPoint(worldX, worldY, threshold = 0.01) {
        let nearestIndex = -1;
        let minDistance = threshold;

        for (let i = 0; i < this.positions.length; i++) {
            const dx = this.positions[i][0] - worldX;
            const dy = this.positions[i][1] - worldY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < minDistance) {
                minDistance = distance;
                nearestIndex = i;
            }
        }

        return nearestIndex;
    }

    onResize() {
        const rect = this.container.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';

        this.camera.updateViewport(width, height);
        this.needsRender = true;
    }

    // Rendering
    render() {
        if (!this.regl || !this.positionBuffer) return;

        this.performanceMonitor.beginFrame();

        const projView = this.camera.getProjectionViewMatrix();
        const nPoints = this.positions.length;
        const viewport = this.camera.viewport;

        this.regl.clear({
            color: this.options.backgroundColor,
            depth: 1
        });

        // Render points based on current mode
        if (this.renderMode === 'continuous' && this.continuousData) {
            this.continuousRenderer({
                position: this.positionBuffer,
                value: this.regl.buffer(new Float32Array(this.continuousData.values)),
                flag: this.flagBuffer,
                projView: projView,
                nPoints: nPoints,
                minViewportDimension: Math.min(viewport.width, viewport.height),
                distance: 1.0,
                minValue: this.continuousData.minValue,
                maxValue: this.continuousData.maxValue,
                lowColor: this.continuousData.lowColor,
                highColor: this.continuousData.highColor,
                count: nPoints
            });
        } else {
            this.pointRenderer({
                position: this.positionBuffer,
                color: this.colorBuffer,
                flag: this.flagBuffer,
                projView: projView,
                nPoints: nPoints,
                minViewportDimension: Math.min(viewport.width, viewport.height),
                distance: 1.0,
                count: nPoints
            });
        }

        this.performanceMonitor.endFrame();
        this.needsRender = false;
    }

    startRenderLoop() {
        const renderFrame = () => {
            if (this.needsRender) {
                this.render();
            }
            requestAnimationFrame(renderFrame);
        };
        requestAnimationFrame(renderFrame);
    }

    // Public methods
    fitToData() {
        if (this.camera) {
            this.camera.fitToData();
        }
    }

    resetView() {
        if (this.camera) {
            this.camera.reset();
        }
    }

    getPerformanceStats() {
        return {
            ...this.performanceMonitor.getStats(),
            pointCount: this.positions ? this.positions.length : 0,
            selectedCount: this.selectedIndices.size
        };
    }

    // Cleanup
    destroy() {
        if (this.regl) {
            this.regl.destroy();
        }
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}