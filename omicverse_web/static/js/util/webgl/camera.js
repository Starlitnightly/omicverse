/**
 * Camera controller for WebGL viewport navigation
 * Supports pan, zoom, and reset operations
 */

import { mat3 } from '../math/matrix.js';

export default class Camera {
    constructor(options = {}) {
        this.canvas = options.canvas;
        this.viewport = {
            width: options.width || 800,
            height: options.height || 600
        };

        // Camera state
        this.position = [0, 0];
        this.zoom = 1.0;
        this.minZoom = 0.1;
        this.maxZoom = 100.0;

        // Data bounds
        this.dataBounds = {
            minX: -1,
            maxX: 1,
            minY: -1,
            maxY: 1
        };

        // Event handling
        this.isDragging = false;
        this.lastMousePos = [0, 0];

        this.setupEventHandlers();
    }

    // Set data bounds for auto-fit functionality
    setDataBounds(bounds) {
        this.dataBounds = { ...bounds };
    }

    // Get current projection-view matrix
    getProjectionViewMatrix() {
        const { width, height } = this.viewport;

        // Create projection matrix (orthographic)
        const aspectRatio = width / height;
        const viewWidth = 2.0 / this.zoom;
        const viewHeight = 2.0 / this.zoom;

        let left, right, bottom, top;

        if (aspectRatio > 1) {
            // Wider than tall
            left = -viewWidth * aspectRatio / 2;
            right = viewWidth * aspectRatio / 2;
            bottom = -viewHeight / 2;
            top = viewHeight / 2;
        } else {
            // Taller than wide
            left = -viewWidth / 2;
            right = viewWidth / 2;
            bottom = -viewHeight / aspectRatio / 2;
            top = viewHeight / aspectRatio / 2;
        }

        // Apply camera position
        left += this.position[0];
        right += this.position[0];
        bottom += this.position[1];
        top += this.position[1];

        // Create transformation matrix
        const matrix = mat3.create();

        // Scale and translate to normalized device coordinates
        mat3.set(matrix,
            2.0 / (right - left), 0, -(right + left) / (right - left),
            0, 2.0 / (top - bottom), -(top + bottom) / (top - bottom),
            0, 0, 1
        );

        return matrix;
    }

    // Convert screen coordinates to world coordinates
    screenToWorld(screenX, screenY) {
        const { width, height } = this.viewport;

        // Convert to normalized device coordinates
        const ndcX = (screenX / width) * 2.0 - 1.0;
        const ndcY = -((screenY / height) * 2.0 - 1.0); // Flip Y

        // Apply inverse transformation
        const aspectRatio = width / height;
        const viewWidth = 2.0 / this.zoom;
        const viewHeight = 2.0 / this.zoom;

        let worldX, worldY;

        if (aspectRatio > 1) {
            worldX = ndcX * viewWidth * aspectRatio / 2 + this.position[0];
            worldY = ndcY * viewHeight / 2 + this.position[1];
        } else {
            worldX = ndcX * viewWidth / 2 + this.position[0];
            worldY = ndcY * viewHeight / aspectRatio / 2 + this.position[1];
        }

        return [worldX, worldY];
    }

    // Convert world coordinates to screen coordinates
    worldToScreen(worldX, worldY) {
        const { width, height } = this.viewport;
        const aspectRatio = width / height;
        const viewWidth = 2.0 / this.zoom;
        const viewHeight = 2.0 / this.zoom;

        let ndcX, ndcY;

        if (aspectRatio > 1) {
            ndcX = (worldX - this.position[0]) / (viewWidth * aspectRatio / 2);
            ndcY = (worldY - this.position[1]) / (viewHeight / 2);
        } else {
            ndcX = (worldX - this.position[0]) / (viewWidth / 2);
            ndcY = (worldY - this.position[1]) / (viewHeight / aspectRatio / 2);
        }

        const screenX = (ndcX + 1.0) / 2.0 * width;
        const screenY = (-ndcY + 1.0) / 2.0 * height;

        return [screenX, screenY];
    }

    // Pan camera by delta in world coordinates
    pan(deltaX, deltaY) {
        this.position[0] += deltaX;
        this.position[1] += deltaY;
        this.notifyChange();
    }

    // Zoom camera at specific point
    zoomAt(factor, centerX = 0, centerY = 0) {
        const oldZoom = this.zoom;
        this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * factor));

        if (this.zoom !== oldZoom) {
            // Adjust position to zoom towards center point
            const zoomChange = this.zoom / oldZoom;
            this.position[0] = centerX + (this.position[0] - centerX) / zoomChange;
            this.position[1] = centerY + (this.position[1] - centerY) / zoomChange;
            this.notifyChange();
        }
    }

    // Fit data to viewport
    fitToData(padding = 0.1) {
        const { minX, maxX, minY, maxY } = this.dataBounds;

        const dataWidth = maxX - minX;
        const dataHeight = maxY - minY;
        const dataCenterX = (minX + maxX) / 2;
        const dataCenterY = (minY + maxY) / 2;

        // Add padding
        const paddedWidth = dataWidth * (1 + padding);
        const paddedHeight = dataHeight * (1 + padding);

        // Calculate zoom to fit data
        const { width, height } = this.viewport;
        const aspectRatio = width / height;
        const dataAspectRatio = paddedWidth / paddedHeight;

        let targetZoom;
        if (dataAspectRatio > aspectRatio) {
            // Data is wider relative to viewport
            targetZoom = 2.0 / (paddedWidth * aspectRatio);
        } else {
            // Data is taller relative to viewport
            targetZoom = 2.0 / paddedHeight;
        }

        // Set camera state
        this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, targetZoom));
        this.position[0] = dataCenterX;
        this.position[1] = dataCenterY;

        this.notifyChange();
    }

    // Reset camera to default state
    reset() {
        this.position[0] = 0;
        this.position[1] = 0;
        this.zoom = 1.0;
        this.notifyChange();
    }

    // Update viewport size
    updateViewport(width, height) {
        this.viewport.width = width;
        this.viewport.height = height;
        this.notifyChange();
    }

    // Event handling setup
    setupEventHandlers() {
        if (!this.canvas) return;

        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));
        this.canvas.addEventListener('contextmenu', e => e.preventDefault());

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this));
        this.canvas.addEventListener('touchmove', this.onTouchMove.bind(this));
        this.canvas.addEventListener('touchend', this.onTouchEnd.bind(this));
    }

    // Mouse event handlers
    onMouseDown(event) {
        if (event.button === 0) { // Left button
            this.isDragging = true;
            this.lastMousePos = [event.clientX, event.clientY];
            this.canvas.style.cursor = 'grabbing';
            event.preventDefault();
        }
    }

    onMouseMove(event) {
        if (this.isDragging) {
            const deltaX = event.clientX - this.lastMousePos[0];
            const deltaY = event.clientY - this.lastMousePos[1];

            // Convert pixel delta to world delta
            const worldDelta = this.screenDeltaToWorldDelta(deltaX, deltaY);
            this.pan(-worldDelta[0], worldDelta[1]); // Negative X for natural drag feeling

            this.lastMousePos = [event.clientX, event.clientY];
            event.preventDefault();
        }
    }

    onMouseUp(event) {
        if (event.button === 0) {
            this.isDragging = false;
            this.canvas.style.cursor = 'grab';
            event.preventDefault();
        }
    }

    onWheel(event) {
        event.preventDefault();

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        const worldPos = this.screenToWorld(mouseX, mouseY);
        const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;

        this.zoomAt(zoomFactor, worldPos[0], worldPos[1]);
    }

    // Touch event handlers (basic implementation)
    onTouchStart(event) {
        if (event.touches.length === 1) {
            const touch = event.touches[0];
            this.isDragging = true;
            this.lastMousePos = [touch.clientX, touch.clientY];
            event.preventDefault();
        }
    }

    onTouchMove(event) {
        if (this.isDragging && event.touches.length === 1) {
            const touch = event.touches[0];
            const deltaX = touch.clientX - this.lastMousePos[0];
            const deltaY = touch.clientY - this.lastMousePos[1];

            const worldDelta = this.screenDeltaToWorldDelta(deltaX, deltaY);
            this.pan(-worldDelta[0], worldDelta[1]);

            this.lastMousePos = [touch.clientX, touch.clientY];
            event.preventDefault();
        }
    }

    onTouchEnd(event) {
        this.isDragging = false;
        event.preventDefault();
    }

    // Helper methods
    screenDeltaToWorldDelta(screenDeltaX, screenDeltaY) {
        const { width, height } = this.viewport;
        const aspectRatio = width / height;
        const viewWidth = 2.0 / this.zoom;
        const viewHeight = 2.0 / this.zoom;

        let worldDeltaX, worldDeltaY;

        if (aspectRatio > 1) {
            worldDeltaX = (screenDeltaX / width) * viewWidth * aspectRatio;
            worldDeltaY = (screenDeltaY / height) * viewHeight;
        } else {
            worldDeltaX = (screenDeltaX / width) * viewWidth;
            worldDeltaY = (screenDeltaY / height) * viewHeight / aspectRatio;
        }

        return [worldDeltaX, worldDeltaY];
    }

    // Callback for camera changes
    onChange = null;

    notifyChange() {
        if (this.onChange) {
            this.onChange({
                position: [...this.position],
                zoom: this.zoom,
                projectionView: this.getProjectionViewMatrix()
            });
        }
    }

    // Get current camera state
    getState() {
        return {
            position: [...this.position],
            zoom: this.zoom,
            viewport: { ...this.viewport },
            dataBounds: { ...this.dataBounds }
        };
    }

    // Restore camera state
    setState(state) {
        this.position = [...state.position];
        this.zoom = state.zoom;
        if (state.viewport) {
            this.viewport = { ...state.viewport };
        }
        if (state.dataBounds) {
            this.dataBounds = { ...state.dataBounds };
        }
        this.notifyChange();
    }
}