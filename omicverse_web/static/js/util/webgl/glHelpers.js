/**
 * WebGL helpers and utilities
 * Based on CellxGene's GL rendering system
 */

// Point rendering flags for shader
export const glPointFlags = `
    void getFlags(float flag, out bool isBackground, out bool isSelected, out bool isHighlight) {
        int iFlag = int(flag);
        isBackground = (iFlag & 1) != 0;
        isSelected = (iFlag & 2) != 0;
        isHighlight = (iFlag & 4) != 0;
    }
`;

// Point size calculation
export const glPointSize = `
    float pointSize(float nPoints, float minViewportDimension, bool isSelected, bool isHighlight) {
        // Base size calculation based on number of points and viewport
        float baseSize = max(2.0, min(8.0, minViewportDimension / sqrt(nPoints) * 0.1));

        // Size modifiers
        if (isHighlight) {
            return baseSize * 2.0;
        } else if (isSelected) {
            return baseSize * 1.5;
        } else {
            return baseSize;
        }
    }
`;

// Color utilities
export function createColorBuffer(regl, colors, nPoints) {
    // Ensure we have RGB values for each point
    const colorData = new Float32Array(nPoints * 3);

    for (let i = 0; i < nPoints; i++) {
        const offset = i * 3;
        if (Array.isArray(colors[i])) {
            // RGB array
            colorData[offset] = colors[i][0] / 255;
            colorData[offset + 1] = colors[i][1] / 255;
            colorData[offset + 2] = colors[i][2] / 255;
        } else if (typeof colors[i] === 'string') {
            // Hex color
            const rgb = hexToRgb(colors[i]);
            colorData[offset] = rgb.r / 255;
            colorData[offset + 1] = rgb.g / 255;
            colorData[offset + 2] = rgb.b / 255;
        } else {
            // Default gray
            colorData[offset] = 0.7;
            colorData[offset + 1] = 0.7;
            colorData[offset + 2] = 0.7;
        }
    }

    return regl.buffer(colorData);
}

export function createPositionBuffer(regl, positions) {
    // Positions should be [[x1, y1], [x2, y2], ...]
    const positionData = new Float32Array(positions.length * 2);

    for (let i = 0; i < positions.length; i++) {
        positionData[i * 2] = positions[i][0];
        positionData[i * 2 + 1] = positions[i][1];
    }

    return regl.buffer(positionData);
}

export function createFlagBuffer(regl, flags) {
    return regl.buffer(new Float32Array(flags));
}

// Viewport and projection utilities
export function createProjectionMatrix(width, height, padding = 50) {
    // Create orthographic projection matrix
    const left = -padding;
    const right = width + padding;
    const bottom = height + padding;
    const top = -padding;

    const mat = new Float32Array([
        2 / (right - left), 0, 0,
        0, 2 / (top - bottom), 0,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), 1
    ]);

    return mat;
}

export function normalizeCoordinates(coordinates, width, height) {
    // Normalize coordinates to viewport space
    const normalized = [];

    for (let i = 0; i < coordinates.length; i++) {
        const [x, y] = coordinates[i];
        normalized.push([
            (x - width / 2) / (width / 2),
            (y - height / 2) / (height / 2)
        ]);
    }

    return normalized;
}

// Color utilities
export function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 128, g: 128, b: 128 };
}

export function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// Generate discrete color palette
export function generateDiscreteColors(n) {
    const colors = [];
    const hueStep = 360 / n;

    for (let i = 0; i < n; i++) {
        const hue = i * hueStep;
        const rgb = hslToRgb(hue / 360, 0.7, 0.6);
        colors.push([rgb.r, rgb.g, rgb.b]);
    }

    return colors;
}

export function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h * 6) % 2 - 1));
    const m = l - c / 2;

    let r, g, b;

    if (h < 1/6) {
        [r, g, b] = [c, x, 0];
    } else if (h < 2/6) {
        [r, g, b] = [x, c, 0];
    } else if (h < 3/6) {
        [r, g, b] = [0, c, x];
    } else if (h < 4/6) {
        [r, g, b] = [0, x, c];
    } else if (h < 5/6) {
        [r, g, b] = [x, 0, c];
    } else {
        [r, g, b] = [c, 0, x];
    }

    return {
        r: Math.round((r + m) * 255),
        g: Math.round((g + m) * 255),
        b: Math.round((b + m) * 255)
    };
}

// Performance monitoring
export function createPerformanceMonitor() {
    const stats = {
        frameCount: 0,
        lastTime: performance.now(),
        fps: 0,
        renderTime: 0
    };

    return {
        beginFrame() {
            stats.frameStartTime = performance.now();
        },

        endFrame() {
            const now = performance.now();
            stats.renderTime = now - stats.frameStartTime;
            stats.frameCount++;

            if (now - stats.lastTime > 1000) {
                stats.fps = stats.frameCount;
                stats.frameCount = 0;
                stats.lastTime = now;
            }
        },

        getStats() {
            return {
                fps: stats.fps,
                renderTime: stats.renderTime.toFixed(2)
            };
        }
    };
}

// Viewport utilities
export function getViewportInfo(canvas) {
    const rect = canvas.getBoundingClientRect();
    const devicePixelRatio = window.devicePixelRatio || 1;

    return {
        width: canvas.width,
        height: canvas.height,
        displayWidth: rect.width,
        displayHeight: rect.height,
        devicePixelRatio,
        minDimension: Math.min(canvas.width, canvas.height)
    };
}

// WebGL capability detection
export function detectWebGLCapabilities(gl) {
    const capabilities = {
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS),
        maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
        maxFragmentTextures: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
        floatTextures: false,
        instancing: false
    };

    // Check for float texture support
    const floatExt = gl.getExtension('OES_texture_float');
    if (floatExt) {
        capabilities.floatTextures = true;
    }

    // Check for instanced rendering
    const instanceExt = gl.getExtension('ANGLE_instanced_arrays');
    if (instanceExt) {
        capabilities.instancing = true;
    }

    return capabilities;
}