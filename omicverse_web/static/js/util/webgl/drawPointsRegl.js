/**
 * High-performance point rendering using Regl
 * Optimized for large-scale single-cell visualization
 */

import { glPointFlags, glPointSize } from './glHelpers.js';

export default function createPointRenderer(regl) {
    return regl({
        vert: `
        precision mediump float;

        attribute vec2 position;
        attribute vec3 color;
        attribute float flag;

        uniform float distance;
        uniform mat3 projView;
        uniform float nPoints;
        uniform float minViewportDimension;

        varying lowp vec4 fragColor;

        const float zBottom = 0.99;
        const float zMiddle = 0.0;
        const float zTop = -1.0;

        // Import flag utilities
        ${glPointFlags}

        // Import point size calculation
        ${glPointSize}

        void main() {
            bool isBackground, isSelected, isHighlight;
            getFlags(flag, isBackground, isSelected, isHighlight);

            float size = pointSize(nPoints, minViewportDimension, isSelected, isHighlight);
            gl_PointSize = size * pow(distance, 0.5);

            float z = isBackground ? zBottom : (isHighlight ? zTop : zMiddle);
            vec3 xy = projView * vec3(position, 1.0);
            gl_Position = vec4(xy.xy, z, 1.0);

            float alpha = isBackground ? 0.6 : 1.0;
            fragColor = vec4(color, alpha);
        }`,

        frag: `
        precision mediump float;
        varying lowp vec4 fragColor;

        void main() {
            // Create circular points
            vec2 coord = gl_PointCoord.xy - 0.5;
            float distance = length(coord);

            if (distance > 0.5) {
                discard;
            }

            // Anti-aliasing
            float alpha = fragColor.a * (1.0 - smoothstep(0.4, 0.5, distance));
            gl_FragColor = vec4(fragColor.rgb, alpha);
        }`,

        attributes: {
            position: regl.prop('position'),
            color: regl.prop('color'),
            flag: regl.prop('flag')
        },

        uniforms: {
            distance: regl.prop('distance'),
            projView: regl.prop('projView'),
            nPoints: regl.prop('nPoints'),
            minViewportDimension: regl.prop('minViewportDimension')
        },

        count: regl.prop('count'),

        primitive: 'points',

        blend: {
            enable: true,
            func: {
                srcRGB: 'src alpha',
                srcAlpha: 1,
                dstRGB: 'one minus src alpha',
                dstAlpha: 1
            }
        },

        depth: {
            enable: true,
            mask: false,
            func: 'less'
        }
    });
}

// Optimized renderer for continuous color mapping
export function createContinuousPointRenderer(regl) {
    return regl({
        vert: `
        precision mediump float;

        attribute vec2 position;
        attribute float value;
        attribute float flag;

        uniform float distance;
        uniform mat3 projView;
        uniform float nPoints;
        uniform float minViewportDimension;
        uniform float minValue;
        uniform float maxValue;
        uniform vec3 lowColor;
        uniform vec3 highColor;

        varying lowp vec4 fragColor;

        ${glPointFlags}
        ${glPointSize}

        vec3 interpolateColor(float t, vec3 color1, vec3 color2) {
            return mix(color1, color2, t);
        }

        void main() {
            bool isBackground, isSelected, isHighlight;
            getFlags(flag, isBackground, isSelected, isHighlight);

            float size = pointSize(nPoints, minViewportDimension, isSelected, isHighlight);
            gl_PointSize = size * pow(distance, 0.5);

            float z = isBackground ? 0.99 : (isHighlight ? -1.0 : 0.0);
            vec3 xy = projView * vec3(position, 1.0);
            gl_Position = vec4(xy.xy, z, 1.0);

            // Color interpolation based on value
            float normalizedValue = (value - minValue) / (maxValue - minValue);
            normalizedValue = clamp(normalizedValue, 0.0, 1.0);

            vec3 color = interpolateColor(normalizedValue, lowColor, highColor);
            float alpha = isBackground ? 0.6 : 1.0;
            fragColor = vec4(color, alpha);
        }`,

        frag: `
        precision mediump float;
        varying lowp vec4 fragColor;

        void main() {
            vec2 coord = gl_PointCoord.xy - 0.5;
            float distance = length(coord);

            if (distance > 0.5) {
                discard;
            }

            float alpha = fragColor.a * (1.0 - smoothstep(0.4, 0.5, distance));
            gl_FragColor = vec4(fragColor.rgb, alpha);
        }`,

        attributes: {
            position: regl.prop('position'),
            value: regl.prop('value'),
            flag: regl.prop('flag')
        },

        uniforms: {
            distance: regl.prop('distance'),
            projView: regl.prop('projView'),
            nPoints: regl.prop('nPoints'),
            minViewportDimension: regl.prop('minViewportDimension'),
            minValue: regl.prop('minValue'),
            maxValue: regl.prop('maxValue'),
            lowColor: regl.prop('lowColor'),
            highColor: regl.prop('highColor')
        },

        count: regl.prop('count'),
        primitive: 'points',

        blend: {
            enable: true,
            func: {
                srcRGB: 'src alpha',
                srcAlpha: 1,
                dstRGB: 'one minus src alpha',
                dstAlpha: 1
            }
        },

        depth: {
            enable: true,
            mask: false,
            func: 'less'
        }
    });
}

// Selection overlay renderer
export function createSelectionRenderer(regl) {
    return regl({
        vert: `
        precision mediump float;

        attribute vec2 position;
        attribute float selected;

        uniform mat3 projView;
        uniform float selectionRadius;

        varying float vSelected;

        void main() {
            vSelected = selected;

            if (selected > 0.5) {
                gl_PointSize = selectionRadius;
                vec3 xy = projView * vec3(position, 1.0);
                gl_Position = vec4(xy.xy, -0.5, 1.0);
            } else {
                gl_PointSize = 0.0;
                gl_Position = vec4(0.0, 0.0, 1.0, 1.0);
            }
        }`,

        frag: `
        precision mediump float;
        varying float vSelected;

        void main() {
            if (vSelected < 0.5) {
                discard;
            }

            vec2 coord = gl_PointCoord.xy - 0.5;
            float distance = length(coord);

            // Selection ring
            if (distance > 0.5 || distance < 0.3) {
                discard;
            }

            gl_FragColor = vec4(1.0, 0.8, 0.0, 0.8);
        }`,

        attributes: {
            position: regl.prop('position'),
            selected: regl.prop('selected')
        },

        uniforms: {
            projView: regl.prop('projView'),
            selectionRadius: regl.prop('selectionRadius')
        },

        count: regl.prop('count'),
        primitive: 'points',

        blend: {
            enable: true,
            func: {
                srcRGB: 'src alpha',
                srcAlpha: 1,
                dstRGB: 'one minus src alpha',
                dstAlpha: 1
            }
        }
    });
}