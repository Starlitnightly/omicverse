/**
 * sc-deckgl.js — WebGL scatter renderer for large single-cell datasets.
 *
 * Replaces the raster (Datashader/matplotlib PNG) renderer for datasets that
 * exceed the Plotly scattergl performance limit (~200 K cells).
 *
 * Advantages over the raster approach:
 *   • Full interactivity: zoom / pan (native deck.gl controller)
 *   • Hover tooltip with per-cell metadata
 *   • Smooth position animation between embeddings (CPU lerp → GPU upload)
 *   • Color fade animation when only coloring changes (for n ≤ 300 K)
 *   • Runs on ANY Mac GPU (Metal/WebGL2 via Apple's Safari/Chrome drivers)
 *
 * Requires: deck.gl loaded via CDN before this script (global `deck` object).
 *
 * Binary wire format from /api/plot_gpu:
 *   [0–3]               uint32 LE  n_cells
 *   [4–7]               uint32 LE  json_len
 *   [8 .. 8+json_len)   UTF-8 JSON metadata
 *   [padded to 8 bytes] float32[n*2]  positions  x0,y0,x1,y1,…
 *   [above + n*8]       uint8[n*4]   colors     r,g,b,a per cell
 *   [above + n*4]       float32[n]   hover_values (cat-code or numeric, NaN=none)
 */
(function (global) {
    'use strict';

    // -------------------------------------------------------------------------
    // Easing (same curve as the Plotly animation in sc-plot.js)
    // -------------------------------------------------------------------------
    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    // -------------------------------------------------------------------------
    // parsePlotGPUBuffer – decode the binary blob from /api/plot_gpu
    // -------------------------------------------------------------------------
    function parsePlotGPUBuffer(buf) {
        const dv      = new DataView(buf);
        const n       = dv.getUint32(0, true);
        const jsonLen = dv.getUint32(4, true);

        // JSON metadata
        const metaBytes = new Uint8Array(buf, 8, jsonLen);
        const meta      = JSON.parse(new TextDecoder().decode(metaBytes));

        // Alignment padding so positions start on an 8-byte boundary
        const totalHdr = 8 + jsonLen;
        const pad      = (8 - (totalHdr % 8)) % 8;
        const posStart = totalHdr + pad;      // divisible by 8 → OK for Float32Array
        const colStart = posStart + n * 8;    // n*8 is div by 8 → OK for Uint8Array
        const hovStart = colStart + n * 4;    // div by 4 → OK for Float32Array

        // Zero-copy typed-array views (requires posStart / colStart / hovStart to be
        // multiples of the element size — guaranteed by the alignment above).
        const positions    = new Float32Array(buf.slice(posStart, posStart + n * 8));
        const colors       = new Uint8Array  (buf.slice(colStart, colStart + n * 4));
        const hoverValues  = new Float32Array(buf.slice(hovStart, hovStart + n * 4));

        return { n, meta, positions, colors, hoverValues };
    }

    // -------------------------------------------------------------------------
    // parseColorOnlyBuffer – decode the compact binary from /api/plot_gpu_colors
    // Format:
    //   [0-3]              uint32 n_cells
    //   [4-7]              uint32 json_len
    //   [8 .. 8+json_len)  UTF-8 JSON metadata
    //   [padded to 4 bytes] uint8[n*4]  colors (r,g,b,a)
    //   [above + n*4]       float32[n]  hover_values
    // -------------------------------------------------------------------------
    function parseColorOnlyBuffer(buf) {
        const dv      = new DataView(buf);
        const n       = dv.getUint32(0, true);
        const jsonLen = dv.getUint32(4, true);

        const metaBytes = new Uint8Array(buf, 8, jsonLen);
        const meta      = JSON.parse(new TextDecoder().decode(metaBytes));

        const totalHdr = 8 + jsonLen;
        const pad      = (4 - (totalHdr % 4)) % 4;
        const colStart = totalHdr + pad;
        const hovStart = colStart + n * 4;

        const colors      = new Uint8Array  (buf.slice(colStart, colStart + n * 4));
        const hoverValues = new Float32Array(buf.slice(hovStart, hovStart + n * 4));

        return { n, meta, colors, hoverValues };
    }

    // -------------------------------------------------------------------------
    // DeckGLRenderer
    // -------------------------------------------------------------------------
    function DeckGLRenderer(container) {
        this.container    = container;
        this.deckgl       = null;
        this.canvas       = null;
        this._positions   = null;   // Float32Array – currently displayed (x,y interleaved)
        this._colors      = null;   // Uint8Array   – currently displayed (r,g,b,a)
        this._hoverValues = null;   // Float32Array – per-cell value for tooltip
        this._meta        = null;   // last metadata object from server
        this._n           = 0;
        this._animReq     = null;   // current requestAnimationFrame id
        this._viewState   = null;   // deck.gl view state (controlled)
        this._radius      = 3.5;    // current point radius (pixels)
        this._opacity     = 0.80;   // current layer opacity (0–1)
    }

    // ------------------------------------------------------------------
    // init – create canvas and Deck instance
    // Returns false if deck.gl is unavailable (caller can fall back).
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype.init = function () {
        if (!global.deck) {
            console.error('[DeckGLRenderer] deck.gl not found – add CDN script to HTML.');
            return false;
        }

        // Canvas fills the container (container must be position:relative)
        const canvas = document.createElement('canvas');
        canvas.id = 'deckgl-canvas';
        canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;display:block;';
        this.container.style.position = 'relative';
        this.container.appendChild(canvas);
        this.canvas = canvas;

        this._viewState = { target: [0, 0, 0], zoom: 0, minZoom: -20, maxZoom: 40 };

        const self = this;
        this.deckgl = new deck.Deck({
            canvas:     canvas,
            views:      new deck.OrthographicView({ id: 'main' }),
            controller: true,
            // Controlled view state: we drive it, the controller updates it via callback
            viewState:  this._viewState,
            onViewStateChange: function (_ref) {
                self._viewState = _ref.viewState;
                self.deckgl.setProps({ viewState: self._viewState });
            },
            layers:     [],
            getTooltip: function (_ref2) { return self._buildTooltip(_ref2.index); },
            // Transparent background – let CSS background-color show through
            parameters: { clearColor: [0, 0, 0, 0] },
        });

        return true;
    };

    // ------------------------------------------------------------------
    // _buildTooltip – called by deck.gl on hover
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype._buildTooltip = function (index) {
        if (index < 0 || !this._positions) return null;

        const x = this._positions[index * 2];
        const y = this._positions[index * 2 + 1];

        let valStr = '';
        if (this._hoverValues && !isNaN(this._hoverValues[index])) {
            const rawVal = this._hoverValues[index];
            const m      = this._meta;
            if (m && m.is_categorical && m.category_labels) {
                const code = Math.round(rawVal);
                valStr = m.category_labels[code] !== undefined
                    ? m.category_labels[code] : String(code);
            } else {
                valStr = rawVal.toFixed(4);
            }
        }

        const label = (this._meta && this._meta.color_label) ? this._meta.color_label + ': ' : '';
        const isDark = document.documentElement.classList.contains('app-skin-dark');

        return {
            html: [
                '<div style="font-size:11px;line-height:1.7;padding:2px 0;">',
                '<b>Cell ' + index + '</b><br/>',
                'x\u202f=\u202f' + x.toFixed(3) + ',\u2002y\u202f=\u202f' + y.toFixed(3),
                valStr ? '<br/>' + label + '<b>' + valStr + '</b>' : '',
                '</div>',
            ].join(''),
            style: {
                backgroundColor: isDark ? '#1f2937' : '#ffffff',
                border:          '1px solid ' + (isDark ? '#4b5563' : '#d1d5db'),
                borderRadius:    '5px',
                padding:         '5px 9px',
                color:           isDark ? '#e5e7eb' : '#1f2937',
                fontSize:        '11px',
                maxWidth:        '220px',
                zIndex:          9999,
                pointerEvents:   'none',
                boxShadow:       '0 2px 8px rgba(0,0,0,0.15)',
            }
        };
    };

    // ------------------------------------------------------------------
    // _makeLayer – build a ScatterplotLayer from typed arrays
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype._makeLayer = function (positions, colors, id) {
        const n       = positions.length / 2;
        const isAnim  = id === 'cells-anim';
        return new deck.ScatterplotLayer({
            id:             id || 'cells',
            data: {
                length:     n,
                attributes: {
                    getPosition:  { value: positions, size: 2 },
                    getFillColor: { value: colors,    size: 4 },
                },
            },
            getRadius:      this._radius,
            radiusUnits:    'pixels',
            opacity:        this._opacity,
            pickable:       !isAnim,
            autoHighlight:  !isAnim,
            highlightColor: [255, 215, 0, 230],
            stroked:        false,
            filled:         true,
            updateTriggers: { getFillColor: [colors], getRadius: [this._radius] },
        });
    };

    /**
     * updateStyle – instantly apply new radius and opacity without re-fetching data.
     * Called by the point-size / opacity sliders.
     */
    DeckGLRenderer.prototype.updateStyle = function (radius, opacity) {
        this._radius  = radius;
        this._opacity = opacity;
        if (!this._positions || !this._colors) return;  // no data yet
        // Rebuild layer with the same data but new visual properties
        this.deckgl.setProps({ layers: [this._makeLayer(this._positions, this._colors)] });
    };

    // ------------------------------------------------------------------
    // _fitView – zoom/pan to show all points
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype._fitView = function (positions) {
        const n = positions.length / 2;
        if (n === 0) return;

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < n; i++) {
            const x = positions[i * 2], y = positions[i * 2 + 1];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
        const cx    = (minX + maxX) / 2;
        const cy    = (minY + maxY) / 2;
        const w     = this.canvas ? this.canvas.clientWidth  : 800;
        const h     = this.canvas ? this.canvas.clientHeight : 500;
        const xSpan = maxX - minX || 1;
        const ySpan = maxY - minY || 1;
        const zoom  = Math.log2(Math.min(w * 0.88 / xSpan, h * 0.88 / ySpan));

        this._viewState = {
            target:  [cx, cy, 0],
            zoom:    zoom,
            minZoom: -20,
            maxZoom: 40,
        };
        this.deckgl.setProps({ viewState: this._viewState });
    };

    // ------------------------------------------------------------------
    // _cancelAnim – stop any in-progress animation
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype._cancelAnim = function () {
        if (this._animReq !== null) {
            cancelAnimationFrame(this._animReq);
            this._animReq = null;
        }
    };

    // ------------------------------------------------------------------
    // setData – immediate update (no animation)
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype.setData = function (positions, colors, meta, hoverValues) {
        this._cancelAnim();
        this._positions   = positions;
        this._colors      = colors;
        this._meta        = meta;
        this._hoverValues = hoverValues;
        this._n           = positions.length / 2;

        this.deckgl.setProps({ layers: [this._makeLayer(positions, colors)] });
        this._fitView(positions);
    };

    // ------------------------------------------------------------------
    // animateToPositions – smooth embedding switch  (requestAnimationFrame lerp)
    // Uses a spatial sample during animation, snaps to full data at end.
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype.animateToPositions = function (
        newPositions, newColors, meta, hoverValues, duration
    ) {
        duration = duration || 600;

        const oldPositions = this._positions || new Float32Array(newPositions.length);
        const n            = Math.min(oldPositions.length, newPositions.length) / 2;

        // Sampled subset for animation (keeps CPU lerp cheap even for 1 M cells)
        const SAMPLE = Math.min(60000, n);
        const step   = Math.max(1, Math.floor(n / SAMPLE));
        const idx    = [];
        for (let i = 0; i < n; i += step) idx.push(i);
        if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);

        const ns      = idx.length;
        const animPos = new Float32Array(ns * 2);
        const animCol = new Uint8Array(ns * 4);

        // Sample new colors for the animation trace
        for (let k = 0; k < ns; k++) {
            const i = idx[k];
            animCol[k * 4]     = newColors[i * 4];
            animCol[k * 4 + 1] = newColors[i * 4 + 1];
            animCol[k * 4 + 2] = newColors[i * 4 + 2];
            animCol[k * 4 + 3] = newColors[i * 4 + 3];
        }

        this._cancelAnim();
        const startTime = performance.now();
        const self      = this;

        const animate = function () {
            const t     = Math.min(1.0, (performance.now() - startTime) / duration);
            const eased = easeInOutCubic(t);

            for (let k = 0; k < ns; k++) {
                const i = idx[k];
                animPos[k * 2]     = oldPositions[i * 2]     + (newPositions[i * 2]     - oldPositions[i * 2])     * eased;
                animPos[k * 2 + 1] = oldPositions[i * 2 + 1] + (newPositions[i * 2 + 1] - oldPositions[i * 2 + 1]) * eased;
            }

            self.deckgl.setProps({
                layers: [self._makeLayer(new Float32Array(animPos), animCol, 'cells-anim')],
            });

            if (t < 1.0) {
                self._animReq = requestAnimationFrame(animate);
            } else {
                // Animation done — switch to full dataset
                self._positions   = newPositions;
                self._colors      = newColors;
                self._meta        = meta;
                self._hoverValues = hoverValues;
                self._n           = newPositions.length / 2;
                self.deckgl.setProps({ layers: [self._makeLayer(newPositions, newColors)] });
                self._fitView(newPositions);
                self._animReq = null;
            }
        };

        this._animReq = requestAnimationFrame(animate);
    };

    // ------------------------------------------------------------------
    // animateToColors – smooth color fade (only for n ≤ 300 K, else snap)
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype.animateToColors = function (
        newColors, meta, hoverValues, duration
    ) {
        const n = this._n;
        // Skip animation for very large datasets (CPU lerp of 4M bytes at 60fps is OK
        // but let's keep it safe on older Macs with ≤ 300K threshold).
        if (n > 300000 || !this._colors) {
            this._colors      = newColors;
            this._meta        = meta;
            this._hoverValues = hoverValues;
            this.deckgl.setProps({ layers: [this._makeLayer(this._positions, newColors)] });
            return;
        }

        duration           = duration || 350;
        const positions    = this._positions;
        const oldColors    = this._colors.slice();  // snapshot
        const currentColors = new Uint8Array(n * 4);

        this._cancelAnim();
        const startTime = performance.now();
        const self      = this;

        const animate = function () {
            const t     = Math.min(1.0, (performance.now() - startTime) / duration);
            const eased = easeInOutCubic(t);

            for (let i = 0; i < n * 4; i++) {
                currentColors[i] = (oldColors[i] + (newColors[i] - oldColors[i]) * eased + 0.5) | 0;
            }

            self.deckgl.setProps({
                layers: [self._makeLayer(positions, currentColors)],
            });

            if (t < 1.0) {
                self._animReq = requestAnimationFrame(animate);
            } else {
                self._colors      = newColors;
                self._meta        = meta;
                self._hoverValues = hoverValues;
                self._animReq     = null;
            }
        };

        this._animReq = requestAnimationFrame(animate);
    };

    // ------------------------------------------------------------------
    // show / hide / destroy
    // ------------------------------------------------------------------
    DeckGLRenderer.prototype.show = function () {
        if (this.canvas) this.canvas.style.display = 'block';
        const wrap = document.getElementById('deckgl-wrap');
        if (wrap) wrap.style.display = 'block';
    };

    DeckGLRenderer.prototype.hide = function () {
        if (this.canvas) this.canvas.style.display = 'none';
        const wrap = document.getElementById('deckgl-wrap');
        if (wrap) wrap.style.display = 'none';
    };

    DeckGLRenderer.prototype.destroy = function () {
        this._cancelAnim();
        if (this.deckgl) {
            this.deckgl.finalize();
            this.deckgl = null;
        }
        if (this.canvas && this.canvas.parentElement) {
            this.canvas.parentElement.removeChild(this.canvas);
        }
        this.canvas       = null;
        this._positions   = null;
        this._colors      = null;
        this._hoverValues = null;
        this._meta        = null;
        this._n           = 0;
    };

    // -------------------------------------------------------------------------
    // Exports
    // -------------------------------------------------------------------------
    global.DeckGLRenderer      = DeckGLRenderer;
    global.parsePlotGPUBuffer   = parsePlotGPUBuffer;
    global.parseColorOnlyBuffer = parseColorOnlyBuffer;

})(window);
