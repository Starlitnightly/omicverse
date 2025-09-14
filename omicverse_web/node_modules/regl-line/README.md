# Regl Line

[![npm version](https://img.shields.io/npm/v/regl-line.svg?color=7f99ff&style=flat-square)](https://www.npmjs.com/package/regl-line)
[![build status](https://img.shields.io/github/actions/workflow/status/flekschas/regl-line/build.yml?branch=master&color=a17fff&style=flat-square)](https://github.com/flekschas/regl-line/actions?query=workflow%3Abuild)
[![gzipped size](https://img.badgesize.io/https:/unpkg.com/regl-line/dist/regl-line.min.js?color=e17fff&compression=gzip&style=flat-square)](https://bundlephobia.com/result?p=regl-line)
[![code style prettier](https://img.shields.io/badge/code_style-prettier-80a1ff.svg?style=flat-square)](https://github.com/prettier/prettier)
[![regl-line demo](https://img.shields.io/badge/demo-online-f264ab.svg?style=flat-square)](https://flekschas.github.io/regl-line/)

> A regl function to conveniently draw flat 2D and 3D lines.

<p align="center">
  <img src="https://flekschas.github.io/regl-line/teaser.gif" />
</p>

<p align="center">
  <a href="https://flekschas.github.io/regl-line/">Click here to see ☝️ in action!</a>
</p>

This small library is inspired by [Regl's line example](http://regl.party/examples?line) and Matt Deslauriers' [wonderful blog post on drawing lines in WebGL](https://mattdesl.svbtle.com/drawing-lines-is-hard).

## Install

```
npm -i regl-line
```

## Getting started

```javascript
import createRegl from 'regl';
import createCamera from 'canvas-orbit-camera';
import createLine from 'regl-line';

// Setup the canvas
const canvas = document.getElementById('canvas');
const { width, height } = canvas.getBoundingClientRect();
canvas.width = width * resize.scale;
canvas.height = height * resize.scale;

// Setup Regl
const regl = createRegl(canvas);
const camera = createCamera(canvas);

// Create a line
const line = createLine(regl, {
  width: 2,
  color: [0.8, 0.2, 0.0, 1.0],
  is2d: true,
  // Flat list of normalized-device coordinates
  points: [-0.9, +0.9, +0.9, +0.9, +0.9, -0.9, -0.9, -0.9, -0.9, +0.85],
});

// Draw
regl.frame(() => {
  regl.clear({ color: [0, 0, 0, 1], depth: 1 });
  camera.tick();
  line.draw({ view: camera.view() });
});
```

For a complete example, see [example/index.js](example/index.js).

### Draw Multiple Lines At Once

To draw multiple lines, you can pass a list of lists of flat point coordinates to `setPoints()` or the constructor.

```javascript
line.setPoints([
  [-0.8, +0.9, +0.8, +0.9], // top line
  [+0.9, +0.8, +0.9, -0.8], // right line
  [+0.8, -0.9, -0.8, -0.9], // bottom line
  [-0.9, -0.8, -0.9, +0.8], // left line
]);
```

### Variable Line Color

To give each line an individual color, you have to do 2 things. First, you have to specify **all** the colors you plan to use.

```javascript
line.setStyle({
  color: [
    [0, 1, 1, 1], // cyan
    [1, 1, 0, 1], // yellow
  ],
});
```

Next, when you set the points (with `setPoints()`), specify an array of indices to associate lines with your previously specified colors.

```javascript
line.setPoints(points, {
  colorIndices: [
    0, //    top line will be cyan
    1, //  right line will be yellow
    0, // bottom line will be cyan
    1, //   left line will be yellow
  ],
});
```

#### Color Gradient

You could even go one step further and specify the color for each point on the line using a list of list of indices.

```javascript
line.setPoints(points, {
  colorIndices: [
    [0, 0, 1, 1], //    top line will have a cyan to yellow gradient
    [1, 1, 0, 0], //  right line will have a yellow to cyan gradient
    [0, 1, 0, 1], // bottom line will have a cyan, yellow, cyan, yellow gradient
    [0, 1, 1, 0], //   left line will have a cyan, yellow, cyan gradient
  ],
});
```

### Variable Line Opacity

To adjust, you can adjust the line width using

```javascript
line.setPoints(points, {
  opacities: [
    0.25, //    top line will have an opacity of 0.25
    0.5, //  right line will have an opacity of 0.5
    0.75, // bottom line will have an opacity of 0.75
    1.0, //   left line will have an opacity of 1.0
  ],
});
```

Similar to [color gradient](#color-gradient), you can also specify the opacity for each point on the line using a list of list of numbers.

### Variable Line Width

To adjust, you can adjust the line width using

```javascript
line.setPoints(points, {
  widths: [
    1, //    top line will have a width of 1
    2, //  right line will have a width of 2
    3, // bottom line will have a width of 3
    4, //   left line will have a width of 4
  ],
});
```

Similar to [color gradient](#color-gradient), you can also specify the width for each point on the line using a list of list of numbers.

## API

### Constructor

<a name="createLine" href="#createLine">#</a> <b>createLine</b>(<i>regl</i>, <i>options = {}</i>)

Create a line instance.

Args:

1. `regl` [regl]: Regl instance to be used for drawing the line.
2. `options` [object]: An object with the following props to customize the line creator.
   - `projection` [[mat4](http://glmatrix.net/docs/module-mat4.html)]: projection matrix (Defaut: _identity matrix_)
   - `model` [[mat4](http://glmatrix.net/docs/module-mat4.html)]: model matrix (Defaut: _identity matrix_)
   - `view` [[mat4](http://glmatrix.net/docs/module-mat4.html)]: view matrix (Defaut: _identity matrix_)
   - `points` [array]: flat list of normalized-device coordinates alternating x,y if `is2d` is `true` or x,y,z. (Defaut: `[]`). To draw multiple lines at once, pass in a list of lists of coordinates.
   - `widths` [array]: flat array of point-wise widths, i.e., the line width at every point. (Defaut: `[]`)
   - `color` [array]: a quadruple of floats (RGBA) ranging in [0,1] defining the color of the line. (Defaut: `[0.8, 0.5, 0, 1]`)
   - `width` [number]: uniform line width scalar. This number sets the base line width. (Defaut: `1`)
   - `miter` [boolean]: if `true` line segments are [miter joined](https://en.wikipedia.org/wiki/Miter_joint). (Defaut: `true`)
   - `is2d` [boolean]: if `true` points are expected to have only x,y coordinates otherwise x,y,z coordinates are expected. (Defaut: `false`)
   - `zPos2d` [number]: if `is2d` is `true` this value defines the uniform z coordinate. (Defaut: `0`)

Returns: `line` instance.

### Methods

<a name="line.clear" href="#line.clear">#</a> line.<b>clear</b>()

Clears all of the data to remove the drawn line.

<a name="line.destroy" href="#line.destroy">#</a> line.<b>destroy</b>()

Destroys all related objects to free memory.

<a name="line.draw" href="#line.draw">#</a> line.<b>draw</b>({ <i>projection</i>, <i>model</i>, <i>view</i> })

Draws the line according to the `projection`, `model`, and `view` matrices.

Args:

1 `options` [object]:

- `projection` [[mat4](http://glmatrix.net/docs/module-mat4.html)]: projection matrix (Defaut: _identity matrix_)
- `model` [[mat4](http://glmatrix.net/docs/module-mat4.html)]: model matrix (Defaut: _identity matrix_)
- `view` [[mat4](http://glmatrix.net/docs/module-mat4.html)]: view matrix (Defaut: _identity matrix_)

<a name="line.getBuffer" href="#line.getBuffer">#</a> line.<b>getBuffer</b>()

Get a reference to the point, width, and color index [buffer objects](http://regl.party/api#buffers). This can be useful for efficient animations.

Returns: `{ points, widths, colorIndices }`

<a name="line.getData" href="#line.getData">#</a> line.<b>getData</b>()

Get a reference to the buffers' [typed data arrays](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Typed_arrays).

Returns: `{ points, widths, colorIndices }`

<a name="line.getPoints" href="#line.getPoints">#</a> line.<b>getPoints</b>()

Get the original list of points defining the line.

Return: flat `array` of points

<a name="line.getStyle" href="#line.getStyle">#</a> line.<b>getStyle</b>()

Get all the style settings.

Returns: `{ color, miter, width }`

<a name="line.setPoints" href="#line.setPoints">#</a> line.<b>setPoints</b>(<i>points</i>, <i>widths</i>, <i>is2d</i>)

Set points defining the line, the point-wise widths, and change the dimensionality.

Args:

1. `points` [array]: flat list of normalized-device coordinates alternating x,y if `is2d` is `true` or x,y,z. To draw multiple lines at once, pass in a list of lists of coordinates.
2. `widths` [array]: flat array of point-wise widths, i.e., the line width at every point.
3. `is2d` [boolean]: if `true` points are expected to have only x,y coordinates otherwise x,y,z coordinates are expected.

<a name="line.setStyle" href="#line.setStyle">#</a> line.<b>setStyle</b>({ <i>color</i>, <i>miter</i>, <i>width</i> })

Args:

1. `option` [object]:
   - `color` [array]: a quadruple of floats (RGBA) ranging in [0,1] defining the color of the line.
   - `width` [number]: uniform line width scalar. This number sets the base line width.
   - `miter` [boolean]: if `true` line segments are [miter joined](https://en.wikipedia.org/wiki/Miter_joint).
