import createOriginalRegl from 'regl';

import {
  DEFAULT_IMAGE_LOAD_TIMEOUT,
  GL_EXTENSIONS,
  IMAGE_LOAD_ERROR,
  W_NAMES,
  Z_NAMES,
} from './constants.js';

/**
 * Get the max value of an array. helper method to be used with `Array.reduce()`.
 * @param   {number}  max  Accumulator holding the max value.
 * @param   {number}  x  Current value.
 * @return  {number}  Max value.
 */
export const arrayMax = (max, x) => (max > x ? max : x);

/**
 * Check if all GL extensions are supported and enabled and warn otherwise
 * @param   {import('regl').Regl}  regl  Regl instance to be tested
 * @param   {boolean}  silent  If `true` the function will not print `console.warn` statements
 * @return  {boolean}  If `true` all required GL extensions are supported
 */
export const checkReglExtensions = (regl, silent) => {
  if (!regl) {
    return false;
  }
  return GL_EXTENSIONS.reduce((every, extension) => {
    if (!regl.hasExtension(extension)) {
      if (!silent) {
        // biome-ignore lint/suspicious/noConsole: This is a legitimately useful warning
        console.warn(
          `WebGL: ${extension} extension not supported. Scatterplot might not render properly`,
        );
      }
      return false;
    }
    return every;
  }, true);
};

/**
 * Create a new Regl instance with `GL_EXTENSIONS` enables
 * @param   {HTMLCanvasElement}  canvas  Canvas element to be rendered on
 * @return  {import('regl').Regl}  New Regl instance
 */
export const createRegl = (canvas) => {
  const gl = canvas.getContext('webgl', {
    antialias: true,
    preserveDrawingBuffer: true,
  });
  const extensions = [];

  // Needed to run the tests properly as the headless-gl doesn't support all
  // extensions, which is fine for the functional tests.
  for (const extension of GL_EXTENSIONS) {
    if (gl.getExtension(extension)) {
      extensions.push(extension);
    } else {
      // biome-ignore lint/suspicious/noConsole: This is a legitimately useful warning
      console.warn(
        `WebGL: ${extension} extension not supported. Scatterplot might not render properly`,
      );
    }
  }

  return createOriginalRegl({ gl, extensions });
};

/**
 * L2 distance between a pair of 2D points
 * @param   {number}  x1  X coordinate of the first point
 * @param   {number}  y1  Y coordinate of the first point
 * @param   {number}  x2  X coordinate of the second point
 * @param   {number}  y2  Y coordinate of the first point
 * @return  {number}  L2 distance
 */
export const dist = (x1, y1, x2, y2) =>
  Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);

/**
 * Get the bounding box of a set of 2D positions
 * @param   {array}  positions2d  2D positions to be checked
 * @return  {array}  Quadruple of form `[xMin, yMin, xMax, yMax]` defining the
 *  bounding box
 */
// biome-ignore lint/style/useNamingConvention: BBox stands for BoundingBox
export const getBBox = (positions2d) => {
  let xMin = Number.POSITIVE_INFINITY;
  let xMax = Number.NEGATIVE_INFINITY;
  let yMin = Number.POSITIVE_INFINITY;
  let yMax = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < positions2d.length; i += 2) {
    xMin = positions2d[i] < xMin ? positions2d[i] : xMin;
    xMax = positions2d[i] > xMax ? positions2d[i] : xMax;
    yMin = positions2d[i + 1] < yMin ? positions2d[i + 1] : yMin;
    yMax = positions2d[i + 1] > yMax ? positions2d[i + 1] : yMax;
  }

  return [xMin, yMin, xMax, yMax];
};

/**
 * Test whether a bounding box is actually specifying an area
 * @param   {array}  bBox  The bounding box to be checked
 * @return  {array}  `true` if the bounding box is valid
 */
// biome-ignore lint/style/useNamingConvention: BBox stands for BoundingBox
export const isValidBBox = ([xMin, yMin, xMax, yMax]) =>
  Number.isFinite(xMin) &&
  Number.isFinite(yMin) &&
  Number.isFinite(xMax) &&
  Number.isFinite(yMax) &&
  xMax - xMin > 0 &&
  yMax - yMin > 0;

const REGEX_HEX_TO_RGB = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;

/**
 * Convert a HEX-encoded color to an RGB-encoded color
 * @param   {string}  hex  HEX-encoded color string.
 * @param   {boolean}  isNormalize  If `true` the returned RGB values will be
 *   normalized to `[0,1]`.
 * @return  {array}  Triple holding the RGB values.
 */
export const hexToRgb = (hex, isNormalize = false) =>
  hex
    .replace(REGEX_HEX_TO_RGB, (_m, r, g, b) => `#${r}${r}${g}${g}${b}${b}`)
    .substring(1)
    .match(/.{2}/g)
    .map((x) => Number.parseInt(x, 16) / 255 ** isNormalize);

export const isConditionalArray = (a, condition, { minLength = 0 } = {}) =>
  Array.isArray(a) && a.length >= minLength && a.every(condition);

export const isPositiveNumber = (x) => !Number.isNaN(+x) && +x >= 0;

export const isStrictlyPositiveNumber = (x) => !Number.isNaN(+x) && +x > 0;

/**
 * Create a function to limit choices to a predefined list
 * @param   {array}  choices  Array of acceptable choices
 * @param   {*}  defaultOption  Default choice
 * @return  {function}  Function limiting the choices
 */
export const limit = (choices, defaultChoice) => (choice) =>
  choices.indexOf(choice) >= 0 ? choice : defaultChoice;

/**
 * Promised-based image loading
 * @param {string}  src  Remote image source, i.e., a URL
 * @param {boolean} isCrossOrigin If `true` allow loading image from a source of another origin.
 * @return  {Promise<HTMLImageElement>}  Promise resolving to the image once its loaded
 */
export const loadImage = (
  src,
  isCrossOrigin = false,
  timeout = DEFAULT_IMAGE_LOAD_TIMEOUT,
) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    if (isCrossOrigin) {
      image.crossOrigin = 'anonymous';
    }
    image.src = src;
    image.onload = () => {
      resolve(image);
    };
    const rejectPromise = () => {
      reject(new Error(IMAGE_LOAD_ERROR));
    };
    image.onerror = rejectPromise;
    setTimeout(rejectPromise, timeout);
  });

/**
 * @deprecated Please use `scatterplot.createTextureFromUrl(url)`
 *
 * Create a Regl texture from an URL.
 * @param   {import('regl').Regl}  regl  Regl instance used for creating the texture.
 * @param   {string}  url  Source URL of the image.
 * @return  {Promise<import('regl').Texture2D>}  Promise resolving to the texture object.
 */
export const createTextureFromUrl = (
  regl,
  url,
  timeout = DEFAULT_IMAGE_LOAD_TIMEOUT,
) =>
  new Promise((resolve, reject) => {
    loadImage(
      url,
      url.indexOf(window.location.origin) !== 0 && url.indexOf('base64') === -1,
      timeout,
    )
      .then((image) => {
        resolve(regl.texture(image));
      })
      .catch((error) => {
        reject(error);
      });
  });

/**
 * Convert a HEX-encoded color to an RGBA-encoded color
 * @param   {string}  hex  HEX-encoded color string.
 * @param   {boolean}  isNormalize  If `true` the returned RGBA values will be
 *   normalized to `[0,1]`.
 * @return  {array}  Triple holding the RGBA values.
 */
export const hexToRgba = (hex, isNormalize = false) => [
  ...hexToRgb(hex, isNormalize),
  255 ** !isNormalize,
];

const REGEX_IS_HEX = /(^#[0-9A-F]{6}$)|(^#[0-9A-F]{3}$)/i;

/**
 * Tests if a string is a valid HEX color encoding
 * @param   {string}  hex  HEX-encoded color string.
 * @return  {boolean}  If `true` the string is a valid HEX color encoding.
 */
export const isHex = (hex) => REGEX_IS_HEX.test(hex);

/**
 * Tests if a number is in `[0,1]`.
 * @param   {number}  x  Number to be tested.
 * @return  {boolean}  If `true` the number is in `[0,1]`.
 */
export const isNormFloat = (x) => x >= 0 && x <= 1;

/**
 * Tests if an array consist of normalized numbers that are in `[0,1]` only.
 * @param   {array}  a  Array to be tested
 * @return  {boolean}  If `true` the array contains only numbers in `[0,1]`.
 */
export const isNormFloatArray = (a) => Array.isArray(a) && a.every(isNormFloat);

/**
 * Computes the cross product to determine the orientation of three points
 * @param   {number}  x1  X-coordinate of first point
 * @param   {number}  y1  Y-coordinate of first point
 * @param   {number}  x2  X-coordinate of second point
 * @param   {number}  y2  Y-coordinate of second point
 * @param   {number}  px  X-coordinate of test point
 * @param   {number}  py  Y-coordinate of test point
 * @return  {number}  Positive if counterclockwise, negative if clockwise
 */
function crossProduct(x1, y1, x2, y2, px, py) {
  return (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1);
}

/**
 * Determines if a point lies within a polygon using the non-zero winding rule.
 * This handles self-intersecting polygons and overlapping areas correctly.
 * @param   {Array}  polygon  1D list of vertices defining the polygon [x1,y1,x2,y2,...]
 * @param   {Array}  point    Tuple of the form [x,y] to be tested
 * @return  {boolean} True if point lies within the polygon
 */
export const isPointInPolygon = (polygon, [px, py] = []) => {
  let winding = 0;

  for (let i = 0, j = polygon.length - 2; i < polygon.length; i += 2) {
    const x1 = polygon[i];
    const y1 = polygon[i + 1];
    const x2 = polygon[j];
    const y2 = polygon[j + 1];

    if (y1 <= py) {
      if (y2 > py) {
        const orientation = crossProduct(x1, y1, x2, y2, px, py);
        if (orientation > 0) {
          winding++;
        }
      }
    } else if (y2 <= py) {
      const orientation = crossProduct(x1, y1, x2, y2, px, py);
      if (orientation < 0) {
        winding--;
      }
    }

    j = i;
  }

  return winding !== 0;
};

/**
 * Tests if a variable is a string
 * @param   {*}  s  Variable to be tested
 * @return  {boolean}  If `true` variable is a string
 */
export const isString = (s) => typeof s === 'string' || s instanceof String;

/**
 * Tests if a number is an interger and in `[0,255]`.
 * @param   {number}  x  Number to be tested.
 * @return  {boolean}  If `true` the number is an interger and in `[0,255]`.
 */
export const isUint8 = (x) => Number.isInteger(x) && x >= 0 && x <= 255;

/**
 * Tests if an array consist of Uint8 numbers only.
 * @param   {array}  a  Array to be tested.
 * @return  {boolean}  If `true` the array contains only Uint8 numbers.
 */
export const isUint8Array = (a) => Array.isArray(a) && a.every(isUint8);

/**
 * Tests if an array is encoding an RGB color.
 * @param   {array}  rgb  Array to be tested
 * @return  {boolean}  If `true` the array hold a triple of Uint8 numbers or
 *   a triple of normalized floats.
 */
export const isRgb = (rgb) =>
  rgb.length === 3 && (isNormFloatArray(rgb) || isUint8Array(rgb));

/**
 * Tests if an array is encoding an RGBA color.
 * @param   {array}  rgb  Array to be tested
 * @return  {boolean}  If `true` the array hold a quadruple of Uint8 numbers or
 *   a quadruple of normalized floats.
 */
export const isRgba = (rgba) =>
  rgba.length === 4 && (isNormFloatArray(rgba) || isUint8Array(rgba));

/**
 * Test if a color is multiple colors
 * @param   {*}  color  To be tested
 * @return  {boolean}  If `true`, `color` is an array of colors.
 */
export const isMultipleColors = (color) =>
  Array.isArray(color) &&
  color.length > 0 &&
  (Array.isArray(color[0]) || isString(color[0]));

/**
 * Test if two arrays contain the same primitive values
 */
export const isSameElements = (a, b) =>
  Array.isArray(a) && Array.isArray(b) && a.every((value, i) => value === b[i]);

/**
 * Test if two arrays contain the same RGBA quadruples
 */
export const isSameRgbas = (a, b) => {
  if (!(Array.isArray(a) && Array.isArray(b)) || a.length !== b.length) {
    return false;
  }

  if (a.length === 0) {
    return true;
  }

  // We need to test whether a and b are arrays of RGBA quadruples
  if (!(Array.isArray(a[0]) && Array.isArray(b[0]))) {
    return false;
  }

  return a.every(([r1, g1, b1, a1], i) => {
    const [r2, g2, b2, a2] = b[i];
    return r1 === r2 && g1 === g2 && b1 === b2 && a1 === a2;
  });
};

/**
 * Fast version of `Math.max`. Based on
 *   https://jsperf.com/math-min-max-vs-ternary-vs-if/24 `Math.max` is not
 *   very fast
 * @param   {number}  a  Value A
 * @param   {number}  b  Value B
 * @return  {boolean}  If `true` A is greater than B.
 */
export const max = (a, b) => (a > b ? a : b);

/**
 * Fast version of `Math.min`. Based on
 *   https://jsperf.com/math-min-max-vs-ternary-vs-if/24 `Math.max` is not
 *   very fast
 * @param   {number}  a  Value A
 * @param   {number}  b  Value B
 * @return  {boolean}  If `true` A is smaller than B.
 */
export const min = (a, b) => (a < b ? a : b);

/**
 * Normalize an array
 * @param   {array}  a  Array to be normalized.
 * @return  {array}  Normalized array.
 */
export const normNumArray = (a) =>
  a.map((x) => x / a.reduce(arrayMax, Number.NEGATIVE_INFINITY));

/**
 * Convert a color to an RGBA color
 * @param   {*}  color  Color to be converted. Currently supports:
 *   HEX, RGB, or RGBA.
 * @param   {boolean}  isNormalize  If `true` the returned RGBA values will be
 *   normalized to `[0,1]`.
 * @return  {array}  Quadruple defining an RGBA color.
 */
export const toRgba = (color, shouldNormalize) => {
  if (isRgba(color)) {
    const isNormalized = isNormFloatArray(color);
    if (
      (shouldNormalize && isNormalized) ||
      !(shouldNormalize || isNormalized)
    ) {
      return color;
    }
    if (shouldNormalize && !isNormalized) {
      return color.map((x) => x / 255);
    }
    return color.map((x) => x * 255);
  }

  if (isRgb(color)) {
    const base = 255 ** !shouldNormalize;
    const isNormalized = isNormFloatArray(color);

    if (
      (shouldNormalize && isNormalized) ||
      !(shouldNormalize || isNormalized)
    ) {
      return [...color, base];
    }
    if (shouldNormalize && !isNormalized) {
      return [...color.map((x) => x / 255), base];
    }
    return [...color.map((x) => x * 255), base];
  }

  if (isHex(color)) {
    return hexToRgba(color, shouldNormalize);
  }

  // biome-ignore lint/suspicious/noConsole: This is a legitimately useful warning
  console.warn(
    'Only HEX, RGB, and RGBA are handled by this function. Returning white instead.',
  );
  return shouldNormalize ? [1, 1, 1, 1] : [255, 255, 255, 255];
};

/**
 * Flip the key-value pairs of an object
 * @param {object} obj - Object to be flipped
 * @return {object} Flipped object
 */
export const flipObj = (obj) =>
  Object.entries(obj).reduce((out, [key, value]) => {
    if (out[value]) {
      out[value] = [...out[value], key];
    } else {
      out[value] = key;
    }
    return out;
  }, {});

export const rgbBrightness = (rgb) =>
  0.21 * rgb[0] + 0.72 * rgb[1] + 0.07 * rgb[2];

/**
 * Clip a number between min and max
 * @param {number} value - The value to be clipped
 * @param {number} minValue - The minimum value
 * @param {number} maxValue - The maximum value
 * @return {number} The clipped value
 */
export const clip = (value, minValue, maxValue) =>
  Math.min(maxValue, Math.max(minValue, value));

/**
 * Convert object- or array-oriented points to array-oriented points
 * @param {import('./types').Points} points - The point data
 * @return {number[][]} Array-oriented points
 */
export const toArrayOrientedPoints = (points) =>
  new Promise((resolve, reject) => {
    if (!points || Array.isArray(points)) {
      resolve(points);
    } else {
      const length =
        Array.isArray(points.x) || ArrayBuffer.isView(points.x)
          ? points.x.length
          : 0;

      const getX =
        (Array.isArray(points.x) || ArrayBuffer.isView(points.x)) &&
        ((i) => points.x[i]);
      const getY =
        (Array.isArray(points.y) || ArrayBuffer.isView(points.y)) &&
        ((i) => points.y[i]);
      const getL =
        (Array.isArray(points.line) || ArrayBuffer.isView(points.line)) &&
        ((i) => points.line[i]);

      // biome-ignore lint/style/useNamingConvention: LO stands for line and order
      const getLO =
        (Array.isArray(points.lineOrder) ||
          ArrayBuffer.isView(points.lineOrder)) &&
        ((i) => points.lineOrder[i]);

      const components = Object.keys(points);
      const getZ = (() => {
        const z = components.find((c) => Z_NAMES.has(c));
        return (
          z &&
          (Array.isArray(points[z]) || ArrayBuffer.isView(points[z])) &&
          ((i) => points[z][i])
        );
      })();
      const getW = (() => {
        const w = components.find((c) => W_NAMES.has(c));
        return (
          w &&
          (Array.isArray(points[w]) || ArrayBuffer.isView(points[w])) &&
          ((i) => points[w][i])
        );
      })();

      if (getX && getY && getZ && getW && getL && getLO) {
        resolve(
          points.x.map((x, i) => [
            x,
            getY(i),
            getZ(i),
            getW(i),
            getL(i),
            getLO(i),
          ]),
        );
      } else if (getX && getY && getZ && getW && getL) {
        resolve(
          Array.from({ length }, (_, i) => [
            getX(i),
            getY(i),
            getZ(i),
            getW(i),
            getL(i),
          ]),
        );
      } else if (getX && getY && getZ && getW) {
        resolve(
          Array.from({ length }, (_, i) => [
            getX(i),
            getY(i),
            getZ(i),
            getW(i),
          ]),
        );
      } else if (getX && getY && getZ) {
        resolve(Array.from({ length }, (_, i) => [getX(i), getY(i), getZ(i)]));
      } else if (getX && getY) {
        resolve(Array.from({ length }, (_, i) => [getX(i), getY(i)]));
      } else {
        reject(new Error('You need to specify at least x and y'));
      }
    }
  });

export const isHorizontalLine = (annotation) =>
  Number.isFinite(annotation.y) && !('x' in annotation);

export const isVerticalLine = (annotation) =>
  Number.isFinite(annotation.x) && !('y' in annotation);

export const isDomRect = (annotation) =>
  Number.isFinite(annotation.x) &&
  Number.isFinite(annotation.y) &&
  Number.isFinite(annotation.width) &&
  Number.isFinite(annotation.height);

export const isRect = (annotation) =>
  Number.isFinite(annotation.x1) &&
  Number.isFinite(annotation.y1) &&
  Number.isFinite(annotation.x2) &&
  Number.isFinite(annotation.x2);

export const isPolygon = (annotation) =>
  'vertices' in annotation && annotation.vertices.length > 1;

export const insertionSort = (array) => {
  const end = array.length;
  for (let i = 1; i < end; i++) {
    // Choosing the first element in our unsorted subarray
    const current = array[i];
    // The last element of our sorted subarray
    let j = i - 1;
    while (j > -1 && current < array[j]) {
      array[j + 1] = array[j];
      j--;
    }
    array[j + 1] = current;
  }
  return array;
};
