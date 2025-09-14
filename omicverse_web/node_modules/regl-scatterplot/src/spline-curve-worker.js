/* eslint-env worker */
/* eslint no-restricted-globals: 1 */

const worker = function worker() {
  const state = {};

  /**
   * Catmull-Rom interpolation
   * @param {number} t - Progress value
   * @param {array} p0 - First point
   * @param {array} p1 - Second point
   * @param {array} p2 - Third point
   * @param {array} p3 - Forth point
   * @return {number} Interpolated value
   */
  const catmullRom = (t, p0, p1, p2, p3) => {
    const v0 = (p2 - p0) * 0.5;
    const v1 = (p3 - p1) * 0.5;
    return (
      (2 * p1 - 2 * p2 + v0 + v1) * t * t * t +
      (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t * t +
      v0 * t +
      p1
    );
  };

  /**
   * Interpolate a point with Catmull-Rom
   * @param {number} t - Progress value
   * @param {array} points - Key points
   * @param {number}  maxPointIdx - Highest point index. Same as array.length - 1
   * @return {array} Interpolated point
   */
  const interpolatePoint = (t, points, maxPointIdx) => {
    const p = maxPointIdx * t;

    const intPoint = Math.floor(p);
    const weight = p - intPoint;

    const p0 = points[Math.max(0, intPoint - 1)];
    const p1 = points[intPoint];
    const p2 = points[Math.min(maxPointIdx, intPoint + 1)];
    const p3 = points[Math.min(maxPointIdx, intPoint + 2)];

    return [
      catmullRom(weight, p0[0], p1[0], p2[0], p3[0]),
      catmullRom(weight, p0[1], p1[1], p2[1], p3[1]),
    ];
  };

  /**
   * Square distance
   * @param {number} x1 - First x coordinate
   * @param {number} y1 - First y coordinate
   * @param {number} x2 - Second x coordinate
   * @param {number} y2 - Second y coordinate
   * @return {number} Distance
   */
  const sqDist = (x1, y1, x2, y2) => (x1 - x2) ** 2 + (y1 - y2) ** 2;

  /**
   * Douglas Peucker square segment distance
   * Implementation from https://github.com/mourner/simplify-js
   * @author Vladimir Agafonkin
   * @copyright Vladimir Agafonkin 2013
   * @license BSD
   * @param {array} p - Point
   * @param {array} p1 - First boundary point
   * @param {array} p2 - Second boundary point
   * @return {number} Distance
   */
  const sqSegDist = (p, p1, p2) => {
    let x = p1[0];
    let y = p1[1];
    let dx = p2[0] - x;
    let dy = p2[1] - y;

    if (dx !== 0 || dy !== 0) {
      const t = ((p[0] - x) * dx + (p[1] - y) * dy) / (dx * dx + dy * dy);

      if (t > 1) {
        x = p2[0];
        y = p2[1];
      } else if (t > 0) {
        x += dx * t;
        y += dy * t;
      }
    }

    dx = p[0] - x;
    dy = p[1] - y;

    return dx * dx + dy * dy;
  };

  /**
   * Douglas Peucker step function
   * Implementation from https://github.com/mourner/simplify-js
   * @author Vladimir Agafonkin
   * @copyright Vladimir Agafonkin 2013
   * @license BSD
   * @param   {[type]}  points  [description]
   * @param   {[type]}  first  [description]
   * @param   {[type]}  last  [description]
   * @param   {[type]}  tolerance  [description]
   * @param   {[type]}  simplified  [description]
   * @return  {[type]}  [description]
   */
  // biome-ignore lint/style/useNamingConvention: DP stands for Douglas Peucker
  const simplifyDPStep = (points, first, last, tolerance, simplified) => {
    let maxDist = tolerance;
    let index;

    for (let i = first + 1; i < last; i++) {
      const dist = sqSegDist(points[i], points[first], points[last]);

      if (dist > maxDist) {
        index = i;
        maxDist = dist;
      }
    }

    if (maxDist > tolerance) {
      if (index - first > 1) {
        simplifyDPStep(points, first, index, tolerance, simplified);
      }
      simplified.push(points[index]);
      if (last - index > 1) {
        simplifyDPStep(points, index, last, tolerance, simplified);
      }
    }
  };

  /**
   * Douglas Peucker. Implementation from https://github.com/mourner/simplify-js
   * @author Vladimir Agafonkin
   * @copyright Vladimir Agafonkin 2013
   * @license BSD
   * @param {array} points - List of points to be simplified
   * @param {number} tolerance - Tolerance level. Points below this distance level will be ignored
   * @return {array} Simplified point list
   */
  const simplifyDouglasPeucker = (points, tolerance) => {
    const last = points.length - 1;
    const simplified = [points[0]];

    simplifyDPStep(points, 0, last, tolerance, simplified);
    simplified.push(points[last]);

    return simplified;
  };

  /**
   * Interpolate intermediate points between key points
   * @param {array} points - Fixed key points
   * @param {number} options.maxIntPointsPerSegment - Maximum number of points between two key points
   * @param {number} options.tolerance - Simplification tolerance
   * @return {array} Interpolated points including key points
   */
  const interpolatePoints = (
    points,
    { maxIntPointsPerSegment = 100, tolerance = 0.002 } = {},
  ) => {
    const numPoints = points.length;
    const maxPointIdx = numPoints - 1;

    const maxOutPoints = maxPointIdx * maxIntPointsPerSegment + 1;
    const sqTolerance = tolerance ** 2;

    let outPoints = [];
    let prevPoint;

    // Generate interpolated points where the squared-distance between points
    // is larger than sqTolerance
    for (let i = 0; i < numPoints - 1; i++) {
      let segmentPoints = [points[i].slice(0, 2)];
      prevPoint = points[i];

      for (let j = 1; j < maxIntPointsPerSegment; j++) {
        const t = (i * maxIntPointsPerSegment + j) / maxOutPoints;
        const intPoint = interpolatePoint(t, points, maxPointIdx);

        // Check squared distance simplification
        if (
          sqDist(prevPoint[0], prevPoint[1], intPoint[0], intPoint[1]) >
          sqTolerance
        ) {
          segmentPoints.push(intPoint);
          prevPoint = intPoint;
        }
      }

      // Add next key point. Needed for the simplification algorithm
      segmentPoints.push(points[i + 1]);
      // Simplify interpolated points using the douglas-peuckner algorithm
      segmentPoints = simplifyDouglasPeucker(segmentPoints, sqTolerance);
      // Add simplified points without the last key point, which is added
      // anyway in the next segment
      outPoints = outPoints.concat(
        segmentPoints.slice(0, segmentPoints.length - 1),
      );
    }
    outPoints.push(points[points.length - 1].slice(0, 2));

    return outPoints.flat();
  };

  /**
   * Group points by line assignment (the fifth component of a point)
   * @param {array} points - Flat list of points
   * @return {array} List of lists of ordered points by line
   */
  const groupPoints = (points) => {
    const groupedPoints = {};

    const isOrdered = !Number.isNaN(+points[0][5]);
    // biome-ignore lint/complexity/noForEach: somehow for .. of does not work in a worker
    points.forEach((point) => {
      const segId = point[4];

      if (!groupedPoints[segId]) {
        groupedPoints[segId] = [];
      }

      if (isOrdered) {
        groupedPoints[segId][point[5]] = point;
      } else {
        groupedPoints[segId].push(point);
      }
    });

    // The filtering ensures that non-existing array entries are removed
    // biome-ignore lint/complexity/noForEach: somehow for .. of does not work in a worker
    Object.entries(groupedPoints).forEach((idPoints) => {
      groupedPoints[idPoints[0]] = idPoints[1].filter((v) => v);
      // Store the first point as the reference
      groupedPoints[idPoints[0]].reference = idPoints[1][0];
    });

    return groupedPoints;
  };

  self.onmessage = function onmessage(event) {
    const numPoints = event.data.points ? +event.data.points.length : 0;

    if (!numPoints) {
      self.postMessage({ error: new Error('No points provided') });
    }

    state.points = event.data.points;

    const groupedPoints = groupPoints(event.data.points);

    self.postMessage({
      points: Object.entries(groupedPoints).reduce(
        (curvePoints, idAndPoints) => {
          curvePoints[idAndPoints[0]] = interpolatePoints(
            idAndPoints[1],
            event.data.options,
          );
          // Make sure the reference is passed on
          curvePoints[idAndPoints[0]].reference = idAndPoints[1].reference;
          return curvePoints;
        },
        {},
      ),
    });
  };
};

export default worker;
