import { createWorker } from '@flekschas/utils';
import workerFn from './spline-curve-worker.js';

const createSplineCurve = (
  points,
  options = { tolerance: 0.002, maxIntPointsPerSegment: 100 },
) =>
  new Promise((resolve, reject) => {
    const worker = createWorker(workerFn);

    worker.onmessage = (e) => {
      if (e.data.error) {
        reject(e.data.error);
      } else {
        resolve(e.data.points);
      }
      worker.terminate();
    };

    worker.postMessage({ points, options });
  });

export default createSplineCurve;
