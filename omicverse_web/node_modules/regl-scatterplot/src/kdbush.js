// biome-ignore lint/style/useNamingConvention: KDBush is a library name
import createKDBushClass from './kdbush-class.js';
import workerFn from './kdbush-worker.js';

// biome-ignore lint/style/useNamingConvention: KDBush is a library name
const KDBush = createKDBushClass();
const WORKER_THRESHOLD = 1000000;

const createWorker = (fn) => {
  const kdbushStr = createKDBushClass.toString();
  const fnStr = fn.toString();
  const workerStr =
    // biome-ignore lint/style/useTemplate: Prefer one assignment per line
    `const createKDBushClass = ${kdbushStr};` +
    'KDBush = createKDBushClass();' +
    `const createWorker = ${fnStr};` +
    'createWorker();';

  const blob = new Blob([workerStr], { type: 'text/javascript' });
  const workerUrl = URL.createObjectURL(blob);
  const worker = new Worker(workerUrl, { name: 'KDBush' });

  // Clean up URL
  URL.revokeObjectURL(workerUrl);

  return worker;
};

/**
 * Create KDBush from an either point data or an existing spatial index
 * @param {import('./types').Points | ArrayBuffer} pointsOrIndex - Points or KDBush index
 * @param {Partial<import('./types').CreateKDBushOptions>} options - Options for configuring the index and its creation
 * @return {Promise<KDBush>} KDBush instance
 */
const createKdbush = (
  pointsOrIndex,
  options = { nodeSize: 16, useWorker: undefined },
) =>
  new Promise((resolve, reject) => {
    if (pointsOrIndex instanceof ArrayBuffer) {
      resolve(KDBush.from(pointsOrIndex));
    } else if (
      (pointsOrIndex.length < WORKER_THRESHOLD ||
        options.useWorker === false) &&
      options.useWorker !== true
    ) {
      const index = new KDBush(pointsOrIndex.length, options.nodeSize);
      for (const pointOrIndex of pointsOrIndex) {
        index.add(pointOrIndex[0], pointOrIndex[1]);
      }
      index.finish();
      resolve(index);
    } else {
      const worker = createWorker(workerFn);

      worker.onmessage = (e) => {
        if (e.data.error) {
          reject(e.data.error);
        } else {
          resolve(KDBush.from(e.data));
        }
        worker.terminate();
      };

      worker.postMessage({ points: pointsOrIndex, nodeSize: options.nodeSize });
    }
  });

export default createKdbush;
