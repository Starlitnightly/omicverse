export default () => {
  addEventListener('message', (event) => {
    const points = event.data.points;

    if (points.length === 0) {
      self.postMessage({ error: new Error('Invalid point data') });
    }

    // biome-ignore lint/correctness/noUndeclaredVariables: KDBush is made available during compilation
    const index = new KDBush(points.length, event.data.nodeSize);

    for (const [x, y] of points) {
      index.add(x, y);
    }

    index.finish();

    postMessage(index.data, [index.data]);
  });
};
