/**
 * Calculates exponential moving average of 2D points
 * @param {[number, number][]} values - Array of numbers to average
 * @param {number} halfLife - Number of steps after which weight becomes half
 * @param {number} windowSize - Maximum number of previous values to consider
 * @returns {number} The exponential moving average
 */
export const exponentialMovingAverage = (values, halfLife, windowSize) => {
  if (values.length === 0) {
    return 0;
  }

  if (values.length === 1) {
    return values[0];
  }

  // Calculate decay factor from `halfLife` such that weight = 0.5 when the
  // step is `halfLife`
  const decayBase = 2 ** (-1 / halfLife);

  // Limit to window size
  const startIdx = Math.max(0, values.length - windowSize);
  const relevantValues = values.slice(startIdx);

  let weightedSumX = 0;
  let weightedSumY = 0;
  let weightSum = 0;

  // Calculate weighted sum starting from most recent value
  for (let i = relevantValues.length - 1; i >= 0; i--) {
    const steps = relevantValues.length - 1 - i;
    const weight = decayBase ** steps;

    weightedSumX += relevantValues[i][0] * weight;
    weightedSumY += relevantValues[i][1] * weight;
    weightSum += weight;
  }

  return [weightedSumX / weightSum, weightedSumY / weightSum];
};
