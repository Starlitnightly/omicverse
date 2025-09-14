/**
 * TypedCrossfilter main export module
 * High-performance multi-dimensional filtering for JavaScript
 */

export { default as ImmutableTypedCrossfilter, createCrossfilter } from './crossfilter.js';
export { default as BitArray } from './bitArray.js';
export {
    createDimension,
    DimensionTypes,
    NumericDimension,
    CategoricalDimension,
    StringDimension
} from './dimensions.js';

export {
    sortArray,
    lowerBound,
    upperBound,
    binarySearch,
    lowerBoundIndirect,
    upperBoundIndirect,
    equalRange,
    mergeSortedArrays,
    createSortIndex
} from './sort.js';