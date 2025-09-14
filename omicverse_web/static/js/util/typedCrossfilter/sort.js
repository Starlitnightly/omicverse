/**
 * High-performance sorting utilities for TypedCrossfilter
 * Includes binary search and sorting algorithms optimized for typed arrays
 */

// Sort typed array in place with indices tracking
export function sortArray(arr, indices = null) {
    if (indices === null) {
        indices = new Array(arr.length);
        for (let i = 0; i < arr.length; i++) {
            indices[i] = i;
        }
    }

    // Use native sort with custom comparator
    indices.sort((a, b) => {
        const valA = arr[a];
        const valB = arr[b];

        if (valA < valB) return -1;
        if (valA > valB) return 1;
        return 0;
    });

    return indices;
}

// Binary search for lower bound (first element >= target)
export function lowerBound(arr, target, start = 0, end = arr.length) {
    let left = start;
    let right = end;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

// Binary search for upper bound (first element > target)
export function upperBound(arr, target, start = 0, end = arr.length) {
    let left = start;
    let right = end;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

// Binary search for exact match
export function binarySearch(arr, target, start = 0, end = arr.length) {
    const pos = lowerBound(arr, target, start, end);
    if (pos < end && arr[pos] === target) {
        return pos;
    }
    return -1;
}

// Binary search with indirect indexing
export function lowerBoundIndirect(arr, indices, target, start = 0, end = indices.length) {
    let left = start;
    let right = end;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[indices[mid]] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

export function upperBoundIndirect(arr, indices, target, start = 0, end = indices.length) {
    let left = start;
    let right = end;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[indices[mid]] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

// Find range of elements matching target
export function equalRange(arr, target, start = 0, end = arr.length) {
    const lower = lowerBound(arr, target, start, end);
    const upper = upperBound(arr, target, start, end);
    return [lower, upper];
}

// Optimized merge operation for sorted arrays
export function mergeSortedArrays(arr1, arr2) {
    const result = new Array(arr1.length + arr2.length);
    let i = 0, j = 0, k = 0;

    while (i < arr1.length && j < arr2.length) {
        if (arr1[i] <= arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }

    while (i < arr1.length) {
        result[k++] = arr1[i++];
    }

    while (j < arr2.length) {
        result[k++] = arr2[j++];
    }

    return result;
}

// Create sort index for maintaining original positions
export function createSortIndex(length) {
    const indices = new Array(length);
    for (let i = 0; i < length; i++) {
        indices[i] = i;
    }
    return indices;
}