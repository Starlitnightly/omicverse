/**
 * Dimension implementations for TypedCrossfilter
 * Supports different data types with optimized filtering
 */

import { sortArray, lowerBound, upperBound, lowerBoundIndirect, upperBoundIndirect } from './sort.js';

// Base dimension class
export class BaseDimension {
    constructor(name, data) {
        this.name = name;
        this.data = data;
        this.length = data.length;
        this.sortIndex = null;
        this.sortedValues = null;
        this.isIndexed = false;
    }

    // Build index for fast range queries
    buildIndex() {
        if (this.isIndexed) return;

        this.sortIndex = sortArray(this.data);
        this.sortedValues = new Array(this.length);

        for (let i = 0; i < this.length; i++) {
            this.sortedValues[i] = this.data[this.sortIndex[i]];
        }

        this.isIndexed = true;
    }

    // Get indices for range selection
    getIndicesInRange(minValue, maxValue) {
        if (!this.isIndexed) {
            this.buildIndex();
        }

        const startIdx = lowerBound(this.sortedValues, minValue);
        const endIdx = upperBound(this.sortedValues, maxValue);

        const result = [];
        for (let i = startIdx; i < endIdx; i++) {
            result.push(this.sortIndex[i]);
        }

        return result;
    }

    // Get all indices
    getAllIndices() {
        const result = new Array(this.length);
        for (let i = 0; i < this.length; i++) {
            result[i] = i;
        }
        return result;
    }

    // Select operation - returns selection specification
    select(spec) {
        if (spec.mode === 'all') {
            return {
                mode: 'all',
                indices: this.getAllIndices()
            };
        } else if (spec.mode === 'range') {
            return {
                mode: 'range',
                indices: this.getIndicesInRange(spec.min, spec.max),
                min: spec.min,
                max: spec.max
            };
        } else if (spec.mode === 'values') {
            return {
                mode: 'values',
                indices: this.getIndicesForValues(spec.values),
                values: spec.values
            };
        }

        throw new Error(`Unknown selection mode: ${spec.mode}`);
    }

    rename(newName) {
        return new this.constructor(newName, this.data);
    }
}

// Numeric dimension for continuous data
export class NumericDimension extends BaseDimension {
    constructor(name, data) {
        super(name, data);
        this.dataType = 'numeric';
    }

    getIndicesForValues(values) {
        const result = new Set();

        for (let value of values) {
            for (let i = 0; i < this.length; i++) {
                if (this.data[i] === value) {
                    result.add(i);
                }
            }
        }

        return Array.from(result);
    }

    // Get statistics for the dimension
    getStats() {
        let min = Infinity;
        let max = -Infinity;
        let sum = 0;
        let count = 0;

        for (let i = 0; i < this.length; i++) {
            const value = this.data[i];
            if (typeof value === 'number' && !isNaN(value)) {
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
                count++;
            }
        }

        return {
            min: min === Infinity ? 0 : min,
            max: max === -Infinity ? 0 : max,
            mean: count > 0 ? sum / count : 0,
            count: count
        };
    }
}

// Categorical dimension for discrete data
export class CategoricalDimension extends BaseDimension {
    constructor(name, data) {
        super(name, data);
        this.dataType = 'categorical';
        this.categories = null;
        this.categoryMap = null;
        this.buildCategories();
    }

    buildCategories() {
        const categorySet = new Set();
        for (let i = 0; i < this.length; i++) {
            categorySet.add(this.data[i]);
        }

        this.categories = Array.from(categorySet).sort();
        this.categoryMap = new Map();

        for (let i = 0; i < this.categories.length; i++) {
            this.categoryMap.set(this.categories[i], i);
        }
    }

    getIndicesForValues(values) {
        const result = [];
        const valueSet = new Set(values);

        for (let i = 0; i < this.length; i++) {
            if (valueSet.has(this.data[i])) {
                result.push(i);
            }
        }

        return result;
    }

    // Get category counts
    getCategoryCounts() {
        const counts = new Map();

        for (let category of this.categories) {
            counts.set(category, 0);
        }

        for (let i = 0; i < this.length; i++) {
            const value = this.data[i];
            if (counts.has(value)) {
                counts.set(value, counts.get(value) + 1);
            }
        }

        return counts;
    }

    getCategories() {
        return this.categories.slice(); // Return copy
    }
}

// String dimension for text data
export class StringDimension extends CategoricalDimension {
    constructor(name, data) {
        // Convert all values to strings
        const stringData = data.map(val => String(val));
        super(name, stringData);
        this.dataType = 'string';
    }

    // Support partial string matching
    getIndicesForPattern(pattern, caseSensitive = false) {
        const regex = new RegExp(pattern, caseSensitive ? 'g' : 'gi');
        const result = [];

        for (let i = 0; i < this.length; i++) {
            if (regex.test(this.data[i])) {
                result.push(i);
            }
        }

        return result;
    }
}

// Factory function to create appropriate dimension type
export function createDimension(name, data) {
    // Detect data type
    const sampleSize = Math.min(100, data.length);
    let numericCount = 0;
    let stringCount = 0;

    for (let i = 0; i < sampleSize; i++) {
        const value = data[i];
        if (typeof value === 'number' && !isNaN(value)) {
            numericCount++;
        } else if (typeof value === 'string') {
            stringCount++;
        }
    }

    // Decision logic
    if (numericCount > sampleSize * 0.8) {
        return new NumericDimension(name, data);
    } else if (stringCount > 0) {
        // Check if it's categorical (limited unique values)
        const uniqueValues = new Set(data.slice(0, sampleSize));
        if (uniqueValues.size < sampleSize * 0.5) {
            return new CategoricalDimension(name, data);
        } else {
            return new StringDimension(name, data);
        }
    } else {
        return new CategoricalDimension(name, data);
    }
}

// Dimension type registry
export const DimensionTypes = {
    numeric: NumericDimension,
    categorical: CategoricalDimension,
    string: StringDimension
};