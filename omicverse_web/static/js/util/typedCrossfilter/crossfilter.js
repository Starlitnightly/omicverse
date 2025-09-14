/**
 * Immutable TypedCrossfilter implementation
 * High-performance multi-dimensional filtering for large datasets
 * Based on CellxGene's filtering architecture
 */

import BitArray from './bitArray.js';
import { createDimension, DimensionTypes } from './dimensions.js';

export default class ImmutableTypedCrossfilter {
    constructor(data, dimensions = {}, selectionCache = {}) {
        this.data = data;
        this.selectionCache = selectionCache; // { bitArray, ... }
        this.dimensions = dimensions; // { name: { id, dim, name, selection } }
        Object.freeze(this);
    }

    // Basic accessors
    size() {
        return this.data.length;
    }

    all() {
        return this.data;
    }

    // Data management
    setData(data) {
        if (this.data === data) return this;
        // Drop cache when data changes
        return new ImmutableTypedCrossfilter(data, this.dimensions);
    }

    // Dimension management
    dimensionNames() {
        return Object.keys(this.dimensions);
    }

    hasDimension(name) {
        return !!this.dimensions[name];
    }

    addDimension(name, type, ...rest) {
        const { data } = this;
        const { bitArray } = this.selectionCache;

        if (this.dimensions[name] !== undefined) {
            throw new Error(`Adding duplicate dimension name ${name}`);
        }

        this._clearSelectionCache();

        // Allocate dimension ID in bit array
        let id;
        if (bitArray) {
            id = bitArray.allocDimension();
            bitArray.selectAll(id);
        }

        // Create dimension
        const DimensionType = DimensionTypes[type] || createDimension;
        const dim = type ? new DimensionType(name, data, ...rest) : createDimension(name, data);
        Object.freeze(dim);

        const dimensions = {
            ...this.dimensions,
            [name]: {
                id,
                dim,
                name,
                selection: dim.select({ mode: 'all' })
            }
        };

        return new ImmutableTypedCrossfilter(data, dimensions, { bitArray });
    }

    delDimension(name) {
        const { data } = this;
        const { bitArray } = this.selectionCache;
        const dimensions = { ...this.dimensions };

        if (dimensions[name] === undefined) {
            throw new ReferenceError(`Unable to delete unknown dimension ${name}`);
        }

        const { id } = dimensions[name];
        delete dimensions[name];
        this._clearSelectionCache();

        if (bitArray) {
            bitArray.freeDimension(id);
        }

        return new ImmutableTypedCrossfilter(data, dimensions, { bitArray });
    }

    renameDimension(oldName, newName) {
        const { [oldName]: dim, ...dimensions } = this.dimensions;
        const { data, selectionCache } = this;

        const newDimensions = {
            ...dimensions,
            [newName]: {
                ...dim,
                name: newName,
                dim: dim.dim.rename(newName)
            }
        };

        return new ImmutableTypedCrossfilter(data, newDimensions, selectionCache);
    }

    // Selection operations
    select(name, spec) {
        if (!this.dimensions[name]) {
            throw new ReferenceError(`Unknown dimension: ${name}`);
        }

        const { data, selectionCache } = this;
        const dimensions = { ...this.dimensions };
        const dimInfo = dimensions[name];

        // Update dimension selection
        const newSelection = dimInfo.dim.select(spec);
        dimensions[name] = {
            ...dimInfo,
            selection: newSelection
        };

        // Update bit array cache
        const { bitArray } = selectionCache;
        if (bitArray && dimInfo.id !== undefined) {
            // Clear dimension
            bitArray.selectNone(dimInfo.id);

            // Set selected indices
            if (newSelection.indices) {
                bitArray.setIndices(newSelection.indices, dimInfo.id, true);
            }
        }

        this._clearSelectionCache();
        return new ImmutableTypedCrossfilter(data, dimensions, { bitArray });
    }

    selectAll(name) {
        return this.select(name, { mode: 'all' });
    }

    selectNone(name) {
        return this.select(name, { mode: 'range', min: Infinity, max: -Infinity });
    }

    selectRange(name, min, max) {
        return this.select(name, { mode: 'range', min, max });
    }

    selectValues(name, values) {
        return this.select(name, { mode: 'values', values });
    }

    // Get current selection for a dimension
    getSelection(name) {
        if (!this.dimensions[name]) {
            throw new ReferenceError(`Unknown dimension: ${name}`);
        }
        return this.dimensions[name].selection;
    }

    // Get filtered data
    allFiltered() {
        const selectedIndices = this._getSelectedIndices();
        return selectedIndices.map(idx => this.data[idx]);
    }

    // Get selected indices
    getSelectedIndices() {
        return this._getSelectedIndices();
    }

    // Count selected items
    countSelected() {
        return this._getSelectedIndices().length;
    }

    // Get dimension statistics for selected data
    getDimensionStats(name) {
        if (!this.dimensions[name]) {
            throw new ReferenceError(`Unknown dimension: ${name}`);
        }

        const dim = this.dimensions[name].dim;
        const selectedIndices = this._getSelectedIndices();

        if (dim.dataType === 'numeric') {
            return this._getNumericStats(dim, selectedIndices);
        } else if (dim.dataType === 'categorical') {
            return this._getCategoricalStats(dim, selectedIndices);
        }

        return null;
    }

    // Performance monitoring
    getPerformanceInfo() {
        const { bitArray } = this.selectionCache;
        return {
            dataSize: this.data.length,
            dimensionCount: Object.keys(this.dimensions).length,
            memoryUsage: bitArray ? bitArray.getMemoryUsage() : 0,
            cacheStatus: !!bitArray
        };
    }

    // Private methods
    _getSelectedIndices() {
        const dimensionIds = Object.values(this.dimensions).map(d => d.id).filter(id => id !== undefined);

        if (dimensionIds.length === 0) {
            // No dimensions, return all indices
            const result = new Array(this.data.length);
            for (let i = 0; i < this.data.length; i++) {
                result[i] = i;
            }
            return result;
        }

        const { bitArray } = this._ensureBitArray();
        return bitArray.getIntersection(dimensionIds);
    }

    _ensureBitArray() {
        let { bitArray } = this.selectionCache;

        if (!bitArray) {
            // Create new bit array
            const numDimensions = Math.max(8, Object.keys(this.dimensions).length * 2);
            bitArray = new BitArray(this.data.length, numDimensions);

            // Initialize dimensions
            for (let [name, dimInfo] of Object.entries(this.dimensions)) {
                if (dimInfo.id === undefined) {
                    dimInfo.id = bitArray.allocDimension();
                }

                // Set selection
                if (dimInfo.selection && dimInfo.selection.indices) {
                    bitArray.setIndices(dimInfo.selection.indices, dimInfo.id, true);
                } else {
                    bitArray.selectAll(dimInfo.id);
                }
            }

            this.selectionCache.bitArray = bitArray;
        }

        return this.selectionCache;
    }

    _clearSelectionCache() {
        // Keep bitArray but clear other cache
        const { bitArray } = this.selectionCache;
        this.selectionCache = bitArray ? { bitArray } : {};
    }

    _getNumericStats(dim, selectedIndices) {
        if (selectedIndices.length === 0) {
            return { min: 0, max: 0, mean: 0, count: 0 };
        }

        let min = Infinity;
        let max = -Infinity;
        let sum = 0;
        let count = 0;

        for (let idx of selectedIndices) {
            const value = dim.data[idx];
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

    _getCategoricalStats(dim, selectedIndices) {
        const counts = new Map();

        for (let idx of selectedIndices) {
            const value = dim.data[idx];
            counts.set(value, (counts.get(value) || 0) + 1);
        }

        return {
            categories: Array.from(counts.keys()),
            counts: counts,
            totalCount: selectedIndices.length
        };
    }
}

// Factory function
export function createCrossfilter(data) {
    return new ImmutableTypedCrossfilter(data);
}