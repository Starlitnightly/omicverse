/**
 * BitArray implementation for high-performance selection filtering
 * Based on CellxGene's TypedCrossfilter system
 */

export default class BitArray {
    constructor(length, numDimensions = 8) {
        this.length = length;
        this.numDimensions = numDimensions;
        this.bitsPerElement = 32;
        this.arrayLength = Math.ceil(length * numDimensions / this.bitsPerElement);

        // Use Uint32Array for bitwise operations
        this.bits = new Uint32Array(this.arrayLength);
        this.freeDimensions = [];

        // Initialize all dimensions as free
        for (let i = 0; i < numDimensions; i++) {
            this.freeDimensions.push(i);
        }
    }

    allocDimension() {
        if (this.freeDimensions.length === 0) {
            throw new Error("No free dimensions available");
        }
        return this.freeDimensions.pop();
    }

    freeDimension(dimensionId) {
        if (dimensionId >= 0 && dimensionId < this.numDimensions) {
            this.freeDimensions.push(dimensionId);
            // Clear the dimension's bits
            this.clearDimension(dimensionId);
        }
    }

    clearDimension(dimensionId) {
        const mask = ~this._getDimensionMask(dimensionId);
        for (let i = 0; i < this.length; i++) {
            const arrayIndex = this._getArrayIndex(i, dimensionId);
            this.bits[arrayIndex] &= mask;
        }
    }

    selectAll(dimensionId) {
        const mask = this._getDimensionMask(dimensionId);
        for (let i = 0; i < this.length; i++) {
            const arrayIndex = this._getArrayIndex(i, dimensionId);
            this.bits[arrayIndex] |= mask;
        }
    }

    selectNone(dimensionId) {
        this.clearDimension(dimensionId);
    }

    set(index, dimensionId, value) {
        if (index < 0 || index >= this.length) {
            throw new Error(`Index ${index} out of bounds [0, ${this.length})`);
        }

        const arrayIndex = this._getArrayIndex(index, dimensionId);
        const mask = this._getDimensionMask(dimensionId);

        if (value) {
            this.bits[arrayIndex] |= mask;
        } else {
            this.bits[arrayIndex] &= ~mask;
        }
    }

    get(index, dimensionId) {
        if (index < 0 || index >= this.length) {
            throw new Error(`Index ${index} out of bounds [0, ${this.length})`);
        }

        const arrayIndex = this._getArrayIndex(index, dimensionId);
        const mask = this._getDimensionMask(dimensionId);
        return (this.bits[arrayIndex] & mask) !== 0;
    }

    // Get indices where all specified dimensions are selected
    getIntersection(dimensionIds) {
        const result = [];

        for (let i = 0; i < this.length; i++) {
            let allSelected = true;

            for (let dimId of dimensionIds) {
                if (!this.get(i, dimId)) {
                    allSelected = false;
                    break;
                }
            }

            if (allSelected) {
                result.push(i);
            }
        }

        return result;
    }

    // Get selection state for all items in a dimension
    getDimensionSelection(dimensionId) {
        const result = new Array(this.length);

        for (let i = 0; i < this.length; i++) {
            result[i] = this.get(i, dimensionId);
        }

        return result;
    }

    // Count selected items in dimension
    countSelected(dimensionId) {
        let count = 0;
        for (let i = 0; i < this.length; i++) {
            if (this.get(i, dimensionId)) {
                count++;
            }
        }
        return count;
    }

    // Batch operations for performance
    setRange(startIndex, endIndex, dimensionId, value) {
        for (let i = startIndex; i < endIndex && i < this.length; i++) {
            this.set(i, dimensionId, value);
        }
    }

    setIndices(indices, dimensionId, value) {
        for (let index of indices) {
            this.set(index, dimensionId, value);
        }
    }

    // Private helper methods
    _getArrayIndex(itemIndex, dimensionId) {
        const bitPosition = itemIndex * this.numDimensions + dimensionId;
        return Math.floor(bitPosition / this.bitsPerElement);
    }

    _getDimensionMask(dimensionId) {
        return 1 << (dimensionId % this.bitsPerElement);
    }

    // Debug methods
    toString() {
        const result = [];
        for (let i = 0; i < this.length; i++) {
            const itemBits = [];
            for (let d = 0; d < this.numDimensions; d++) {
                itemBits.push(this.get(i, d) ? '1' : '0');
            }
            result.push(`[${i}]: ${itemBits.join('')}`);
        }
        return result.join('\n');
    }

    // Get memory usage in bytes
    getMemoryUsage() {
        return this.bits.byteLength;
    }
}