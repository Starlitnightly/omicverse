/**
 * High-performance data manager with FlatBuffers support
 * Handles data loading, caching, and processing for large-scale visualizations
 */

class DataManager {
    constructor() {
        this.currentData = null;
        this.schema = null;
        this.cache = new Map();
        this.isLoading = false;

        // Performance monitoring
        this.performanceStats = {
            totalDataSize: 0,
            cacheHits: 0,
            cacheMisses: 0,
            loadTime: 0
        };

        // Event handlers
        this.onDataLoad = null;
        this.onDataError = null;
        this.onProgress = null;
    }

    // Load data from server
    async loadData(dataInfo) {
        if (this.isLoading) {
            throw new Error('Data loading already in progress');
        }

        this.isLoading = true;
        const startTime = performance.now();

        try {
            // Get schema first
            const schemaResponse = await fetch('/api/schema');
            if (!schemaResponse.ok) {
                throw new Error('Failed to load data schema');
            }

            const schemaData = await schemaResponse.json();
            this.schema = schemaData.schema;

            // Load basic data
            await this.loadEssentialData();

            this.performanceStats.loadTime = performance.now() - startTime;
            this.currentData = dataInfo;

            if (this.onDataLoad) {
                this.onDataLoad(this.currentData, this.schema);
            }

            console.log(`Data loaded in ${this.performanceStats.loadTime.toFixed(2)}ms`);

        } catch (error) {
            console.error('Data loading failed:', error);
            if (this.onDataError) {
                this.onDataError(error);
            }
            throw error;
        } finally {
            this.isLoading = false;
        }
    }

    // Load essential data needed for initial visualization
    async loadEssentialData() {
        // Load observation annotations (categorical data for UI)
        const obsColumns = this.schema.annotations.obs.columns
            .filter(col => col.type === 'categorical')
            .slice(0, 10) // Limit initial load
            .map(col => col.name);

        if (obsColumns.length > 0) {
            await this.loadObservationData(obsColumns);
        }

        // Load first available embedding
        const embeddings = this.schema.layout.obs;
        if (embeddings.length > 0) {
            await this.loadEmbeddingData(embeddings[0].name);
        }
    }

    // Load observation data (categorical/continuous annotations)
    async loadObservationData(columns = null, chunkIndex = 0) {
        const cacheKey = `obs_${columns ? columns.join(',') : 'all'}_${chunkIndex}`;

        if (this.cache.has(cacheKey)) {
            this.performanceStats.cacheHits++;
            return this.cache.get(cacheKey);
        }

        this.performanceStats.cacheMisses++;

        try {
            let url = '/api/data/obs';
            const params = new URLSearchParams();

            if (columns && columns.length > 0) {
                columns.forEach(col => params.append('columns', col));
            }
            if (chunkIndex > 0) {
                params.append('chunk', chunkIndex);
            }

            if (params.toString()) {
                url += '?' + params.toString();
            }

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to load observation data: ${response.statusText}`);
            }

            const fbs_data = await response.arrayBuffer();
            const decoded_data = this.decodeFlatBuffer(fbs_data);

            // Cache the result
            this.cache.set(cacheKey, decoded_data);
            this.performanceStats.totalDataSize += fbs_data.byteLength;

            return decoded_data;

        } catch (error) {
            console.error('Failed to load observation data:', error);
            throw error;
        }
    }

    // Load embedding coordinates
    async loadEmbeddingData(embeddingName, chunkIndex = 0) {
        const cacheKey = `embedding_${embeddingName}_${chunkIndex}`;

        if (this.cache.has(cacheKey)) {
            this.performanceStats.cacheHits++;
            return this.cache.get(cacheKey);
        }

        this.performanceStats.cacheMisses++;

        try {
            let url = `/api/data/embedding/${embeddingName}`;
            if (chunkIndex > 0) {
                url += `?chunk=${chunkIndex}`;
            }

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to load embedding data: ${response.statusText}`);
            }

            const fbs_data = await response.arrayBuffer();
            const decoded_data = this.decodeFlatBuffer(fbs_data);

            // Cache the result
            this.cache.set(cacheKey, decoded_data);
            this.performanceStats.totalDataSize += fbs_data.byteLength;

            return decoded_data;

        } catch (error) {
            console.error('Failed to load embedding data:', error);
            throw error;
        }
    }

    // Load gene expression data
    async loadExpressionData(geneNames, cellIndices = null) {
        const cacheKey = `expression_${geneNames.join(',')}_${cellIndices ? cellIndices.length : 'all'}`;

        if (this.cache.has(cacheKey)) {
            this.performanceStats.cacheHits++;
            return this.cache.get(cacheKey);
        }

        this.performanceStats.cacheMisses++;

        try {
            const response = await fetch('/api/data/expression', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    genes: geneNames,
                    cell_indices: cellIndices
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to load expression data: ${response.statusText}`);
            }

            const fbs_data = await response.arrayBuffer();
            const decoded_data = this.decodeFlatBuffer(fbs_data);

            // Cache the result
            this.cache.set(cacheKey, decoded_data);
            this.performanceStats.totalDataSize += fbs_data.byteLength;

            return decoded_data;

        } catch (error) {
            console.error('Failed to load expression data:', error);
            throw error;
        }
    }

    // Filter cells based on criteria
    async filterCells(filters) {
        try {
            const response = await fetch('/api/filter', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(filters)
            });

            if (!response.ok) {
                throw new Error(`Failed to filter cells: ${response.statusText}`);
            }

            const result = await response.json();
            return result.filtered_indices;

        } catch (error) {
            console.error('Failed to filter cells:', error);
            throw error;
        }
    }

    // Compute differential expression
    async computeDifferentialExpression(group1Indices, group2Indices, options = {}) {
        try {
            const response = await fetch('/api/differential_expression', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    group1_indices: group1Indices,
                    group2_indices: group2Indices,
                    method: options.method || 'wilcoxon',
                    n_genes: options.n_genes || 100
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to compute differential expression: ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            console.error('Failed to compute differential expression:', error);
            throw error;
        }
    }

    // Decode FlatBuffer data (placeholder - would use actual FlatBuffers JS library)
    decodeFlatBuffer(arrayBuffer) {
        // For now, we'll assume the server sends JSON for compatibility
        // In a full implementation, this would use FlatBuffers JS decoder
        try {
            // This is a simplified decoder - replace with actual FlatBuffers implementation
            const decoder = new TextDecoder();
            const jsonString = decoder.decode(arrayBuffer);
            return JSON.parse(jsonString);
        } catch (error) {
            console.error('Failed to decode FlatBuffer:', error);
            return null;
        }
    }

    // Get available embeddings
    getAvailableEmbeddings() {
        if (!this.schema) return [];
        return this.schema.layout.obs.map(emb => ({
            name: emb.name,
            displayName: emb.name.replace('X_', '').toUpperCase()
        }));
    }

    // Get available observation columns
    getObservationColumns() {
        if (!this.schema) return { categorical: [], continuous: [] };

        const categorical = this.schema.annotations.obs.columns
            .filter(col => col.type === 'categorical')
            .map(col => ({
                name: col.name,
                categories: col.categories
            }));

        const continuous = this.schema.annotations.obs.columns
            .filter(col => col.type === 'continuous')
            .map(col => ({ name: col.name }));

        return { categorical, continuous };
    }

    // Get data statistics
    getDataStats() {
        return {
            ...this.performanceStats,
            cacheSize: this.cache.size,
            memoryUsage: `${(this.performanceStats.totalDataSize / 1024 / 1024).toFixed(2)} MB`,
            cacheHitRatio: this.performanceStats.cacheHits / (this.performanceStats.cacheHits + this.performanceStats.cacheMisses)
        };
    }

    // Clear cache
    clearCache() {
        this.cache.clear();
        this.performanceStats.totalDataSize = 0;
        this.performanceStats.cacheHits = 0;
        this.performanceStats.cacheMisses = 0;
    }

    // Prefetch data for better performance
    async prefetchData(embeddingName) {
        try {
            // Prefetch embedding data
            await this.loadEmbeddingData(embeddingName);

            // Prefetch common observation columns
            const obsColumns = this.getObservationColumns();
            const commonColumns = obsColumns.categorical.slice(0, 5).map(col => col.name);

            if (commonColumns.length > 0) {
                await this.loadObservationData(commonColumns);
            }

            console.log(`Prefetched data for ${embeddingName}`);

        } catch (error) {
            console.warn('Prefetch failed:', error);
        }
    }
}

// Export for use in other modules
window.DataManager = DataManager;