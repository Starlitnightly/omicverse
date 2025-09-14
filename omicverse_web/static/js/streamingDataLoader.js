/**
 * Streaming data loader for handling large datasets
 * Implements progressive loading and viewport-based data fetching
 */

class StreamingDataLoader {
    constructor(dataManager) {
        this.dataManager = dataManager;
        this.chunkSize = 50000; // Points per chunk
        this.loadedChunks = new Set();
        this.totalChunks = 0;
        this.currentData = null;

        // Streaming state
        this.isStreaming = false;
        this.streamingProgress = 0;

        // Viewport-based loading
        this.viewportBounds = null;
        this.visibleChunks = new Set();

        // Event handlers
        this.onProgress = null;
        this.onChunkLoaded = null;
        this.onStreamComplete = null;
    }

    // Initialize streaming for large datasets
    async initializeStreaming(schema) {
        try {
            this.totalChunks = schema.chunk_info?.total_chunks || 1;
            this.chunkSize = schema.chunk_info?.chunk_size || 50000;

            console.log(`Initialized streaming: ${this.totalChunks} chunks, ${this.chunkSize} points per chunk`);

            return {
                totalChunks: this.totalChunks,
                chunkSize: this.chunkSize,
                supportsStreaming: this.totalChunks > 1
            };

        } catch (error) {
            console.error('Failed to initialize streaming:', error);
            throw error;
        }
    }

    // Load data progressively
    async loadDataStreaming(embeddingName, options = {}) {
        if (this.isStreaming) {
            console.warn('Streaming already in progress');
            return;
        }

        this.isStreaming = true;
        this.streamingProgress = 0;
        this.loadedChunks.clear();

        try {
            // Load initial chunk first for immediate visualization
            const initialChunk = await this.loadChunk(embeddingName, 0);

            if (this.onChunkLoaded) {
                this.onChunkLoaded(initialChunk, 0, this.totalChunks);
            }

            this.loadedChunks.add(0);
            this.streamingProgress = 1 / this.totalChunks;

            if (this.onProgress) {
                this.onProgress(this.streamingProgress, 'Loaded initial data');
            }

            // Load remaining chunks in background
            if (this.totalChunks > 1) {
                this.loadRemainingChunks(embeddingName, options);
            } else {
                this.completeStreaming();
            }

            return initialChunk;

        } catch (error) {
            this.isStreaming = false;
            throw error;
        }
    }

    // Load remaining chunks in background
    async loadRemainingChunks(embeddingName, options) {
        const concurrency = options.concurrency || 2; // Number of parallel requests
        const priority = options.priority || 'background'; // 'background' or 'immediate'

        // Create chunk loading queue
        const chunks = [];
        for (let i = 1; i < this.totalChunks; i++) {
            chunks.push(i);
        }

        // Prioritize chunks based on viewport if available
        if (this.viewportBounds) {
            chunks.sort((a, b) => {
                const distanceA = this.getChunkDistanceFromViewport(a);
                const distanceB = this.getChunkDistanceFromViewport(b);
                return distanceA - distanceB;
            });
        }

        // Load chunks with controlled concurrency
        const loadChunkBatch = async (chunkIndices) => {
            const promises = chunkIndices.map(async (chunkIndex) => {
                try {
                    const chunkData = await this.loadChunk(embeddingName, chunkIndex);
                    this.loadedChunks.add(chunkIndex);
                    this.streamingProgress = this.loadedChunks.size / this.totalChunks;

                    if (this.onChunkLoaded) {
                        this.onChunkLoaded(chunkData, chunkIndex, this.totalChunks);
                    }

                    if (this.onProgress) {
                        this.onProgress(
                            this.streamingProgress,
                            `Loaded ${this.loadedChunks.size}/${this.totalChunks} chunks`
                        );
                    }

                    return chunkData;

                } catch (error) {
                    console.error(`Failed to load chunk ${chunkIndex}:`, error);
                    return null;
                }
            });

            return Promise.all(promises);
        };

        // Process chunks in batches
        for (let i = 0; i < chunks.length; i += concurrency) {
            const batch = chunks.slice(i, i + concurrency);

            if (priority === 'immediate') {
                await loadChunkBatch(batch);
            } else {
                // Background loading with delay to avoid blocking UI
                setTimeout(() => loadChunkBatch(batch), i * 100);
            }

            // Check if streaming was cancelled
            if (!this.isStreaming) {
                break;
            }
        }

        // Complete streaming when all chunks are processed
        if (this.isStreaming && this.loadedChunks.size === this.totalChunks) {
            this.completeStreaming();
        }
    }

    // Load a single data chunk
    async loadChunk(embeddingName, chunkIndex) {
        try {
            // Load embedding chunk
            const embeddingData = await this.dataManager.loadEmbeddingData(embeddingName, chunkIndex);

            // Load basic observation data for this chunk
            const obsData = await this.dataManager.loadObservationData(null, chunkIndex);

            return {
                chunkIndex: chunkIndex,
                positions: embeddingData.x ? embeddingData.x.map((x, i) => [x, embeddingData.y[i]]) : [],
                observations: obsData,
                size: embeddingData.x ? embeddingData.x.length : 0
            };

        } catch (error) {
            console.error(`Failed to load chunk ${chunkIndex}:`, error);
            throw error;
        }
    }

    // Update viewport bounds for priority loading
    updateViewport(bounds) {
        this.viewportBounds = bounds;

        // Determine which chunks are visible
        this.updateVisibleChunks();

        // Load visible chunks if not already loaded
        this.loadVisibleChunks();
    }

    updateVisibleChunks() {
        if (!this.viewportBounds || !this.currentData) return;

        this.visibleChunks.clear();

        // Simple approach: mark chunks as visible based on index
        // In a real implementation, this would use spatial indexing
        for (let i = 0; i < this.totalChunks; i++) {
            // For now, consider all chunks visible
            // TODO: Implement proper spatial bounds checking
            this.visibleChunks.add(i);
        }
    }

    async loadVisibleChunks() {
        if (!this.currentData) return;

        const chunksToLoad = [];
        for (let chunkIndex of this.visibleChunks) {
            if (!this.loadedChunks.has(chunkIndex)) {
                chunksToLoad.push(chunkIndex);
            }
        }

        if (chunksToLoad.length === 0) return;

        console.log(`Loading ${chunksToLoad.length} visible chunks`);

        // Load visible chunks with high priority
        for (let chunkIndex of chunksToLoad.slice(0, 3)) { // Limit to 3 chunks at once
            try {
                const chunkData = await this.loadChunk(this.currentData.embeddingName, chunkIndex);
                this.loadedChunks.add(chunkIndex);

                if (this.onChunkLoaded) {
                    this.onChunkLoaded(chunkData, chunkIndex, this.totalChunks);
                }

            } catch (error) {
                console.warn(`Failed to load visible chunk ${chunkIndex}:`, error);
            }
        }
    }

    getChunkDistanceFromViewport(chunkIndex) {
        // Simple distance calculation - in a real implementation,
        // this would calculate actual spatial distance from viewport center
        return Math.abs(chunkIndex - Math.floor(this.totalChunks / 2));
    }

    // Complete streaming process
    completeStreaming() {
        this.isStreaming = false;
        this.streamingProgress = 1.0;

        if (this.onStreamComplete) {
            this.onStreamComplete({
                totalChunks: this.totalChunks,
                loadedChunks: this.loadedChunks.size,
                totalPoints: this.loadedChunks.size * this.chunkSize
            });
        }

        console.log(`Streaming complete: ${this.loadedChunks.size}/${this.totalChunks} chunks loaded`);
    }

    // Cancel streaming
    cancelStreaming() {
        this.isStreaming = false;
        console.log('Streaming cancelled');
    }

    // Get loading statistics
    getLoadingStats() {
        return {
            isStreaming: this.isStreaming,
            progress: this.streamingProgress,
            chunksLoaded: this.loadedChunks.size,
            totalChunks: this.totalChunks,
            visibleChunks: this.visibleChunks.size,
            loadedDataPoints: this.loadedChunks.size * this.chunkSize
        };
    }

    // Prefetch data around current viewport
    async prefetchAroundViewport(embeddingName, radius = 2) {
        if (!this.viewportBounds) return;

        const currentCenter = Math.floor(this.totalChunks / 2); // Simplified center calculation
        const chunksToPreload = [];

        for (let i = Math.max(0, currentCenter - radius);
             i < Math.min(this.totalChunks, currentCenter + radius + 1);
             i++) {
            if (!this.loadedChunks.has(i)) {
                chunksToPreload.push(i);
            }
        }

        console.log(`Prefetching ${chunksToPreload.length} chunks around viewport`);

        // Load with low priority
        for (let chunkIndex of chunksToPreload) {
            setTimeout(async () => {
                try {
                    const chunkData = await this.loadChunk(embeddingName, chunkIndex);
                    this.loadedChunks.add(chunkIndex);

                    if (this.onChunkLoaded) {
                        this.onChunkLoaded(chunkData, chunkIndex, this.totalChunks);
                    }

                } catch (error) {
                    console.warn(`Prefetch failed for chunk ${chunkIndex}:`, error);
                }
            }, chunkIndex * 500); // Staggered loading
        }
    }

    // Memory management - unload distant chunks
    unloadDistantChunks(keepRadius = 5) {
        if (!this.viewportBounds || this.totalChunks <= keepRadius * 2) return;

        const currentCenter = Math.floor(this.totalChunks / 2);
        const chunksToUnload = [];

        for (let chunkIndex of this.loadedChunks) {
            const distance = Math.abs(chunkIndex - currentCenter);
            if (distance > keepRadius) {
                chunksToUnload.push(chunkIndex);
            }
        }

        if (chunksToUnload.length > 0) {
            console.log(`Unloading ${chunksToUnload.length} distant chunks to save memory`);

            chunksToUnload.forEach(chunkIndex => {
                this.loadedChunks.delete(chunkIndex);
                // Note: In a real implementation, you'd also free the actual data
            });
        }
    }
}

// Export for use in other modules
window.StreamingDataLoader = StreamingDataLoader;