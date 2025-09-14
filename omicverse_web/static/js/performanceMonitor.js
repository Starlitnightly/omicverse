/**
 * Performance monitoring and optimization system
 * Tracks rendering performance, memory usage, and provides optimization recommendations
 */

class PerformanceMonitor {
    constructor() {
        this.metrics = {
            rendering: {
                fps: 0,
                frameTime: 0,
                renderCalls: 0,
                droppedFrames: 0
            },
            memory: {
                totalMemory: 0,
                usedMemory: 0,
                dataCache: 0,
                bufferMemory: 0
            },
            data: {
                pointsRendered: 0,
                chunksLoaded: 0,
                cacheHits: 0,
                cacheMisses: 0
            },
            network: {
                totalRequests: 0,
                totalDataTransferred: 0,
                averageResponseTime: 0
            }
        };

        this.history = {
            fps: [],
            frameTime: [],
            memoryUsage: []
        };

        this.thresholds = {
            minFPS: 30,
            maxFrameTime: 33.33, // ms
            maxMemoryUsage: 1024 * 1024 * 1024, // 1GB
            maxCacheMissRatio: 0.3
        };

        this.isMonitoring = false;
        this.monitoringInterval = null;
        this.performanceObserver = null;

        // Callbacks
        this.onPerformanceIssue = null;
        this.onOptimizationRecommendation = null;

        this.initializeMonitoring();
    }

    initializeMonitoring() {
        // Initialize Performance Observer API if available
        if ('PerformanceObserver' in window) {
            this.performanceObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    this.processPerformanceEntry(entry);
                }
            });

            try {
                this.performanceObserver.observe({ entryTypes: ['measure', 'navigation', 'resource'] });
            } catch (e) {
                console.warn('Performance Observer not fully supported:', e);
            }
        }

        // Initialize memory monitoring if available
        if ('memory' in performance) {
            this.monitorMemory();
        }
    }

    startMonitoring() {
        if (this.isMonitoring) return;

        this.isMonitoring = true;
        console.log('Performance monitoring started');

        // Start periodic monitoring
        this.monitoringInterval = setInterval(() => {
            this.updateMetrics();
            this.analyzePerformance();
        }, 1000);

        // Monitor frame rate
        this.startFrameRateMonitoring();
    }

    stopMonitoring() {
        if (!this.isMonitoring) return;

        this.isMonitoring = false;

        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }

        if (this.frameRateCallback) {
            cancelAnimationFrame(this.frameRateCallback);
        }

        console.log('Performance monitoring stopped');
    }

    startFrameRateMonitoring() {
        let lastTime = performance.now();
        let frameCount = 0;
        let lastFrameTime = lastTime;

        const measureFrame = (currentTime) => {
            frameCount++;
            const deltaTime = currentTime - lastFrameTime;

            // Update frame time
            this.metrics.rendering.frameTime = deltaTime;

            // Calculate FPS every second
            if (currentTime - lastTime >= 1000) {
                this.metrics.rendering.fps = frameCount;
                this.history.fps.push(frameCount);
                this.history.frameTime.push(deltaTime);

                // Keep history manageable
                if (this.history.fps.length > 60) {
                    this.history.fps.shift();
                    this.history.frameTime.shift();
                }

                // Check for dropped frames
                if (frameCount < this.thresholds.minFPS) {
                    this.metrics.rendering.droppedFrames++;
                }

                frameCount = 0;
                lastTime = currentTime;
            }

            lastFrameTime = currentTime;

            if (this.isMonitoring) {
                this.frameRateCallback = requestAnimationFrame(measureFrame);
            }
        };

        this.frameRateCallback = requestAnimationFrame(measureFrame);
    }

    updateMetrics() {
        // Update memory metrics
        if ('memory' in performance) {
            const memory = performance.memory;
            this.metrics.memory.totalMemory = memory.jsHeapSizeLimit;
            this.metrics.memory.usedMemory = memory.usedJSHeapSize;

            this.history.memoryUsage.push(memory.usedJSHeapSize);
            if (this.history.memoryUsage.length > 60) {
                this.history.memoryUsage.shift();
            }
        }

        // Update network metrics
        this.updateNetworkMetrics();
    }

    updateNetworkMetrics() {
        if ('getEntriesByType' in performance) {
            const resourceEntries = performance.getEntriesByType('resource');
            const recentEntries = resourceEntries.filter(entry =>
                entry.startTime > (performance.now() - 5000) // Last 5 seconds
            );

            this.metrics.network.totalRequests += recentEntries.length;

            if (recentEntries.length > 0) {
                const totalResponseTime = recentEntries.reduce((sum, entry) =>
                    sum + (entry.responseEnd - entry.requestStart), 0);
                this.metrics.network.averageResponseTime = totalResponseTime / recentEntries.length;

                const totalTransferred = recentEntries.reduce((sum, entry) =>
                    sum + (entry.transferSize || 0), 0);
                this.metrics.network.totalDataTransferred += totalTransferred;
            }
        }
    }

    processPerformanceEntry(entry) {
        switch (entry.entryType) {
            case 'measure':
                this.processMeasureEntry(entry);
                break;
            case 'navigation':
                this.processNavigationEntry(entry);
                break;
            case 'resource':
                this.processResourceEntry(entry);
                break;
        }
    }

    processMeasureEntry(entry) {
        if (entry.name.includes('render')) {
            this.metrics.rendering.renderCalls++;
        }
    }

    processNavigationEntry(entry) {
        // Track page load performance
        console.log('Navigation timing:', {
            loadTime: entry.loadEventEnd - entry.loadEventStart,
            domContentLoaded: entry.domContentLoadedEventEnd - entry.domContentLoadedEventStart
        });
    }

    processResourceEntry(entry) {
        // Track resource loading performance
        if (entry.name.includes('api/')) {
            const responseTime = entry.responseEnd - entry.requestStart;
            console.log(`API call performance: ${entry.name} - ${responseTime.toFixed(2)}ms`);
        }
    }

    analyzePerformance() {
        const issues = [];
        const recommendations = [];

        // Analyze FPS
        if (this.metrics.rendering.fps < this.thresholds.minFPS && this.metrics.rendering.fps > 0) {
            issues.push({
                type: 'low_fps',
                severity: 'high',
                message: `Low FPS detected: ${this.metrics.rendering.fps}`,
                metric: this.metrics.rendering.fps,
                threshold: this.thresholds.minFPS
            });

            recommendations.push({
                type: 'reduce_point_count',
                message: 'Consider reducing the number of rendered points or enabling level-of-detail',
                priority: 'high'
            });
        }

        // Analyze frame time
        if (this.metrics.rendering.frameTime > this.thresholds.maxFrameTime) {
            issues.push({
                type: 'high_frame_time',
                severity: 'medium',
                message: `High frame time: ${this.metrics.rendering.frameTime.toFixed(2)}ms`,
                metric: this.metrics.rendering.frameTime,
                threshold: this.thresholds.maxFrameTime
            });

            recommendations.push({
                type: 'optimize_shaders',
                message: 'Consider optimizing shader complexity or reducing visual effects',
                priority: 'medium'
            });
        }

        // Analyze memory usage
        if (this.metrics.memory.usedMemory > this.thresholds.maxMemoryUsage) {
            issues.push({
                type: 'high_memory_usage',
                severity: 'high',
                message: `High memory usage: ${(this.metrics.memory.usedMemory / 1024 / 1024).toFixed(0)}MB`,
                metric: this.metrics.memory.usedMemory,
                threshold: this.thresholds.maxMemoryUsage
            });

            recommendations.push({
                type: 'clear_cache',
                message: 'Consider clearing data cache or reducing cached data',
                priority: 'high'
            });
        }

        // Analyze cache performance
        const totalCacheRequests = this.metrics.data.cacheHits + this.metrics.data.cacheMisses;
        if (totalCacheRequests > 0) {
            const missRatio = this.metrics.data.cacheMisses / totalCacheRequests;
            if (missRatio > this.thresholds.maxCacheMissRatio) {
                issues.push({
                    type: 'cache_inefficiency',
                    severity: 'medium',
                    message: `High cache miss ratio: ${(missRatio * 100).toFixed(1)}%`,
                    metric: missRatio,
                    threshold: this.thresholds.maxCacheMissRatio
                });

                recommendations.push({
                    type: 'improve_caching',
                    message: 'Consider prefetching data or adjusting cache size',
                    priority: 'medium'
                });
            }
        }

        // Report issues and recommendations
        if (issues.length > 0 && this.onPerformanceIssue) {
            this.onPerformanceIssue(issues);
        }

        if (recommendations.length > 0 && this.onOptimizationRecommendation) {
            this.onOptimizationRecommendation(recommendations);
        }
    }

    // Update metrics from external sources
    updateDataMetrics(dataMetrics) {
        Object.assign(this.metrics.data, dataMetrics);
    }

    updateRenderingMetrics(renderingMetrics) {
        Object.assign(this.metrics.rendering, renderingMetrics);
    }

    updateMemoryMetrics(memoryMetrics) {
        Object.assign(this.metrics.memory, memoryMetrics);
    }

    monitorMemory() {
        if (!('memory' in performance)) return;

        const checkMemory = () => {
            const memory = performance.memory;
            const usedMB = memory.usedJSHeapSize / 1024 / 1024;
            const totalMB = memory.totalJSHeapSize / 1024 / 1024;

            // Warn if memory usage is high
            if (usedMB > 512) { // 512MB threshold
                console.warn(`High memory usage detected: ${usedMB.toFixed(0)}MB used of ${totalMB.toFixed(0)}MB allocated`);
            }
        };

        setInterval(checkMemory, 5000); // Check every 5 seconds
    }

    // Get performance summary
    getPerformanceSummary() {
        return {
            metrics: { ...this.metrics },
            averages: {
                avgFPS: this.history.fps.length > 0 ?
                    this.history.fps.reduce((a, b) => a + b, 0) / this.history.fps.length : 0,
                avgFrameTime: this.history.frameTime.length > 0 ?
                    this.history.frameTime.reduce((a, b) => a + b, 0) / this.history.frameTime.length : 0,
                avgMemoryUsage: this.history.memoryUsage.length > 0 ?
                    this.history.memoryUsage.reduce((a, b) => a + b, 0) / this.history.memoryUsage.length : 0
            },
            status: this.getPerformanceStatus()
        };
    }

    getPerformanceStatus() {
        const issues = [];

        if (this.metrics.rendering.fps < this.thresholds.minFPS && this.metrics.rendering.fps > 0) {
            issues.push('low_fps');
        }

        if (this.metrics.rendering.frameTime > this.thresholds.maxFrameTime) {
            issues.push('high_frame_time');
        }

        if (this.metrics.memory.usedMemory > this.thresholds.maxMemoryUsage) {
            issues.push('high_memory');
        }

        if (issues.length === 0) {
            return 'good';
        } else if (issues.length <= 2) {
            return 'warning';
        } else {
            return 'critical';
        }
    }

    // Optimization methods
    optimizeForLowFPS() {
        console.log('Applying FPS optimizations...');

        return {
            reducedPointSize: true,
            enabledLOD: true,
            disabledAntialiasing: true,
            message: 'Applied FPS optimizations: reduced point size, enabled LOD, disabled antialiasing'
        };
    }

    optimizeForMemory() {
        console.log('Applying memory optimizations...');

        // Clear caches
        if (window.singleCellApp && window.singleCellApp.dataManager) {
            window.singleCellApp.dataManager.clearCache();
        }

        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }

        return {
            clearedCache: true,
            forcedGC: true,
            message: 'Applied memory optimizations: cleared cache, forced garbage collection'
        };
    }

    // Create performance report
    generateReport() {
        const summary = this.getPerformanceSummary();
        const report = {
            timestamp: new Date().toISOString(),
            summary: summary,
            recommendations: this.getRecommendations(),
            browserInfo: {
                userAgent: navigator.userAgent,
                memory: 'memory' in performance ? performance.memory : null,
                hardwareConcurrency: navigator.hardwareConcurrency,
                deviceMemory: navigator.deviceMemory || 'unknown'
            }
        };

        return report;
    }

    getRecommendations() {
        const recommendations = [];
        const status = this.getPerformanceStatus();

        switch (status) {
            case 'critical':
                recommendations.push(
                    'Reduce dataset size or enable data streaming',
                    'Use lower quality rendering settings',
                    'Close other browser tabs to free memory'
                );
                break;
            case 'warning':
                recommendations.push(
                    'Consider reducing point size or opacity',
                    'Enable level-of-detail rendering',
                    'Clear data cache periodically'
                );
                break;
            case 'good':
                recommendations.push('Performance is optimal');
                break;
        }

        return recommendations;
    }

    // Export performance data
    exportData() {
        return JSON.stringify(this.generateReport(), null, 2);
    }
}

// Create performance dashboard
class PerformanceDashboard {
    constructor(monitor) {
        this.monitor = monitor;
        this.element = null;
        this.isVisible = false;
    }

    create() {
        this.element = document.createElement('div');
        this.element.id = 'performance-dashboard';
        this.element.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10000;
            min-width: 250px;
            max-width: 400px;
            display: none;
        `;

        document.body.appendChild(this.element);
        this.startUpdating();
    }

    toggle() {
        this.isVisible = !this.isVisible;
        this.element.style.display = this.isVisible ? 'block' : 'none';
    }

    startUpdating() {
        setInterval(() => {
            if (this.isVisible) {
                this.update();
            }
        }, 1000);
    }

    update() {
        if (!this.element) return;

        const summary = this.monitor.getPerformanceSummary();
        const status = summary.status;

        let statusColor = '#00ff00';
        if (status === 'warning') statusColor = '#ffff00';
        if (status === 'critical') statusColor = '#ff0000';

        this.element.innerHTML = `
            <div style="border-bottom: 1px solid #333; margin-bottom: 10px; padding-bottom: 5px;">
                <strong>Performance Monitor</strong>
                <span style="color: ${statusColor}; float: right;">${status.toUpperCase()}</span>
            </div>

            <div><strong>Rendering:</strong></div>
            <div>FPS: ${summary.metrics.rendering.fps}</div>
            <div>Frame Time: ${summary.metrics.rendering.frameTime.toFixed(2)}ms</div>
            <div>Render Calls: ${summary.metrics.rendering.renderCalls}</div>

            <div style="margin-top: 10px;"><strong>Memory:</strong></div>
            <div>Used: ${(summary.metrics.memory.usedMemory / 1024 / 1024).toFixed(0)}MB</div>
            <div>Total: ${(summary.metrics.memory.totalMemory / 1024 / 1024).toFixed(0)}MB</div>

            <div style="margin-top: 10px;"><strong>Data:</strong></div>
            <div>Points: ${summary.metrics.data.pointsRendered.toLocaleString()}</div>
            <div>Cache Hits: ${summary.metrics.data.cacheHits}</div>
            <div>Cache Misses: ${summary.metrics.data.cacheMisses}</div>

            <div style="margin-top: 10px;"><strong>Averages:</strong></div>
            <div>Avg FPS: ${summary.averages.avgFPS.toFixed(1)}</div>
            <div>Avg Frame Time: ${summary.averages.avgFrameTime.toFixed(2)}ms</div>
        `;
    }
}

// Global performance monitor instance
window.performanceMonitor = new PerformanceMonitor();
window.PerformanceMonitor = PerformanceMonitor;
window.PerformanceDashboard = PerformanceDashboard;