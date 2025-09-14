#!/bin/bash

# OmicVerse Single Cell Analysis Platform Quick Start Script
# This script will generate sample data and start the web server

echo "🚀 OmicVerse 单细胞分析平台快速启动"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python 3.8+"
    exit 1
fi

echo "✅ Python3 已找到"

# Change to script directory
cd "$(dirname "$0")"

# Check if sample data exists
if [ ! -f "sample_data.h5ad" ]; then
    echo "📊 生成示例数据..."
    python3 create_sample_data.py --cells 1500 --genes 2500 --clusters 6
    if [ $? -ne 0 ]; then
        echo "❌ 示例数据生成失败"
        exit 1
    fi
else
    echo "✅ 示例数据已存在"
fi

echo ""
echo "🌐 启动网页服务器..."
echo "🔗 请在浏览器中访问: http://localhost:5050"
echo "📁 可以上传示例数据文件: sample_data.h5ad"
echo ""
echo "⌨️  按 Ctrl+C 停止服务器"
echo ""

# Start the server
python3 start_server.py
