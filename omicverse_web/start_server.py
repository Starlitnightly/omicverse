#!/usr/bin/env python3
"""
OmicVerse Single Cell Analysis Platform Launcher

This script provides a convenient way to start the web server with proper
configuration and dependency checking.
"""

import sys
import os
import subprocess
import importlib
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('scanpy', 'Scanpy'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('werkzeug', 'Werkzeug')
    ]
    
    optional_packages = [
        ('omicverse', 'OmicVerse')
    ]
    
    missing_required = []
    missing_optional = []
    
    print("🔍 检查依赖包...")
    
    # Check required packages
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - 已安装")
        except ImportError:
            missing_required.append((package, name))
            print(f"❌ {name} - 未安装")
    
    # Check optional packages
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - 已安装 (可选)")
        except ImportError:
            missing_optional.append((package, name))
            print(f"⚠️  {name} - 未安装 (可选，但推荐安装)")
    
    if missing_required:
        print("\n❌ 缺少必需的依赖包:")
        for package, name in missing_required:
            print(f"   - {name} ({package})")
        print("\n请运行以下命令安装:")
        packages = " ".join([pkg for pkg, _ in missing_required])
        print(f"   pip install {packages}")
        return False
    
    if missing_optional:
        print("\n⚠️  建议安装以下可选包以获得完整功能:")
        for package, name in missing_optional:
            print(f"   - {name} ({package})")
        packages = " ".join([pkg for pkg, _ in missing_optional])
        print(f"   pip install {packages}")
    
    return True

def check_files():
    """Check if required files exist"""
    current_dir = Path(__file__).parent
    required_files = [
        'app.py',
        'single_cell_analysis_standalone.html',
    ]
    
    print("\n🔍 检查必需文件...")
    
    missing_files = []
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    if missing_files:
        print("\n❌ 缺少必需文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    return True

def get_available_port(start_port=5050):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    return None


def _parse_args():
    """Parse launcher CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Launch OmicVerse web server."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "0") or 0),
        help="Port to bind. If omitted, auto-select from 5050.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode.",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable Flask debug mode.",
    )
    parser.set_defaults(debug=False)
    return parser.parse_args()

def main():
    """Main launcher function"""
    args = _parse_args()

    print("🚀 OmicVerse 单细胞分析平台启动器")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ 是必需的")
        print(f"   当前版本: {sys.version}")
        return 1
    
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check files
    if not check_files():
        return 1
    
    # Resolve port (explicit first, then auto-discovery)
    port = args.port if args.port > 0 else get_available_port()
    if port is None:
        print("❌ 无法找到可用端口")
        return 1

    print(f"\n🌐 服务将在端口 {port} 启动")
    
    # Set environment variables
    os.environ['PORT'] = str(port)
    os.environ['FLASK_ENV'] = 'development'
    
    # Start the server
    print("\n🎯 启动服务器...")
    print("-" * 50)
    print(f"📱 本地访问: http://localhost:{port}")
    print(f"🌍 网络访问: http://{args.host}:{port}")
    print("⌨️  按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        sys.path.insert(0, str(Path(__file__).parent))
        from app import app
        app.run(debug=args.debug, host=args.host, port=port, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
        return 0
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
