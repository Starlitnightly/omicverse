#!/usr/bin/env python3
"""
OmicVerse 统计功能使用示例

这个示例展示了如何在生成报告时启用或禁用统计功能。
"""

import scanpy as sc
import omicverse as ov
import os

def example_with_analytics():
    """启用统计功能的示例"""
    print("📊 示例1：启用统计功能")
    
    # 假设我们有一个示例数据集
    # adata = sc.datasets.pbmc68k_reduced()
    
    # 生成报告时启用统计（默认行为）
    # ov.generate_scRNA_report(
    #     adata,
    #     output_path="report_with_analytics.html",
    #     enable_analytics=True,  # 默认为True
    #     analytics_id="PROJ-001"  # 可选的项目ID
    # )
    
    print("✅ 报告已生成，包含匿名统计功能")
    print("   统计信息将帮助改进 OmicVerse")

def example_without_analytics():
    """禁用统计功能的示例"""
    print("\n🚫 示例2：禁用统计功能")
    
    # 方法1：函数参数禁用
    # ov.generate_scRNA_report(
    #     adata,
    #     output_path="report_no_analytics.html",
    #     enable_analytics=False  # 明确禁用
    # )
    
    print("✅ 报告已生成，不包含任何统计功能")

def example_env_variable():
    """使用环境变量禁用统计"""
    print("\n🌍 示例3：使用环境变量禁用统计")
    
    # 设置环境变量
    os.environ['OMICVERSE_ANALYTICS'] = 'false'
    
    # 即使不设置 enable_analytics=False，统计也会被禁用
    # ov.generate_scRNA_report(
    #     adata,
    #     output_path="report_env_disabled.html"
    # )
    
    print("✅ 环境变量已设置，所有报告都将禁用统计")
    print("   要重新启用，请删除环境变量或设置为 'true'")

def show_privacy_info():
    """显示隐私保护信息"""
    print("\n🔒 隐私保护说明")
    print("=" * 50)
    
    privacy_info = {
        "收集的信息": [
            "报告生成时间",
            "匿名用户ID（机器信息hash）",
            "操作系统类型",
            "浏览器语言（仅查看时）",
            "基本使用统计"
        ],
        "不收集的信息": [
            "个人姓名或邮箱",
            "分析数据内容",
            "文件路径或名称",
            "IP地址",
            "具体位置信息"
        ],
        "用途": [
            "了解软件使用情况",
            "改进软件功能",
            "优化用户体验",
            "技术支持和开发"
        ]
    }
    
    for category, items in privacy_info.items():
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   • {item}")

def test_analytics_settings():
    """测试不同的统计设置"""
    print("\n🧪 测试统计设置")
    print("=" * 50)
    
    # 测试不同的环境变量值
    test_values = ['false', 'true', 'no', 'yes', '0', '1', 'off', 'on']
    
    for value in test_values:
        os.environ['OMICVERSE_ANALYTICS'] = value
        env_analytics = os.environ.get('OMICVERSE_ANALYTICS', '').lower()
        disabled = env_analytics in ['false', 'no', '0', 'off', 'disable']
        
        status = "禁用" if disabled else "启用"
        print(f"   OMICVERSE_ANALYTICS='{value}' → {status}")
    
    # 清理环境变量
    if 'OMICVERSE_ANALYTICS' in os.environ:
        del os.environ['OMICVERSE_ANALYTICS']

if __name__ == "__main__":
    print("🚀 OmicVerse 统计功能使用示例")
    print("=" * 60)
    
    # 运行所有示例
    example_with_analytics()
    example_without_analytics()
    example_env_variable()
    show_privacy_info()
    test_analytics_settings()
    
    print("\n✅ 示例演示完成!")
    print("\n💡 提示:")
    print("   - 统计功能完全匿名且可选")
    print("   - 帮助我们改进 OmicVerse")
    print("   - 您可以随时禁用")
    print("   - 详细信息请查看文档") 