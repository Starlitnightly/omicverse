# OmicVerse Analytics Configuration
"""
OmicVerse 统计追踪配置

这个模块提供了对报告使用情况的统计追踪功能。
所有数据都是匿名的，不会收集个人隐私信息。

收集的信息包括：
- 报告生成时间
- 匿名用户ID（基于机器信息的hash）
- 操作系统信息
- 浏览器类型（仅在打开报告时）
- 地理位置（仅国家/地区级别）
"""

# 默认配置
DEFAULT_CONFIG = {
    # 是否启用统计
    'enable_analytics': True,
    
    # 统计服务器地址
    'analytics_endpoint': 'https://analytics.omicverse.org/track.gif',
    
    # 备用Google Analytics ID（如果需要）
    'google_analytics_id': None,
    
    # 项目标识符
    'project_id': 'omicverse-scrna',
    
    # 数据保留策略
    'data_retention_days': 365,
    
    # 隐私设置
    'privacy_mode': True,  # 启用隐私保护模式
    'hash_user_info': True,  # 对用户信息进行hash处理
    'collect_ip': False,  # 不收集IP地址
    'collect_detailed_ua': False,  # 不收集详细的用户代理信息
}

# 统计事件类型
EVENT_TYPES = {
    'REPORT_GENERATED': 'report_generated',
    'REPORT_VIEWED': 'report_viewed',
    'SECTION_VIEWED': 'section_viewed',
    'THEME_SWITCHED': 'theme_switched',
}

# 统计字段定义
TRACKED_FIELDS = {
    'timestamp': 'ISO格式时间戳',
    'event_type': '事件类型',
    'user_hash': '匿名用户标识',
    'session_id': '会话ID',
    'report_id': '报告标识',
    'platform': '操作系统平台',
    'language': '浏览器语言',
    'timezone': '时区',
    'screen_resolution': '屏幕分辨率',
    'viewport_size': '浏览器窗口大小',
}

def get_privacy_notice():
    """获取隐私声明"""
    return """
    📊 统计说明：
    
    OmicVerse 会收集匿名的使用统计信息，用于：
    • 了解软件使用情况
    • 改进软件功能
    • 提供更好的用户体验
    
    我们承诺：
    ✅ 所有数据都是匿名的
    ✅ 不收集个人隐私信息
    ✅ 不收集具体文件内容
    ✅ 遵循 GDPR 等隐私法规
    
    您可以通过设置 enable_analytics=False 来禁用统计功能。
    """

def get_opt_out_instructions():
    """获取退出统计的说明"""
    return """
    如何禁用统计：
    
    在生成报告时设置：
    ```python
    ov.generate_scRNA_report(
        adata, 
        enable_analytics=False
    )
    ```
    
    或设置环境变量：
    ```bash
    export OMICVERSE_ANALYTICS=false
    ```
    """ 