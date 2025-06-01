#!/usr/bin/env python3
"""
OmicVerse Analytics Sender
独立的analytics数据发送工具，不依赖HTML报告生成
模拟lzy_tt.html中成功的发送格式
"""

import requests
import platform
import hashlib
import os
from datetime import datetime
from urllib.parse import urlencode
import uuid


def get_geolocation():
    """获取完整的地理位置信息"""
    try:
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            geo_data = response.json()
            # 返回完整的地理位置数据
            return geo_data
    except:
        pass
    return {
        'status': 'fail',
        'country': 'Unknown',
        'countryCode': 'XX',
        'region': 'Unknown',
        'regionName': 'Unknown',
        'city': 'Unknown',
        'timezone': 'UTC',
        'isp': 'Unknown',
        'query': 'Unknown'
    }


def generate_user_hash():
    """生成匿名用户hash"""
    machine_info = f"{platform.node()}-{platform.system()}-{platform.machine()}"
    return hashlib.md5(machine_info.encode()).hexdigest()[:8]


def send_analytics(analytics_id, event_type='report_view', **kwargs):
    """
    发送analytics数据到服务器 - 标准版
    
    Parameters:
    -----------
    analytics_id : str
        项目或报告的唯一标识符
    event_type : str
        事件类型，默认为'report_view'
    **kwargs : dict
        额外的参数
        
    Returns:
    --------
    bool : 发送是否成功
    
    Note:
    -----
    包含以下标准字段：
    - id, user, ts (核心字段)
    - platform, ua, lang, tz, country (环境信息)
    - ref (引用页面，Python环境为空)
    
    与send_analytics_full的区别：
    - 标准版：9个基础字段
    - 完整版：17个字段，包含详细地理信息(城市、ISP、坐标等)
    """
    
    # 基础数据收集
    user_hash = generate_user_hash()
    timestamp = datetime.now().isoformat()
    platform_info = platform.system() + " " + platform.release()
    
    # 获取地理位置
    print("🌍 正在获取地理位置...")
    geo_data = get_geolocation()
    print(f"🌍 检测到国家: {geo_data.get('country', 'Unknown')}")
    if geo_data.get('city', 'Unknown') != 'Unknown':
        print(f"🏙️ 城市: {geo_data.get('city')}, {geo_data.get('regionName')}")
    if geo_data.get('timezone', 'UTC') != 'UTC':
        print(f"🕐 时区: {geo_data.get('timezone')}")
    
    # 构建数据包 - 使用服务器期望的参数名称
    data = {
        # 核心字段  
        'id': analytics_id,
        'user': user_hash,
        'ts': timestamp,
        
        # 基础环境信息
        'platform': platform_info,
        'ua': f'Python-OmicVerse/{platform.python_version()}',
        'lang': 'en-US',  # 可以从环境变量获取
        'tz': geo_data.get('timezone', 'UTC'),
        'country': geo_data.get('country', 'Unknown'),
        'ref': '',        # Python环境没有referrer
        
        **kwargs          # 额外参数
    }
    
    # 只在有意义的情况下添加可选字段
    if event_type != 'report_view':
        data['event_type'] = event_type
    
    print(f"📊 发送Analytics数据: {analytics_id}")
    print(f"👤 用户Hash: {user_hash}")
    print(f"🌍 国家: {data['country']}")
    print(f"🕐 时区: {data['tz']}")
    print(f"🌐 语言: {data['lang']}")
    print(f"💻 平台: {platform_info}")
    print(f"⏰ 时间: {timestamp}")
    
    # 发送请求 - 模拟lzy_tt.html中的方式
    analytics_endpoint = os.environ.get('OMICVERSE_ANALYTICS_ENDPOINT', 'http://8.130.139.217/track.gif')
    
    try:
        # 方法1: 使用GET请求（模拟img.src方式）
        params = urlencode(data)
        url = f"{analytics_endpoint}?{params}"
        
        print(f"📡 发送到: {analytics_endpoint}")
        print(f"📋 参数数量: {len(data)} 个字段")
        print(f"🔍 关键参数: lang={data['lang']}, tz={data['tz']}, country={data['country']}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Analytics数据发送成功!")
            return True
        else:
            print(f"❌ 发送失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 发送失败: {str(e)}")
        return False


def send_analytics_simple(analytics_id):
    """
    简化版发送函数 - 使用默认参数
    
    Parameters:
    -----------
    analytics_id : str
        项目标识符
        
    Note:
    -----
    实际上调用send_analytics，包含基础环境信息：
    - 项目ID、用户hash、时间戳
    - 平台信息、用户代理、语言、时区、国家
    """
    return send_analytics(analytics_id)


def send_analytics_detailed(analytics_id, project_name=None, user_note=None):
    """
    详细版发送函数，包含额外信息
    
    Parameters:
    -----------
    analytics_id : str
        项目标识符
    project_name : str
        项目名称
    user_note : str
        用户备注
    """
    extra_data = {}
    if project_name:
        extra_data['project_name'] = project_name
    if user_note:
        extra_data['user_note'] = user_note
        
    return send_analytics(analytics_id, **extra_data)


def send_analytics_full(analytics_id, event_type='report_view', **kwargs):
    """
    完整版发送函数，包含所有地理位置信息
    
    Parameters:
    -----------
    analytics_id : str
        项目或报告的唯一标识符
    event_type : str
        事件类型，默认为'report_view'
    **kwargs : dict
        额外的参数
        
    Returns:
    --------
    bool : 发送是否成功
    
    Note:
    -----
    使用与ip-api.com完全一致的字段名称和数据结构
    """
    
    # 基础数据收集
    user_hash = generate_user_hash()
    timestamp = datetime.now().isoformat()
    platform_info = platform.system() + " " + platform.release()
    
    # 获取完整地理位置
    print("🌍 正在获取完整地理位置信息...")
    geo_data = get_geolocation()
    
    # 显示详细信息
    if geo_data.get('status') == 'success':
        print(f"✅ 地理位置获取成功:")
        print(f"   🌍 国家: {geo_data.get('country')} ({geo_data.get('countryCode')})")
        print(f"   🏙️ 城市: {geo_data.get('city')}, {geo_data.get('regionName')}")
        print(f"   📍 地区: {geo_data.get('region')} ({geo_data.get('regionName')})")
        print(f"   📮 邮编: {geo_data.get('zip', 'N/A')}")
        print(f"   🕐 时区: {geo_data.get('timezone')}")
        print(f"   🌐 ISP: {geo_data.get('isp')}")
        print(f"   🏢 组织: {geo_data.get('org')}")
        print(f"   📍 坐标: {geo_data.get('lat')}, {geo_data.get('lon')}")
        print(f"   🔌 AS: {geo_data.get('as')}")
        print(f"   🌐 查询IP: {geo_data.get('query')}")
    else:
        print(f"⚠️ 地理位置获取失败，使用默认值")
    
    # 构建完整数据包 - 使用与ip-api.com完全一致的字段名称
    data = {
        # 核心字段 (必需)
        'id': analytics_id,
        'user': user_hash,
        'ts': timestamp,
        
        # 基础环境信息 
        'platform': platform_info,
        'ua': f'Python-OmicVerse/{platform.python_version()}',
        'lang': 'en-US',  
        'tz': geo_data.get('timezone', 'UTC'),          # ✅ 向后兼容
        'timezone': geo_data.get('timezone', 'UTC'),    # ✅ 与ip-api.com完全一致
        'ref': '',  # Python环境没有referrer
        
        # 基础地理信息 (必需)
        'country': geo_data.get('country', 'Unknown'),
        
        # 详细地理信息 - 使用ip-api.com的确切字段名称
        'status': geo_data.get('status', 'success'),         # ✅ ip-api状态字段
        'countryCode': geo_data.get('countryCode', 'XX'),    # ✅ 驼峰式，与ip-api一致
        'region': geo_data.get('region', 'Unknown'),         # ✅ 州/省代码 (如 "CA")
        'regionName': geo_data.get('regionName', 'Unknown'), # ✅ 州/省全名 (如 "California")
        'city': geo_data.get('city', 'Unknown'),             # ✅ 城市名
        'zip': geo_data.get('zip', ''),                      # ✅ 邮编 (字段名为zip，不是zipCode)
        'lat': geo_data.get('lat', 0),                       # ✅ 纬度 (数字类型)
        'lon': geo_data.get('lon', 0),                       # ✅ 经度 (数字类型)
        'isp': geo_data.get('isp', 'Unknown'),               # ✅ ISP提供商
        'org': geo_data.get('org', 'Unknown'),               # ✅ 组织名 (简化版)
        'as': geo_data.get('as', 'Unknown'),                 # ✅ AS信息
        'query': geo_data.get('query', 'Unknown'),           # ✅ 查询的IP地址
        
        **kwargs          # 额外参数
    }
    
    # 添加事件类型
    if event_type != 'report_view':
        data['event_type'] = event_type
    
    print(f"📊 发送完整Analytics数据: {analytics_id}")
    print(f"👤 用户Hash: {user_hash}")
    print(f"📍 详细位置: {data['city']}, {data['regionName']} ({data['region']}), {data['country']}")
    print(f"📮 邮编: {data['zip']}")
    print(f"🕐 时区: {data['tz']}")  
    print(f"🌐 语言: {data['lang']}")  
    print(f"🌐 ISP: {data['isp']}")
    print(f"🏢 组织: {data['org']}")
    print(f"📍 精确坐标: ({data['lat']}, {data['lon']})")
    print(f"🔌 AS信息: {data['as']}")
    print(f"⏰ 时间: {timestamp}")
    
    # 发送请求
    analytics_endpoint = os.environ.get('OMICVERSE_ANALYTICS_ENDPOINT', 'http://8.130.139.217/track.gif')
    
    try:
        params = urlencode(data)
        url = f"{analytics_endpoint}?{params}"
        
        print(f"📡 发送到: {analytics_endpoint}")
        print(f"📋 参数数量: {len(data)} 个字段")
        print(f"🔍 核心参数: lang={data['lang']}, tz={data['tz']}, country={data['country']}")
        print(f"🌍 地理详情: city={data['city']}, region={data['region']}, regionName={data['regionName']}")
        print(f"📍 坐标与ISP: lat={data['lat']}, lon={data['lon']}, isp={data['isp']}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ 完整Analytics数据发送成功!")
            print(f"📊 发送了 {len(data)} 个字段，完全匹配ip-api.com结构")
            return True
        else:
            print(f"❌ 发送失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 发送失败: {str(e)}")
        return False


def test_analytics_connection():
    """
    测试与analytics服务器的连接
    """
    print("🧪 测试Analytics服务器连接...")
    
    # 测试健康检查端点
    health_endpoint = 'http://8.130.139.217/health'
    try:
        response = requests.get(health_endpoint, timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 服务器健康: {health_data.get('status')}")
            print(f"📅 服务器时间: {health_data.get('timestamp')}")
            print(f"🔖 服务器版本: {health_data.get('version')}")
        else:
            print(f"⚠️ 健康检查失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return False
    
    # 测试debug端点
    debug_endpoint = 'http://8.130.139.217/debug'
    try:
        test_params = {'test': 'connection', 'country': 'TestCountry'}
        response = requests.get(debug_endpoint, params=test_params, timeout=5)
        if response.status_code == 200:
            debug_data = response.json()
            print(f"🔍 Debug测试成功")
            print(f"📊 收到参数: {debug_data.get('url_params', {})}")
        else:
            print(f"⚠️ Debug测试失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Debug测试失败: {e}")
    
    return True


def send_analytics_silent(analytics_id, event_type='report_view', **kwargs):
    """
    静默版发送函数 - 不输出任何信息
    
    Parameters:
    -----------
    analytics_id : str
        项目或报告的唯一标识符
    event_type : str
        事件类型，默认为'report_view'
    **kwargs : dict
        额外的参数
        
    Returns:
    --------
    bool : 发送是否成功
    
    Note:
    -----
    与send_analytics功能完全相同，但不输出任何调试信息
    """
    
    try:
        # 基础数据收集
        user_hash = generate_user_hash()
        timestamp = datetime.now().isoformat()
        platform_info = platform.system() + " " + platform.release()
        
        # 获取地理位置
        geo_data = get_geolocation()
        
        # 构建数据包
        data = {
            # 核心字段  
            'id': analytics_id,
            'user': user_hash,
            'ts': timestamp,
            
            # 基础环境信息
            'platform': platform_info,
            'ua': f'Python-OmicVerse/{platform.python_version()}',
            'lang': 'en-US',
            'tz': geo_data.get('timezone', 'UTC'),
            'country': geo_data.get('country', 'Unknown'),
            'ref': '',
            
            **kwargs
        }
        
        if event_type != 'report_view':
            data['event_type'] = event_type
        
        # 发送请求
        analytics_endpoint = os.environ.get('OMICVERSE_ANALYTICS_ENDPOINT', 'http://8.130.139.217/track.gif')
        
        params = urlencode(data)
        url = f"{analytics_endpoint}?{params}"
        
        response = requests.get(url, timeout=10)
        return response.status_code == 200
        
    except:
        return False


def send_analytics_full_silent(analytics_id, event_type='report_view', **kwargs):
    """
    静默版完整发送函数 - 不输出任何信息
    
    Parameters:
    -----------
    analytics_id : str
        项目或报告的唯一标识符
    event_type : str
        事件类型，默认为'report_view'
    **kwargs : dict
        额外的参数
        
    Returns:
    --------
    bool : 发送是否成功
    
    Note:
    -----
    与send_analytics_full功能完全相同，但不输出任何调试信息
    """
    
    try:
        # 基础数据收集
        user_hash = generate_user_hash()
        timestamp = datetime.now().isoformat()
        platform_info = platform.system() + " " + platform.release()
        
        # 获取完整地理位置
        geo_data = get_geolocation()
        
        # 构建完整数据包
        data = {
            # 核心字段
            'id': analytics_id,
            'user': user_hash,
            'ts': timestamp,
            
            # 基础环境信息 
            'platform': platform_info,
            'ua': f'Python-OmicVerse/{platform.python_version()}',
            'lang': 'en-US',  
            'tz': geo_data.get('timezone', 'UTC'),
            'timezone': geo_data.get('timezone', 'UTC'),
            'ref': '',
            
            # 基础地理信息
            'country': geo_data.get('country', 'Unknown'),
            
            # 详细地理信息 - ip-api.com兼容
            'status': geo_data.get('status', 'success'),
            'countryCode': geo_data.get('countryCode', 'XX'),
            'region': geo_data.get('region', 'Unknown'),
            'regionName': geo_data.get('regionName', 'Unknown'),
            'city': geo_data.get('city', 'Unknown'),
            'zip': geo_data.get('zip', ''),
            'lat': geo_data.get('lat', 0),
            'lon': geo_data.get('lon', 0),
            'isp': geo_data.get('isp', 'Unknown'),
            'org': geo_data.get('org', 'Unknown'),
            'as': geo_data.get('as', 'Unknown'),
            'query': geo_data.get('query', 'Unknown'),
            
            **kwargs
        }
        
        if event_type != 'report_view':
            data['event_type'] = event_type
        
        # 发送请求
        analytics_endpoint = os.environ.get('OMICVERSE_ANALYTICS_ENDPOINT', 'http://8.130.139.217/track.gif')
        
        params = urlencode(data)
        url = f"{analytics_endpoint}?{params}"
        
        response = requests.get(url, timeout=10)
        return response.status_code == 200
        
    except:
        return False


def send_analytics_simple_silent(analytics_id):
    """
    静默版简化发送函数 - 不输出任何信息
    
    Parameters:
    -----------
    analytics_id : str
        项目标识符
        
    Returns:
    --------
    bool : 发送是否成功
    """
    return send_analytics_silent(analytics_id)


if __name__ == "__main__":
    # 测试用例
    print("🧬 OmicVerse Analytics Sender 测试")
    print("=" * 50)
    
    # 测试连接
    if test_analytics_connection():
        print("\n" + "=" * 50)
        
        # 发送简化版测试数据
        test_id_simple = f"TEST-SIMPLE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\n📤 测试1: 标准版Analytics (包含基础环境信息)")
        print("-" * 30)
        success1 = send_analytics_simple(test_id_simple)
        
        print(f"\n" + "=" * 50)
        
        # 发送完整版测试数据
        test_id_full = f"TEST-FULL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\n📤 测试2: 完整版Analytics (包含详细地理信息)")
        print("-" * 30)
        success2 = send_analytics_full(test_id_full, project_name="Test Project", analysis_type="scRNA-seq")
        
        # 总结
        print(f"\n" + "=" * 50)
        print(f"📊 测试结果总结:")
        print(f"✅ 标准版: {'成功' if success1 else '失败'} - ID: {test_id_simple}")
        print(f"✅ 完整版: {'成功' if success2 else '失败'} - ID: {test_id_full}")
        
        if success1 or success2:
            print(f"\n🎉 测试成功! 你可以在dashboard中查看这些记录")
            print(f"📋 Dashboard: http://8.130.139.217/dashboard")
        else:
            print(f"\n❌ 所有测试失败")
    
    print(f"\n📖 使用方法:")
    print(f"# 标准版 (推荐日常使用，包含基础环境信息)")
    print(f"import omicverse.single._analytics_sender as sender")
    print(f"sender.send_analytics_simple('YOUR-PROJECT-ID')")
    print(f"")
    print(f"# 完整版 (包含详细地理信息：城市、ISP、坐标等)")
    print(f"sender.send_analytics_full('YOUR-PROJECT-ID', project_name='My Analysis')")
    print(f"")
    print(f"# 静默版本 (不输出任何调试信息)")
    print(f"success = sender.send_analytics_simple_silent('YOUR-PROJECT-ID')")
    print(f"success = sender.send_analytics_full_silent('YOUR-PROJECT-ID')")
    print(f"# 返回值: True=成功, False=失败") 