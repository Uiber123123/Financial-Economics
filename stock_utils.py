"""
股票工具模块 - 处理股票列表和选择
"""
import json
import pandas as pd


def load_stock_list(file_path='Sticky_Stock.txt'):
    """
    从JSON文件加载股票列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 处理代码格式 - 移除后缀.SZ/.SH
        df['code'] = df['dm'].str.split('.').str[0]
        df['name'] = df['mc']
        df['exchange'] = df['jys']
        
        # 创建显示名称（代码-名称）
        df['display_name'] = df['code'] + ' - ' + df['name']
        
        return df[['code', 'name', 'exchange', 'display_name']]
    
    except Exception as e:
        print(f"加载股票列表失败: {e}")
        # 返回默认股票列表
        default_stocks = [
            {'code': '000001', 'name': '平安银行', 'exchange': 'SZ', 'display_name': '000001 - 平安银行'},
            {'code': '000002', 'name': '万科A', 'exchange': 'SZ', 'display_name': '000002 - 万科A'},
            {'code': '600000', 'name': '浦发银行', 'exchange': 'SH', 'display_name': '600000 - 浦发银行'},
        ]
        return pd.DataFrame(default_stocks)


def create_stock_selector_dict(stock_df):
    """
    创建股票选择器的字典 {显示名称: 代码}
    """
    return dict(zip(stock_df['display_name'], stock_df['code']))


def get_stock_name(stock_df, code):
    """
    根据代码获取股票名称
    """
    result = stock_df[stock_df['code'] == code]['name']
    if len(result) > 0:
        return result.iloc[0]
    return code

