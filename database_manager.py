"""
数据库管理模块 - 从SQLite数据库读取股票和因子数据
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


class DatabaseManager:
    """
    数据库管理器
    """
    
    def __init__(self, db_path='stock_data.db'):
        """初始化数据库连接"""
        self.db_path = db_path
    
    def _get_connection(self):
        """获取数据库连接（线程安全）"""
        try:
            # 使用check_same_thread=False以支持多线程（只读操作是安全的）
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return conn
        except Exception as e:
            raise Exception(f"无法连接到数据库 {self.db_path}: {e}")
    
    def close(self):
        """关闭数据库连接（保留用于兼容性）"""
        pass
    
    def get_stock_data(self, stock_code, start_date, end_date):
        """
        获取单只股票的价格数据
        
        参数:
            stock_code: 股票代码（字符串）
            start_date: 开始日期（字符串，格式：YYYY-MM-DD 或 YYYYMMDD）
            end_date: 结束日期（字符串，格式：YYYY-MM-DD 或 YYYYMMDD）
        
        返回:
            DataFrame: 包含日期、股票代码、收盘价的数据框
        """
        # 转换日期格式
        if len(start_date) == 8:  # YYYYMMDD
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        if len(end_date) == 8:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        
        query = """
            SELECT date, stock_code, adj_close
            FROM stock_prices
            WHERE stock_code = ?
            AND date BETWEEN ? AND ?
            ORDER BY date
        """
        
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=(stock_code, start_date, end_date))
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.columns = ['日期', '股票代码', '收盘']
                
            return df
        except Exception as e:
            print(f"查询股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_multiple_stocks_data(self, stock_codes, start_date, end_date):
        """
        获取多只股票的价格数据
        """
        all_data = []
        
        for code in stock_codes:
            df = self.get_stock_data(code, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def get_factor_returns(self, start_date=None, end_date=None, weight_type='tmv'):
        """
        获取三因子收益率数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            weight_type: 'tmv' (流通市值加权) 或 'mc' (总市值加权)
        
        返回:
            DataFrame: 包含日期和三因子收益率
        """
        # 转换日期格式
        if start_date and len(start_date) == 8:
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        if end_date and len(end_date) == 8:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        
        # 选择列
        if weight_type == 'tmv':
            cols = "date, rmrf_tmv as MKT, smb_tmv as SMB, hml_tmv as HML"
        else:  # mc
            cols = "date, rmrf_mc as MKT, smb_mc as SMB, hml_mc as HML"
        
        if start_date and end_date:
            query = f"""
                SELECT {cols}
                FROM factor_returns
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            """
            params = (start_date, end_date)
        else:
            query = f"""
                SELECT {cols}
                FROM factor_returns
                ORDER BY date
            """
            params = ()
        
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
            return df
        except Exception as e:
            print(f"查询因子数据失败: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_available_stocks(self):
        """
        获取所有可用股票列表
        """
        query = """
            SELECT DISTINCT stock_code
            FROM stock_prices
            ORDER BY stock_code
        """
        
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
            return df['stock_code'].tolist()
        except Exception as e:
            print(f"查询股票列表失败: {e}")
            return []
        finally:
            conn.close()
    
    def get_date_range(self):
        """
        获取数据库中的日期范围
        """
        query = """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM stock_prices
        """
        
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
            return df.iloc[0]['min_date'], df.iloc[0]['max_date']
        except Exception as e:
            print(f"查询日期范围失败: {e}")
            return None, None
        finally:
            conn.close()
    
    def get_stock_count(self):
        """
        获取股票数量
        """
        query = "SELECT COUNT(DISTINCT stock_code) as count FROM stock_prices"
        
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
            return df.iloc[0]['count']
        except Exception as e:
            print(f"查询股票数量失败: {e}")
            return 0
        finally:
            conn.close()


def calculate_returns(df):
    """
    计算股票收益率（对数收益率）
    """
    # 按股票代码和日期排序
    df = df.sort_values(['股票代码', '日期']).copy()
    
    # 计算对数收益率
    # 先计算每只股票的价格比率，然后取对数
    df['收益率'] = df.groupby('股票代码')['收盘'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    return df


def prepare_portfolio_data(stock_codes, start_date, end_date, db_manager=None):
    """
    准备投资组合所需的数据
    返回收益率矩阵
    """
    if db_manager is None:
        db_manager = DatabaseManager()
    
    # 获取股票数据
    df = db_manager.get_multiple_stocks_data(stock_codes, start_date, end_date)
    
    if df.empty:
        return None
    
    # 计算收益率
    df = calculate_returns(df)
    
    # 转换为宽格式（日期x股票）
    returns_matrix = df.pivot(index='日期', columns='股票代码', values='收益率')
    returns_matrix = returns_matrix.dropna()
    
    return returns_matrix


# 创建全局数据库管理器实例
_db_manager = None

def get_risk_free_rate(start_date, end_date):
    """
    获取无风险利率（框定日期内的日几何平均利率）
    从RESSET_BDDRFRET_1.xlsx文件读取
    """
    try:
        import os
        file_path = 'RESSET_BDDRFRET_1.xlsx'
        
        if not os.path.exists(file_path):
            print(f"无风险利率文件不存在: {file_path}，使用默认值")
            return 0.00001  # 默认日利率
        
        df_rf = pd.read_excel(file_path)
        
        # 转换日期格式
        if len(start_date) == 8:  # YYYYMMDD
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        if len(end_date) == 8:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # 假设文件中有日期和收益率列
        # 需要根据实际列名调整
        if '日期_Date' in df_rf.columns:
            df_rf['日期'] = pd.to_datetime(df_rf['日期_Date'])
        elif '日期' in df_rf.columns:
            df_rf['日期'] = pd.to_datetime(df_rf['日期'])
        
        # 筛选日期范围
        mask = (df_rf['日期'] >= start_date_dt) & (df_rf['日期'] <= end_date_dt)
        df_filtered = df_rf[mask]
        
        if len(df_filtered) == 0:
            print("未找到指定日期范围的无风险利率数据，使用默认值")
            return 0.00001
        
        # 获取无风险利率列（需要根据实际列名调整）
        if 'daily_rate' in df_filtered.columns:
            daily_rates = df_filtered['daily_rate']
        elif '日收益率' in df_filtered.columns:
            daily_rates = df_filtered['日收益率']
        else:
            # 尝试找到包含"收益率"的列
            rate_cols = [col for col in df_filtered.columns if '收益率' in col or 'rate' in col.lower()]
            if rate_cols:
                daily_rates = df_filtered[rate_cols[0]]
            else:
                print("未找到无风险利率列，使用默认值")
                return 0.00001
        
        # 计算日几何平均收益率
        # 几何平均: (1+r1)*(1+r2)*...*(1+rn))^(1/n) - 1
        if len(daily_rates) > 0:
            geometric_mean = np.prod(1 + daily_rates.values) ** (1 / len(daily_rates)) - 1
            return geometric_mean
        else:
            return 0.00001
            
    except Exception as e:
        print(f"获取无风险利率失败: {e}，使用默认值")
        return 0.00001  # 默认日利率约0.001%


def get_db_manager():
    """获取全局数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

