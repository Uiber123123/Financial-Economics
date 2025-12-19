"""
数据库构建脚本 - 将Excel文件合并到SQLite数据库
"""
import pandas as pd
import sqlite3
import os
from glob import glob
from datetime import datetime


def create_database(db_path='stock_data.db'):
    """
    创建SQLite数据库
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建股票价格表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            stock_code TEXT NOT NULL,
            date DATE NOT NULL,
            adj_close REAL NOT NULL,
            PRIMARY KEY (stock_code, date)
        )
    ''')
    
    # 创建索引以加速查询
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stock_code ON stock_prices(stock_code)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_date ON stock_prices(date)
    ''')
    
    # 创建三因子数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS factor_returns (
            date DATE PRIMARY KEY,
            rmrf_tmv REAL,
            smb_tmv REAL,
            hml_tmv REAL,
            rmrf_mc REAL,
            smb_mc REAL,
            hml_mc REAL
        )
    ''')
    
    conn.commit()
    return conn


def load_stock_data_files(data_dir='Data Source Construction', conn=None):
    """
    加载所有股票数据文件
    """
    pattern = os.path.join(data_dir, 'RESSET_DRESSTK_*.xlsx')
    files = glob(pattern)
    
    print(f"找到 {len(files)} 个股票数据文件")
    
    total_records = 0
    chunk_size = 5000  # 每次插入的最大行数
    
    for i, file_path in enumerate(files, 1):
        try:
            print(f"正在处理 [{i}/{len(files)}]: {os.path.basename(file_path)}", end='')
            
            # 读取Excel文件
            df = pd.read_excel(file_path)
            
            # 处理不同列数的文件
            num_cols = len(df.columns)
            
            if num_cols == 5:
                # 2021年后的数据：5列，忽略第1列和第3列
                # 保留第2列(股票代码)、第4列(日期)、第5列(复权价格)
                df = df.iloc[:, [1, 3, 4]]
                df.columns = ['stock_code', 'date', 'adj_close']
            elif num_cols == 3:
                # 2021年前的数据：3列
                df.columns = ['stock_code', 'date', 'adj_close']
            else:
                print(f" - 跳过：列数异常({num_cols}列)")
                continue
            
            # 转换股票代码为6位字符串格式（补零）
            df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
            
            # 处理日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 清理数据
            df = df.dropna()
            df = df[df['adj_close'] > 0]  # 移除非正价格
            
            # 分批写入数据库以避免SQLite变量限制
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(df))
                chunk = df.iloc[start_idx:end_idx]
                
                chunk.to_sql('stock_prices', conn, if_exists='append', index=False)
                
                # 每处理10个chunk提交一次
                if (chunk_idx + 1) % 10 == 0:
                    conn.commit()
            
            total_records += len(df)
            print(f" - 完成，{len(df)} 条记录")
            
        except Exception as e:
            print(f" - 错误: {e}")
            continue
    
    print(f"\n总共导入 {total_records} 条股票价格记录")
    return total_records


def load_factor_data(data_dir='Data Source Construction', conn=None):
    """
    加载三因子数据
    """
    file_path = os.path.join(data_dir, 'RESSET_THRFACDAT_DAILY_1.xlsx')
    
    if not os.path.exists(file_path):
        print(f"三因子数据文件不存在: {file_path}")
        return 0
    
    print(f"正在处理三因子数据: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_excel(file_path)
        
        # 重命名列
        df.columns = ['date', 'rmrf_tmv', 'smb_tmv', 'hml_tmv', 'rmrf_mc', 'smb_mc', 'hml_mc']
        
        # 处理日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        # 清理数据
        df = df.dropna()
        
        # 写入数据库
        df.to_sql('factor_returns', conn, if_exists='replace', index=False)
        
        print(f"导入 {len(df)} 条因子数据记录")
        return len(df)
        
    except Exception as e:
        print(f"加载三因子数据失败: {e}")
        return 0


def get_database_stats(conn):
    """
    获取数据库统计信息
    """
    cursor = conn.cursor()
    
    # 股票数量
    cursor.execute("SELECT COUNT(DISTINCT stock_code) FROM stock_prices")
    num_stocks = cursor.fetchone()[0]
    
    # 总记录数
    cursor.execute("SELECT COUNT(*) FROM stock_prices")
    num_records = cursor.fetchone()[0]
    
    # 日期范围
    cursor.execute("SELECT MIN(date), MAX(date) FROM stock_prices")
    date_range = cursor.fetchone()
    
    # 因子数据记录数
    cursor.execute("SELECT COUNT(*) FROM factor_returns")
    num_factors = cursor.fetchone()[0]
    
    print("\n" + "="*60)
    print("数据库统计信息")
    print("="*60)
    print(f"股票数量: {num_stocks}")
    print(f"价格记录数: {num_records:,}")
    print(f"日期范围: {date_range[0]} 至 {date_range[1]}")
    print(f"因子数据记录数: {num_factors}")
    print("="*60)


def main():
    """
    主函数
    """
    print("="*60)
    print("股票数据库构建程序")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = datetime.now()
    
    # 检查数据目录
    if not os.path.exists('Data Source Construction'):
        print("错误: 未找到 'Data Source Construction' 目录")
        print("请确保数据文件位于该目录下")
        return
    
    # 检查数据库是否已存在
    if os.path.exists('stock_data.db'):
        print("⚠️  警告: 数据库文件已存在！")
        print("   当前数据库将被保留，新数据将追加到现有数据中")
        print("   如需重新构建，请先删除 stock_data.db 文件\n")
    
    # 创建数据库
    print("步骤 1: 创建/连接数据库...")
    conn = create_database('stock_data.db')
    print("数据库准备完成\n")
    
    # 加载股票数据
    print("步骤 2: 加载股票价格数据...")
    print("提示: 这可能需要5-15分钟，请耐心等待...\n")
    load_stock_data_files('Data Source Construction', conn)
    print()
    
    # 加载因子数据
    print("步骤 3: 加载三因子数据...")
    load_factor_data('Data Source Construction', conn)
    print()
    
    # 显示统计信息
    print("步骤 4: 生成统计信息...")
    get_database_stats(conn)
    
    # 提交并关闭
    print("\n正在提交数据...")
    conn.commit()
    conn.close()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration.total_seconds():.1f} 秒")
    print("\n✅ 数据库构建完成！")
    print(f"   数据库文件: stock_data.db")
    print(f"   文件大小: {os.path.getsize('stock_data.db') / (1024*1024):.1f} MB")
    print("\n现在可以运行 'streamlit run app.py' 启动系统")


if __name__ == "__main__":
    main()

