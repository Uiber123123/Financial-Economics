"""
主应用程序 - Streamlit Web界面
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# 导入自定义模块
import portfolio_optimizer as po
import backtesting as bt
import stock_utils as su
import database_manager as dbm

# 页面配置
st.set_page_config(
    page_title="金融经济学学习系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 简单的用户认证
def check_login():
    """简单的登录验证"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.title("🔐 登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        
        if st.button("登录"):
            # 简单验证（用户名: admin, 密码: admin123）
            if username == "张洲宁" and password == "202302234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("用户名或密码错误！")
        
        #st.info("默认账号: admin / admin123")
        return False
    
    return True


def main():
    """主函数"""
    
    if not check_login():
        return
    
    # 侧边栏
    st.sidebar.title("📊 金融经济学学习系统")
    st.sidebar.write(f"欢迎，{st.session_state.get('username', 'admin')}！")
    
    if st.sidebar.button("退出登录"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # 功能选择
    page = st.sidebar.radio(
        "选择功能",
        ["📥 数据下载", "📊 投资组合优化", "📈 回测分析"]
    )
    
    if page == "📥 数据下载":
        page_data_download()
    elif page == "📊 投资组合优化":
        page_portfolio_optimization()
    elif page == "📈 回测分析":
        page_backtesting()


def page_data_download():
    """数据下载页面"""
    st.title("📥 股票数据下载")
    
    # 加载股票列表
    if 'stock_list' not in st.session_state:
        st.session_state.stock_list = su.load_stock_list()
    
    stock_dict = su.create_stock_selector_dict(st.session_state.stock_list)
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_count = min(10, len(stock_dict))
        selected_stock_names = st.multiselect(
            "选择股票（可多选）",
            options=list(stock_dict.keys()),
            default=list(stock_dict.keys())[:default_count],
            help="可以通过搜索框快速查找股票"
        )
        stock_codes = [stock_dict[name] for name in selected_stock_names]
        st.info(f"已选择 {len(stock_codes)} 只股票")
    
    with col2:
        end_date = datetime(2025, 9, 30)
        start_date = end_date - timedelta(days=3653)
        date_start = st.date_input("开始日期", start_date)
        date_end = st.date_input("结束日期", end_date)
    
    if st.button("获取数据", type="primary"):
        with st.spinner("正在从数据库获取数据..."):
            start_str = date_start.strftime("%Y%m%d")
            end_str = date_end.strftime("%Y%m%d")
            
            db_manager = dbm.get_db_manager()
            data = db_manager.get_multiple_stocks_data(stock_codes, start_str, end_str)
            
            if not data.empty:
                st.success(f"成功获取 {len(stock_codes)} 只股票的数据！")
                
                # 存储到session state
                st.session_state.stock_data = data
                st.session_state.stock_codes = stock_codes
                st.session_state.date_range = (start_str, end_str)
                
                # 显示数据预览和统计
                st.subheader("数据预览")
                st.dataframe(data.head(20), width='stretch')
                
                st.subheader("数据统计")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("股票数量", len(stock_codes))
                col2.metric("总记录数", len(data))
                col3.metric("日期范围", f"{data['日期'].min().date()} 至 {data['日期'].max().date()}")
                col4.metric("平均交易日", len(data) // len(stock_codes))
                
                # 下载按钮
                csv = data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载CSV文件",
                    data=csv,
                    file_name=f"stock_data_{start_str}_{end_str}.csv",
                    mime="text/csv"
                )
            else:
                st.error("未能获取数据，请检查股票代码和日期！")


def page_portfolio_optimization():
    """投资组合优化页面"""
    st.title("📊 投资组合优化 - 马科维茨有效前沿")
    
    # 检查是否有数据
    if 'stock_data' not in st.session_state:
        st.warning("请先在'数据下载'页面获取股票数据！")
        return
    
    data = st.session_state.stock_data
    stock_codes = st.session_state.stock_codes
    stock_df = st.session_state.get('stock_list', pd.DataFrame())
    
    # 创建股票名称映射
    stock_names_map = {}
    if not stock_df.empty:
        for code in stock_codes:
            name = su.get_stock_name(stock_df, code)
            stock_names_map[code] = f"{code} - {name}"
    else:
        stock_names_map = {code: code for code in stock_codes}
    
    # 选择股票
    st.subheader("选择股票")
    selected_stock_displays = st.multiselect(
        "选择要分析的股票",
        options=[stock_names_map[code] for code in stock_codes],
        default=[stock_names_map[code] for code in stock_codes[:min(5, len(stock_codes))]],
        help="建议选择2-10只股票以获得最佳效果"
    )
    
    # 从显示名称提取代码
    selected_stocks = [code for code in stock_codes if stock_names_map[code] in selected_stock_displays]
    
    if len(selected_stocks) < 2:
        st.warning("请至少选择2只股票！")
        return
    
    # 参数设置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 日期范围选择
        data['日期'] = pd.to_datetime(data['日期'])
        date_range = st.slider(
            "选择时间范围",
            min_value=data['日期'].min().date(),
            max_value=data['日期'].max().date(),
            value=(data['日期'].min().date(), data['日期'].max().date())
        )
    
    with col2:
        # 风险厌恶系数
        risk_aversion = st.slider(
            "风险厌恶系数 A",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="效用函数: U = E(r) - 0.005*A*σ². A越大表示越厌恶风险"
        )
    
    with col3:
        # 做空约束选择
        allow_short = st.radio(
            "投资约束",
            options=["不允许做空", "允许做空"],
            index=0,
            help="允许做空：使用解析解（快速）\n不允许做空：使用数值优化"
        )
        allow_short_bool = (allow_short == "允许做空")
    
    if st.button("计算有效前沿", type="primary"):
        with st.spinner("正在计算..."):
            # 筛选数据
            mask = (data['日期'].dt.date >= date_range[0]) & (data['日期'].dt.date <= date_range[1])
            filtered_data = data[mask & data['股票代码'].isin(selected_stocks)]
            
            # 计算收益率（使用database_manager模块）
            filtered_data = dbm.calculate_returns(filtered_data)
            
            # 准备收益率矩阵
            returns_matrix = filtered_data.pivot(index='日期', columns='股票代码', values='收益率')
            returns_matrix = returns_matrix.dropna()
            
            if returns_matrix.empty or len(returns_matrix) < 20:
                st.error("数据不足，请选择更长的时间范围或更多股票！")
                return
            
            # 更新 selected_stocks 为实际可用的股票代码（防止dropna后长度不匹配）
            actual_stocks = returns_matrix.columns.tolist()
            if len(actual_stocks) < len(selected_stocks):
                st.warning(f"注意：有 {len(selected_stocks) - len(actual_stocks)} 只股票因数据不足被剔除")
                selected_stocks = actual_stocks
            
            # 计算均值和协方差
            mean_returns = returns_matrix.mean()
            cov_matrix = returns_matrix.cov()
            
            # 获取无风险利率（日几何平均利率）
            date_start_str = date_range[0].strftime("%Y%m%d")
            date_end_str = date_range[1].strftime("%Y%m%d")
            risk_free_rate = dbm.get_risk_free_rate(date_start_str, date_end_str)
            
            st.info(f"计算方法: {'解析解（允许做空）' if allow_short_bool else '数值优化（不允许做空）'}")
            st.info(f"无风险日收益率: {risk_free_rate:.6f} (年化约 {risk_free_rate*252:.2%})")
            
            # 统一使用数值优化方法计算
            method_name = "二次规划（允许做空）" if allow_short_bool else "二次规划（非负约束）"
            st.markdown(f"### 🔢 使用{method_name}求解")
            
            # 计算三种最优组合
            max_sharpe_result = po.max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, allow_short=allow_short_bool)
            min_var_result = po.min_variance(mean_returns, cov_matrix, allow_short=allow_short_bool)
            w_max_utility, error_utility = po.optimal_utility_portfolio(mean_returns, cov_matrix, risk_aversion, allow_short=allow_short_bool)
            
            if not max_sharpe_result.success or not min_var_result.success:
                st.error("优化失败！请选择相关性较低的股票或更长的时间范围。")
                return
            
            utility_result = {'x': w_max_utility} if w_max_utility is not None else None
            
            # 绘制有效前沿
            fig, _, _, _ = po.plot_efficient_frontier(
                mean_returns, cov_matrix, risk_free_rate, selected_stocks,
                risk_aversion=risk_aversion, use_analytical=allow_short_bool
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # 准备策略结果用于样本内回测
            strategies_results = {}
            
            # 显示最优投资组合
            def display_portfolio(title, icon, weights, stock_codes, show_utility=False):
                """统一显示投资组合信息"""
                st.markdown("---")
                st.subheader(f"{icon} {title}")
                
                weights_df = po.get_portfolio_weights_df(weights, stock_codes)
                p_return, p_std = po.calculate_portfolio_performance(weights, mean_returns, cov_matrix)
                sharpe = (p_return - risk_free_rate * 252) / p_std
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(weights_df, width='stretch')
                with col2:
                    st.metric("预期年化收益率", f"{p_return:.2%}")
                    st.metric("年化标准差（风险）", f"{p_std:.2%}")
                    st.metric("夏普比率", f"{sharpe:.2f}")
                    if show_utility:
                        utility = p_return - 0.005 * risk_aversion * (p_std ** 2)
                        st.metric("效用值", f"{utility:.4f}")
            
            method_suffix = "Short" if allow_short_bool else "No-Short"
            
            # 最大夏普比率组合
            display_portfolio("最优投资组合（最大夏普比率）", "🌟", max_sharpe_result.x, selected_stocks)
            strategies_results[f"Max Sharpe ({method_suffix})"] = {
                'weights': max_sharpe_result.x,
                'stock_codes': selected_stocks
            }
            
            # 最小方差组合
            display_portfolio("最小风险组合", "🛡️", min_var_result.x, selected_stocks)
            strategies_results[f"Min Variance ({method_suffix})"] = {
                'weights': min_var_result.x,
                'stock_codes': selected_stocks
            }
            
            # 效用最优组合
            if utility_result is not None:
                display_portfolio(f"效用最优组合（风险厌恶系数 A={risk_aversion}）", "💜", 
                                utility_result['x'], selected_stocks, show_utility=True)
                strategies_results[f"Max Utility ({method_suffix})"] = {
                    'weights': utility_result['x'],
                    'stock_codes': selected_stocks
                }
            
            # 样本内表现分析
            st.markdown("---")
            st.subheader("📊 样本内表现分析")
            
            # 绘制累计收益和滚动夏普比率
            perf_fig = po.plot_portfolio_performance_comparison(strategies_results, returns_matrix)
            st.plotly_chart(perf_fig, width='stretch')
            
            # 计算并显示性能指标表格
            st.subheader("📈 性能指标汇总")
            
            metrics_list = []
            for strategy_name, result in strategies_results.items():
                metrics = po.calculate_performance_metrics(
                    result['weights'], returns_matrix, result['stock_codes'], risk_free_rate
                )
                if metrics:
                    metrics['Strategy'] = strategy_name
                    metrics_list.append(metrics)
            
            if metrics_list:
                metrics_df = pd.DataFrame(metrics_list)
                cols = ['Strategy', 'Total Return', 'Annual Return', 'Annual Volatility', 
                       'Sharpe Ratio', 'Max Drawdown', 'Final Value']
                metrics_df = metrics_df[cols]
                
                st.dataframe(
                    metrics_df.style.format({
                        'Total Return': '{:.2%}',
                        'Annual Return': '{:.2%}',
                        'Annual Volatility': '{:.2%}',
                        'Sharpe Ratio': '{:.2f}',
                        'Max Drawdown': '{:.2%}',
                        'Final Value': '{:.2f}'
                    }),
                    width='stretch'
                )
                
                # 下载按钮
                csv = metrics_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载性能指标",
                    data=csv,
                    file_name=f"portfolio_performance_{'short' if allow_short_bool else 'noshort'}.csv",
                    mime="text/csv"
                )


def page_backtesting():
    """回测分析页面"""
    st.title("📈 样本外回测分析")
    
    # 检查是否有数据
    if 'stock_data' not in st.session_state:
        st.warning("请先在'数据下载'页面获取股票数据！")
        return
    
    data = st.session_state.stock_data
    stock_codes = st.session_state.stock_codes
    
    # 选择股票
    selected_stocks = st.multiselect(
        "选择回测股票",
        stock_codes,
        default=stock_codes[:min(5, len(stock_codes))]
    )
    
    if len(selected_stocks) < 2:
        st.warning("请至少选择2只股票！")
        return
    
    # 设置参数
    col1, col2 = st.columns(2)
    
    with col1:
        train_ratio = st.slider("训练集比例", 0.5, 0.8, 0.7, 0.05)
    
    with col2:
        rebalance_freq = st.selectbox("再平衡频率", ["不再平衡", "月度", "季度"])
    
    if st.button("运行回测", type="primary"):
        with st.spinner("正在运行回测..."):
            # 筛选数据
            filtered_data = data[data['股票代码'].isin(selected_stocks)].copy()
            filtered_data = dbm.calculate_returns(filtered_data)
            
            # 准备收益率矩阵
            returns_matrix = filtered_data.pivot(index='日期', columns='股票代码', values='收益率')
            returns_matrix = returns_matrix.dropna()
            
            if len(returns_matrix) < 60:
                st.error("数据不足，请选择更长的时间范围！")
                return
            
            # 分割训练集和测试集
            train_data, test_data = bt.split_train_test(returns_matrix, train_ratio)
            
            st.info(f"训练集: {train_data.index[0].date()} 至 {train_data.index[-1].date()} ({len(train_data)} 天)")
            st.info(f"测试集: {test_data.index[0].date()} 至 {test_data.index[-1].date()} ({len(test_data)} 天)")
            
            # 在训练集上优化
            mean_returns = train_data.mean()
            cov_matrix = train_data.cov()
            
            # 获取无风险利率（使用训练集日期范围）
            train_start = train_data.index[0].strftime("%Y%m%d")
            train_end = train_data.index[-1].strftime("%Y%m%d")
            risk_free_rate = dbm.get_risk_free_rate(train_start, train_end)
            
            # 计算不同策略
            max_sharpe_result = po.max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
            min_var_result = po.min_variance(mean_returns, cov_matrix)
            equal_weight = pd.Series(1.0 / len(selected_stocks), index=selected_stocks)
            
            strategies = {
                'Max Sharpe': pd.Series(max_sharpe_result.x, index=selected_stocks),
                'Min Variance': pd.Series(min_var_result.x, index=selected_stocks),
                'Equal Weight': equal_weight
            }
            
            # 回测
            backtest_results = bt.backtest_multiple_strategies(strategies, test_data, risk_free_rate)
            
            # 显示累计收益图
            st.subheader("📊 累计收益对比")
            fig = bt.plot_backtest_results(backtest_results)
            st.plotly_chart(fig, width='stretch')
            
            # 显示性能指标
            st.subheader("📈 性能指标对比")
            metrics_df = bt.create_metrics_comparison_table(backtest_results)
            
            st.dataframe(
                metrics_df.style.format({
                    'Total Return': '{:.2%}',
                    'Annual Return': '{:.2%}',
                    'Annual Volatility': '{:.2%}',
                    'Sharpe Ratio': '{:.2f}',
                    'Sortino Ratio': '{:.2f}',
                    'Max Drawdown': '{:.2%}',
                    'Calmar Ratio': '{:.2f}',
                    'Win Rate': '{:.2%}'
                }),
                width='stretch'
            )
            
            # 性能指标柱状图
            st.subheader("📊 关键指标可视化")
            fig2 = bt.plot_metrics_comparison(metrics_df)
            st.plotly_chart(fig2, width='stretch')
            
            # 滚动夏普比率
            st.subheader("📉 滚动夏普比率（60天窗口）")
            fig3 = bt.plot_rolling_metrics(backtest_results, window=60)
            st.plotly_chart(fig3, width='stretch')
            
            # 下载回测结果
            csv = metrics_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载回测结果",
                data=csv,
                file_name="backtest_results.csv",
                mime="text/csv"
            )
            
            # 最优策略推荐
            best_strategy = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax(), 'Strategy']
            st.success(f"🌟 推荐策略: {best_strategy} （基于最高夏普比率）")


if __name__ == "__main__":
    main()

