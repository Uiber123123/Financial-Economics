"""
投资组合优化模块 - 马科维茨理论和有效前沿
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import plotly.graph_objects as go


def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    计算投资组合的收益率和风险
    """
    portfolio_return = np.sum(mean_returns * weights) * 252  # 年化收益
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # 年化标准差
    return portfolio_return, portfolio_std


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    计算负夏普比率（用于最小化）
    """
    p_return, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (p_return - risk_free_rate * 252) / p_std
    return -sharpe


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, allow_short=False):
    """
    计算最大夏普比率的投资组合（使用数值优化）
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if allow_short:
        bounds = tuple((-2, 2) for _ in range(num_assets))
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        args=args,
        options={'maxiter': 1000}
    )
    
    return result


def min_variance(mean_returns, cov_matrix, allow_short=False):
    """
    计算最小方差投资组合（使用数值优化）
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if allow_short:
        bounds = tuple((-2, 2) for _ in range(num_assets))
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(
        lambda w, mr, cm: calculate_portfolio_performance(w, mr, cm)[1],
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        args=args,
        options={'maxiter': 1000}
    )
    
    return result


def efficient_frontier_qp(mean_returns, cov_matrix, num_portfolios=100, allow_short=True):
    """
    使用二次规划计算有效前沿（完整的双曲线）
    基于数值优化方法而非解析解
    """
    num_assets = len(mean_returns)
    
    # 首先找到最小方差组合
    min_var_result = min_variance(mean_returns, cov_matrix, allow_short=allow_short)
    if not min_var_result.success:
        return None, None, "无法计算最小方差组合"
    
    min_return, min_std = calculate_portfolio_performance(min_var_result.x, mean_returns, cov_matrix)
    
    # 找到最大收益率（单一资产或组合）
    max_return = np.max(mean_returns) * 252
    
    # 生成目标收益率范围 - 从最小方差点向两侧延伸，展示双曲线
    # 下半部分：从一个较低的收益率到最小方差收益率
    lower_returns = np.linspace(min_return * 0.5, min_return, num_portfolios // 3)
    # 上半部分（有效前沿）：从最小方差收益率到最大收益率
    upper_returns = np.linspace(min_return, max_return * 0.95, num_portfolios * 2 // 3)
    target_returns = np.concatenate([lower_returns[:-1], upper_returns])
    
    results_std = []
    results_return = []
    
    for target in target_returns:
        # 约束条件：权重和为1，目标收益率等于target
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix)[0] - target}
        ]
        
        # 边界条件
        if allow_short:
            bounds = tuple((-2, 2) for _ in range(num_assets))  # 允许做空，但限制范围避免极端值
        else:
            bounds = tuple((0, 1) for _ in range(num_assets))
        
        initial_guess = num_assets * [1. / num_assets]
        
        # 最小化方差
        result = minimize(
            lambda w, mr, cm: calculate_portfolio_performance(w, mr, cm)[1],
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            args=(mean_returns, cov_matrix),
            options={'maxiter': 1000}
        )
        
        if result.success:
            p_return, p_std = calculate_portfolio_performance(result.x, mean_returns, cov_matrix)
            results_return.append(p_return)
            results_std.append(p_std)
        else:
            # 如果优化失败，跳过这个点
            continue
    
    if len(results_return) < 10:
        return None, None, "有效前沿计算点数不足"
    
    return np.array(results_return), np.array(results_std), None


def efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_portfolios=100):
    """
    计算有效前沿（带非负约束版本）
    """
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    # 获取最小和最大收益率
    min_var_result = min_variance(mean_returns, cov_matrix)
    min_return, _ = calculate_portfolio_performance(min_var_result.x, mean_returns, cov_matrix)
    max_return = np.max(mean_returns) * 252
    
    target_returns = np.linspace(min_return, max_return * 0.95, num_portfolios)
    
    for i, target in enumerate(target_returns):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix)[0] - target}
        )
        
        num_assets = len(mean_returns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]
        
        result = minimize(
            lambda w, mr, cm: calculate_portfolio_performance(w, mr, cm)[1],
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            args=(mean_returns, cov_matrix),
            options={'maxiter': 500}
        )
        
        if result.success:
            p_return, p_std = calculate_portfolio_performance(result.x, mean_returns, cov_matrix)
            sharpe = (p_return - risk_free_rate * 252) / p_std
            
            results[0, i] = p_std
            results[1, i] = p_return
            results[2, i] = sharpe
            weights_record.append(result.x)
        else:
            results[:, i] = np.nan
            weights_record.append(np.full(num_assets, np.nan))
    
    return results, weights_record


def generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, num_portfolios=5000):
    """
    生成随机投资组合（用于可视化）
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        p_return, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (p_return - risk_free_rate * 252) / p_std
        
        results[0, i] = p_std
        results[1, i] = p_return
        results[2, i] = sharpe
    
    return results


def optimal_utility_portfolio(mean_returns, cov_matrix, risk_aversion, allow_short=False):
    """
    计算效用函数最优组合（使用数值优化）
    U = E(r) - 0.005 * A * sigma^2
    其中A为风险厌恶系数
    """
    def negative_utility(w, mr, cm, A):
        p_ret, p_std = calculate_portfolio_performance(w, mr, cm)
        utility = p_ret - 0.005 * A * (p_std ** 2)
        return -utility
    
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if allow_short:
        bounds = tuple((-2, 2) for _ in range(num_assets))
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(
        negative_utility,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        args=(mean_returns, cov_matrix, risk_aversion),
        options={'maxiter': 1000}
    )
    
    if result.success:
        return result.x, None
    else:
        return None, "优化失败"


def backtest_portfolio_performance(weights, returns_df, stock_codes):
    """
    计算投资组合的样本内表现
    """
    # 确保weights和returns_df的列对应
    common_stocks = [code for code in stock_codes if code in returns_df.columns]
    
    if len(common_stocks) == 0:
        return None, None
    
    weights_aligned = weights[returns_df.columns.get_indexer(common_stocks)]
    weights_aligned = weights_aligned / np.sum(weights_aligned)
    
    # 计算组合收益
    portfolio_returns = (returns_df[common_stocks] * weights_aligned).sum(axis=1)
    
    # 计算累计收益
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # 计算滚动夏普比率（60天窗口）
    rolling_sharpe = (portfolio_returns.rolling(window=60).mean() / 
                     portfolio_returns.rolling(window=60).std() * np.sqrt(252))
    
    return cumulative_returns, rolling_sharpe


def plot_portfolio_performance_comparison(strategies_results, returns_df):
    """
    绘制多个策略的样本内表现对比
    strategies_results: {策略名称: weights}
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative Returns', 'Rolling Sharpe Ratio (60-Day)'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12
    )
    
    colors = {
        'Max Sharpe (Analytical)': 'green',
        'Max Sharpe (Numerical)': 'darkgreen',
        'Min Variance (Analytical)': 'blue',
        'Min Variance (Numerical)': 'darkblue',
        'Max Utility (Analytical)': 'purple',
        'Max Utility (Numerical)': 'darkmagenta'
    }
    
    for strategy_name, result in strategies_results.items():
        weights = result['weights']
        stock_codes = result['stock_codes']
        
        cum_returns, rolling_sharpe = backtest_portfolio_performance(weights, returns_df, stock_codes)
        
        if cum_returns is not None:
            # 累计收益曲线
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns,
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=colors.get(strategy_name, 'gray'), width=2),
                    hovertemplate=f'<b>{strategy_name}</b><br>Date: %{{x}}<br>Cumulative Return: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 滚动夏普比率
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=colors.get(strategy_name, 'gray'), width=1.5),
                    showlegend=False,
                    hovertemplate=f'<b>{strategy_name}</b><br>Date: %{{x}}<br>Sharpe: %{{y:.2f}}<extra></extra>'
                ),
                row=2, col=1
            )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Rolling Sharpe Ratio", row=2, col=1)
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def calculate_performance_metrics(weights, returns_df, stock_codes, risk_free_rate=0):
    """
    计算投资组合的性能指标
    """
    common_stocks = [code for code in stock_codes if code in returns_df.columns]
    if len(common_stocks) == 0:
        return None
    
    weights_aligned = weights[returns_df.columns.get_indexer(common_stocks)]
    weights_aligned = weights_aligned / np.sum(weights_aligned)
    
    portfolio_returns = (returns_df[common_stocks] * weights_aligned).sum(axis=1)
    
    # 计算指标
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate * 252) / annual_vol if annual_vol > 0 else 0
    
    # 最大回撤
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Final Value': cumulative.iloc[-1] if len(cumulative) > 0 else 1.0
    }


def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, stock_names, 
                           risk_aversion=None, use_analytical=True):
    """
    绘制有效前沿图（全部使用数值优化方法）
    - use_analytical=True: 允许做空
    - use_analytical=False: 不允许做空
    """
    allow_short = use_analytical
    
    # 创建图表
    fig = go.Figure()
    
    # ========== 计算有效前沿双曲线（使用二次规划） ==========
    mu_frontier, sigma_frontier, error_msg = efficient_frontier_qp(
        mean_returns, cov_matrix, num_portfolios=200, allow_short=allow_short
    )
    
    if error_msg:
        # 如果计算失败，显示错误
        fig.add_annotation(
            text=f"⚠️ {error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,255,0,0.5)"
        )
        return fig, None, None, None
    
    # 找到最小方差点的位置，分割上下两部分
    min_var_idx = np.argmin(sigma_frontier)
    
    # 下半部分（无效前沿）
    if min_var_idx > 0:
        fig.add_trace(go.Scatter(
            x=sigma_frontier[:min_var_idx+1],
            y=mu_frontier[:min_var_idx+1],
            mode='lines',
            line=dict(color='lightcoral', width=2, dash='dot'),
            name='Inefficient Frontier',
            hovertemplate='<b>Inefficient Frontier</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
    
    # 上半部分（有效前沿）
    fig.add_trace(go.Scatter(
        x=sigma_frontier[min_var_idx:],
        y=mu_frontier[min_var_idx:],
        mode='lines',
        line=dict(color='red', width=4),
        name='Efficient Frontier',
        hovertemplate='<b>Efficient Frontier</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))
    
    # ========== 计算关键组合（使用数值优化） ==========
    max_sharpe_result = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, allow_short=allow_short)
    min_var_result = min_variance(mean_returns, cov_matrix, allow_short=allow_short)
    
    max_sharpe_return, max_sharpe_std = calculate_portfolio_performance(
        max_sharpe_result.x, mean_returns, cov_matrix
    )
    min_var_return, min_var_std = calculate_portfolio_performance(
        min_var_result.x, mean_returns, cov_matrix
    )
    
    # 最大夏普比率组合
    fig.add_trace(go.Scatter(
        x=[max_sharpe_std],
        y=[max_sharpe_return],
        mode='markers',
        marker=dict(color='green', size=15, symbol='star'),
        name='Max Sharpe Ratio',
        hovertemplate='<b>Max Sharpe Portfolio</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))
    
    # 最小方差组合
    fig.add_trace(go.Scatter(
        x=[min_var_std],
        y=[min_var_return],
        mode='markers',
        marker=dict(color='blue', size=15, symbol='diamond'),
        name='Min Variance',
        hovertemplate='<b>Min Variance Portfolio</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))
    
    # 效用最优组合
    utility_result = None
    utility_return = None
    utility_std = None
    
    if risk_aversion is not None:
        w_utility, error_msg_utility = optimal_utility_portfolio(
            mean_returns, cov_matrix, risk_aversion, allow_short=allow_short
        )
        if not error_msg_utility and w_utility is not None:
            utility_return, utility_std = calculate_portfolio_performance(
                w_utility, mean_returns, cov_matrix
            )
            fig.add_trace(go.Scatter(
                x=[utility_std],
                y=[utility_return],
                mode='markers',
                marker=dict(color='purple', size=15, symbol='pentagon'),
                name=f'Optimal Utility (A={risk_aversion})',
                hovertemplate=f'<b>Utility Optimal (A={risk_aversion})</b><br>Risk: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>'
            ))
            utility_result = {'x': w_utility}
    
    # 添加无风险资产点
    rf_annual = risk_free_rate * 252
    fig.add_trace(go.Scatter(
        x=[0],
        y=[rf_annual],
        mode='markers',
        marker=dict(color='black', size=12, symbol='x', line=dict(width=2)),
        name='Risk-Free Asset',
        hovertemplate=f'<b>Risk-Free Asset</b><br>Return: {rf_annual:.2%}<extra></extra>'
    ))
    
    # 添加资本市场线（CML）- 缩短范围以突出双曲线
    sharpe_slope = (max_sharpe_return - rf_annual) / max_sharpe_std if max_sharpe_std > 0 else 0
    
    # 限制CML范围：只到切点组合的1.2倍，避免过长的直线掩盖双曲线
    cml_x = np.linspace(0, max_sharpe_std * 1.2, 50)
    cml_y = rf_annual + sharpe_slope * cml_x
    
    fig.add_trace(go.Scatter(
        x=cml_x,
        y=cml_y,
        mode='lines',
        line=dict(color='darkgreen', width=2, dash='dash'),
        name='CML (with Risk-Free Asset)',
        hovertemplate='<b>Capital Market Line</b><br>Risk: %{x:.2%}<br>Expected Return: %{y:.2%}<extra></extra>'
    ))
    
    # 自动调整坐标轴范围 - 突出双曲线
    all_x = list(sigma_frontier)
    all_y = list(mu_frontier)
    all_x.extend([0, max_sharpe_std, min_var_std])
    all_y.extend([rf_annual, max_sharpe_return, min_var_return])
    
    if utility_std is not None and utility_return is not None:
        all_x.append(utility_std)
        all_y.append(utility_return)
    
    # 设置坐标轴范围 - 重点展示双曲线区域
    x_min = 0
    x_max = max(sigma_frontier) * 1.1  # 以双曲线的最大值为基准
    y_range = max(all_y) - min(all_y)
    y_min = min(all_y) - y_range * 0.05
    y_max = max(all_y) + y_range * 0.05
    
    # 构建标题
    if allow_short:
        title = 'Efficient Frontier - Hyperbola (Numerical QP - Shorting Allowed)'
    else:
        title = 'Efficient Frontier - Constrained (Numerical QP - No Shorting)'
    
    fig.update_layout(
        title=title,
        xaxis_title='Risk (Annual Std Dev)',
        yaxis_title='Expected Return (Annual)',
        xaxis=dict(
            tickformat='.1%',
            range=[x_min, x_max]
        ),
        yaxis=dict(
            tickformat='.1%',
            range=[y_min, y_max]
        ),
        hovermode='closest',
        showlegend=True,
        height=650,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    return fig, max_sharpe_result, min_var_result, utility_result


def get_portfolio_weights_df(weights, stock_codes, stock_names=None):
    """
    将投资组合权重转换为DataFrame
    """
    if stock_names is None:
        stock_names = stock_codes
    
    df = pd.DataFrame({
        'Stock Code': stock_codes,
        'Stock Name': stock_names,
        'Weight': weights,
        'Weight %': weights * 100
    })
    df = df.sort_values('Weight', ascending=False)
    df = df[df['Weight'] > 0.001]  # 只显示权重大于0.1%的股票
    return df

