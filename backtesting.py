"""
回测模块 - 样本外投资组合回测和性能评估
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def split_train_test(returns_df, train_ratio=0.7):
    """
    分割训练集和测试集
    """
    split_point = int(len(returns_df) * train_ratio)
    train_data = returns_df.iloc[:split_point]
    test_data = returns_df.iloc[split_point:]
    return train_data, test_data


def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    计算夏普比率
    """
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=0):
    """
    计算索提诺比率（只考虑下行风险）
    """
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    return sortino


def calculate_max_drawdown(cumulative_returns):
    """
    计算最大回撤
    """
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_dd = drawdown.min()
    return max_dd


def calculate_win_rate(returns):
    """
    计算胜率
    """
    positive_days = (returns > 0).sum()
    total_days = len(returns[returns != 0])
    if total_days == 0:
        return 0
    win_rate = positive_days / total_days
    return win_rate


def calculate_calmar_ratio(returns, cumulative_returns):
    """
    计算卡玛比率（年化收益/最大回撤）
    """
    annual_return = (1 + returns.mean()) ** 252 - 1
    max_dd = abs(calculate_max_drawdown(cumulative_returns))
    if max_dd == 0:
        return 0
    calmar = annual_return / max_dd
    return calmar


def backtest_portfolio(weights, test_returns, risk_free_rate=0):
    """
    回测投资组合
    """
    # 确保weights和test_returns的列对应
    common_stocks = list(set(weights.index) & set(test_returns.columns))
    weights_aligned = weights[common_stocks]
    test_returns_aligned = test_returns[common_stocks]
    
    # 归一化权重
    weights_aligned = weights_aligned / weights_aligned.sum()
    
    # 计算组合收益
    portfolio_returns = (test_returns_aligned * weights_aligned).sum(axis=1)
    
    # 计算累计收益
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # 计算性能指标
    metrics = {
        'Total Return': cumulative_returns.iloc[-1] - 1,
        'Annual Return': (1 + portfolio_returns.mean()) ** 252 - 1,
        'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(portfolio_returns, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(portfolio_returns, risk_free_rate),
        'Max Drawdown': calculate_max_drawdown(cumulative_returns),
        'Calmar Ratio': calculate_calmar_ratio(portfolio_returns, cumulative_returns),
        'Win Rate': calculate_win_rate(portfolio_returns),
        'Total Days': len(portfolio_returns),
        'Positive Days': (portfolio_returns > 0).sum(),
        'Negative Days': (portfolio_returns < 0).sum()
    }
    
    return portfolio_returns, cumulative_returns, metrics


def backtest_multiple_strategies(strategies_dict, test_returns, risk_free_rate=0):
    """
    回测多个策略
    strategies_dict: {'策略名称': weights_series}
    """
    results = {}
    
    for strategy_name, weights in strategies_dict.items():
        portfolio_returns, cumulative_returns, metrics = backtest_portfolio(
            weights, test_returns, risk_free_rate
        )
        results[strategy_name] = {
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'metrics': metrics
        }
    
    return results


def plot_backtest_results(backtest_results):
    """
    绘制回测结果
    """
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative Returns', 'Daily Returns'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12
    )
    
    # 累计收益曲线
    for strategy_name, results in backtest_results.items():
        fig.add_trace(
            go.Scatter(
                x=results['cumulative_returns'].index,
                y=results['cumulative_returns'],
                mode='lines',
                name=strategy_name,
                hovertemplate=f'<b>{strategy_name}</b><br>Date: %{{x}}<br>Cumulative Return: %{{y:.2%}}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 日收益率
    for strategy_name, results in backtest_results.items():
        fig.add_trace(
            go.Scatter(
                x=results['returns'].index,
                y=results['returns'],
                mode='lines',
                name=strategy_name,
                showlegend=False,
                line=dict(width=1),
                hovertemplate=f'<b>{strategy_name}</b><br>Date: %{{x}}<br>Daily Return: %{{y:.2%}}<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Return", tickformat='.0%', row=1, col=1)
    fig.update_yaxes(title_text="Daily Return", tickformat='.1%', row=2, col=1)
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_metrics_comparison_table(backtest_results):
    """
    创建性能指标对比表
    """
    metrics_list = []
    
    for strategy_name, results in backtest_results.items():
        metrics = results['metrics'].copy()
        metrics['Strategy'] = strategy_name
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    
    # 调整列顺序
    cols = ['Strategy', 'Total Return', 'Annual Return', 'Annual Volatility', 
            'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 
            'Win Rate', 'Total Days', 'Positive Days', 'Negative Days']
    df = df[cols]
    
    return df


def plot_metrics_comparison(metrics_df):
    """
    绘制性能指标对比图
    """
    # 选择关键指标
    key_metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=key_metrics,
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, metric in enumerate(key_metrics):
        row, col = positions[i]
        
        # 为Max Drawdown取绝对值
        values = metrics_df[metric]
        if metric == 'Max Drawdown':
            values = abs(values)
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Strategy'],
                y=values,
                name=metric,
                showlegend=False,
                text=[f'{v:.2%}' if metric in ['Annual Return', 'Max Drawdown', 'Win Rate'] else f'{v:.2f}' 
                      for v in values],
                textposition='outside'
            ),
            row=row, col=col
        )
        
        # 设置y轴格式
        if metric in ['Annual Return', 'Max Drawdown', 'Win Rate']:
            fig.update_yaxes(tickformat='.0%', row=row, col=col)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Performance Metrics Comparison"
    )
    
    return fig


def calculate_rolling_sharpe(returns, window=60, risk_free_rate=0):
    """
    计算滚动夏普比率
    """
    excess_returns = returns - risk_free_rate
    rolling_sharpe = (excess_returns.rolling(window=window).mean() / 
                     excess_returns.rolling(window=window).std() * np.sqrt(252))
    return rolling_sharpe


def plot_rolling_metrics(backtest_results, window=60):
    """
    绘制滚动指标
    """
    fig = go.Figure()
    
    for strategy_name, results in backtest_results.items():
        rolling_sharpe = calculate_rolling_sharpe(results['returns'], window)
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name=f'{strategy_name} (Rolling Sharpe)',
            hovertemplate=f'<b>{strategy_name}</b><br>Date: %{{x}}<br>Rolling Sharpe: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Rolling Sharpe Ratio ({window}-Day Window)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        height=400
    )
    
    return fig

