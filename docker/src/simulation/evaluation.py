import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def compute_backtest_metrics(equity: pd.Series,
                             rf_rate: float = 0.0,
                             periods_per_year: int = 365) -> dict:

    # 일별 수익률
    returns = equity.pct_change().dropna()

    # 누적 수익률
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    # 연평균 복리 수익률 (CAGR)
    years = periods_per_year / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1

    # 연환산 변동성
    vol = returns.std(ddof=0) * np.sqrt(periods_per_year)

    # 샤프 비율
    sharpe = (returns.mean() * periods_per_year - rf_rate) / vol

    # 소티노 비율
    neg_returns = returns[returns < 0]
    downside_vol = neg_returns.std(ddof=0) * np.sqrt(periods_per_year)
    sortino = (returns.mean() * periods_per_year - rf_rate) / downside_vol

    # 최대 낙폭 (Max Drawdown)
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    max_dd = drawdowns.min()

    # 칼마 비율 (CAGR / |Max Drawdown|)
    calmar = cagr / abs(max_dd)

    # 승률 (win rate)
    win_rate = (returns > 0).sum() / len(returns)

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Daily Win Rate': win_rate
    }


def compute_performance_metrics(
    trade_returns,            # list or 1D-array: 각 트레이드 수익률
    equity_curve=None,        # pd.Series: 인덱스=날짜, 값=전일 종가 기반 누적 자산 가치
    benchmark_returns=None,   # pd.Series: 인덱스=날짜, 값=벤치마크 일일 수익률
    var_confidence=0.95       # VaR 계산 시 신뢰수준
):
    trades = np.asarray(trade_returns)
    n = len(trades)
    if n == 0:
        return {}
    wins = trades[trades > 1]
    losses = trades[trades < 1]

    # 1) 거래단위 지표
    win_rate = len(wins) / (len(wins) + len(losses))
    avg_win = wins.mean() if wins.size > 0 else 0.0
    avg_loss = losses.mean() if losses.size > 0 else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    profit_factor = (wins.sum() / -losses.sum()) if losses.sum() != 0 else np.nan

    # 2) 꼬리 위험: VaR, CVaR
    var_level = np.percentile(trades, (1 - var_confidence) * 100)
    cvar = trades[trades <= var_level].mean()

    # 3) 분포 특성: 왜도·첨도
    skewness = skew(trades)
    kurt = kurtosis(trades)

    metrics = {
        'Trade Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Expectancy': expectancy,
        'Profit Factor': profit_factor,
        f'VaR ({int(var_confidence*100)}%)': var_level,
        f'CVaR ({int(var_confidence*100)}%)': cvar,
        'Skewness': skewness,
        'Kurtosis': kurt,
    }

    # 4) 드로우다운 관련 (equity_curve 필요)
    if equity_curve is not None:
        # 일별 수익률
        returns = equity_curve.pct_change().fillna(0)
        # 누적 수익 곡선
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = cum / peak - 1

        # MDD 및 지속기간
        max_dd = dd.min()
        # 최대 Drawdown 구간 길이 계산
        durations = (dd == 0).astype(int).groupby((dd == 0).astype(int).cumsum()).cumsum()
        max_dd_duration = durations.max()
        # Ulcer Index
        ulcer_index = np.sqrt((dd**2).mean())

        metrics.update({
            'Max Drawdown': max_dd,
            'Max DD Duration (days)': max_dd_duration,
            'Ulcer Index': ulcer_index
        })

    # 5) 벤치마크 대비 알파·베타·정보비율 (둘 다 필요)
    if equity_curve is not None and benchmark_returns is not None:
        df = pd.DataFrame({
            'strat': equity_curve.pct_change(),
            'bench': benchmark_returns
        }).dropna()
        cov = df['strat'].cov(df['bench'])
        beta = cov / df['bench'].var()
        alpha = df['strat'].mean() - beta * df['bench'].mean()
        ir = alpha / (df['strat'] - beta * df['bench']).std()

        metrics.update({
            'Beta': beta,
            'Alpha': alpha,
            'Information Ratio': ir
        })

    return pd.Series(metrics)


def evaluate(df, start_date, end_date, periods_per_year):
    pd.set_option('mode.chained_assignment',  None)
    df["date"] = pd.to_datetime(df['ts'] * 1e3).dt.date
    daily_equity = df.groupby('date')['equity'].last()
    all_date = pd.date_range(start=start_date, end=end_date)

    daily_equity = daily_equity.reindex(all_date).fillna(method='ffill')
    daily_equity.index.name = 'date'
    daily_equity.fillna(1e6, inplace=True)

    return {
        **compute_performance_metrics(df.v),
        **compute_backtest_metrics(daily_equity, periods_per_year=periods_per_year),
    }
