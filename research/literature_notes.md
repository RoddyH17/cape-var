# CAPE Literature Synthesis

*Compiled via systematic web search — 2026-03-11*

## 关键论文

### 1. Campbell & Shiller (1988) — 理论基础
**Journal of Finance, 43(3)**

对数线性近似恒等式：
```
log(P_t/D_t) = k + ρ·E[log(R_{t+1})] + (1-ρ)·E[log(D_{t+1}/D_t)]
```
可测方程：
```
R_{t,t+10} = α + β·log(1/CAPE_t) + ε    β > 0 (高CAPE → 低收益)
```
统计要点：Newey-West HAC 处理重叠数据。

### 2. Davis et al. (2018) — Fair-Value CAPE / ECY
**Journal of Portfolio Management, 44(3)**

2步 VAR 框架：

**Step 1 — VAR(12)**，状态变量：
```
X_t = [1/CAPE, y_real_10yr, CPI_inflation, σ_stock, σ_bond]
```

**Step 2 — 10yr return decomposition:**
```
r̂_10yr = Δlog(CAPE) + g_earnings + y_dividend
        ≈ VAR-implied CAPE expansion + 2% + dividend yield
```

ECY 简化：`ECY = 1/CAPE - y_real_10yr`

| 模型 | 10yr RMSE |
|------|----------|
| 传统 CAPE OLS | 6.6% |
| Fair-Value CAPE VAR | 3.8% (**-42%**) |
| ML-VAR hybrid | 2.6% (-61%) |

*注：Roddy 实习报告为 40% 改善，与 Davis 基准一致。*

### 3. Asness (2012) — PE-收益分位映射
**AQR White Paper**

十分位分析（非参数）：

| CAPE 分位 | 10yr 实际年化收益 |
|----------|----------------|
| 最低十位 | ~10%+ |
| 中位数 | ~6-7% |
| 最高十位 | **0.5%** (范围: -6.1% ~ +6.3%) |

单调性明确；2012年估值水平下预期实际收益仅 0.9%/年。

### 4. Kenourgios et al. (2021) — CAPE5 跨市场验证
**Operational Research, 22(4)**

希腊 FTSE/ASE (1997–2018)；数据短 → CAPE5 = 5年平均 EPS。

| 预测变量 | 5yr 收益 R² |
|---------|-----------|
| **CAPE** | **0.76** |
| CAPE5 | 0.74 |
| P/E | 0.02 |
| P/BV | 不显著 |

预测力随时间跨度增强。

## 实现对照

我们的 `cape_var_model.py` 实现路径：

1. `fit_baseline_shiller()` → Campbell-Shiller OLS baseline
2. `compute_excess_cape_yield()` → Davis ECY (单因子简化)
3. `fit_var_cape()` → VAR([EY, CapIdx, Y10]) → 接近 Davis Step 1 spec
4. `compare_models()` → RMSE 对比，目标 ~40% 改善

## TODO 升级方向

- [ ] 完整 Davis 5变量 VAR：加入 CPI、σ_stock、σ_bond
- [ ] GRU/LSTM 替换 VAR 第二阶段（ML-VAR hybrid）
- [ ] CAPE5 variant 测试（非美市场适用）
- [ ] FRED 数据接口：TIPS yield → 真实 y_real
