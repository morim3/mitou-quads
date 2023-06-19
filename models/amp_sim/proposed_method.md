
# 提案手法について



## Default Strategy Parameters


### ハイパーパラメータ

- `n_dim`: 関数の定義域の次元 $n$
- `n_samples`: サンプル数 $\mu$

- ステップサイズ制御
  - `c_sigma`: $c_\sigma = \dfrac{\mu_{\rm eff} + 2}{n + \mu_{\rm eff} + 5}$
  - `d_sigma`: $d_\sigma = 1 + 2 \max \par{0, \sqrt{\dfrac{\mu_{\rm eff} - 1}{n + 1} - 1}} + c_\sigma$
- `norm_const`: ノルムの期待値 $r = \ex\bra{\norm{\ndist(\bm 0, I)}}$

### パラメータ

- `mean`: $\bm m$
- `cov`: $\bm C$
- `threshold`: $\theta$
- `step_size`: $\sigma$
- `cov_path`: $\bm p_c$
- `th_path`: $\bm p_\theta$
- `step_path`: $\bm p_\sigma$

### 入力

- サンプル点 $\bm x_i$
- サンプル点での関数値 $f(\bm x_i)$

### 更新則

- `mean`: 重み付きのサンプル平均 $\displaystyle \bm m^{(g+1)} = \sum_{i=1}^\mu w_i \bm x_i^{(g+1)}$
  - `w_mu`: 組み替え重み $\bm w_\mu \propto \par{\log \dfrac{\mu + 1}{i}}_{i=1}^{\mu}$
  - `mu_eff`: 分散のスケーリング項 $\mu_{\rm eff} = 1 / \sum_{i=1}^\mu w_i^2$
- `cov`: 
- `cov_path`: $\bm p_c^{(g+1)} = (1 - c_c) \bm p_c^{(g)} + h_\sigma \sqrt{c_c(2 - c_c)\mu_{\rm eff}}\ang{\bm y}_w^{(g)}$
  - `y`: $\displaystyle \ang{\bm y}_w^{(g)} = \sum_{i=1}^\mu w_i \frac{\bm x_i^{(g)} - \bm m^{(g)}}{\sigma^{(g)}}$
- `threshold`: $\theta^{(g+1)} = c_\theta \theta^{(g)} + (1 - c_\theta)\hat\theta^{(g+1)}$
  - `np.quantile(accepted_val, q=hp.quantile)`: $\hat\theta^{(g+1)}$
- `th_path`: $\bm p_\theta^{(g+1)} = 0.7 \bm p_\theta^{(g)} + 0.3 \par{\log \theta^{(g+1)} - \log \theta^{(g)}}$
- `step_size`: $\sigma^{(g+1)} = \sigma^{(g)} \times \exp \par{\dfrac{c_\sigma}{d_\sigma} \times \par{\dfrac{\norm{\bm p_\sigma}}{r} - 1}}$
- `step_path`: $\bm p_\sigma^{(g+1)} = (1 - c_\sigma) \bm p_\theta^{(g)} + \sqrt{c_\sigma(2 - c_\sigma) \mu_{\rm eff}} \par{\bm C^{(g)}}^{-\frac{1}{2}} \ang{\bm y}_w^{(g)}$
  - `diagD`, `B`: 共分散の対角化 $C^{(g)} = B^{(g)}T^{(g)}\par{B^{(g)}}^\tp$
  - `C_`: 共分散の逆行列の平方根 $\par{\bm C^{(g)}}^{-\frac{1}{2}} = B^{(g)}\par{T^{(g)}}^{-\frac{1}{2}}\par{B^{(g)}}^\tp$
