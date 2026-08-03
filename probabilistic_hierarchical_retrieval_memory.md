# 概率化层次检索记忆：面向冻结预训练模型的长期内部状态存储

## 1. 文档目的

本文定义一个面向冻结预训练 Transformer 的长期检索记忆结构。它以 CAMELoT 的 training-free、逐层关联记忆为直接基线，但不再依赖单一固定 cosine threshold 决定槽位写入，而是引入：

- 已有簇归属与新簇创建的概率判断；
- 有时间顺序的原始记录 payload；
- 基于模型实际使用行为的记录保留和过期机制；
- 对过大、宽泛或多模态簇的动态 split；
- 对碎片化簇的 merge；
- 分层 ANN 路由、簇内精确 K/V 重排和原生 attention。

目标不是声称聚类后的表示与完整历史 attention 严格等价，而是在受控内存和计算预算下，尽可能保留可被未来查询重新访问的历史信息。

## 2. 核心研究问题

固定 threshold 写入存在三个主要问题：

1. 不同层、不同 head 和不同上下文阶段的 K/residual 分布尺度不同，一个全局阈值难以稳定工作。
2. 高频大簇会通过 count prior 或 centroid 扩张产生 rich-get-richer，持续吸收边缘记录。
3. 单个 centroid 可能无法代表多模态簇，使簇内真正相关记录在第一阶段检索时不可见。

本研究要验证：

```text
概率簇后验 + 时间/效用衰减 + 动态 split/merge
```

是否能在相同内存、读取数量和模型 backbone 下，比 CAMELoT 的固定 hard threshold 获得更好的长期召回与语言建模表现。

## 3. 设计边界

### 3.1 初始范围

- 冻结 pretrained decoder-only Transformer；
- 不训练或修改主体模型参数；
- 记忆模块在推理过程中在线更新；
- 第一阶段使用模型原生 K 作为索引表示，以保证与 CAMELoT 公平对比；
- payload 保存对应历史记录的原始 K/V 或可重建 K/V 的 token span；
- 当前窗口继续使用完整原生 KV；
- 记忆召回只作为当前窗口之外的补充。

### 3.2 后续索引表示

在 K 索引基线成立后，再比较：

- K 作为 ANN 索引；
- Q 作为历史查询状态索引；
- residual stream 作为索引；
- K/residual 联合索引；
- 当前 residual 检索历史 post-state，再返回后续 span 的 K/V。

无论使用哪种索引，最终进入 attention 的 payload 应尽量保持为历史原始 K/V，而不是把索引 centroid 直接当作完整记忆内容。

## 4. 总体架构

```text
current hidden state / K / Q / residual
                  |
                  v
          top-level ANN routing
                  |
          parent index slots
                  |
        posterior child routing
                  |
             leaf slots
                  |
        ordered record posting lists
                  |
      exact K or residual reranking
                  |
          selected historical K/V
                  |
 current native Q attends [memory K/V + window K/V]
```

记忆是逐层维护的。默认不同层不共享 centroid、密度参数和 threshold，因为各层内部几何分布通常不同。

## 5. 数据结构

### 5.1 Memory Record

每条历史记录至少包含：

```text
record:
  id
  layer
  head_or_kv_group
  source_token_or_span
  index_vector
  original_key
  original_value

  write_time
  sequence_order
  last_retrieval_time
  retrieval_count
  attention_contribution_ema
  counterfactual_gain_ema

  active_weight
  superseded_by
  conflict_group
  source_authority
```

`source_token_or_span` 用于需要时重建更完整的历史 K/V。若存储预算允许，优先保存量化后的原始 K/V，避免重复执行历史 forward。

### 5.2 Leaf Slot

叶子 slot 是索引簇，不是单一压缩事实：

```text
leaf_slot:
  id
  layer
  centroid
  effective_count
  resultant_length
  concentration
  weighted_scatter

  record_ids
  ordered_time_range
  usage_ema
  routing_regret_ema
  value_conflict_score

  provisional_child_a
  provisional_child_b
  split_statistics
```

### 5.3 Parent Slot

parent slot 只用于分层路由：

```text
parent_slot:
  routing_centroid
  child_ids
  aggregate_radius
  aggregate_time_range
  routing_usage
```

parent slot 不直接作为最终 K/V 注入模型。最终 payload 必须来自叶子记录或明确验证过的叶子级摘要。

## 6. 写入与新簇后验

设新记录索引向量为归一化向量 \(x_t\)。对于已有簇 \(C_i\)：

\[
P(z_t=i\mid x_t,H_t)
\propto
(N_i^{eff}(t)+\epsilon)^\gamma
p(x_t\mid C_i)
\]

新簇概率为：

\[
P(z_t=\mathrm{new}\mid x_t,H_t)
\propto
\alpha_t p_0(x_t)
\]

归一化得到：

\[
P_{\mathrm{new}}=
\frac{\alpha_t p_0(x_t)}
{\alpha_t p_0(x_t)+
\sum_i(N_i^{eff}(t)+\epsilon)^\gamma p(x_t\mid C_i)}
\]

写入规则：

```text
if P_new > tau_new:
    create leaf slot
else:
    assign to argmax_i P(z=i | x_t, H_t)
```

### 6.1 vMF Likelihood

对单位球面上的 K 或 residual，可使用 von Mises-Fisher likelihood：

\[
\log p(x_t\mid C_i)
=
\log C_d(\kappa_i)+\kappa_i\mu_i^\top x_t
\]

其中：

- \(\mu_i\) 是加权方向均值；
- \(\kappa_i\) 表示簇的集中程度；
- 宽簇和紧簇不再共享同一个相似度 threshold。

### 6.2 避免退化成换皮 threshold

如果所有簇的 \(\kappa_i\) 相同、count 不参与、base density 为常数且 \(\alpha\) 固定，那么该决策仍会退化为固定 cosine threshold。

因此概率模型的有效信息必须至少包括以下一项：

- per-cluster concentration；
- temporal effective count；
- model-utility prior；
- local density；
- value conflict；
- layer-specific calibration。

## 7. Rich-Get-Richer 控制

普通 count prior：

\[
P(z=i)\propto n_i
\]

会使大簇更容易继续吸收新记录。采用 tempered count：

\[
P(z=i)\propto(N_i^{eff}+\epsilon)^\gamma,
\qquad 0\leq\gamma<1
\]

但仅压缩 count 不够。系统还必须：

- 让旧记录的有效质量衰减；
- 剔除低价值记录；
- 对宽簇降低 likelihood concentration；
- 对高 conflict 或高 routing regret 簇执行 split；
- 对持续增长但不能提高召回的簇设置容量惩罚。

## 8. 时间、模型行为与记录保留

### 8.1 记录权重

每条记录的动态权重：

\[
w_j(t)=
e^{-\lambda_a(t-t_j)}
e^{-\lambda_i(t-t_j^{last})}
(\epsilon+\bar g_j)^\beta
(1-p_j^{superseded})
\]

其中：

- \(t_j\)：写入时间；
- \(t_j^{last}\)：最后有效召回时间；
- \(\bar g_j\)：模型收益的 EMA；
- \(p_j^{superseded}\)：被后续记录覆盖的概率。

slot 的有效统计量：

\[
N_i^{eff}(t)=\sum_{j\in C_i}w_j(t)
\]

\[
m_i(t)=\sum_{j\in C_i}w_j(t)x_j,
\qquad
\mu_i(t)=\frac{m_i(t)}{\|m_i(t)\|}
\]

### 8.2 模型收益信号

由低成本到高成本依次使用：

1. 召回次数和最后召回时间；
2. 被选入最终 K/V rerank 的次数；
3. attention output contribution；
4. 有记忆与无记忆输出 logits 的 KL；
5. 对正确 next token 的 log-probability 增益；
6. 周期性 counterfactual ablation。

attention weight 不能单独作为价值，因为它不直接等于对最终输出的因果贡献。

### 8.3 保留决策

记录应在其预期未来收益低于存储成本时过期：

\[
P(\mathrm{reuse}\mid H_t)
E[\mathrm{gain}\mid r_j]
\leq
\lambda_{\mathrm{mem}}\operatorname{bytes}(r_j)
\]

预算控制器动态调整 \(\lambda_{\mathrm{mem}}\)：

\[
\lambda_{\mathrm{mem},t+1}
=
\left[
\lambda_{\mathrm{mem},t}
+\eta(\operatorname{Memory}_t-B)
\right]_+
\]

这样 threshold 与模型行为、记录顺序和实际预算共同变化，而不是固定 TTL。

### 8.4 可删除性约束

如果 slot 只保留 centroid、count 和平均 KV，就无法准确移除其中一条记录。支持簇内过期必须采用以下至少一种形式：

- 保留每条记录的索引向量和权重；
- 保留时间 bucket 的 sufficient statistics；
- 保留可独立删除的 micro-cluster；
- 保留 posting list，并允许异步重建 slot 统计量。

## 9. 时间冲突与状态更新

对于：

```text
t1: Alice lives in Paris
t2: Alice moved to London
```

不能将两个 V 简单平均。记录应保留时间顺序并建立：

```text
r_t2 supersedes r_t1
```

读取策略由查询决定：

- 当前状态查询：优先最新且未被覆盖的记录；
- 历史查询：根据指定时间返回旧记录；
- 无时间约束：返回变更链或让模型同时看到相关版本；
- 权威冲突：结合 source authority，而不是只看 recency。

slot 的 centroid 可以合并索引信息，但 payload 默认不合并冲突 value。

## 10. 动态 Split

### 10.1 分割不能只看 token 数

高频但紧密的簇不应仅因记录多而拆分。真正的 split 条件是：

> 当前 slot 已经不能作为其内部记录的可靠路由代表。

### 10.2 簇离散度

维护加权 resultant length：

\[
\bar R_i=
\frac{\left\|\sum_{j\in C_i}w_jx_j\right\|}
{\sum_{j\in C_i}w_j}
\]

\(\bar R_i\) 越低，簇越分散。

### 10.3 多模态收益

单中心损失：

\[
J_1=\sum_jw_j(1-\cos(x_j,\mu_i))
\]

候选双中心损失：

\[
J_2=
\sum_jw_j
\min_{c\in\{a,b\}}
(1-\cos(x_j,\mu_c))
\]

分割收益：

\[
G_{\mathrm{split}}=
\frac{J_1-J_2}{J_1}
\]

### 10.4 Routing Regret

对实际查询 \(q\)，定义：

\[
r_i(q)=
\max_{j\in C_i}\cos(q,k_j)
-\cos(q,\mu_i)
\]

如果某条内部记录与查询高度匹配，但 slot centroid 与查询不匹配，那么该记录可能在 ANN 第一阶段被漏召回。维护：

\[
\bar r_i=\operatorname{EMA}_{q\rightarrow C_i}r_i(q)
\]

高 routing regret 是比单纯簇内方差更贴近真实检索行为的 split 信号。

### 10.5 Value 和时间冲突

候选归类距离可以组合：

\[
D(j,c)=
\lambda_k(1-\cos(k_j,\mu_c))
+\lambda_v d(v_j,\nu_c)
+\lambda_t d_{\mathrm{temporal}}(t_j,c)
\]

时间项用于检测连续状态切换或有效期，不应无条件把新旧记录分开。

### 10.6 SplitScore

\[
\operatorname{SplitScore}_i=
a\,\operatorname{dispersion}_i
+b\,\operatorname{multimodality}_i
+c\,\operatorname{routingRegret}_i
+d\,\operatorname{valueConflict}_i
-\lambda_{\mathrm{slot}}
\]

触发候选分割至少要求：

```text
effective_count > minimum_child_mass
and
(
  resultant_length < minimum_concentration
  or split_gain > minimum_gain
  or routing_regret > maximum_regret
  or value_conflict > maximum_conflict
)
```

### 10.7 在线候选子中心

每个叶子 slot 持续维护两个 provisional child centroids：

1. 新记录正常写入 parent leaf；
2. 同时归类到两个候选子中心之一；
3. 在线更新 \(J_1\)、\(J_2\)、child mass 和稳定性；
4. split gain 持续超过阈值后异步扫描 posting list；
5. 正式建立两个 child leaf slots；
6. 原 slot 转为 parent routing node。

### 10.8 接受分割

\[
\Delta=
\mathcal L(C_a)+\mathcal L(C_b)
-\mathcal L(C_i)
-\lambda_{\mathrm{slot}}
\]

仅在以下条件满足时接受：

- \(\Delta>0\)；
- 两个子簇都有最低有效质量；
- 改善持续多个更新周期；
- routing regret 明显降低；
- 采样查询的 retrieval miss 降低；
- 额外槽位成本低于预期模型收益。

## 11. 动态 Merge

只有 split 会产生持续碎片化，因此必须有反向 merge：

```text
merge candidates:
  neighboring leaf centroids
  overlapping temporal validity
  low value conflict
  low merged routing regret
```

merge 接受条件：

\[
\Delta_{\mathrm{merge}}=
\mathcal L(C_{ab})
-\mathcal L(C_a)
-\mathcal L(C_b)
+\lambda_{\mathrm{slot}}
>0
\]

采用迟滞控制：

- split threshold 高于 merge threshold；
- split/merge 后设置 cooldown；
- 先执行记录过期，再判断 split；
- 合并前检查时间冲突和 supersession 链；
- 禁止短周期来回振荡。

## 12. 分层 Retrieval

### 12.1 第一阶段：ANN 路由

使用当前 token 的索引状态检索 top-level parent/leaf slots：

```text
route_score =
  index_similarity
  + optional recency prior
  + optional utility prior
```

第一阶段只需要高 recall，不直接决定最终注入模型的 K/V。

### 12.2 第二阶段：子簇后验

在候选 parent 内计算：

\[
P(C_{child}\mid q,C_{parent})
\]

选择若干 leaf slots。

### 12.3 第三阶段：簇内精确重排

从 leaf posting list 取候选记录，用当前原生 Q 与历史原始 K 重新计算：

\[
s_j=
\frac{q^\top k_j}{\sqrt d}
+b_{\mathrm{time}}(q,r_j)
+b_{\mathrm{authority}}(q,r_j)
\]

最终只将 top-k 历史 K/V 注入当前 attention。索引相似度不替代原生 QK score。

### 12.4 当前窗口合并

\[
K_{\mathrm{aug}}=
[K_{\mathrm{memory}};K_{\mathrm{window}}]
\]

\[
V_{\mathrm{aug}}=
[V_{\mathrm{memory}};V_{\mathrm{window}}]
\]

当前 Q 对增强后的 K/V 执行正常 attention。

### 12.5 重复项的 count correction

如果多个完全相同的 K/V 被压缩成一个代表，完整 attention 中它们具有 multiplicity：

\[
A(q)=
\frac{\sum_i n_i e^{s_i}v_i}
{\sum_j n_j e^{s_j}}
\]

精确重复项可通过：

\[
s_i'=s_i+\log n_i
\]

恢复其总权重。对非完全相同或 value 冲突的记录不能使用该修正代替原始 payload。

## 13. 写入、读取和维护流程

### 13.1 写入

```text
1. extract per-layer index state and original K/V
2. retrieve candidate slots
3. compute existing-cluster and new-cluster posterior
4. create slot or append record to selected leaf
5. update weighted sufficient statistics
6. update temporal/conflict links
7. update provisional split state
```

### 13.2 读取

```text
1. build current routing state
2. ANN retrieve parent/leaf candidates
3. route to child leaves
4. fetch record candidates
5. rerank using current native Q against original historical K
6. apply temporal and authority policy
7. augment current-window K/V
8. record observed retrieval utility
```

### 13.3 周期维护

```text
1. decay record weights
2. evict records below expected-utility threshold
3. rebuild affected sufficient statistics
4. evaluate pending splits
5. evaluate neighboring merges
6. compact posting lists and ANN index
7. recalibrate per-layer density parameters
```

维护应异步或分批执行，避免每个 token 都承担全量管理成本。

## 14. 与 CAMELoT 的公平对比

保持以下部分完全一致：

- pretrained backbone；
- memory-augmented layers；
- K/V 提取方式；
- read top-k；
- 当前窗口长度；
- memory byte budget；
- 最终 attention 注入方式；
- 数据顺序和评估任务。

只逐步替换 write/maintenance policy。

### 14.1 必须比较的版本

```text
A. CAMELoT-Hard
   fixed cosine threshold R

B. Adaptive-Quantile
   threshold follows recent similarity distribution

C. vMF-Posterior
   per-cluster concentration and tempered count

D. Temporal-vMF
   posterior + effective count + record expiry

E. Hierarchical-vMF
   posterior + expiry + dynamic split/merge
```

Adaptive-Quantile 是必要对照，用于判断收益来自“阈值可变”还是来自真正的 cluster-conditioned uncertainty。

### 14.2 Pareto 比较

不能只比较 CAMELoT 默认 \(R=0.93\)。必须：

- 扫描 hard threshold；
- 扫描 \(\alpha,\tau_{\mathrm{new}},\gamma\)；
- 匹配平均 slot 数或总 memory bytes；
- 比较 quality-memory-latency Pareto frontier；
- 报告每层 slot 分布和写入率。

## 15. 评估任务

### 15.1 控制任务

- 完全重复 K/V；
- 同 key、不同时间 value；
- 相似 key、不同 value；
- 高频大簇与低频小簇；
- 簇逐渐漂移；
- 单簇转为双模态；
- topic burst 后恢复；
- 已删除或被覆盖事实查询；
- absent-fact 查询。

### 15.2 长期对话

- 用户偏好长期保持；
- 偏好更新；
- 人物属性变化；
- 旧事实与最新事实区分；
- 跨多轮 paraphrase 召回；
- 时间限定问题；
- 权威来源冲突。

### 15.3 长上下文任务

- language-model perplexity；
- needle retrieval；
- multi-needle retrieval；
- RULER 风格长上下文任务；
- 长文实体和属性跟踪；
- 代码变量、函数和接口历史召回。

## 16. 指标

### 16.1 任务质量

- perplexity；
- exact recall；
- latest-fact accuracy；
- historical-fact accuracy；
- conflict resolution accuracy；
- absent-fact abstention；
- wrong-memory injection。

### 16.2 检索质量

- parent routing recall；
- leaf routing recall；
- record top-k recall；
- routing regret；
- false merge rate；
- duplicate slot rate；
- split purity；
- split/merge oscillation rate。

### 16.3 存储和性能

- slot count；
- record count；
- bytes per layer；
- ANN index bytes；
- write latency；
- read latency；
- maintenance latency；
- K/V reconstruction cost；
- eviction and rebuild frequency。

### 16.4 模型行为

- target-token logit lift；
- output-logit KL；
- attention output contribution；
- counterfactual memory gain；
- recalled-record utilization rate。

## 17. 关键消融

- raw count vs tempered count；
- permanent count vs temporal effective count；
- fixed TTL vs utility-based expiry；
- attention weight utility vs counterfactual utility；
- no split vs size-only split vs routing-regret split；
- no merge vs merge with hysteresis；
- centroid payload vs original K/V payload；
- K index vs residual index；
- ANN-only score vs native QK rerank；
- no count correction vs \(\log n\) correction；
- global parameters vs per-layer calibration。

## 18. 复杂度与实现风险

### 18.1 写入成本

朴素全槽搜索为：

\[
O(N\cdot |\mathrm{slots}|)
\]

需要 ANN、分层路由或批量写入降低成本。

### 18.2 payload 增长

slot consolidation 不等于 payload 不增长。如果保留所有原始记录，存储仍可能随历史增长。系统必须通过：

- 过期；
- supersession；
- 精确重复压缩；
- 量化；
- span 级存储；
- 冷热分层；
- 可重建 payload；

控制实际 bytes。

### 18.3 内部表示漂移

同一语义在不同上下文和位置产生的 K/residual 可能显著不同。动态 split 可能把语义相同但几何漂移的记录分开，也可能错误合并几何接近但语义不同的记录。

因此必须测量：

- layer/head 差异；
- 位置和 RoPE 影响；
- 上下文噪声；
- query-conditioned routing regret；
- 同事实跨时间的几何漂移。

### 18.4 价值网络反馈环

如果只根据历史召回次数决定保留，已经被频繁召回的记录会更容易继续存活，形成新的 rich-get-richer。必须保留：

- 探索概率；
- 最低新记录保护期；
- 使用次数去偏；
- counterfactual gain 校准；
- 冷门事实的均匀抽样评估。

## 19. 最小实现顺序

### Phase 1: CAMELoT 可复现基线

- 固定 K 索引；
- 固定 threshold；
- 原始 CAMELoT 平均 K/V；
- 记录 slot growth 和 retrieval quality。

### Phase 2: 概率写入

- per-layer vMF statistics；
- tempered effective count；
- explicit new-cluster posterior；
- 与 threshold sweep 做 Pareto 对比。

### Phase 3: 原始 payload 与时间

- slot 作为 index；
- posting list 保存 ordered records；
- native QK rerank；
- utility/time decay；
- supersession 和 conflict policy。

### Phase 4: Split/Merge

- provisional two-center statistics；
- routing regret；
- asynchronous split；
- merge hysteresis；
- 分层 ANN。

### Phase 5: 索引表示扩展

- residual indexing；
- K/residual joint indexing；
- post-state and subsequent-span retrieval；
- 跨模型和跨层稳定性分析。

## 20. 成功条件

在相同 memory bytes 和 read top-k 下，Hierarchical-vMF 相对 CAMELoT-Hard 应满足：

- 更高的长期 recall；
- 更低的 stale-fact recall；
- 更低的 false merge；
- 更低的 routing regret；
- slot 增长不因动态 split 失控；
- split/merge 不发生高频振荡；
- language-model quality 不因错误 memory injection 明显下降；
- 维护成本不抵消读取节省。

只有在长期事实召回、冲突处理、absent-fact 控制和跨时间更新上均成立，才可以进一步讨论接近无限历史访问的能力。

## 21. 研究定位

用概率化、时间相关、模型行为校准的层次簇作为内部状态索引，同时保留可精确重排的历史 K/V 或 token-span payload。

相对 CAMELoT，核心增量是：

- hard threshold 变为 cluster-conditioned posterior；
- 永久 count 变为 temporal effective mass；
- slot 内部从单一平均表示变为有序记录集合；
- 固定平面槽位变为可 split/merge 的层次索引；
- centroid 直接注入变为索引路由后对原始 K/V 重排；
- recency eviction 变为预期模型收益与存储成本之间的动态决策。

最需要验证的不是聚类指标本身，而是这些机制是否真实降低模型的长期检索错误，并在可接受的存储和延迟下提高最终任务质量。
