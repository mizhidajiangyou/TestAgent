# 节点⑫ · LINK-S5b / S6b / T12b / S7：links 上任务包链 + 离线复算 CLI + 一次真机 E2E

归档 HEAD：`242d386`，`working_tree_dirty=true`（本会话先跑门、后按里程碑入库，测得的是未提
交的工作树）。开关语义按 v15 §10：**adapter 默认关闭**，因此本归档里 FH2.4/2.6/2.7 的既有基
线零改动。

## 门（12 道）

| 门 | 结果 | 读数 |
| --- | --- | --- |
| ruff check | PASS | All checks passed! |
| ruff format --check | PASS | 169 文件 |
| mypy | PASS | 98 源文件 0 错 |
| pytest tests/ -q | PASS | 1010 passed |
| TestArchitectureGate | PASS | 7 passed |
| engine golden | PASS | 16 passed |
| tasks validate --strict | PASS | _example / gui / perf / testcase |
| links golden 三 stage | PASS | graph / planner / fixture |
| R6 benchmark | PASS | graph pair 1.0/0.9167；selected 命中含 `POST /coupons → POST /orders` |
| links baseline | PASS | 13 端点、模块 {cart:4, orders:4, products:5}、L0=725 字符 |
| **links parity-off** | PASS | 5/5 fixture 在 `LINKS_ENABLED=true`、包未 opt-in 时八维 diff=0 且不写 sidecar |

`links-baseline` / `links-parity-off` 自本轮起进入常驻门禁（归档器 `--only` 可单跑）。

## 交付

* **接线（S5b/S6b/T12b）**：`pipeline/links_pass.py` 一次 run 四个触点——L0/L1 批次材料、
  phase2 由盲切 2 改 L3a cluster、`l3b` 一个 selected 路径一个 unit、最终 artifact 上跑
  Gate1/2/3；`PipelineResult.links_report` 承载报告，`output/<session>.links.json` 落 sidecar。
* **开关**：包 `pipeline.links.enabled=false` + 配置 `LINKS_ENABLED` + 本次 run `--links` 三道
  闸；`split.by=clusters` 在无 links 时逐字节退回 `batch_size` 分组且标签格式相同，故录制的
  prompt 指纹不动。
* **程序独占身份列**：`STAGE_SOURCE` 把 `phase2_api` 映射进 schema 值域 `phase2`；
  `note_forgeries` 在 **raw items** 上读取模型自报的 `path_id`/`source_stage`（质量转换器不再
  拷贝这两列 ⇒ "忽略"与"记录"同时成立）；`normalize_history` 复制外部基线并清空其身份列，原
  值只入审计。
* **去重命名空间**：`dedup.namespace_field="path_id"`；links-off 该列恒空 ⇒ 分组与冻结一致。
* **S7 CLI**：`testagent links-check --input --spec [--run-metadata] [-o]`，冷进程 `--help` 已
  冒烟。带 sidecar 只按**该 run 记录的契约**复算，以 `payload ↔ canonical_sha256 ↔ path_id`
  三方校验拒绝事后改分母；无 sidecar 为观察模式，固定 `selection unknown / contract coverage
  n/a`，绝不从 `case.path_id` 反推 selected。退出码 0=干净、1=未履约或候选未覆盖、2=输入/契约
  /哈希不自洽；只报告，不改写 artifact。
* **web**：请求位 `links` + 响应 `links_report` **同一对象透传**（不重算率）；links-off 响应字
  节不变，12 格契约仍全绿。

## 开关矩阵读数（`tests/test_links_wiring.py`，9 项）

off：phase1 4 unit + phase2 **7 个 batch**、无 L0、无身份列、无报告；on：phase2 **3 个
cluster**、`l3b ≤ LINKS_L3B_MAX_CALLS`、每 unit 带 L0、只有 L3b 持 `path_id`、报告含逐阶段
`rejected_rate`。另证：伪造被忽略并计数、外部基线身份被洗、并发两 run 的 `LinksPass` 与路径
集合互不污染、未声明 `pipeline.links` 的包（perf）即使 `LINKS_ENABLED=true` 也不启用。

## 真机 E2E（.env 模型 qwen3.7-flash，额度正常）

* `testagent testcase -r <ecommerce 需求> -s examples/ecommerce_swagger.json --links` → 60 条
  用例 + sidecar + 快照；sidecar：`planned=1 candidate=1 selected=1 l3b_units=1`、
  `r6_edges=0`、`r6_abstentions=22`。
* `links-check --run-metadata` 复算：**`Pc61912b81c96: FAILED (pairs 0/1)`**、`pair_rate 0.0`、
  `attempted=1 selected=1`、退出码 1。
* **诚实结论**：本轮 L3b unit 未产出可保留用例（artifact `source_stage` 只有 phase1 29 /
  phase2 31），Gate 给 REJECTED 而非假绿；"候选未履约仍留在分母"在真机上成立。散文边零命中
  是候选池只有 1 条路径的直接原因，属规则面事实而非接线缺陷。
* 同轮真机另跑出两个质量线死点并修（详见 `20260921-QL2-prompt-fidelity-242d386`）：raw 审计
  在新链 `raw_calls: 0 / match: false`（修后真机复跑 `raw_calls=5, rows_sum=30,
  artifact_count=30, match=True`）；`json_mode` 从未到达 provider。
* 可重放的零 token 复验命令：`.venv/bin/python scripts/parity_links_off.py`；
  `.venv/bin/testagent links-check --input <artifact.json> --spec <spec.json> [--run-metadata
  <session>.links.json]`。真机产物（cases.json / sidecar / 复算报告）落在运行时输出目录，未随
  本归档入库——验收件是否复制进 `examples/acceptance/` 留待 output 处置一并裁决。
