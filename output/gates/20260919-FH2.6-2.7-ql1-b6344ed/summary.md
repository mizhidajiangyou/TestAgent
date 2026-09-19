# 节点⑨ · FH2.6（web 切换）+ FH2.7（conversation 切换）+ QL-1（质量线接线）

归档 HEAD：`b6344ed`；工作区干净（归档前测得）。按 plan-k §9.3，本目录**先于 FH2.8 删除
commit 落盘**，作为删除门的证据基线。

## 门（10 道，逐门独立子进程取真实退出码）

| 门 | 结果 | 读数 |
| --- | --- | --- |
| ruff check | PASS | testagent/ + tests/ + scripts/ |
| ruff format --check | PASS | 160 文件 |
| mypy strict | PASS | 94 源文件 0 错 |
| pytest tests/ -q | PASS | 968 passed |
| TestArchitectureGate | PASS | 7 passed（含新「pipeline 禁读 settings 单例」） |
| engine golden | PASS | 四轨迹 diff=0（本轮未重录引擎 golden） |
| tasks validate --strict | PASS | _example / gui / perf / testcase |
| links golden 三 stage | PASS | graph / planner / fixture |
| R6 benchmark | PASS | graph pair 1.0/0.9167、selected directed 1.0/0.8889 |

## 交付与证据

**FH2.6 web 切换（4f73474）**：`POST /api/generate` 内部改走 `tasks/testcase` +
`PipelineExecutor`，外部 HTTP 契约由 12 格基线守住——基线录于**改动前** commit
`75680a4`（`meta.git_sha` 为凭）。切换后重放：**11/12 格逐字节相同**（status、响应
schema、错误文案、CSV 字节、markdown 正文、count/historical_count）；唯一差异
`ok-with-swagger` count 4→2，裁决为「接受-修后行为」：旧链只在存在历史基线时才去重
生成结果内部重复（去重逻辑长在 `_merge_historical_cases` 的 `if historical_cases`
分支里），新链按已批准的 H1 维度 `duplicate_generated` 恒去重。E4 只放行
session_id、机器路径前缀、报告时间戳三类噪声。

**FH2.7 conversation 切换（c89e74b）**：会话生成提示词不再调 `build_testcase_prompt`
/`build_performance_prompt`/`build_gui_test_prompt`，改渲染 `tasks/<pkg>/prompts`；
提示词文本字节由 5 格基线（录于 4f73474）证明零漂移，另有 1 格测试把 `templates/`
回退路径打桩为失败，证明文本确实来自任务包而非旧目录（防「字节相同但没换源」）。

**QL-1 质量线接线（7b62fe9 + 58ce5a7）**：plan-k 未排期的缺口——T1~T13 只有旧生成器
一个消费者，新链零引用，导致 FH2.4「修后行为首录」录进了没有质量线的链。已抽为
`pipeline/quality_pass.py` 单一实现并接入新链。等价证据
（`tests/test_quality_chain_equivalence.py`，同脚本双链）：LLM 调用次数相等、
id/title/endpoint/test_type/priority/steps/expected_results/tags 序列逐字段相等、
两侧 executability 均有定级；被接受的那一处差异单独成测试，不藏在绿灯里。

## 已知遗留（本节点未覆盖）

- 17 键产物 dict 的 `binds` / `scenario_*` / `covers_obligations` 列在两条链上都可能
  为空（取决于模型是否输出），本轮只保证**不降级**，不声称新链更严。
- 旧链 vs 新链等价比较未覆盖 review 开/关两态的产物差异（review 语义改进按设计保留，
  程序身份字段由缺陷⑦的 `_restore_code_identity` 恢复）。

## 复核命令

```bash
.venv/bin/python scripts/gate_archive.py --node FH2.6-2.7-ql1   # 复跑本目录
.venv/bin/python -m pytest tests/test_migration_parity_web.py tests/test_migration_parity_prompts.py \
    tests/test_quality_chain_equivalence.py -q                   # 契约 / 提示词 / 双链等价
TESTAGENT_RECORD_PARITY=1 .venv/bin/python -m pytest tests/test_migration_parity_web.py -q \
    && git diff --stat tests/fixtures/                           # 期望只差 provenance 两行
```

结论：**PASS**。下一节点 = ⑩（FH2.8 两个删除门），其证据基线即本目录。
