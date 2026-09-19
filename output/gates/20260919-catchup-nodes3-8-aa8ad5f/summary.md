# 补录归档 · plan-k 节点 ③④⑤⑥⑦⑧

**为什么是"补录"**：节点 ③（LINK-S1a+S1b）、④（LINK-S2 规则层+消费层）、⑤（LINK-S4 纯函数地基）、
⑥（FH-H parity harness）、⑦（FH2.4 testcase parity）、⑧（FH2.5 perf/gui parity）当初只在
commit message 里声称过门，`output/gates/` 无记录 —— 按 plan-k §9.3「无记录视为门未过」，本目录是
把它们补成可复核证据，而不是新结论。

**为什么现在补可信**：⑦⑧ 的「重放 minimal diff=0」在 2026-09-19 之前是**自我证明**——每次
`pytest` 都会把刚跑出的观察值写回基线（`record_fixture` 无开关），门禁抓不到漂移；另外
`pipeline/runtime.py` 曾从 `get_settings()` 单例取 `output_token_cap`（注入被忽略、取值取决于谁先
预热缓存），以及嵌套 Settings section 各自读 cwd 下的 `.env`（本机
`OPENAI_MAX_OUTPUT_TOKENS=32768` 会被录进基线）。三处在 `fdbd117` 修复后，基线才是跨机器、
跨 cwd 稳定的比较对象。本归档在修复之后的 `aa8ad5f` 上重跑，等于给 ⑥⑦⑧ 换上一个真门禁重验。

## 门的实况（逐门，全部独立子进程取真实退出码）

| 门 | 结果 | 关键读数 |
| --- | --- | --- |
| ruff check | PASS | testagent/ + tests/ + scripts/ |
| ruff format --check | PASS | 155 文件 |
| mypy strict | PASS | 93 源文件 0 错 |
| pytest tests/ -q | PASS | 928 passed（修前同一条命令 12 failed + 4 errors） |
| TestArchitectureGate | PASS | 含新增「pipeline 禁读 settings 单例」断言 |
| engine golden | PASS | 四轨迹 diff=0（未重录） |
| tasks validate --strict | PASS | _example / gui / perf / testcase 四包 |
| links golden（S0 硬门） | PASS | graph / planner / fixture 三 stage diff=0 |
| R6 benchmark graph | PASS | pair P=1.0 R=0.9167 |
| R6 benchmark selected | PASS | directed P=1.0 R=0.8889 |

## 与各节点 DoD 的对应

- **⑥ FH-H**：`tests/parity_harness.py` 自测 11 项（record→replay diff=0 两种观察、漂移检出、
  指纹稳定性、凭据守卫、录制必须是显式开关、缺基线必须 fail loud）。公共接口冻结后本次的改动
  属 harness 变更（plan-k §8.2「禁回写」需同步升级两条 parity 线）——已按规程同时升级
  FH2.4/FH2.5 两侧：`ensure_fixture` + `bare_settings`。
- **⑦ FH2.4**：五维矩阵 `tc-a1..a5` 在仓库根（`.env` 在场）显式重录 → 与已提交基线仅差
  `git_sha`/`recorded_utc` 两行，行为零变化（Artifact + Fingerprint + Failure-Semantics 三门合一）。
- **⑧ FH2.5**：`gui-baseline` / `perf-baseline-k6` 同上；`test_gui_contract_compilable` 从
  "含 import pytest" 升级为真 `compile()`。
- **③④⑤ links**：links golden 三 stage + R6 双 stage 由常驻 pytest 与本次归档双覆盖。

## 复核命令

```bash
.venv/bin/python scripts/gate_archive.py --node <milestone> --note "..."   # 生成/复跑本目录
.venv/bin/python -m pytest tests/ -q
TESTAGENT_RECORD_PARITY=1 .venv/bin/python -m pytest tests/test_migration_parity_testcase.py \
    tests/test_migration_parity_perf_gui.py -q && git diff --stat tests/fixtures/   # 期望只差 meta 两行
```

结论：**PASS（作为补录证据）**。真正的节点 ⑨~⑬ 尚未开工，见 `task.md` §二.1。
