# 节点⑨后续 · QL-2 前置：新链 prompt 保真门（修"模板拿到的是对象 repr"这一类缺陷）

归档 HEAD：`242d386`，`working_tree_dirty=true` —— 本会话按用户指示先跑门、后入库，所以
归档测得的是**未提交的工作树**；紧随其后的 commit 内容就是这份树。12 道门逐门独立子进程
取真实退出码。

## 门

| 门 | 结果 | 读数 |
| --- | --- | --- |
| ruff check | PASS | testagent/ + tests/ + scripts/ |
| ruff format --check | PASS | 169 文件 |
| mypy | PASS | 98 源文件 0 错 |
| pytest tests/ -q | PASS | 1010 passed |
| TestArchitectureGate | PASS | 7 passed（pipeline 不 import 旧层、不读 settings 单例） |
| engine golden | PASS | 16 passed（引擎事件基线未动） |
| tasks validate --strict | PASS | _example / gui / perf / testcase |
| links golden 三 stage | PASS | graph / planner / fixture |
| R6 benchmark | PASS | graph pair 1.0/0.9167、selected directed 1.0/0.8889 |
| links baseline | PASS | 13 端点、模块 {cart:4, orders:4, products:5}、L0=725 字符 |
| links parity-off | PASS | 5/5 fixture 在 `LINKS_ENABLED=true`、包未 opt-in 时八维 diff=0 |

## 缺陷清单（全部先有字节证据，再动手）

新增 `tests/test_migration_parity_pipeline_prompts.py`：同一个 deterministic fake 同时驱动旧链
（`TestCaseGenerator` / `GUITestGenerator` / `PerformanceGenerator`）与新链（`PipelineExecutor`
+ `tasks/*`），逐 unit 比较真实发出的 `(system, user)` 字节。首跑读数：testcase 四个场景
（plain / chinese / json-mode / historical）**每次调用的 system 与 user 全不同**，call 数一度
26 vs 50（脚本化响应耗尽后两臂各自重问；改成"按 prompt 出答"的 fake 后才对齐 3 vs 3）。逐项：

1. `{{ endpoints }}` / `{{ requirements }}` 在旧链是**字符串**（批次富签名、单条需求文本），
   新链把 `ctx.parsed` 的**对象列表**送进模板 ⇒ prompt 里是 `[APIEndpoint(...)]` repr；
2. 批切片失效：phase2 每个 batch 都拿到**全量**端点 ⇒ "batch=2" 只减调用数不改提示词；
3. system prompt 丢了 ERROR CONTRACT、语言提示、JSON-mode 包装与历史基线后缀（manifest 只有
   `inline:<裸基>`）——错误契约正是防两阶段各编一套状态码的东西；
4. `already_covered`（含 T8 场景身份列表）从未注入 ⇒ Phase 2 重做 Phase 1；
5. `historical_cases` 以 list 进模板（旧链是文本摘要）；
6. T9 权威值表未附加到新链 user prompt；
7. 单位级重编号泄漏进"已覆盖"摘要（旧链此处是模型自报的 `TC-XXX`）；
8. 无 swagger 时 `TaskPackage.render` 的 `{**synthetic, **context}` 兜底把**校验用样例**送进
   生产 prompt（负对照实测：同一模板不给该键时样例确实出现）。

## 单一真相源落点

`config/prompt_contract.py`（契约片段 + 组装顺序 + 覆盖文本）、`parsers/swagger_parser.py`
（富签名）、`pipeline/scenario.covered_identities`（T8 身份投影）各一处持有；旧
`prompt_builder` / `testcase_generator` / `quality_pass` 全部改为委托。新链侧：
`system_prompt: "generation:<base>"`（哪些基提历史基线由 `HISTORICAL_AWARE_BASES` 决定；脚本族
`SCRIPT_BASES` 不带错误契约）、**`manifest.prompt_views`** 按包声明渲染（testcase=富签名、
gui/perf=纯文本，旧链三者本就不一致，全局猜一个会改坏另一族）、`StageSpec.coverage_input`
声明跨阶段数据流、`runtime._render_unit_prompts` 为两臂唯一装配点、`inputs.parse_inputs` 预置
派生视图空值以杜绝 synthetic 兜底。另补 `testcase_to_full_dict` 的 `path_id`/`source_stage`
两列（此前往返即丢），并把"examples 只准用真实选项"做成机器门（旧示例里有个不存在的
`--no-review`）。

## 基线时效（逐 fixture 说明后才重录）

`testcase/tc-a1..a5`、`gui/gui-baseline`、`perf/perf-baseline-k6` 七个 fixture 重录。
`git diff` 字段分类：`user_sha256` 46 行、`system_sha256` 45 行、新增列 `path_id`/
`source_stage` 各 22 行、provenance 若干；**artifact 内容、label、params、response_text 一行未
动**。移动原因即上表 ①–⑧，而新哈希是"旧链那一份"（由本归档的保真门逐调用证明）。
`gui-baseline` 二次移动单独归因于 ⑧ 与 review context 换行保真（去掉多余 `rstrip`，Jinja 已
丢弃模板自身末行）。

## 由本节点带出的证据链缺陷（同批修复）

* 归档器边跑边写 `output/gates/<新目录>/`，同一轮里的 pytest 门扫到"有日志无 manifest"的
  半成品 ⇒ 每个归档自证 FAIL（观察污染被观察对象）。改为在 `output/.gates-staging/` 成稿后
  一次性搬入位，且拒绝覆盖已存在的归档。
* 入库证据烙着操作者机器：卫生门只扫 `.md/.json/.py/...`，**`.log` 不在扫描面**，所以日志里的
  `/Users/<name>/...` 一直是漏网之鱼。补：`SCAN_SUFFIXES` 加 `.log` + 归档器写日志统一替换为
  `<repo>`；`manifest.executor` 不再记录操作系统用户名。负对照：往被跟踪的 `.log` 里塞一条
  home 路径 → 门 FAIL 并点名该文件，还原 → 恢复通过。
* text unit 被校验器拒绝时完全静默（真机 gui 两次拿到不可编译脚本才暴露）：现把校验类型、
  返回长度与具体编译错误写进 WARNING；空 text artifact 不再被计数成 1 item（曾打印
  "Done: 1 item(s)" 而文件 0 字节）。
