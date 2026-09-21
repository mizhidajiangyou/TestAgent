# 节点⑩（前半）· B5.4 删前基线：perf / gui 旧生成器待拆

归档 HEAD：`bdb5a0c`，`working_tree_dirty=true`（归档在删前跑，删除动作按用户要求需逐项
批准）。plan-k §9.1 要求"删除门的 gate 归档必须先于删除 commit 落盘"——本目录就是那份删前
读数：12 道门全 PASS，且此时旧链仍在，所以旧链对照类门（prompt oracle 的对拍、perf 指纹
parity）都还有活体尺子可校验。

## 门（12 道）

| 门 | 结果 | 读数 |
| --- | --- | --- |
| ruff check / format | PASS | 171 文件已格式化 |
| mypy | PASS | 98 源文件 0 错 |
| pytest tests/ -q | PASS | 1027 passed |
| TestArchitectureGate | PASS | 7 passed |
| engine golden | PASS | 16 passed |
| tasks validate --strict | PASS | 四包 ✓ |
| links golden / R6 两道 | PASS | graph・planner・fixture；pair 1.0/0.9167 |
| links baseline | PASS | 13 端点 / L0 725 字符 |
| links parity-off | PASS | 5/5 格 diff=0 |

## 旧链残余引用面（`scripts/symbol_gate.py` 实测，非估计）

perf/gui 一族共 **64 处**，按持有者分：

| 位置 | 处数 | 归属 |
| --- | --- | --- |
| tests/test_gui_test_generator.py + test_performance_generator.py | 32 | 只测旧链，随旧链删 |
| testagent/container.py | 7 | provider `performance_generator` / `gui_generator` |
| tests/test_perf_task.py | 5 | **拿旧链当尺子**的指纹 parity（删前须先固化，见下） |
| testagent/cli/commands/generate_{gui,perf}.py + commands/__init__.py | 5 | 被任务包命令遮蔽的旧入口 |
| tests/test_migration_parity_pipeline_prompts.py | 5 | 录制工具，随旧链删（尺子已存档） |
| testagent/generators/__init__.py + 两个旧生成器模块 | 5 | 本体 |
| tests/test_e2e_pipeline.py | 3 | 第 4 步用旧 perf 生成器，需改指任务包 |
| 其余（testagent/pipeline/validators.py 注释、tests/test_conversation.py、oracle 的禁用清单字符串等） | — | 文本提及，非调用 |

## 删前必须落地的三件事（本归档时刻已完成两件）

1. **prompt oracle 已入库**（commit `c4efe22`）：10 个场景的完整 `(system,user)` 存档 + AST
   自证"本门不得 import 旧层" ⇒ 删除旧链后保真门仍然成立。
2. **校验器直测与"模板缺失不静默兜底"已入库**（commit `bdb5a0c`）：旧链测过的
   `strip_fences`/`python_compile`/xml 判序/when 守卫改由新链直测；旧 builder 吞模板异常改用
   inline 兜底文案，任务包链必须失败出声。
3. **perf 指纹 parity 尚未固化**：`tests/test_perf_task.py` 的 16 格矩阵（format × review ×
   language × fenced）与 3 个默认值/覆盖/空响应格仍直接调 `PerformanceGenerator` 当尺子；
   它必须先录成 fixture，B5.4 才能真正删。

## 不能在本门删除的东西（留给 B7.3）

`templates/performance_prompt.j2` 与 `templates/gui_test_prompt.j2` 此时仍被旧
`prompt_builder.build_performance_prompt` / `build_gui_test_prompt` 渲染；`test_perf_task.py`
里"包模板与旧模板反向改名后逐字节相同"两格也依赖这两个文件。它们随 B7.3 删 `build_*` 一起走。

## 批准边界

本归档之后不再自动删任何文件：删除清单（含每个文件"能力由谁接管"）交用户逐项批准。
