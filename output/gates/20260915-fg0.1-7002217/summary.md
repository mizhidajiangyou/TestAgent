# 节点① FG0.1 落档 · summary

**结论：PASS（2026-09-15，commit 7002217）**

## 裁决内容（FG0.1 四项必填，plan-k §1.1）

1. **A/B 选择**：方案 A（用户 2026-09-15 确认沿用 fix-plan §6.2 D1=A）；T1~T10 → 质量里程碑门 → 修后重录 fixtures → B6b.1~4 → B7 → 删除门 → links 收尾。方案 B 存档于 plan-k 附录 A，不执行。
2. **fixture 重录时点**：阶段 A 收尾重录门（plan-k §6.4 第 3 条）。
3. **窗口期 testcase 链允许改动范围**：仅 fix-plan 内 T1~T15 改动；gui/perf 生成器、prompt_builder 生成类函数、6 个 j2 全程冻结（F1/F2）。
4. **字段先建方**：`response_schemas`+resolver→T3；五身份字段→T8；`binds`/`executability`→T10；`path_id`/`source_stage`+LINKS_*→LINK-S1a（物理落盘按 plan-l L-1 单写者序列）。

## 首日核验结果（plan-k §11）

- 全量门 7 项全绿：pytest **583 passed**（0 fail）、ruff check / format --check / mypy strict / tasks validate --strict / engine golden 四轨迹 diff=0 / plan-l §6 arch 断言 3 passed。
- 基线现状与 plan-k §1.3 零偏差：`tasks/`={_example,perf}；无 `build_engine_generate_unit`；web 未切 `pipeline_executor`；质量/links 新标识全仓零命中；`output/gates/` 可入库（check-ignore exit=1）。
- 基线计数说明：task.md §五 旧基线 567 → 本次 583，增量来自 plan-i/plan-j 轮新增测试，全绿无回归。

## 与上一节点的差异

首个门节点；无前序。预备线（FH1.5 卫生 4 commit）已在本节点前落地：d4a5aeb（gates 例外）、6f33b19（.gitignore 补全 + rm --cached）、caeb622（CLI/examples 入库）、7002217（task-history 入库）。

## 遗留待用户动作

- 根目录 6 个散落文件（bak-task.md/dev.md/overview.md/renwu.md/t.b.md/task-summary.md）处置方式待裁决（未预先进 .gitignore）。
- FH0.2 的 v2.1 评审 4 项条件（C3.6/C3.7/C4.5/C4.6）待方案作者确认（不阻塞 T1~T10）。
