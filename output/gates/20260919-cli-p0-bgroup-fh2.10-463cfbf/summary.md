# 批次归档 · CLI 文本产物 P0 + B 组 + README（FH2.10 一项）

归档 HEAD：`463cfbf`，工作区干净。10 道门全 PASS（pytest 979 passed / mypy strict 94 文件 / ruff+format 覆盖 testagent+tests+scripts / engine golden diff=0 / links golden 三 stage / R6 双 stage / validate --strict 四包）。

## 这一批修了什么

- **P0（真实模型才暴露）**：`testagent perf|gui -o x.js|.py` 写出的文件是 JSON 编码的字符串（首字符 `"`、换行全成字面 `\\n`、脚本不可运行）。真因是动态 CLI 选项 `click.Option(["--format","-f"], "output_format", ...)` 的第二位置参数其实是 `show_default`，参数名仍为 `format`，而 `_run_task` pop 的是 `output_format` → 永远落到硬编码 `json` writer。修前 `real_perf2.js` 11387B/真实换行 0/字面 290；修后 `real_perf3.js` 9772B/真实换行 281/字面 0。回归断言改为"不得以引号开头 + 行序列等于脚本"（旧断言只 `compile()`，而 JSON 字符串字面量本身是合法 Python 表达式，所以 979 项测试全绿仍漏）。
- **撤回一条误判**：曾判"模型把脚本包成 JSON 字符串"并加了 `unwrap_json_string`；快照证明 artifact 本就是真换行脚本，引号是写盘加的 → 该投机代码已删除。
- **B 组三项**：会话 id 拼路径未校验（新增单一持有者 `validate_session_id`，三个入径点）、`required: null` 让签名渲染器抛 TypeError（compact+rich 各一处）、CI `docker login -p` 明文入 argv 改 `--password-stdin`。
- **FH2.10 一项**：CI 加 `testagent tasks validate --strict` 硬门；README 补任务包命令表 + parity 重录纪律 + 会话审计/恢复说明。

## 真实模型验收（额度正常）

`testagent testcase`（bookstore 3 需求 + 9 端点，review off）：51 条，steps/expected_results 全有，越权端点 0，(title,endpoint,type) 重复 0，executability 31 INTEGRATION / 20 DRAFT，closure 0.0~1.0，22 条义务声明；`testagent review` REVIEWED；`perf`/`gui` 产物修后可运行。

## 未做

FH2.8 删除门（前置：testcase 侧迁移期对比固化）、FH2.9、CI 改用 uv.lock（本机无法在不破坏 .venv 下验证）、T12b/S5b/S6b/S7、发布动作（按指示不做，见 task.md Checklist）。
