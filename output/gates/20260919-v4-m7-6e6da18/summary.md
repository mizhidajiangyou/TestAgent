# v4 M7 发布门

归档 HEAD：`6e6da18`；UTC：`2026-09-19T04:44:38Z`。
本目录早于 `scripts/gate_archive.py`，故只有 manifest（逐门读数在下表），无独立日志文件。

| 门 | 结果 |
|---|---|
| `ruff check + format` | PASS |
| `mypy strict` | 92 files, 0 errors |
| `pytest tests/ -q` | 863 passed |
| `engine golden` | 16 passed, diff=0 |
| `TestArchitectureGate` | 6 passed |
| `tasks validate --strict` | PASS |
| `CLI discovery (--help x4)` | PASS |

## 真实模型验收

| 项 | 记录 |
|---|---|
| model | "qwen3.7-flash-2026-07-15 (DashScope primary fallback chain)" |
| review_mode_b_json | {"status": "REVIEWED", "run_id": "ebc55e41fd24", "exit": 0, "ids_preserved": true} |
| review_mode_b_md | {"status": "REVIEWED", "run_id": "67315650c88c", "fences_balanced": true} |
| review_mode_a | {"status": "REVIEWED", "run_id": "cd117927fa7b"} |
| single_concurrency_1 | {"cases": 15, "capabilities_saved": {"split_mode": "single", "concurrency": 1}, "serial_log": true, "swagger_warn": true} |
| auto_default | {"cases": 40, "capabilities": "auto/3(settings)"} |
| gui_with_reference | {"selected": "6/6", "compile_ok": true} |
| gui_without_reference | {"done": true} |

验收中修复：publication._validate_serialized made format-aware (MD was being JSON-parsed); BOM check restructured

**结论**：`PASS`
