# TestAgent

**Requirements + API Document -> Agent Pipeline -> Test Cases + Performance Scripts + Test Reports**

TestAgent 是一个 AI 驱动的测试资产生成工具：输入需求文档和 Swagger/OpenAPI 接口文档，自动产出结构化测试用例、性能测试脚本（k6 / JMeter）、GUI 测试脚本（Playwright）以及测试报告，并支持通过连续对话对生成的资产进行校验和迭代优化。

## 特性

- 解析 Swagger/OpenAPI 规范（支持 URL、JSON/YAML 文件、dict 三种输入；OpenAPI 3 响应 schema 抽取与 `$ref` 递归解析；Swagger 2.0 显式降级）
- 解析需求文档（支持 Markdown、纯文本、JSON，以及 PDF/DOCX/HTML/PPTX 等二进制格式）
- 基于 LLM（OpenAI / Azure OpenAI）生成测试用例，覆盖正向/边界/负向/集成场景
- **多模型 fallback**：`OPENAI_MODEL` 支持逗号分隔配置多个模型，首选模型失败时自动尝试后续模型
- **多轮交叉校验 Review**：`REVIEW_ENABLED=true` 时，奇数轮用非首选模型、偶数轮用首选模型交替评审，单模型场景自动降级并告警
- **质量确定性门禁**（随生成自动运行）：义务覆盖账本（需求 AC + 规格事实）、场景身份去重（受控词表 + 规范化变体键）、一致性检查（需求 vs 规格的四类确定性交叉检查 + 策略裁决）、可执行性门禁（占位符闭合 Gate-A / 绑定契约 Gate-B / 三级定级）、会话级 CASES_BUDGET 预算（裁条有账）
- **生成全程可审计**：每次会话落 `output/sessions/<sid>/`——逐响应 raw 原文、调用元数据流水、全链路对账表（merge − 去重 − 裁剪 = 产物数）、义务/去重/预算/一致性/可执行性五份报告
- **跨模块关联（links）**：从规格与需求散文派生确定性关系图（R1 字段流 / R2 资源 ID / R3-R4 生命周期 / R6 中文散列模板），规划跨模块路径并按预算选择，配三道验证 Gate 与 pair/path 双口径覆盖率（`LINKS_*` 配置，当前管线级能力、CLI 接线随任务包迁移推进）
- **单文档模式与并发覆盖**：`--split/--no-split` 选择章节切分或整文单单元输入；`--concurrency` 本次调用覆盖并发度
- **独立评审服务**：`testagent review` 对已有产物做带需求参考（mode A）或纯质量（mode B）的分片评审，复用引擎 ReviewLoop、状态机与保留率护栏，产物/报告原子发布（no-clobber、in-place 自动备份）
- 基于 LLM 生成可执行的性能测试脚本（k6 或 JMeter JMX，含 XML 完整性校验）
- **GUI 测试生成**：基于需求 + 目标 URL 生成 Playwright Python 测试脚本（stagehand 风格的健壮定位器与 `expect()` 断言）；`--with-testcases` 严格导入已有用例作为补充参考（坏条目在 LLM 调用前失败）
- **会话式精炼引擎**：`testagent chat` 提供交互式连续对话，对已生成的测试用例/脚本/代码进行 generate → validate → refine 迭代优化（langgraph 风格的状态管理与版本链）
- **健壮文档解析**：PDF/DOCX/HTML/PPTX 三级回退（docling → 格式专属库 → 纯文本），自动探测可用后端
- 生成 Markdown / JSON 格式的测试用例报告和性能测试报告模板，测试用例支持导出 CSV（Excel 友好）
- 分层配置（环境变量 > .env > 默认值）、依赖注入容器、CLI 命令行界面

## 项目结构

```
TestAgent/
├── testagent/                 # 主包
│   ├── config/                # 配置层（settings、数据模型、日志）
│   ├── parsers/               # 解析层（Swagger 含 $ref 解析、需求文档、DocumentParser）
│   ├── engine/                # AI 引擎层（LLM 客户端 + 模型档案、Prompt 构建、会话引擎、
│   │                          #  截断感知生成循环、raw 审计转储）
│   ├── pipeline/              # 任务包管道 + 质量域纯函数（义务/一致性/绑定/场景身份/
│   │                          #  可执行性/归一化/context_builder/links 图-规划-门禁）
│   ├── generators/            # 生成层（测试用例、性能脚本、GUI 脚本）
│   ├── artifact/              # 评审产物层（严格加载、分片、发布）
│   ├── review/                # 独立评审服务（复用引擎 ReviewLoop）
│   ├── orchestration/         # 能力编排（输入适配、并发覆盖、GUI 用例导入）
│   ├── reports/               # 报告层（测试用例报告、性能报告）
│   ├── cli/                   # CLI 命令包（commands/ 含 tasks、checkpoint、review）
│   └── container.py           # 依赖注入容器
├── tasks/                     # 任务包（_example 最小示例）
├── templates/                 # Jinja2 Prompt 模板
├── examples/                  # 示例输入文件（含 R6 benchmark 三件套、bookstore E2E 输入）
├── scripts/                   # 基线测量 / links golden / R6 benchmark 脚本
├── tests/                     # 单元测试 + 端到端测试 + 架构门禁 + links golden fixtures
├── docker/                    # Docker 镜像与编排
├── k8s/                       # Kubernetes 部署清单
└── monitoring/                # Prometheus / Grafana 监控配置
```

## 安装

要求 Python >= 3.14。推荐使用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境与依赖：

```bash
uv sync --frozen --all-extras        # 按 uv.lock 精确安装（与 CI 完全一致）
source .venv/bin/activate
```

依赖有改动时必须同步锁文件，否则 CI 的 `uv lock --check` 会红：

```bash
uv lock && uv lock --check
```

（`uv pip install -e ".[dev]"` 仍可用，但它按"满足下界的最新版"解析，
不与 `uv.lock` 一致 —— 门禁与镜像就该锁在同一份解析结果上。）

## 配置

复制环境变量模板并填写 API Key：

```bash
cp .env.example .env
```

关键配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `OPENAI_MODEL` | 模型名，支持逗号分隔多个（首选 + fallback + review 评审） | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | API 地址（可指向兼容服务） | `https://api.openai.com/v1` |
| `OPENAI_MAX_OUTPUT_TOKENS` | 单次最大输出 token（防截断） | `16000` |
| `OPENAI_MAX_CONCURRENCY` | 并发请求上限（可用 `--concurrency` 按次覆盖） | `5` |
| `OPENAI_MODEL_PROFILE` | 显式指定模型档案（deepseek-v4 / qwen3.8 / openai-reasoning / openai-classic / generic-openai-compatible），空 = 按模型名自动匹配 | 空 |
| `OPENAI_STREAM` | 流式输出（实时进度日志） | `true` |
| `AZURE_OPENAI_ENABLED` | 启用 Azure OpenAI | `false` |
| `REVIEW_ENABLED` | 生成链二次复检开关（多轮交叉校验；不影响独立 `testagent review` 命令） | `false` |
| `REVIEW_MAX_ROUNDS` | 生成链 Review 轮数（奇数轮非首选模型、偶数轮首选模型） | `2` |
| `OUTPUT_LANGUAGE` | 输出语言，`chinese` 或 `english` | `chinese` |
| `AUDIT_DUMP_ENABLED` | 会话审计落盘（raw 响应 + 对账表 + 报告）；`false` 回滚到静默行为 | `true` |
| `CONFLICT_POLICY` | 规格 vs 需求冲突裁决：`strict`（不代做决策）/ `spec_first` / `requirement_first` | `strict` |
| `CASES_BUDGET` | 会话级用例总量上限（义务覆盖用例优先保留；`0` 关闭） | `60` |
| `SPLIT_MODE` | 需求输入模式：`auto` 章节切分 / `single` 整文单单元 | `auto` |
| `LINKS_ENABLED` | 跨模块 links 管线总开关（R6 单独受 `LINKS_PROSE_ENABLED` 控制） | `true` |
| `SCRIPT_FORMAT` | 默认性能脚本格式 | `k6` |
| `OUTPUT_DIR` / `TASKS_DIR` / `TASKS_DISABLE` | 输出目录 / 任务包目录 / 任务包屏蔽开关 | 见 `.env.example` |

完整清单见 `.env.example`；查看当前生效配置：

```bash
testagent config
```

## 使用

### 生成测试用例

```bash
# JSON 格式（默认）
testagent generate-tests \
  --swagger examples/sample_swagger.json \
  --requirements examples/sample_requirements.md \
  --output ./output/testcases.json

# Markdown / CSV（CSV 为 UTF-8 BOM，Excel 直接打开中文不乱码）
testagent generate-tests -s examples/sample_swagger.json -r examples/sample_requirements.md -f markdown -o ./output/testcases.md
testagent generate-tests -s examples/sample_swagger.json -r examples/sample_requirements.md -f csv -o ./output/testcases.csv
```

每次运行自动落会话审计目录 `output/sessions/<session_id>/`：逐响应 `.raw.txt`、`raw_calls.jsonl` 元数据流水、`reconciliation.json` 全链路对账表，以及一致性/义务/去重/预算/可执行性报告（有发现时）。任何用例被确定性门禁移除都逐条有账，不存在静默丢弃。

#### 单文档模式与并发覆盖

```bash
# 整文作为单一需求单元（Phase1 一次扇出），并发 1 串行执行
testagent generate-tests -r long_doc.md --no-split --concurrency 1 -o output/cases.json

# 章节切分（默认），本次并发覆盖为 3
testagent generate-tests -r requirements.md --split --concurrency 3 -o output/cases.json
```

- `single` 模式保留全文不切分、不截断；超长时仅 WARN（截断恢复兜底保持开启），与 Swagger 同用时提醒 Phase 2 仍按端点分批。
- 优先级：显式 CLI > resume 保存值 > 环境变量/.env > 默认值；生效值随会话记录保存，`--resume` 时自动继承。
- `single` 模式无逐条验收标准，义务覆盖率与章节模式口径不同（日志与元数据均已注明）。

### 二次复检（Review，生成链内）

设置 `REVIEW_ENABLED=true` 后，生成完首批用例会再起一个**全新的对话（无上下文）**，把当前用例 JSON + 接口 + 需求喂给 LLM 做整体复检。多模型时默认 2 轮交叉校验（奇数轮非首选模型、偶数轮首选模型）。

```bash
OPENAI_MODEL="gpt-4o-mini,gpt-4o" REVIEW_ENABLED=true REVIEW_MAX_ROUNDS=2 \
  testagent generate-tests -s examples/sample_swagger.json -r examples/sample_requirements.md -o ./output/testcases.json
```

`REVIEW_MAX_ROUNDS` 控制总轮数：调到 3-4 提质量，调到 1 提速度，设 0 或 `REVIEW_ENABLED=false` 跳过。单轮解析失败保留上一轮结果不丢弃。

### 独立评审（`testagent review`）

对**已有产物**执行独立评审，不重新生成。支持 JSON（对象数组或 `{"test_cases": [...]}` envelope）、Markdown、TXT（UTF-8）。

```bash
# Mode A：带需求参考评审（参考缺失/解析失败整体 FAILED，不静默降级）
testagent review --doc cases.json -r requirements.md --rounds 2 -o cases_final.json

# Mode B：纯质量评审（结构/清晰度/重复/可执行性；不推断未提供的业务规则）
testagent review --doc cases.json --rounds 2

# 原地评审（自动创建 <源文件>.bak.<run_id> 字节备份）
testagent review --doc cases.json --in-place

# Markdown 文档评审（逻辑块分片，代码围栏/表格不从中间切开）
testagent review --doc notes.md -o notes_final.md
```

行为要点：

- 文档按逻辑条目/块分片（`REVIEW_CHUNK_SIZE`，同时受 `REVIEW_MAX_PROMPT_CHARS` 字符预算约束）；单个不可拆块超预算记 `oversized_atomic_item` 并零调用失败，不截半个对象。
- 分片状态机：`REVIEWED`（评审采纳且保留率 ≥ 70% 相对原片）/ `REVIEW_REJECTED`（保留率不足或引擎护栏拒收）/ `REVIEW_FAILED`（异常/不可解析）；失败片超过 `REVIEW_MAX_CHUNK_FAILURE_RATIO` 整文回滚为原始产物。
- 只有整文 `REVIEWED` 才写评审产物；`REJECTED/FAILED/DISABLED` 只发布报告，源文件不变。
- 产物与报告均为原子、不覆盖已有文件的发布（no-clobber）；报告含 run_id、源/输出哈希与逐分片账本，是评审状态的权威来源。退出码：`0` 正常完成、`1` 评审失败或发布失败、`2` 用法错误或禁用。
- Mode B 不校验需求覆盖，也不保证优于原文。

### 生成性能测试脚本

```bash
# k6 脚本（默认）
testagent generate-perf -s examples/sample_swagger.json --base-url https://api.example.com

# JMeter JMX 脚本
testagent generate-perf -s examples/sample_swagger.json -f jmeter -o ./output/perf.jmx
```

### 生成 GUI 测试脚本（Playwright）

```bash
# 基础用法
testagent generate-gui -r examples/sample_requirements.md --url https://example.com

# 带 Swagger 上下文（API-aware GUI 测试）
testagent generate-gui -r requirements.md -s swagger.json --url https://app.example.com

# 导入已有测试用例作为补充参考（严格校验：坏条目/重复 id 在 LLM 调用前失败）
testagent generate-gui -r requirements.md --with-testcases cases.json --url https://example.com
```

参考用例按 high > medium > low 优先级在数量（`GUI_REFERENCE_MAX_CASES`）与字符（`GUI_REFERENCE_MAX_CHARS`）双预算下选择，选择账目（selected/omitted 及原因）结构化输出；API-only 场景不会伪造页面选择器。默认不带 `--with-testcases` 时行为与原路径完全一致。

生成的脚本可直接用 pytest 运行：

```bash
pytest output/gui_test.py --browser chromium
```

### 连续对话精炼（Chat）

`testagent chat` 提供交互式对话界面，可对已生成的测试用例、性能脚本、GUI 脚本进行 generate → validate → refine 迭代优化。

```bash
testagent chat -r requirements.md -s swagger.json
testagent chat -r requirements.md --session my-session-1   # 恢复会话
```

内置命令：`generate` / `refine` / `validate` / `save` / `history` / `exit`。

### 任务包管道（Task Packages）

除内置命令外，TestAgent 支持以**任务包**方式声明式扩展新任务：一个目录 + `manifest.json` + Jinja2 模板，即可获得一个新的 CLI 命令（无需写 Python）。

```bash
testagent tasks list --all
testagent tasks validate --strict   # CI 模式，任何 manifest/模板问题 exit 1
testagent _example --text "hello world" -o out/echo.json
```

**失败语义与恢复**：每个生成单元按封闭七态分类（SUCCESS / EMPTY / INVALID / TIMEOUT / PROVIDER_ERROR / VALIDATION_ERROR / CANCELLED），各自映射恢复策略。

**快照（pre_review_snapshot）**：review 前原子写入产物快照，用于 review 异常时找回已生成产物：

```bash
testagent checkpoint list
testagent checkpoint recover <session_id> --save-as out/recovered.json
```

`TASKS_DISABLE` 可屏蔽单个任务包作回滚开关。

### 运行生成的脚本

```bash
k6 run ./output/perf_test.js
jmeter -n -t ./output/perf_test.jmx -l results.jtl
pytest ./output/gui_test.py --browser chromium
```

## 开发

```bash
# 运行测试（860+ 用例：端到端流水线 + 多模型 fallback + 多轮 review + 会话引擎 +
# GUI 生成器 + 文档解析器 + 截断恢复引擎 + 任务包管道 E2E + 质量域纯函数
# （义务/一致性/绑定/场景去重/可执行性/归一化）+ links 图/规划/门禁 + 架构门禁）
pytest

# 代码检查（ruff lint + format，含 scripts/）
ruff check testagent/ tests/ scripts/
ruff format --check testagent/ tests/ scripts/

# 严格类型检查
mypy testagent/

# 任务包严格校验（CI 模式）
testagent tasks validate --strict

# 引擎行为 golden（四轨迹重放必须 diff=0）
pytest tests/test_engine_event_baseline.py -q

# links golden 重放 + R6 benchmark（关系抽取三层指标 + 方向硬门 + selected 门）
python scripts/golden_links.py --stage all
python scripts/benchmark_r6.py --stage graph
python scripts/benchmark_r6.py --stage selected
python scripts/measure_baseline.py
```

架构约束（由 `tests/test_pipeline_e2e.py::TestArchitectureGate` 强制）：

- `pipeline/` 不得 import 旧 `generators` / `prompt_builder` / `conversation`（新层与旧生成器只在组合根装配）；
- pipeline / web / cli 新层不得调用旧 `build_*_prompt` seam；
- engine 域无关（零 import pipeline、零引用 TestCase/APIEndpoint）；
- 共享模型字段（TestCase/APIEndpoint/Settings）唯一声明 + 基线字段顺序锁死；
- links 域对共享字段禁用带默认值的 `getattr`（跨域读取必须走强类型 accessor）。

## 部署

### Docker

CI/CD 流水线已自动将镜像推送到 Docker Hub，可直接拉取使用，无需本地构建：

```bash
docker pull mzdjy/testagent:latest
```

镜像入口为 `testagent`，`docker run mzdjy/testagent:latest <args>` 等价于直接调用 CLI。结果默认写到容器内 `/app/output`，用 `-v` 把宿主目录挂进去即可在当前目录拿到产物。

#### 快速使用（Docker Hub 镜像）

```bash
# 1) 推荐：把当前目录挂为 /work，输入文件， output 提前创建给777权限。
docker run --rm -v "$PWD":/work -v "$PWD/output":/work/output -w /work \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    -e OPENAI_MODEL="$OPENAI_MODEL" \
    mzdjy/testagent:latest generate-tests \
    -r requirements.md -s swagger.json \
    -o output/testcases.json

# 2) 仅挂载 output 目录（输入用 URL，无需本地文件）
docker run --rm -v "$PWD/output":/app/output \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    mzdjy/testagent:latest generate-tests \
    -s https://petstore3.swagger.io/api/v3/openapi.json \
    -o output/testcases.json

# 3) 增量生成：以历史用例为基线，只补净新用例
docker run --rm -v "$PWD":/work -w /work \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    mzdjy/testagent:latest generate-tests \
    -r new_requirements.md -H output/testcases.json \
    -o output/testcases.json

# 4) 独立评审已有产物（原地评审自动备份）
docker run --rm -v "$PWD":/work -w /work \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    mzdjy/testagent:latest review \
    --doc output/testcases.json -r requirements.md -o output/testcases_final.json

# 5) 生成 GUI 测试脚本（Playwright），导入已有用例作参考
docker run --rm -v "$PWD":/work -w /work \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    mzdjy/testagent:latest generate-gui \
    -r requirements.md --with-testcases output/testcases.json --url https://example.com \
    -o output/gui_test.py

# 6) 启动 Web GUI（可被其他平台通过 iframe 嵌入）
docker run --rm -p 8000:8000 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    mzdjy/testagent:latest serve --host 0.0.0.0 --port 8000
```

#### 本地自建镜像

需要修改镜像内容时也可从源码构建：

```bash
docker build -f docker/Dockerfile -t testagent .

docker run --rm -v "$PWD":/work -w /work \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    testagent generate-tests -r requirements.md -s swagger.json \
    -o output/testcases.json

docker run --rm -v "$PWD/output":/app/output \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    testagent generate-tests -s https://example.com/openapi.json -o output/testcases.json

docker run --rm -p 8000:8000 -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    testagent serve --host 0.0.0.0 --port 8000
```

也可以用 docker compose：

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm testagent \
    generate-tests -r /work/requirements.md -s /work/swagger.json -o output/testcases.json
docker compose -f docker/docker-compose.yml up serve        # Web GUI
docker compose -f docker/docker-compose.yml up -d prometheus grafana  # 监控
```

> 镜像内置 `documents` + `web` 可选依赖，支持 PDF/DOCX/HTML/PPTX 解析与 Web GUI；以非 root 用户运行。

### Kubernetes

```bash
kubectl create secret generic testagent-secrets --from-env-file=.env
kubectl apply -f k8s/
```

## 监控

项目内置 Prometheus + Grafana 配置：

```bash
docker compose -f docker/docker-compose.yml up -d prometheus grafana
```

- Prometheus: `http://localhost:9090`（配置见 `monitoring/prometheus.yml`）
- Grafana: `http://localhost:3000`（默认账号 admin/admin，仪表板见 `monitoring/grafana-dashboard.json`）

## 文档与证据纪律

### 入库文档的引用纪律（有测试强制）

仓库被克隆后读者只能看到入库文件，因此：

- **入库正文只允许引用入库路径**：`examples/`（验收与样例产物）、`tests/`、`scripts/`、
  `testagent/`、`tasks/`、`output/gates/`。本地计划文档、`output/` 根下的运行产物、
  `output/guidang/` 一律不得出现在入库正文里。
- **引用不了就把结论抄进正文**：门禁读数、修前/修后字节数这类证据，写数字而不是写路径。
- 运行期输出的示例命令统一写 `./output/...` 前缀（表示"你跑出来的文件"，不是仓库内容引用）。
- 任何入库文件都不得含机器绝对路径（`/Users/...`、`/home/...`）、用户名或密钥；
  门禁归档的 `manifest.json` / `*.log` 由 `scripts/gate_archive.py` 写 `<repo>` 占位。

强制实现见 `tests/test_repo_hygiene.py`（三条：入库引用可达、归档结构完整、无个人路径）。

### 任务包命令与迁移 parity 纪律

除 `generate-tests` / `generate-perf` / `generate-gui`（迁移期保留、将被删除门移除）外，
同一能力已有任务包版命令：

```bash
testagent tasks list && testagent tasks validate --strict   # 包契约门（CI 已内置为硬门）
testagent testcase -r examples/bookstore_requirements.md -s examples/bookstore_swagger.json -o cases.json
testagent perf   -s examples/bookstore_swagger.json --base-url https://api.example.com -o load.js
testagent gui    -r examples/bookstore_requirements.md --url https://bookstore.example.com -o test_gui.py
testagent review --doc cases.json -r requirements.md --rounds 2 -o cases_reviewed.json
```

**parity fixtures 纪律**（为什么不能随手重录）：

- `tests/fixtures/migration/{testcase,perf,gui,web_contract,conversation_prompts}/*.json`
  是"改动前录、改动后放"的基线；默认 `pytest` **只读**这些基线做比对。
- 只有显式 `TESTAGENT_RECORD_PARITY=1` 才会写盘；重录必须在 commit message 里逐差异解释
  （golden 一旦在改动后重录，就变成自我证明，门禁失去抓漂移的能力）。
- 每个里程碑跑 `.venv/bin/python scripts/gate_archive.py --node <里程碑>`：逐门独立子进程
  取真实退出码，归档 `output/gates/<日期>-<节点>-<sha>/{manifest.json,*.log}`；无归档视为门未过。
- 会话审计与恢复：`<output_dir>/sessions/<session>/`（raw 原文 + reconciliation/义务/预算/去重/可执行性
  报告）、`<output_dir>/<session>.pre_review_snapshot.json`（评审炸了用它恢复：`testagent checkpoint list|recover`）。
  `--resume <session-id>` / `checkpoint recover <session-id>` 的 id 会做路径安全校验。

## 扩展路线

Apache License 2.0，见 [LICENSE](LICENSE)。


- 管道迁移收尾：web / conversation 已切任务包新链（FH2.6 / FH2.7，质量线 T1~T13 亦已接入）。
  剩余：两个删除门 FH2.8（B5.4 perf/gui、B6b.5 testcase）与 FH2.9（旧提示词构造器清理）；
  删除门前须把"迁移期对比"固化成永久 golden，之后才是 links 接线（S5b / S6b / S7）。
  各门的过门证据见 `output/gates/`（本仓库唯一入库的运行输出目录）。
- 性能结果分析：接入 `.jtl` / k6 JSON 结果，产出带指标的完整性能报告
- 会话持久化：将会话状态序列化到磁盘/数据库，支持跨进程恢复

