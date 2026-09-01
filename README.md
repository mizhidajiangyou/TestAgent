# TestAgent

**Requirements + API Document -> Agent Pipeline -> Test Cases + Performance Scripts + Test Reports**

TestAgent 是一个 AI 驱动的测试资产生成工具：输入需求文档和 Swagger/OpenAPI 接口文档，自动产出结构化测试用例、性能测试脚本（k6 / JMeter）、GUI 测试脚本（Playwright）以及测试报告，并支持通过连续对话对生成的资产进行校验和迭代优化。

## 特性

- 解析 Swagger/OpenAPI 规范（支持 URL、JSON/YAML 文件、dict 三种输入）
- 解析需求文档（支持 Markdown、纯文本、JSON，以及 PDF/DOCX/HTML/PPTX 等二进制格式）
- 基于 LLM（OpenAI / Azure OpenAI）生成测试用例，覆盖正向/边界/负向/集成场景
- **多模型 fallback**：`OPENAI_MODEL` 支持逗号分隔配置多个模型，首选模型失败时自动尝试后续模型
- **多轮交叉校验 Review**：`REVIEW_ENABLED=true` 时，奇数轮用非首选模型、偶数轮用首选模型交替评审，单模型场景自动降级并告警
- 基于 LLM 生成可执行的性能测试脚本（k6 或 JMeter JMX，含 XML 完整性校验）
- **GUI 测试生成**：基于需求 + 目标 URL 生成 Playwright Python 测试脚本（stagehand 风格的健壮定位器与 `expect()` 断言）
- **会话式精炼引擎**：`testagent chat` 提供交互式连续对话，对已生成的测试用例/脚本/代码进行 generate → validate → refine 迭代优化（langgraph 风格的状态管理与版本链）
- **健壮文档解析**：PDF/DOCX/HTML/PPTX 三级回退（docling → 格式专属库 → 纯文本），自动探测可用后端
- 生成 Markdown / JSON 格式的测试用例报告和性能测试报告模板，测试用例支持导出 CSV（Excel 友好）
- 分层配置（环境变量 > .env > 默认值）、依赖注入容器、CLI 命令行界面

## 项目结构

```
TestAgent/
├── testagent/                 # 主包
│   ├── config/                # 配置层（settings、数据模型、日志）
│   ├── parsers/               # 解析层（Swagger、需求文档、DocumentParser）
│   ├── engine/                # AI 引擎层（LLM 客户端 + 模型档案、Prompt 构建、会话引擎）
│   ├── pipeline/              # 任务包管道（manifest/executor/registry/校验/写出）
│   ├── generators/            # 生成层（测试用例、性能脚本、GUI 脚本、截断恢复引擎）
│   ├── reports/               # 报告层（测试用例报告、性能报告）
│   ├── utils/                 # 工具函数
│   ├── cli/                   # CLI 命令包（commands/ 含 tasks、checkpoint）
│   └── container.py           # 依赖注入容器
├── tasks/                     # 任务包（_example 最小示例）
├── templates/                 # Jinja2 Prompt 模板
├── examples/                  # 示例输入文件
├── tests/                     # 单元测试 + 端到端测试 + 架构门禁
├── docker/                    # Docker 镜像与编排
├── k8s/                       # Kubernetes 部署清单
└── monitoring/                # Prometheus / Grafana 监控配置
```

## 安装

要求 Python >= 3.14。推荐使用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境与依赖：

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[dev]"
```

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
| `OPENAI_TIMEOUT` | 单次请求超时（秒） | `300` |
| `OPENAI_MAX_OUTPUT_TOKENS` | 单次最大输出 token（防截断） | `16000` |
| `OPENAI_MODEL_PROFILE` | 显式指定模型档案（deepseek-v4 / qwen3.8 / openai-reasoning / openai-classic / generic-openai-compatible），空 = 按模型名自动匹配 | 空 |
| `OPENAI_REASONING_EFFORT` | 首轮推理强度意图（low/medium/high/disabled），空 = 模型默认 | 空 |
| `OPENAI_BLOCKING_HARD_TIMEOUT` | 单次阻塞调用的硬墙钟上限（秒）。显式配置严格生效且必须 ≥ `OPENAI_TIMEOUT`；未配置 = max(超时×2, 600) | `0`（未配置） |
| `OPENAI_STREAM` | 流式输出（实时进度日志） | `true` |
| `AZURE_OPENAI_ENABLED` | 启用 Azure OpenAI | `false` |
| `PERF_BASE_URL` | 性能测试目标地址 | `https://api.example.com` |
| `PERF_VIRTUAL_USERS` | 虚拟用户数 | `100` |
| `PERF_DURATION_SECONDS` | 压测时长（秒） | `300` |
| `OUTPUT_DIR` | 输出目录 | `./output` |
| `TASKS_DIR` | 任务包扫描目录（每个子目录一个 manifest.json → 一个 CLI 命令） | `./tasks` |
| `TASKS_DISABLE` | 屏蔽指定任务包（逗号分隔，迁移期回滚开关） | 空 |
| `CHECKPOINT_KEEP_LAST` | 保留最近 N 个 pre-review 快照 | `50` |
| `SCRIPT_FORMAT` | 默认性能脚本格式 | `k6` |
| `REVIEW_ENABLED` | 生成后是否二次复检（多轮交叉校验） | `false` |
| `REVIEW_MAX_ROUNDS` | Review 轮数（奇数轮用非首选模型、偶数轮用首选模型） | `2` |
| `OUTPUT_LANGUAGE` | 输出语言，`chinese` 或 `english`（用例内容与报告标题） | `chinese` |

查看当前生效配置：

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

# Markdown 格式
testagent generate-tests \
  -s examples/sample_swagger.json \
  -r examples/sample_requirements.md \
  -f markdown \
  -o ./output/testcases.md

# CSV 格式（UTF-8 BOM，Excel 直接打开中文不乱码）
testagent generate-tests \
  -s examples/sample_swagger.json \
  -r examples/sample_requirements.md \
  -f csv \
  -o ./output/testcases.csv
```

Swagger 也可直接传 URL：

```bash
testagent generate-tests -s https://petstore3.swagger.io/api/v3/openapi.json
```

### 多模型 Fallback

`OPENAI_MODEL` 支持逗号分隔配置多个模型，**第一个是首选模型**（用于常规生成），后续是 fallback 候选 + Review 评审员：

```bash
OPENAI_MODEL="gpt-4o-mini,gpt-4o,gpt-4-turbo" testagent generate-tests -s swagger.json
```

- 首选模型在重试 3 次后仍失败时，自动切换到下一个 fallback 模型，每次切换都有日志告警。
- token usage 跨所有子客户端汇总，方便观测各模型消耗。

### 二次复检（Review）

设置 `REVIEW_ENABLED=true` 后，生成完首批用例会再起一个**全新的对话（无上下文）**，把当前用例 JSON + 接口 + 需求喂给 LLM 做整体复检：修正接口不匹配、补齐 CRUD/边界/安全/性能用例、把预期结果改写成可断言的格式、填充 tags、补充清理与幂等用例，最后输出**完整合并后的最终列表**。

**多轮交叉校验**：当配置了多个模型时，Review 默认跑 2 轮，奇数轮用非首选模型、偶数轮用首选模型交替评审，两个模型互相挑刺比单模型自审更能发现盲点。单模型场景会自动降级到同模型 review，但日志会显式 WARNING 提示交叉校验未生效。

```bash
OPENAI_MODEL="gpt-4o-mini,gpt-4o" REVIEW_ENABLED=true REVIEW_MAX_ROUNDS=2 \
  testagent generate-tests \
  -s examples/sample_swagger.json \
  -r examples/sample_requirements.md \
  -f markdown \
  -o ./output/testcases.md
```

`REVIEW_MAX_ROUNDS` 控制总轮数：调到 3-4 提质量，调到 1 提速度，设 0 或 `REVIEW_ENABLED=false` 跳过。单轮解析失败保留上一轮结果不丢弃。

### 输出语言

`OUTPUT_LANGUAGE=chinese`（默认）时，用例内容（标题/步骤/预期结果）和报告（标题/汇总表/类型/优先级）均为中文；设为 `english` 恢复英文输出。

### 生成性能测试脚本

```bash
# k6 脚本（默认）
testagent generate-perf -s examples/sample_swagger.json --base-url https://api.example.com

# JMeter JMX 脚本
testagent generate-perf -s examples/sample_swagger.json -f jmeter -o ./output/perf.jmx
```

输出物：

- `perf_test.js` / `perf_test.jmx`：可直接运行的性能脚本
- `perf_test.md`：性能测试报告模板（含配置、KPI 表格、执行说明）

### 生成 GUI 测试脚本（Playwright）

基于需求文档 + 目标 URL 生成 Playwright Python 测试脚本，使用健壮的定位器策略（`get_by_role` / `get_by_label` / `get_by_text`）和 `expect()` 机器可校验断言（参考 stagehand 的无障碍树定位思路）。

```bash
# 基础用法
testagent generate-gui -r examples/sample_requirements.md --url https://example.com

# 带 Swagger 上下文（API-aware GUI 测试）
testagent generate-gui -r requirements.md -s swagger.json --url https://app.example.com

# 指定输出路径
testagent generate-gui -r requirements.md --url https://example.com -o tests/test_login.py
```

生成的脚本可直接用 pytest 运行：

```bash
pytest output/gui_test.py --browser chromium
```

### 连续对话精炼（Chat）

`testagent chat` 提供交互式对话界面，可对已生成的测试用例、性能脚本、GUI 脚本进行 generate → validate → refine 迭代优化。会话状态在内存中持久化，支持跨轮精炼和版本追溯（参考 langgraph 的 StateGraph + checkpointer 模式）。

```bash
# 带需求 + Swagger 上下文启动对话
testagent chat -r requirements.md -s swagger.json

# 仅带需求
testagent chat -r requirements.md

# 恢复之前的会话
testagent chat -r requirements.md --session my-session-1
```

对话中可用自然语言下指令：

```
> 生成用户注册模块的测试用例
> 给密码字段补充更多边界用例
> 验证当前测试用例
> 精简一下用例描述
> save
> exit
```

内置命令：`generate` / `refine` / `validate` / `save`（保存最新 artifact）/ `history`（查看历史）/ `exit`。

### 任务包管道（Task Packages）

除内置命令外，TestAgent 支持以**任务包**方式声明式扩展新任务：一个目录 + `manifest.json` + Jinja2 模板，即可获得一个新的 CLI 命令（无需写 Python）。内置 `tasks/_example` 是最小示例：

```bash
# 列出发现的任务包（--all 含下划线隐藏包）
testagent tasks list --all

# 校验所有任务包（manifest 严格 schema + 模板 synthetic-context 渲染）
testagent tasks validate          # 报告问题但 exit 0
testagent tasks validate --strict # 任何问题 exit 1（CI 模式）

# 运行隐藏示例包（单阶段 echo：text 输入 → JSON 数组产物）
testagent _example --text "hello world" -o out/echo.json
```

manifest.json 声明输入（swagger/requirements/file/text/choice/int/bool）、阶段（模板 + 拆分策略 + 输出契约）、校验器（jsonschema / python_compile / xml / regex）、合并（baseline 去重 + 重编号）与输出格式。schema 全字段 `extra="forbid"`——拼写错误在校验期即报错，不会静默变成默认值。

**五分钟加一个新任务**：复制 `tasks/_example` → 改 `manifest.json`（name/inputs/stages）→ 放模板 → `testagent tasks validate --strict` → `testagent <name> ...`。模板中可选变量用 `{{ x | default('') }}` 守卫；manifest 的 `template_context` 字段可声明样例上下文用于校验期渲染。

**失败语义与恢复**：每个生成单元按封闭七态分类（SUCCESS / EMPTY / INVALID / TIMEOUT / PROVIDER_ERROR / VALIDATION_ERROR / CANCELLED），各自映射恢复策略——空响应串行重问、超时与 provider 错误走重试 + 模型 fallback、校验失败走定向修复；引擎恢复阶梯已耗尽仍空的单元不再重复请求。

**快照（pre_review_snapshot）**：每次任务运行在 review 前原子写入 `<session_id>.pre_review_snapshot.json`（保留最近 `CHECKPOINT_KEEP_LAST` 个）。它是**产物快照**而非断点续跑——用于 review 阶段异常时找回已生成的产物：

```bash
testagent checkpoint list                      # 列出可恢复的快照
testagent checkpoint recover <session_id> --save-as out/recovered.json
```

与旧 Generator 的关系：任务包管道与现有 `generate-tests` / `generate-perf` / `generate-gui` **并存**（旧命令在迁移期始终优先），迁移按 `output/plan-c.md` 渐进执行；`TASKS_DISABLE` 可屏蔽单个任务包作回滚开关。

### 运行生成的脚本

```bash
# k6
k6 run output/perf_test.js

# JMeter
jmeter -n -t output/perf_test.jmx -l results.jtl

# Playwright GUI 测试
pytest output/gui_test.py --browser chromium
```

## 开发

```bash
# 运行测试（487 个用例：端到端流水线 + 多模型 fallback + 多轮 review + 会话引擎 +
# GUI 生成器 + 文档解析器 + 截断恢复引擎 + 任务包管道 E2E + 架构门禁）
pytest

# 代码检查（ruff lint + format）
ruff check testagent/ tests/
ruff format --check testagent/ tests/

# 严格类型检查
mypy testagent/

# 任务包严格校验（CI 模式，任何 manifest/模板问题 exit 1）
testagent tasks validate --strict
```

架构约束（由 `tests/test_pipeline_e2e.py` 的架构门禁测试强制）：`testagent/pipeline/` 不得 import 旧的 `generators` / `prompt_builder` / `conversation`——新管道与旧生成器通过依赖注入容器在组合根装配，防止迁移期反向耦合。

## 部署

### Docker

CI/CD 流水线已自动将镜像推送到 Docker Hub，可直接拉取使用，无需本地构建：

```bash
# 拉取最新镜像（main 分支构建，对应 GitHub Actions latest tag）
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

# 4) 生成 GUI 测试脚本（Playwright）
docker run --rm -v "$PWD":/work -w /work \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    mzdjy/testagent:latest generate-gui \
    -r requirements.md --url https://example.com \
    -o output/gui_test.py

# 5) 启动 Web GUI（可被其他平台通过 iframe 嵌入）
docker run --rm -p 8000:8000 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    mzdjy/testagent:latest serve --host 0.0.0.0 --port 8000
```

#### 本地自建镜像

需要修改镜像内容时也可从源码构建：

```bash
# 构建镜像（在仓库根目录执行）
docker build -f docker/Dockerfile -t testagent .

# 推荐：把当前目录挂为 /work，输入文件与 output 都在宿主侧
docker run --rm -v "$PWD":/work -w /work \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    testagent generate-tests -r requirements.md -s swagger.json \
    -o output/testcases.json

# 仅挂载 output 目录（输入用 URL 或镜像内置文件）
docker run --rm -v "$PWD/output":/app/output \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    testagent generate-tests -s https://example.com/openapi.json -o output/testcases.json

# 增量生成：以历史用例为基线，只补净新用例
docker run --rm -v "$PWD":/work -w /work -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    testagent generate-tests -r new_requirements.md -H output/testcases.json -o output/testcases.json

# 启动 Web GUI（可被其他平台通过 iframe 嵌入）
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

## 扩展路线

- 多 Agent 协调：引擎层可扩展为多 Agent 流水线（分析 -> 生成 -> 评审）
- 性能结果分析：接入 `.jtl` / k6 JSON 结果，产出带指标的完整性能报告
- 会话持久化：将会话状态序列化到磁盘/数据库，支持跨进程恢复

## License

Apache License 2.0，见 [LICENSE](LICENSE)。
