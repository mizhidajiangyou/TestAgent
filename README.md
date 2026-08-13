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
│   ├── engine/                # AI 引擎层（LLM 客户端、Prompt 构建、会话引擎）
│   ├── generators/            # 生成层（测试用例、性能脚本、GUI 脚本）
│   ├── reports/               # 报告层（测试用例报告、性能报告）
│   ├── utils/                 # 工具函数
│   ├── cli.py                 # CLI 命令
│   └── container.py           # 依赖注入容器
├── templates/                 # Jinja2 Prompt 模板
├── examples/                  # 示例输入文件
├── tests/                     # 单元测试 + 端到端测试
├── docker/                    # Docker 镜像与编排
├── k8s/                       # Kubernetes 部署清单
└── monitoring/                # Prometheus / Grafana 监控配置
```

## 安装

要求 Python >= 3.11。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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
| `AZURE_OPENAI_ENABLED` | 启用 Azure OpenAI | `false` |
| `PERF_BASE_URL` | 性能测试目标地址 | `https://api.example.com` |
| `PERF_VIRTUAL_USERS` | 虚拟用户数 | `100` |
| `PERF_DURATION_SECONDS` | 压测时长（秒） | `300` |
| `OUTPUT_DIR` | 输出目录 | `./output` |
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
# 运行测试（192 个用例，含端到端流水线 + 多模型 fallback + 多轮 review + 会话引擎 + GUI 生成器 + 文档解析器）
pytest

# 代码检查（ruff lint + format）
ruff check testagent/ tests/
ruff format --check testagent/ tests/

# 严格类型检查
mypy testagent/
```

## 部署

### Docker

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run testagent generate-tests -s /app/swagger.json
```

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
