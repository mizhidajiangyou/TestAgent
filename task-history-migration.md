# TestAgent 迁移工程 task.md 历史存档（阶段一 ~ 阶段十三）

> 存档说明：原 task.md（项目经验手册 + 阶段一~十三任务记录）于 2026-09-11 被另一会话（mattpocock skill 安装任务）覆盖。本文件为从会话上下文完整恢复的历史记录，供追溯；后续迁移工程状态以本文件末尾"当前思路记录"为准。

以下是整合了近期（2026-08-21 至 08-22）任务产出与经验后的 TestAgent 项目经验手册完整归纳版。原有结构保持不变，新增内容已融入对应章节。

TestAgent 项目经验手册（归纳版 v2026-08-22）

一、项目架构约定

模块分工：llm_client（LLM 调用）/ prompt_builder（模板渲染）/ generators（生成器）/ parsers（解析器）/ container（DI 容器）/ config/models（数据模型）/ engine（会话引擎、ReviewLoop、并发原语）。
容器注册：新服务在 container.py 以懒加载属性注册（如 conversation_manager），不急于实例化。已迁移至 dependency-injector 框架（DeclarativeContainer + providers.Singleton），访问方式为调用式 container.xxx()。CLI 覆盖配置须走生成器公开 setter（如 set_review_enabled），兼容 DI Singleton。
PromptBuilder 通用方法：render_template(name, **ctx) -> str | None，模板缺失时返回 None，由调用方走 _build_inline_* 内联回退。所有模块共用此方法，不要各自维护 templates_dir。
模板文件：放 templates/*.j2，命名规则 <功能>.prompt.j2 / conversation_<动作>.j2。

二、核心设计模式
模式   做法
会话持久化   SessionStore Protocol + InMemoryStore（默认）+ FileStore；运行时依赖不序列化，恢复时重新注入

Artifact 版本链   generate 产出 v1；refine 每轮 version+1，parent_id 指向上一版

双层校验   程序化快检（循环内快速终止）；LLM 深度评审仅用于 validate 动作

动作推断   优先 context["action"]，其次消息关键词；refine 无 artifact 时回退 generate

可选依赖   init 中 try/except 探测并缓存布尔标志；三级回退链；缺库时仍可 import

异步并发   同步核心 + 异步 shim；gather_with_concurrency Semaphore 限流保序；fan-out 后失败单元串行恢复

错误契约   模块级常量 ERROR_CONTRACT 注入 system prompt；禁止模型自造错误码

增量生成   Phase1（需求）+ Phase2（端点）；Phase2 接收覆盖摘要防重复

Review 公共化   泛型 ReviewLoop[T] 放 engine/，通过钩子注入（build_prompt/parse/call_llm）适配不同生成器；TestCaseGenerator 传重型 call_llm，perf/gui 走内建简单重试；支持真异步 arun 双版本

模型档案驱动   frozen dataclass ModelProfile 声明模型差异（参数名/温度/effort/空回判据），业务代码零 if-else；新模型接入=加档案数据+测试

三、高频踩坑与解法
问题   解法
mypy strict + Optional 窄化   用临时变量 existing = ... 再 if existing is not None: 窄化

mypy strict + 可选库   pyproject 用 ignore_missing_imports=true，不要散落 # type: ignore

python-docx 保序   必须遍历 doc.element.body，按标签构造，勿用独立列表

Starlette 响应头   删除用 del response.headers["key"] 且先检查 in

Ruff 中文标点   pyproject per-file-ignores 豁免全角字符

_extract_json 括号陷阱   按「先出现且能解析」优先匹配

测试 LLM 功能   validate 有二次调用需 side_effect 列表；异步路径必须提供 awaitable

私有属性暴露   CLI 覆盖参数暴露公开 setter，不要直接访问私有属性

空截断 ≠ 超限   恢复杠杆是降并发或串行恢复；严禁降低 max_tokens

finish_reason=="stop" 不可盲信   加 parse+未闭合轻校验兜假 stop

stall 区分空返回与全重复   仅非空零增计 stall；空返回/异常走瞬态重试

SearchReplace 锚点风险   插入方法级代码必须锚定「方法完整结束+下一段落开始」边界；插入后立即 Read 回看+跑测，防止截断现有方法

跨测试文件 import   被依赖文件成隐式契约；共享 mock 放 conftest.py 更稳；CI 报 ModuleNotFoundError 先查 git track 状态

UP046 迁移前提   先 grep 确认无外部 import 旧 TypeVar；改完必跑 mypy strict 验证类内签名

聚合层异常透传   MultiModel 等聚合层尾部必须 isinstance 检测并重抛底层类型化异常，禁包 RuntimeError

思考期日志黑洞   reasoning 流必须 INFO 级可见（首 chunk+周期进度+结束统计）；跨线程归属用 ContextVar

空回分类处理   length+空=预算耗尽（短路+降档）；stop/None+空=瞬态（保留重试）；不分类统一重试是无用重试根因

日志模型名探测   聚合型客户端属性名不同，best-effort 日志按候选列表逐个探测（primary_model/model_name）

配置模型同步   数据模型字段与 Settings 字段需成对维护，否则 CLI 用环境变量默认值时 mypy 报错

四、增量生成三层架构

数据层：容错加载历史用例（文件不存在/解析失败 → 空列表）
Prompt 层：历史摘要注入模板，明确指示"只生成净新用例"
合并层：历史全保留 + 新用例按去重键追加
    去重键 = 标题(小写) + 端点(小写) + 测试类型
测试隔离硬规则：每条用例唯一数据、禁止硬编码邮箱、禁止断言全局计数、创建类必须写清理逻辑

五、Python 版本与包管理（uv）

5.1 版本基线
Python ≥ 3.14，不再兼容 3.9/3.11/3.12
包管理器：uv（pip 兼容模式），不引入 uv.lock

5.2 Python 版本升级四个配置点（必同步）
配置点   位置   作用
解释器门槛   pyproject.toml → requires-python   pip/uv 能否安装

Lint 语法目标   pyproject.toml → [tool.ruff] target-version   UP 规则建议的新语法

类型检查目标   pyproject.toml → [tool.mypy] python_version   mypy 检查版本

运行镜像   docker/Dockerfile → FROM python:X-slim   实际运行环境

5.3 uv 本地工作流
uv venv --python 3.14                  # 创建 venv
uv venv --python 3.14 --clear           # 强制清空重建
source .venv/bin/activate
uv pip install -e ".[dev,documents,web]" # 安装

六、Docker / CI 要点

ENTRYPOINT ["testagent"] + CMD ["--help"]
OUTPUT_DIR=/app/output + -v 挂载
镜像标签：latest（仅 main）+ SHA + v* tag；仓库名小写
docker 作业 needs: test
多架构构建三件套：setup-qemu + setup-buildx + build-push-action
GitHub Actions + uv：setup-python 装 Python + setup-uv 装 uv（不传 python-version）
setup-uv 缓存：无 uv.lock 时必须 enable-cache: false 或用 cache-dependency-glob
setup-python cache: pip 与 uv 不兼容，用 uv 时不带 cache: pip
Node 20 deprecated 告警无需处理

七、验证清单（每次任务结束必跑）

.venv/bin/python -m ruff check testagent/ tests/
.venv/bin/python -m ruff format --check testagent/ tests/
.venv/bin/python -m mypy testagent/
.venv/bin/python -m pytest tests/ -q
.venv/bin/testagent --help

当前测试基线：398 passed；既有 ruff 基线 2 个 UP046 已修复归零。

八、LLM 调用关键契约

返回类型：chat()/achat() 返回 str；需元数据时用 chat_with_meta()/achat_with_meta()（返回 LLMResponse）
JSON 模式：OPENAI_JSON_MODE opt-in，输出契约 {"test_cases":[...]}
流式输出：恒 stream=True，每 3s 打进度日志；provider 拒绝时透明回退 blocking
等待可见性：守护线程 + 主线程每 30s still waiting 日志；思考期 reasoning 流 INFO 级可见（首 chunk "is thinking" + 每 30s 进度 + 结束统计）
并发：OPENAI_MAX_CONCURRENCY 默认 5；跨线程日志归属用 ContextVar（CALL_LABEL），to_thread 自动传播
session id：入口生成 uuid4().hex[:12]，日志前缀 []
空截断处理：按 finish_reason + budget_shared 分类：BUDGET_EXHAUSTED 短路抛类型化异常；TRANSIENT_EMPTY 保留 3× 重试
重试策略：保持同一高上限；恢复序列：DOWNGRADE(一次) → SPLIT(shrink_scope) → RAISE_BUDGET(×2 钳制到 profile cap) → FAIL(salvage)
intent_override 五层透传：用能力探测（getattr intent_capable）而非鸭子类型硬传

九、截断处理方案

v6 最终版：设计+逻辑验证完成，生产代码尚未修改
v10 B 阶段（已落地）：ModelProfile 档案机制 + ReasoningBudgetExhaustedError + 五层恢复链 + 思考期日志可见化 + dotenv 注释修复
核心机制：LLMResponse 富对象 / _is_truncated 截断检测 / pending 期望用例数粒度续传 / 配额感知过滤 / 智能折半 / 五道终止保险 / TruncationPolicy dataclass

十、提示词模板变更检查清单

模板字段须与数据模型对齐（TestCase/CSV_COLUMNS/_dict_to_testcase/_testcase_to_dict）
内联回退 _build_inline_* 要同步
硬字段 vs 软引导：不想扩展模型时降级为 Guidance
Jinja2 渲染验证：全变量 + 最小变量各跑一次

十一、关键文件定位速查
文件   职责
testagent/container.py   DI 容器（DeclarativeContainer，含 review_client/performance_generator 注入）

testagent/engine/llm_client.py   LLM 调用（_chat_core + ModelProfile + ContextVar 日志 + 思考期可见化）

testagent/engine/review.py   泛型 ReviewLoop[T] + ReviewResult + 真异步双版本 + 结构化日志

testagent/engine/model_profiles.py   ModelProfile/RequestIntent/classify_response/recovery_plan + 5 内置档案

testagent/generators/testcase_generator.py   生成器核心（委托 ReviewLoop，传 call_llm 钩子）

testagent/prompt_builder.py   Prompt 构建（ERROR_CONTRACT/render_template/build_script_review_prompt/内联兜底）

testagent/engine/session_store.py   会话持久化（FileStore/InMemoryStore）

testagent/engine/concurrency.py   gather_with_concurrency 并发原语

testagent/cli/   CLI 包（含 --review/--ramp-up/--think-time/--auth-type 选项）

testagent/web/app.py   FastAPI Web 层

testagent/config/settings.py   配置（含 OPENAI_REASONING_EFFORT/CONTINUATION_THINKING_BUDGET/MODEL_PROFILE）

testagent/config/models.py   数据模型（TestCase 10 字段 / PerformanceConfig.auth_type）

templates/*.j2   所有 Jinja2 提示词模板（含 script review 模板）


使用建议：改代码前重点看 一（架构约定）+ 二（设计模式）+ 三（踩坑）；涉及 LLM 相关改动看 八（调用契约）；涉及截断/长文本看 九（v6/v10 方案）；涉及 Review 看 二（ReviewLoop 模式）+ 十一（engine/review.py）；提交前跑 七（验证清单）。

---

# 阶段二：提示词工程化优化 + 真实 API 验证（2026-08-23 ~ 08-26）

## 任务清单（全部完成）
- [x] 盘点 8 个模板与消费方契约（10 字段 schema、endpoint 精确匹配、`## Requirements(Context)` 标题依赖、测试断言标记）
- [x] 联网调研提示词工程最佳实践（Contract→Constrain→Check、process framing、正向指令、去冗余防截断）
- [x] 优化模板：
  - `testcase_prompt.j2`：加角色句 + `## Process` 4 步；Rules 重组为 12 条 `## Hard Constraints`（新增 #8 禁自造码兜底、#9 类型保真）；独立 `## Style & Brevity` 段
  - `api_prompt.j2`：加 `## Process`；修正自相矛盾示例（INVALID_EMAIL → error.details.email）；新增类型保真 + 禁造码两条约束
  - `review_prompt.j2`：加 `## Review Process`；checklist 增至 14 条（新增 #13 禁自造错误码、#14 类型保真）；Hard constraints 加「保持需求域覆盖」
  - `conversation_validate.j2`：补齐缺失的 `gui_script` 分支 + 通用 else 兜底（此前 gui_script 校验 checklist 为空）
  - `performance_prompt.j2` / `gui_test_prompt.j2` / `script_review_prompt.j2` / `conversation_refine.j2`：评估后已达标，不动（避免破坏断言）
- [x] 补充 examples/：`ecommerce_requirements.md`（商品/购物车/订单/优惠券 4 域）+ `ecommerce_swagger.json`（13 端点、内联 schema、enum/required 齐全），解析验证 13 端点签名完整
- [x] 真实 API 4 轮运行 + 评审迭代（见下）
- [x] 工程修复 3 项（见下）
- [x] 全量验证：406 passed（398 基线 + 8 新增）、ruff check/format、mypy 全绿

## 真实运行证据链（output/ 下保留全部产物）
| 轮次 | 配置 | 结果 | 评审结论 |
|---|---|---|---|
| v1 `ecommerce_testcases.json` | 24000 tokens | 26 例（review 两轮截断抢救 55→38→26） | 结构 100% 合规，但：自造码 INVALID_STATUS、UUID 当 integer id、非法 enum "books"、幽灵字段 shippingAddress、优惠券域被截断丢失 |
| v2 `ecommerce_testcases_v2.json` | 32768 tokens | **坍缩到 1 例** | 事故：抢救出的 1 例被 ReviewLoop 无条件采纳，替换全部清单 |
| v3 `ecommerce_testcases_v3.json` | +防回退保护 | 97 例（review 两轮被拦截保留原始） | 防回退生效（日志 `shrank 97 -> 26, keeping previous`），但混入 ~33 条退化桩（空 title/expected_results）+ 10 组重复 |
| v4 `ecommerce_testcases_v4.json` | +桩过滤归一化 | **53 例，达标** | 见下 |

### v4 生产标准评审（程序化审计 /tmp/quality_check.py）
- endpoint 不匹配 0 / 10 字段 schema 违规 0 / 缺状态码断言 0 / 模糊断言 0 / 422 出现 0 / 自造错误码 0 / UUID 当 integer id 0 / 精确重复 0
- 端点覆盖 13/13；优惠券域 4 例保留；分布 functional 15 / boundary 18 / security 10 / negative 8 / integration 2
- 抽检：单行可执行 steps（method+path+headers+body）、`Status <code>; field <op> value` 断言、`<TOKEN>/<ORDER_ID>` 占位、唯一数据隔离、创建类用例带 DELETE/cancel 清理
- 残留小瑕疵（不阻塞）：个别用例 payload 用了非法 enum（books/active，且该例是 403 用例 body 不被校验）；个别状态码 200/201 在 swagger 两可。**结论：达到生产标准**

## 工程修复明细
1. **ReviewLoop 防回退保护**（`engine/review.py`）：列表型产物单轮保留率 <50%（MIN_RETENTION_RATIO）判定该轮失败、保留上一轮产物。根因：截断抢救的残片（1 例）曾被无条件采纳替换 30+ 例。脚本型（str）不受影响。新增 4 个单测（含异步镜像、50% 边界）
2. **退化桩过滤归一化**（`testcase_generator._normalize_cases`）：空 title 或空 expected_results 的桩用例在**历史合并前**剔除（用户基线永不误杀）+ 顺序重编号；review 后再跑一次。同时缩小 review 输入缓解截断。新增 4 个单测
3. **输出预算**（`.env`）：OPENAI_MAX_OUTPUT_TOKENS 24000→32768（qwen profile budget_shared，推理+可见共享预算，上限 131072）
4. **Review 输入瘦身**：review 的 endpoints 注入从 `endpoints_to_text` 改为 `endpoints_to_signature`（带类型/enum，恰是类型保真检查项的比对基准，且更短）

## 本任务经验（全局性经验另行写入 experience.md）
- 提示词「禁止自造码」必须配兜底路径：契约未定义的场景（如 409 状态冲突）明确告知模型「只断言状态码并在 description 注明」，否则模型会造码填空
- 类型保真约束要给出可执行基准：把带类型/enum 的 signature 注入 prompt，检查项才不是空话
- LLM 评审「返回完整清单」模式在长列表下必然截断：抢救前缀 + 保留率阈值是当前可行护栏；根治方向是增量式评审（只返回 added/fixed/removed 差量）
- 边界过滤要区分数据来源：过滤规则只作用于 LLM 产出，用户提供的基线数据必须旁路（本任务历史用例因无 expected_results 被误杀过一次）
- 通用校验模板（conversation_validate）新增产物类型时分支必须同步，缺分支=静默空校验

---

# 阶段三：管道引擎 v3 收口路线图（2026-08-28，产出 output/plan-b.md）

- 输入：output/plan.md（v2 实施计划含代码 + 内嵌 24 条评审意见，结论"有条件通过"）
- 要求：结合评审 P0/P1/P2 修订意见 + 当前基线（v10 截断方案、阶段二 ReviewLoop、412 tests）产出分步规划，每任务按必要性/通用性/功能性/可行性打分 1~10
- [x] 读取 plan.md 全文（v2 计划 + 评审）
- [x] 评审 14 项优先级意见（P0×5 / P1×5 / P2×4）→ 任务全量映射对账
- [x] 撰写 plan-b.md：评分体系（N35%+V25%+G25%+F15%）+ 8 步执行序列 + 28 个任务块（含评分/依赖/门禁/回滚）+ DAG + 与 v10 的对接点
- [x] 验证：脚本重算全部综合分与公式一致（28/28，整数算术+half-up）；14 项评审意见全部映射到任务；Top-10 排序与分值一致；8 步骤齐全
- [x] 修订（用户反馈 B6 可行性最低/语义不明）：B6 拆为 **B6a 引擎泛化**（综合 7.2，V 5→6，四步提交表 + GenericHooks 目标协议 + 引擎改动点逐条列明）与 **B6b testcase 迁移**（综合 6.5）两个可独立交付里程碑；补齐 v10 对接细节——Outcome→UnitStatus 四行映射表（含异常路径映射）、`_chat_core` 零新增修改说明（intent_override 通道已存在）、快照帧 v10 动作覆盖；DAG/关键路径/门禁表/里程碑说明同步更新；重跑一致性验证 29 块全过
- [x] 二次修订（用户反馈：UnitStatus 七态与使用不一致 + B6a 硬依赖 B5 不合理 + 需列出待决策项）：
  1. 删除映射表中未定义的 FAILED/SUCCESS_PARTIAL——"部分产出/引擎已恢复"改为 `UnitResult` 元数据字段（partial / engine_recovered）；B3.1 补"封闭七态纪律"（Enum 引用未定义成员 = 运行时 AttributeError / mypy 报错 / 字符串绕过 = 静默死分支）
  2. B6a 硬依赖改为仅 B3.1，B5 降为软依赖；DAG/关键路径改为 B5∥B6a 并行、B6b 汇合
  3. 新增"八、待用户决策项"D1~D4（B5/B6a 顺序、engine_recovered 后是否再 re-ask、快照 golden 更新纪律、引擎产物 dict vs Generic[T]），均附推荐方案
  4. 重跑一致性验证：29 块评分精确一致、映射表仅七态成员、决策节被正文引用
- [x] 定稿版（用户确认按推荐执行）：产出 **output/plan-c.md**（v4）——D1~D4 推荐方案全部固化为已决策记录（B5→B6a 单人串行 / engine_recovered 仍空不 re-ask / 快照差异可解释则接受+记录 / 引擎产物 dict+适配器），决策落点到具体任务块（B5 D1落点、B3.1+B4.6 D2、B6a 门禁 D3、B6a 协议 D4）；Step 5 拆为 5/5b；架构决策扩为 16 条（12 收口 + 4 已决策）；一致性验证 29 块/14 项映射/七态纪律/D 落点全过，无开放决策项
- [x] v4.1 修订（用户发现 B3.4↔B4.4 循环依赖 + B4.5 过大）：
  1. **循环解除**：根因是"构造"与"渲染"混在一个任务——B3.4 收口为 `build_synthetic_context(manifest)` 纯函数（仅依赖 B1.4），B4.3 的 TaskPackage.validate_renderable 单向消费；文档内留修正说明
  2. **B4.5 五拆**：inputs(7.8)/split(7.7)/validators(7.8)/merge(7.6)/writers(7.4) 各自独立 commit，依赖线性 a→b→d、c/e 可并行；每子任务独立门禁（v2 对应测试要点 + 边界负例）与回滚
  3. DAG/关键路径（…B4.5a→b→d→B4.6…）/Top-10（B4.5 移出，B3.2 8.0 递补第 10）同步更新
  4. 验证升级：脚本解析全部 34 个任务的"依赖"声明建图做环检测（首过即抓出解析误报并修正脚本），33 块评分/14 项映射/七态/D 决策全过

---

# 阶段四：plan-c 执行 —— Step 0~3 + B8 完成（2026-08-28）

按 output/plan-c.md 关键路径执行至 B4.11 + B8；**B5（perf/gui 迁移）起的迁移里程碑未启动**，新架构与旧生成器并存（旧命令优先、TASKS_DISABLE 回滚开关齐备）。

## 已交付（对应用户"执行"指令的范围）

### Step 0：B0 环境验证
- jsonschema 4.26.0 安装 + pyproject dependencies（dev 加 types-jsonschema）
- 基线修复：中间会话"提示词优化"把进度日志 INFO→DEBUG 未同步测试 → 测试改为 DEBUG 级断言（is thinking/结束统计保持 INFO）；基线锁定 406→**487 passed**

### Step 1：引擎加固（llm_client.py，18 项新测试 tests/test_llm_client_engine_hardening.py）
- **B1.1** timeout 语义修正：显式 `OPENAI_BLOCKING_HARD_TIMEOUT` 严格生效（< OPENAI_TIMEOUT 构造期报错）；未配置 = max(timeout×2, 600)
- **B1.2** abandoned worker 资源模型：worker 自报 ident 的登记表（完成自动注销）+ `worker_metrics()` 三指标（active/abandoned/oldest_age）+ 并发超时残留测试（N 并发 → 恰 N 残留）+ 熔断（超阈值跳过重试直接上抛给 fallback）
- **B2.1** 流式降级去粘滞：streak 只计 transport 白名单（APIConnectionError/APITimeoutError/httpx.ReadError/RemoteProtocolError/ReadTimeout），编程 bug 原样上抛；BadRequestError 仅 message 含 "stream" 才永久禁用；3 次连续瞬态失败才禁用；成功重置 streak
- **B2.2** LLMCallTimeoutError：轮询看门狗硬超时 + MAX_TIMEOUT_RETRIES_PER_CALL=1 重试预算 + v10 边界单测（超时 ≠ BUDGET_EXHAUSTED、不触发降档）

### Step 2+3：pipeline 包（13 个新模块 + tasks/_example + 5 个新测试文件 81 项测试）
- **B1.4/B4.2** manifest.py：StrictModel(extra="forbid") 全模型；manifest_version 白名单；name/alias 校验（正则/唯一/≠name/保留命令黑名单）；split.requires input、choice.requires choices 等跨字段校验
- **B1.3** parity.py：STRICT（全字段并集+保序，默认）/ SEMANTIC（白名单+排序，显式选择；多字段必报 diff）/ TEXT 三模式
- **B1.5** 数据流声明：v1 = fan-out + merge（schema docstring 固化）
- **B3.1** status.py：封闭七态 Enum + RECOVERY_POLICY 全覆盖 + UnitResult（partial/engine_recovered 元数据）+ Outcome/异常映射（D2 固化：engine_recovered 仍空不 re-ask）
- **B3.2/B4.5a** inputs.py：INPUT_PARSERS 注册表（7 kind）+ parse_inputs + from_settings 点路径解析 + require_any
- **B4.5b** split.py：single/per_input/batch 三策略 + evaluate_when
- **B4.5c** validators.py：jsonschema Draft202012 逐条校验（错误串带字段路径喂 re-ask）+ 四类 text 校验器（python_compile/xml/regex/contains）+ when 守卫
- **B4.5d** merge.py：baseline 前置 + dedup（keys+normalize）+ renumber
- **B4.5e** writers.py：json/csv/markdown/text 四格式 + resolve_extension（fixed/format 键控/by_input 键控——v2 伪代码补全为可运行实现）+ meta.json 副产物
- **B3.4** synthetic.py：build_synthetic_context 纯函数（template_context > kind 样例 > 管道派生变量）
- **B4.3/B3.5** registry.py：DirectorySource（绝对路径解析，防 chdir 失效）+ Registry 四类冲突构造期检测 + TaskPackage（StrictUndefined + synthetic 渲染校验 + 引用存在性）
- **B4.6** executor.py：PipelineExecutor（generate_unit 注入式）+ gather 并发 + UnitStatus 分派恢复（plain-EMPTY 串行 re-ask / recovered-EMPTY 计失败 / 全空不恢复）+ pre_review_snapshot 原子写 + CHECKPOINT_KEEP_LAST 修剪 + list/recover
- **B4.7** clicommand.py：动态 CLI（legacy 优先冲突策略 + hidden 任务注册 + TASKS_DISABLE）+ cli/__init__ 注册接线
- **B4.4/B4.8** tasks.py / checkpoint.py 命令（SystemExit 传播——click 8.4 ctx.exit 在 standalone=False 下丢码）
- **B4.10** fingerprint.py：RequestFingerprint（model+双 prompt sha+params+label）+ diff
- **B4.9** E2E golden path：`_example` 真跑全链（CLI 注册→解析→渲染→fake LLM→schema 校验→merge→snapshot→writer）+ checkpoint list/recover + validate --strict 正反例
- **B4.11** arch gate：AST 级 import 扫描（pipeline 禁 import generators/prompt_builder/conversation）+ 新层 legacy seam 检查
- container.py：task_registry / pipeline_executor providers（generate_unit 经组合根注入，满足 arch gate）
- settings.py：blocking_hard_timeout + tasks_dir 字段

### B8 配置同步 + 文档
- .env.example：OPENAI_BLOCKING_HARD_TIMEOUT / TASKS_DIR / TASKS_DISABLE / CHECKPOINT_KEEP_LAST
- pyproject：jsonschema 运行依赖 + types-jsonschema dev 依赖
- docker/Dockerfile：COPY tasks/（挂载 /app/tasks 可扩展任务包）
- README：项目结构（pipeline/tasks）、配置表 8 个新变量、任务包管道章节（五分钟新任务教程/失败语义/快照说明/与旧 Generator 并存关系）、开发章节（487 用例 + 架构门禁说明）

## 验证（全绿）
- pytest **487 passed**（406 基线 + 81 新增：engine_hardening 18 / pipeline_units 29 / pipeline_flow 27 / pipeline_e2e 7）
- ruff check + format 全过；mypy strict 0 错误（62 文件）；`testagent tasks validate --strict` ✓；CLI --help 正常
- E2E 证明：`_example --text hello` 经真实 CLI 注册→fake LLM→产物 [{"text","id":"TC-001"}] 落盘 + 快照可 recover

## 执行中的问题与修正（4 个）
1. click 8.4.2 `ctx.exit(code)` 在 CliRunner standalone_mode=False 下 exit_code 恒 0 → 改 `raise SystemExit`
2. dependency-injector provider 捕获 import 时函数引用，monkeypatch 模块属性无效 → 类级 `provider.override` + 测试后 reset
3. 相对路径 TASKS_DIR 在测试 chdir 后失效 → DirectorySource 构造期 resolve()
4. 测试间日志污染（CLI E2E 的 setup_logging 给 `testagent` logger setLevel INFO）→ caplog.at_level 显式指定 logger 名

## 后续（按 plan-c 关键路径继续）
- B5：tasks/perf + tasks/gui 任务包迁移（等价矩阵 format×review×空响应 + 双门槛 parity + 删旧 generator）
- B6a：TruncationEngine 四步泛化（快照冻结→数据钩子→scope→续写；GenericHooks dict 协议）
- B6b/B7：testcase 迁移 → web/chat 指针切换 → grep 门禁删除
- 迁移期回滚开关全部就绪：TASKS_DISABLE / 旧命令优先 / 独立 commit 粒度

---

# 阶段五：B5~B7 迁移计划编写（2026-09-04，待评审）

用户要求：为未启动的 B5~B7 单独写迁移计划，评审后继续执行。

- [x] 代码调研（3 个 Explore agent + 直接核读）：三个旧生成器机制差异（testcase 走 TruncationEngine 扇出；perf/gui 单发不走引擎）、_EngineHooks/arun 精确签名、ReviewLoop 位于 engine/review.py（pipeline 可 import 不违反 arch gate）、web /api/generate 直调 agenerate、conversation import PromptBuilder + DEFAULT_TARGET_URL（跨层耦合）、6 个生成类模板清单
- [x] 发现 6 个前置缺口：G1 pipeline review stage 未实现（ReviewSpec 声明未消费，B5 硬前置）；G2 runtime 不走截断引擎（TruncationSpec 未消费，B6b 前置）；G3 conversation 反向依赖 gui 生成器常量；G4 旧 session record 无等价物；G5 web 直调；G6 prompt_builder 残余职能
- [x] 产出 **output/plan-d.md**（迁移专案，延续 plan→b→c 谱系不覆盖 v2 存档）：17 任务块（B5.0~B5.4 / B6a-0..3 / B6b.1~4 / B7.1~3 / B8'）各含四维评分（8.0~5.7）、文件级改动点、依赖/门禁/回滚；新增待决策 D5~D8（命令命名 / session record 去留 / web 契约范围 / 引擎化时机，均附推荐）；关键路径与 M1~M4 里程碑；**关键顺序修正：B6b.4（删 testcase 旧实现）必须晚于 B7.1/B7.2（web/conversation 切换）**；等价纪律 = 模板逐字节对拍 + 双门槛（parity+fingerprint）+ B6a 快照重放（D3）
- [x] 一致性验证通过：17 块评分公式精确一致、依赖图无环、G/D 引用完备、引擎签名为调研实录、里程碑/门禁/DoD 表齐备

## 本任务经验（规则 4）
1. 写迁移计划前先做逐文件接口调研（Explore agent 并行 + 关键文件亲读）——计划里的签名/行号必须是实录而非记忆，本次 G1/G2 两个"声明未消费"的缺口只有读实现才能发现，直接决定任务排序（B5.0 前置）
2. 跨层隐藏耦合是删除步的地雷：conversation import gui 生成器的 DEFAULT_TARGET_URL 这类"常量借用"在 grep 生成器类名时才暴露——删除任务的 grep 门禁要含 import 级引用
3. 删除顺序受消费者约束：TestCaseGenerator 有 web/conversation 两个存活消费者，其删除（B6b.4）必须排在指针切换（B7.1/2）之后，DAG 要显式表达该约束而非只按编号顺序

## 待用户评审
- D5~D8 四项决策（plan-d.md §二，均附推荐）
- 任务粒度与等价容差点登记方式（B6b.3）

---

# 阶段六：plan-d v1 评审意见吸收 → v2（2026-09-04）

用户评审结论：8.2/10 **条件通过**——不直接开工 B5.0，先收紧四处（Review 契约 / B6a 接口 / Failure Semantics Parity / session metadata）。

- [x] 17 条意见（7 必须 + 3 建议 + 矩阵维度 + 硬原则 + DAG 删除门独立）全量吸收，落点对账表置于 plan-d.md §零
- [x] 四处契约收紧（新增 §三"迁移契约"章，先于代码存在）：
  1. **R1/R2 Review 契约**：生命周期状态机（GENERATED→…→REVIEWING→终态），review 是 Post-process 不进 stages；三态语义 REVIEWED / REVIEW_REJECTED（retention 守卫拒收，保留 original）/ REVIEW_FAILED（异常或不可解析，回退快照内容）——FAILED≠REJECTED 语义区分进 parity 断言
  2. **R3/R4/R5 引擎接口**：EngineContext 收敛续写回调（五散参→单 ctx）；GenericHooks 收敛为 5 核心能力（extract/salvage/scope/dedup/continue），**convert 强制移出引擎**（引擎只产 list[dict]，适配器在外）；EngineSnapshotFrame 改名 EngineEvent（事件日志冻结行为而非实现）
  3. **I1 三门槛**：Artifact + Fingerprint + **Failure Semantics Parity**（八失败形态：empty/timeout/provider error/validation error/engine recovered/review rejected/review failure/cancelled，各断言 UnitStatus+恢复次数+最终 artifact）
  4. **D6 改向**：删 --resume 的 recovery 职责，保留 session metadata 重定义为审计记录（B6b.3 落地）
- [x] 结构调整：B6b 四拆→五拆（B6b.1 runtime / B6b.2 adapter+H1 合同 / B6b.3 task 包+metadata / B6b.4 parity / B6b.5 delete，排障边界清晰）；新增 B5.2a 共享常量脱钩（DEFAULT_TARGET_URL 先行迁出，解 conversation 反向依赖）；B5.4 与 B6b.5 两个删除门各自独立；B5.3 矩阵补输入形态维度（perf 3 形态 / gui 4 组合）；F1 Migration Freeze（B5.0 合入起旧实现只许 regression fix）；F2 "只迁移不增强"升格硬原则；H1 historical merge 六维度合同（以旧实现为准在 B6b.2 固化）；D9 prompt parity 分层（fingerprint=硬门禁，normalized diff=观察）
- [x] 任务块 17→19（+B5.2a，B6b 4→5），评分重算全部公式精确
- [x] 一致性验证通过：19 块评分精确、7+3 映射完备、R1~R5/I1/F1/F2/H1/D6/D9 可核查断言、依赖图无环、B6b.5 晚于 B7.1/2 约束在任务块与 DAG 双处

## 本任务经验（规则 4）
1. 评审驱动的计划修订要先做"意见→落点对账表"再动正文——10 条意见与章节的映射使第二轮评审可逐条核销，避免口头吸收
2. 契约先于代码：Review 三态、引擎 5 能力这类语义设计放进计划独立章节（§三迁移契约），任务块只引用不重复定义——单一真相源，后续实现与 parity 测试都从这章取验收标准
3. "逐字节对拍"这类过强等价声明会把自己逼进死角（空白/换行必然差异）——门禁分层的正确切口是：语义层做观察、指纹层做硬门禁（fingerprint 本来就规整掉了无关差异）

---

# 阶段七：plan-d v2 二轮评审吸收 → v3（2026-09-04）

用户二轮评审：8.0/10 条件通过（四维 9.2/8.0/8.7/7.2；按权重公式复算为 8.5，不影响结论）——可开工 B5.0/B5.2a/B6a-0，但 B5.3/B6b.1/B6b.4 前须固化 6 项 P0 契约。已向用户说明两点：① P0-3"golden 来自 refactor 前行为"的操作化——事件发射点随 B6a-0 引入，改造前轨迹以 test_truncation_recovery 15 项断言为锚，golden 采用"deterministic + correctness 交叉吻合"双重校验；② 评分复算 8.5≠8.1 小勘误。

- [x] 二轮 13 条意见全吸收（P0×6 + P1×6 + 关键路径写法），对账表更新至 §零两轮合璧
- [x] 六项 P0 契约收紧：
  1. **I1 重写**：Failure Semantics Parity = 可观察语义合同（失败分类描述+恢复次数+恢复路径+最终 artifact），UnitStatus 仅为新架构规范化表示；三层映射表（Legacy 可观察行为→迁移合同→Pipeline 表示）
  2. **R6 ReviewHooks[T]**：build_prompt/parse/apply/retention_check 四责任协议先行，hooks 不含 call_llm（ReviewLoop 内置）
  3. **R5 增强**：round=本批次第 N 次 LLM 请求序号（1 起，recovery 不递增）；事件三类别 request/recovery/terminal；golden 双重校验（correctness 断言交叉吻合为 baseline 锚）；四轨迹覆盖
  4. **I2 fingerprint 白名单**：model+双 sha+params+logical_label；禁 session id/时间戳；CALL_LABEL 新旧格式差异的归一化规则在 B6b.3 定死
  5. **D6.1**：session metadata 的 artifact/snapshot 一律 reference{path,sha256}，必含 session_schema_version=1
  6. **Contract Parity 第四门槛**：web 契约以 fixture/schema golden 证明（HTTP status/response schema/error code/error body），B7.1 切换前录制旧路由实况
- [x] 六项 P1：H1 合同区删"预计=旧优先"（改为 B6b.2 代码核对决定）；B6b.4 矩阵三层分层（A 主等价矩阵 / B 独立 failure 套件 8 case / C 组合压力 3 case，不做笛卡尔积）；B5.0+B5.3 加 max_rounds=1/>1 生命周期两路径；删除门两层（Symbol Gate: grep+AST import 扫描 / Behavior Gate: 全量）；永久 migration fixtures（tests/fixtures/migration/ 五子目录，legacy 删除后 parity 载体从实现转移为 fixture，B8' 常驻回归）；B6a-1 引用改"§三 R5 规则"（D3 编号漂移修正，D3 语义保留给 plan-c 快照政策）
- [x] 关键路径改写：Perf/GUI 线与 Testcase 线两条汇合路径；显式声明 B5.* 与 B6a.* 可完全并行、B5.0 是 B6b.3 唯一跨线前置、B5.4 不阻塞 B6b.*
- [x] 一致性验证 12 项全过（评分精确/映射完备/各契约可核查断言/依赖图无环）；修复 Write 工具引入的下划线转义污染（104 处 `\_`→`_`，首修因文件写入未持久重跑确认清零）

## 本任务经验（规则 4）
1. "parity 断言新系统状态与旧一致"是范畴错误——旧系统没有新系统的内部类型；正确形态是"可观察语义合同"三层映射（观察描述→语义合同→规范化表示），契约里必须写明比较对象是 observable 而非 internal
2. 行为冻结类 golden 的锚点问题：观测机制本身是新增的，"改造前轨迹"无法直接录制——用既有 correctness 断言交叉吻合作为人工审定 baseline 的可执行形式；"两次运行一致"只证 deterministic，必须叠加第二重校验
3. 迁移期比较机制 ≠ 永久回归：旧实现删除后等价性即失证——在删除前把 legacy 实况固化为 fixture 文件（input/fingerprint/artifact/failure semantics/events），等价载体从"活代码"转移为"死数据"，这一步决定迁移工程的长期价值
4. 长文写入工具会做 markdown 转义（`_`→`\_`），代码符号名被污染后 grep/断言静默失效——写后应跑一次 `\_` 扫描；且首次修复后要"写后读回"确认持久化（本次遇到首修未持久的异常，同脚本内 read-back 验证才可靠）

---

# 阶段八：plan-e 定稿 + 首批三任务执行（2026-09-04）

三轮评审 8.6/10：批准 B5.0/B5.2a/B6a-0 立即开工，条件为 4 项修订定稿（P0-1 response trace / P0-2 具体动作合同 / P0-3 真实 legacy 轨迹基线 / P1-4 契约归一化 + R6 apply 澄清）。

- [x] **plan-e.md**（执行基线，不膨胀——只载 E1~E5 修订与执行决策，任务/DAG/门禁沿用 plan-d v3）：
  - E1 fixture 六要素（input / request_trace[fingerprint+response] / artifact / failure_semantics / events），重放语义=response trace 作 deterministic fake 脚本，固定 Request+Provider+Expected 三元组
  - E2 timeout/provider error 具体动作合同表（分层次数+顺序；timeout 三层：client timeout-retry≤1 → 模型切换 → executor 串行×1；downgrade/split 禁止——与 plan-c B2.2 冻结语义一致；"禁止单独使用 fallback 宽泛词"）
  - E3 EngineEvent 基线三步程序：legacy 既有日志+fake 计数为观测源录制 → observer 落同批分支点（全量测试证零控制流变化）→ 日志推导序列与 observer 可映射子集逐项吻合+correctness 交叉（correctness≠event-order proof 的补洞）
  - E4 Contract Parity 归一化白名单（request_id/trace_id/timestamp/session_id/path 可归一；status/code/schema/message/业务字段禁归一）
  - E5 apply(original, reviewed)->accepted_artifact——ReviewLoop 不负责 merge，apply 是领域侧唯一替换/合并点，防双重合并

- [x] **B5.2a 常量脱钩**：DEFAULT_TARGET_URL 迁至 testagent/config/constants.py（值冻结注释）；conversation.py 改引新位置；gui 生成器 re-import 保持模块级引用兼容（B5.4 删除时随删）。86 conversation+gui 测试过

- [x] **B6a-0 EngineEvent 冻结**（truncation.py 纯观测改造）：
  - EngineEvent（8 字段 frozen dataclass，event 三类别 request/recovery/terminal，round=本批第 N 次请求序号、recovery 不递增——为此把 calls+=1 移到 call 发射前，行为中性）+ TruncationEngine 可选 observer 参数（默认 None 零开销）
  - 发射点：call（请求前）/ continue/reask（prompt 选择）/ budget_exhausted/downgrade/split/raise_budget（恢复阶梯）/ salvage（截断吸收+终局抢救）/ done/fail（全部终止路径）
  - **E3 证据链落地**（tests/test_engine_event_baseline.py，16 项）：① determinism（双跑逐事件一致）② **log cross-check**——14 条 legacy 分支日志签名（downgrading/splitting/raising/recovery exhausted/coverage complete/call cap/wall time/empty streak/3 transient/scope floor/cannot parse/empty truncation/reasoning budget exhausted）映射动作序列，与 observer 可映射子集逐项相等 ③ correctness 交叉（downgrade 恰一次 sticky、split 后 scope 3→1、caps 16000→16000→32000、terminal 恰一次且在末尾）
  - golden 四轨迹提交（tests/fixtures/migration/engine_events/*.json，含 provenance 与覆盖说明）；TESTAGENT_RECORD_GOLDEN=1 显式重录
  - 零行为变化证明：correctness 15 项 + 全量 520 过

- [x] **B5.0 Review Runtime**（R1/R2/R6/E5 全落地）：
  - pipeline/review_hooks.py：ReviewHooks[T] 四责任协议（build_prompt/parse/apply/retention_check）+ make_list_hooks/make_text_hooks 工厂（apply 均恒等替换；list retention=MIN_RETENTION_RATIO 镜像、text 豁免；empty original 不拒）
  - pipeline/runtime.py：build_review_runner——复用 engine/review.py ReviewLoop 同一实现（轮次交替/每轮内置重试/全败回退 original，机制零漂移）；parse 包装器跟踪 parse_ok 实现 REJECTED/FAILED 分类（parse 成功但未被采纳=retention 拒收，断言 hooks.retention_check 一致性）；REVIEWED 经 hooks.apply 收尾；异常→REVIEW_FAILED 回退快照内容
  - executor：可选 review_runner 注入，snapshot 后执行，终态进 review_meta（status/rounds/reason）
  - manifest ReviewSpec.system_prompt 新字段（file:/inline: 约定，保持 review 请求指纹全任务可控）+ synthetic 加 artifact/round 派生默认（review 模板可过 strict 校验）
  - _example：prompts/review.j2 + manifest 声明（enabled from_settings）
  - container：pipeline_executor 注入 review_runner（组合根装配，arch gate 不破）
  - 17 项测试：协议四责任/三态（REVIEWED/REJECTED 保 original/FAILED 回快照[异常+不可解析两源]/DISABLED）/text 豁免/多轮生命周期（round1 changed→round2 以 refined 为 current；max_rounds=1/>1）/奇 review 偶 primary 交替/声明 system prompt/executor e2e（review 爆炸→artifact==快照）
  - **顺带修复 B4 遗留 bug**：registry.system_prompt 不剥 "inline:" 前缀（_example 的 system prompt 一直在泄漏前缀）

- [x] 问题与修正（3 个）：① 测试 fake 主客户端双重计数（委托链上两处 append）→ 计数只在子客户端；② test_engine_event_baseline 误加 load_dotenv() 把用户本地 .env（OUTPUT_LANGUAGE=english）泄入测试环境致 test_defaults 失败 → 移除（os.getenv 不需要）；③ round 语义：calls 自增在 prompt 选择前致 recovery 事件 round=N+1 → 移到 call 发射前（行为中性）

## ⚠️ Migration Freeze 生效声明（F1，自本记录起）

generators/ 三实现（testcase/performance/gui 生成器）+ prompt_builder 生成类函数 + 6 个生成类 j2 进入 FREEZE：仅许 P0/P1 regression fix（目标=回本基线）；禁止功能修改/prompt 文案/参数/模板改动。违规后果=parity 基线失效，相关 golden/fixtures 必须重录并在 commit 说明。truncation.py（引擎本体）不在此列——B6a 泛化即其改造计划。

## 验证（全绿）
- pytest **520 passed**（487 基线 + 33 新增：engine_event_baseline 16 / pipeline_review 17）
- ruff check + format 全过；mypy strict 0 错误（64 文件）；`tasks validate --strict` ✓（含新 review 模板渲染）
- golden 证据样例（downgrade_split.json）：call→budget_exhausted→downgrade→continue→call→budget_exhausted→split[scope 3→1, pending 6→2]→continue→call→done，round 语义与恢复后状态快照均正确

## 下一步（按 plan-d v3 + plan-e 关键路径）
- B5.1 tasks/perf → B5.2 tasks/gui（fingerprint 对拍=硬门禁，I2 白名单）→ B5.3 矩阵+fixtures（E1 schema 含 response trace）→ B5.4 删除
- B6a-1 数据钩子泛化（golden 重放 diff=0 门禁）

## 本任务经验（规则 4）
1. 观测埋点与控制流的相对次序决定事件语义：`calls+=1` 的位置让"recovery 事件携带上一请求序号还是下一请求序号" silently 漂移——为观测新增的状态读取要显式声明读取时机（发射点+自增点的相对顺序），并用 golden 测试钉死
2. 测试模块顶层的 load_dotenv() 是环境污染物：它把开发者本地 .env 注入整个 pytest 会话的 os.environ，破坏"默认值"类断言且只在模块导入顺序变化后显形——测试读环境变量用 os.getenv 即可，永远不要在测试里 load_dotenv
3. 双客户端 fake（primary/secondary）的脚本索引是独立的：轮次交替测试里"第 N 轮返回什么"取决于该角色客户端自己的调用计数，不是全局计数——构造脚本时要按角色分别排剧

---

# 阶段十：B5.1 tasks/perf 迁移完成（2026-09-04）

按 plan-d v3 B5.1 + plan-e 执行。门禁全绿：**fingerprint 对拍零差异**（I2 白名单）+ `tasks validate --strict` + fake-LLM run-through + 全量 550 passed + ruff/mypy 零错误。

## 交付内容

### B5.1a 管道代码 parity 修复（6 处，均为等价性必需）
1. **inputs.py `_endpoints_signature`**：body 属性补 enum（对齐 legacy `_format_param` 字节级）——此前管道本地实现漏了 body enum，perf review context 渲染会指纹漂移；重构为共享 `_format_param`（params/body 同一格式化）
2. **registry.py `system_prompt(spec, context=None)`**：file: 系统提示词用运行上下文渲染（format/output_language 方差必须到达 system prompt）+ strip()（文件尾换行不进指纹）
3. **manifest.py**：ValidatorSpec 增 `declaration`/`closing`（对齐旧 `_validate_jmx` 四检查边界：声明前缀/闭合后缀/解析/根标签）；ReviewSpec 增 `context_template`（review 上下文块由任务包模板拥有——perf 负载配置形 vs gui 目标 URL 形）；template_context 放宽为 dict[str, Any]（int 算术合法）；template_refs 补 review.system_prompt/context_template 存在性+渲染检查（B5.0 遗漏）
4. **validators.py**：xml 分支按 declaration→closing→parse→root 顺序（镜像旧 `_validate_jmx` 检查次序，pass/fail 边界一致）
5. **review_hooks.py**：`make_text_hooks(render, validate=None)`——parse 内跑 manifest 校验器（镜像旧 perf review parse 的 jmx 守卫：坏评审答案永不清白首轮脚本）；review 四终态常量移驻协议模块（runtime 再导出保持 B5.0 导入路径）
6. **executor.py**：text artifact 归约（merge 后 `[{"script":...}]` → 脚本串，legacy parity：text 任务产物是一个脚本不是条目列表）；REVIEW_DISABLED 不写 meta（legacy 只在 review 真跑过时写 `.meta.json`）；runtime：空 text artifact 跳过 review（legacy `if review_enabled and script`）+ review system prompt 带上下文渲染 + `_review_context_value`（context_template 渲染一次注入 context[2]，rstrip 防文件尾换行入指纹）

### B5.1b tasks/perf 任务包（5 文件）
- **manifest.json**：inputs = swagger（必需，require_any 强制）+ script_format（choice k6/jmeter，from_settings:script_format，`--script-format` 避让动态 CLI 自带的 `--format` 输出格式选项）+ 6 perf 参数（from_settings:perf.*，CLI 选项名对齐旧命令 --base-url/--virtual-users/--duration/--ramp-up/--think-time/--auth-type）；单 stage contract=text；xml(root/declaration/closing) when script_format=jmeter；extension by_input 键控 {.js/.jmx}；fan_out_recover=**false**（对齐旧单发语义：空响应不 re-ask）；merge.renumber.field=""（text 条目不编号）；review.context=["script","round","context_text"]
- **prompts/performance.j2**：旧 performance_prompt.j2 语义等价迁移（仅变量重命名：`endpoints`→`endpoints_text`、`config.X`→`X`；**反向重命名 diff 字节级验证一致**）
- **prompts/script_review.j2**：旧 script_review_prompt.j2 迁移（仅 `script_kind`→`script_format`；反向重命名 diff 验证一致）
- **prompts/performance_system.j2 / script_review_system.j2**：新建——旧 Python 组装的系统提示词模板化（format 分支 + chinese 语言提示拼接，渲染结果与旧逐字节一致，已冒烟验证 4×2 组合）
- **prompts/review_context.j2**：旧 `context_text` 组装形（## Load Configuration 6 行 + ## Endpoint Signatures），渲染字节级等于旧 f-string 拼接

### B5.1c 测试（tests/test_perf_task.py，30 项）
- **指纹对拍矩阵 16 格**：format(k6/jmeter) × review(off/2轮) × language(english/chinese) × 响应形态(plain/fenced)——每格断言 (system,user) 请求序列 + 客户端角色序列（gen=primary，奇轮=secondary，偶轮=primary）+ 请求计数 + 最终 artifact 全等
- **对拍补强**：from_settings 默认解析路径（raw 只给 swagger）、非默认参数全覆盖（CLI override → prompt 到达断言）、空响应 k6 对拍（1 请求/空 artifact/不烧 review 轮）、FingerprintLog 机制诚实性（sha=捕获对、label="perf:script"）
- **模板纪律**：反向重命名 diff ×2（冻结期任何一侧漂移即红）+ strict 校验
- **run-through**：k6/jmeter happy、extension 键控（.js/.jmx）、meta 副产物、review 三态（REVIEWED/含轮内重试恢复/REVIEW_FAILED 快照回退）、jmx 候选守卫拒收、invalid 生成=INVALID 单元、空生成跳过 review、chinese 系统提示词注入

## ⚠️ 已登记等价性差异（B5.3 failure 套件逐项裁决：文档化接受或修复）
| # | 差异 | legacy 可观察 | pipeline 可观察 |
|---|---|---|---|
| D1 | jmeter 无效生成 | `_validate_jmx` raise ValueError（CLI 报错） | INVALID 单元 + 空 artifact + units_failed=1 |
| D2 | 无效生成的 review 修复路径 | review 看到无效脚本（可能修复） | 生成即 INVALID 丢弃，review 不见（E2 "validators 喂 review" 定向修复未实现） |
| D3 | meta.json | `perf_test.meta.json`，{generator,script_format,reviewed,rounds_*} | `perf_test.js.meta.json`，{status,rounds_*}（路径+形状双差异） |
| D4 | timeout/provider error | 异常上抛（CLI 报错） | 失败单元计数 + 空 artifact（fan_out_recover=false 已对齐空响应语义） |
| D5 | 空 swagger 边缘 | context_text 尾随 "\n"（header 后） | rstrip 去掉（退化路径，1 字节） |
| D6 | 性能报告 | CLI 存盘后生成 perf_test.md（reports 层） | 任务命令不生成（报告是 CLI 级步骤，B5.4/B7 处置） |
| D7 | --review/--no-review CLI 旗标 | 每次调用可覆盖 | 仅 REVIEW_ENABLED 设置级（B5.4/B7 CLI parity） |
| D8 | CALL_LABEL 格式 | ""（生成）/ "{format}-script"（review 日志） | "perf:script" / "perf:review"（不泄入请求，已断言；B5.3 fixture 按 I2 归一化） |
| D9 | review 开启 + max_rounds=0 | meta 写出（reviewed=false） | REVIEW_DISABLED 统一不写 meta（退化配置） |

## 验证证据
- pytest **550 passed**（520 基线 + 30 新增 test_perf_task）
- `tasks validate --strict` ✓（_example + perf）；`testagent perf --help` 选项面齐；与 legacy `generate-perf` 并存（冲突策略：legacy 优先，B5.4 删除）
- ruff check/format 全过；mypy strict 0 错误（64 文件）
- 对拍硬门禁证据：矩阵 16 格 `pipe_fake.pairs == legacy_fake.pairs` 全绿（含 chinese 系统提示词、fenced 响应、双轮 review 交替）

## 本任务经验（规则 4）
1. "模板逐字节对拍"的正确落地形态：模板文件本身做**反向重命名 diff**（把新模板的变量名换回旧名后与冻结模板 diff 为零）——既保住"语义等价迁移"的纪律，又允许管道变量命名约定（endpoints_text vs endpoints、flat vs config.X）；比"逐字节复制模板"更诚实，因为复制反而会在渲染期因变量形态不同而漂移
2. 指纹对拍的比较边界选在 **fake LLM 的调用面**（(system,user) 序列）而非 FingerprintLog——legacy 侧没有指纹机制，且 label/params 在两侧是表示层差异；I2 白名单的字段在调用面上天然全等。FingerprintLog 只在新侧做机制诚实性校验
3. 动态 CLI 有保留选项（--format/-f 输出格式）：任务输入名撞保留名时 click 同名选项后者静默覆盖前者——输入名必须让位（script_format + `--script-format`），键控机制用 by_input 而非 format 键控
4. 系统提示词的方差（format/language）是静态 manifest 字段覆盖不了的——file: 系统提示词必须能拿运行上下文渲染；这次是 perf 的 format+language 二维，B6b 的 testcase 还有 json_mode 维度，机制已通用化
5. text 产物要过三层形态：单元层 `{"script":...}` 条目（复用 merge 机器）→ 管道层归约为字符串（snapshot/review/write 都消费字符串）→ 写层 extension 键控；漏归约则 review 会把 list 误判为 JSON 产物走 retention 守卫

---

# 阶段九：用例质量修复方案 testcase-fix-plan.md（2026-09-04，进行中）

## 输入
- output/用例质量排查与解决方案-20260904.md（前次排查方案：R1~R9 根因 + S1~S11 解法）
- 用户提供的对该方案的评审（8.3/10，四个核心修订：① Coverage 以 AC/obligation 为一等公民而非 AC×endpoint 笛卡尔矩阵；② 去重不从自然语言猜意图，scenario identity 入 schema 一等字段；③ 冲突不硬编码优先级，改为检测+策略化裁决；④ 补 executability——placeholder 闭合与跨用例依赖）
- output/plan-links-v7.md / plan-links-v8.md（跨模块关联上下文保真方案，用户要求整合进新方案）

## 要求
- 结合评审 + 实际证据，产出独立新方案 output/testcase-fix-plan.md
- 方案自包含：不引用其他方案的字段/编号，证据直接写进方案
- 允许运行 testagent 获取实际证据；不直接改代码

## 待办
- [x] 提取 v7/v8 关键概念（状态机/关系类型/信任级/binds 契约/三道 gate/指标）——2 个 search agent 并行，结构化提取
- [x] 逐条核实排查文档的代码级断言（R1~R9 行号）——全部属实（truncation.py:92/156/191/219/523、testcase_generator.py:419/666/769/1106/1192、prompt_builder.py:34/122/170、swagger_parser.py:110、settings.py:167、model_profiles.py:156）
- [x] 程序化审计 output/testcases.json（32 条）：可移除重复 22、端点 8/8/8/8、登录 0 条、401/409 越权、漏 name 3/3、4 种占位符风格、limit=1000 脑补
- [x] 真跑 testagent（会话 69f4234c8fde，29 条/51,467 tokens）：全部结构性缺陷复现；新增证据=TC-011≡TC-019 标题逐字相同的跨需求重复未去重（去重键完全相同）+ 丢弃零日志在复跑中继续不可见
- [x] 撰写 output/testcase-fix-plan.md（10 章：证据基线/根因链/四大数据契约/数据流/15 任务评分/实施顺序与决策项/验收/风险/即时缓解/附录）
- [x] 自验证：15 任务评分按权重公式全部重算一致；跨方案引用扫描仅剩头部溯源（非规范引用）；无占位符；无转义污染；TC 归属推断补注推导依据
- [x] 更新 task.md + 经验总结

## 产出
- **output/testcase-fix-plan.md**（待用户评审；决策项 D1~D3 需确认，D1=落地点与迁移冻结关系为关键门禁）
- 证据留存：output/testcases_rerun_20260904.json、output/rerun_evidence_20260904.log、output/sessions/69f4234c8fde.json

## 本任务经验（规则 4）
1. 评审驱动的整改方案，四个修订要落成"数据契约"而不是"行为补丁"：覆盖主体换成义务一等公民、去重键换成 schema 一等字段、冲突换成分类+策略、可执行性换成门禁——契约定了，任务分解与验收标准都是推论
2. "数量从 32 降到 16~22"这类结果指标不能当验收标准（删掉 14 条重要场景也能达标）——验收要写"该有的有没有（义务覆盖）、重复的有没有（场景键残留 0）、胡编的有没有（无依据断言 0）、跑得起来吗（孤儿 0）"，数量降为观察项
3. 复跑对照是区分"偶发/结构"缺陷的最便宜手段：两次运行数量不同（32/29）但缺陷集合完全一致，即证明结构根因；且复跑本身就是新证据源（本次抓到标题逐字相同的跨需求重复——比基线更有说服力的去重真空证据）
4. 写涉及既有代码的方案前，方案里引用的每一处行号都要重新 grep 核实（本次全部命中是因为排查文档当天写的，但这是运气不是常态）；且要检查项目自身约束（本次发现迁移冻结覆盖用例生成器与模板，直接产生落地点决策项 D1——方案若不处理项目约束，执行时必然撞墙）
5. 长方案（139KB）提取概念：按标题 grep 建行号索引后分区读取，抓"枚举值/字段名/失败分类常量"三类信息即可在不搬代码的情况下保住契约细节

---

# 阶段十一：B6a-1 数据钩子泛化完成（2026-09-04，恢复中断会话）

上次执行 plan-e 至 B6a-1 时被意外打断。恢复后盘点：实现主体已在中断期间提交（`83bff0e 评审优化`），遗留 4 处 ruff 问题（3 个测试文件 import 排序 + 1 个未用导入）与 2 处 format 未格式化，门禁未做最终验证。本次收尾：修复遗留 + 全量门禁验证。

## 交付内容（对照 plan-d v3 B6a-1 任务块）
- **_EngineHooks → GenericHooks**（§三 R4）：engine/truncation.py 的 `GenericHooks` 收敛为 5 核心能力 + re-ask 构建器（extract / salvage / scope_key / dedup_key / build_continue_context + build_reask）；**convert 移出引擎**——引擎只产 `list[dict]`，`_to_test_cases` 不再是引擎钩子
- **引擎内部 dict 化**：produced/covered/_merge/_absorb/_salvage_and_return 全部操作 `dict[str, Any]`；非 dict 条目在合并期丢弃（旧转换器行为等价）；无 scope 声明条目归属 primary scope（旧转换器 fallback 首端点的等价规则）
- **TestCaseGenerator 引擎外适配器**：模块级 `_item_scope_key` / `_engine_dedup_key`（dedup 键与旧 `title|endpoint.full_path|test_type.value` 字节级等价，test_type 经枚举归一化）；`arun`/`run` 返回后由生成器调用 `_to_test_cases` 转换
- **产出 pipeline/truncation_hooks.py**：`dict_scope_key` / `dict_dedup_key`（resolved scope 折叠进 endpoint 槽）/ `generic_reask`（与冻结旧 `_build_reask_prompt` 字节级一致，测试锁双向漂移）/ `make_dict_hooks` 工厂（extract/salvage 强制宿主注入——不附带第二个解析器）；B6b.1 从 manifest TruncationSpec 接线
- **引擎域无关性提前钉死**：AST 门禁（零 TestCase/APIEndpoint import 与注解）作为测试常驻（B6a-3 门禁提前满足）

## 门禁证据（plan-d v3 B6a-1 DoD 逐条）
1. **EngineEvent golden 重放 diff=0**：4 条轨迹（normal/salvage/downgrade_split/budget_exhausted）golden 录制于 14:04（B6a-0，provenance 字段内嵌），晚于录制时间的 B6a-1 改动（16:25 提交）下重放逐事件相等——非重录自证
2. **correctness 15 项**：test_truncation_recovery 全过（降档恰一次 sticky / split 后 raise budget 钳制 / 终局 salvage 合并 / slim continue 等）
3. **全量绿**：561 passed（550 基线 + 11 新增 test_truncation_hooks：键等价 3 + reask 等价 2 + dict hooks 3 + 引擎 e2e 2 + 域无关 1）
4. **mypy strict 0 错误**（65 文件）；ruff check/format 全过（含本次修复的 4 处遗留）；`tasks validate --strict` ✓；CLI --help ✓

## 本任务经验（规则 4）
1. 恢复中断任务先重建状态再动手：`git status` + `git log` 区分"已提交/未提交/未跟踪"，本次发现主体已在打断期间提交（83bff0e），剩余只有 lint 残留与未验证门禁——直接重写实现会撞车；判定"完成度"的可靠信号是测试结果而非文件存在
2. "重放 diff=0"类门禁必须核对 golden 的录制时机早于被测改动（mtime + provenance 字段双证）；改动后重录的 golden 是自我证明，没有门禁效力（R5 规则：可解释差异才允许重录并逐事件解释）
3. 等价迁移的钩子泛化，等价性断言要落在"键函数字节级"上：新适配器键 vs 旧转换器产物的键逐字节对拍（含 test_type 枚举归一化、无 scope 归属规则两个边角），这是 golden 重放有意义的前提——键不等价则事件序列相同也可能掩盖产物漂移

---

# 阶段十二：B6a-2 scope 泛化完成（2026-09-04）

按 plan-d v3 B6a-2 + plan-e 执行：`arun(endpoints)→arun(scope_items)` + `scope_item_key` 钩子；shrink/filter/expected 全部改经 scope key；TestCaseGenerator 适配（scope_key=ep.full_path 等价）。引擎对 scope 条目零属性访问，域无关性再进一步。

## 交付内容
- **引擎**（testagent/engine/truncation.py）：
  - `GenericHooks` 新增 `scope_item_key: Callable[[Any], str]`——scope 条目对引擎不透明，键经钩子派生，引擎不再读 `.full_path`
  - `arun`/`run` 参数 `endpoints` → `scope_items`；`scope`/`batch_set`/`expected` 全部由钩子键派生；`shrink_scope` 本就在键上操作（无需改）
  - `filter_to_scope(items, batch_set, expected, scope_key)` 第四参注入**声明键**函数（保留"无声明 → 绕过过滤"分支语义；与覆盖/去重用的**解析键**分离）；`_absorb`/`_salvage_and_return` 传 `hooks.scope_key`
  - `build_continue_context` 原样透传 scope 条目（宿主自渲染），签名不变
  - **提示词文本一字未动**（`build_continue_prompt` 的 "endpoint xcount" 属请求指纹面，冻结）
- **宿主适配**（testcase_generator.py）：模块级 `_scope_item_key(ep) = ep.full_path`（与旧引擎自读完全等价）注入钩子
- **工厂**（pipeline/truncation_hooks.py）：`make_dict_hooks` 增 `scope_item_key` 参数；缺省解析器支持 str（自身即键）/dict（读 scope_field），未知类型抛 TypeError 要求显式注入（B6b.1 显式接线）
- **测试**（3 新增 + 2 适配）：适配器键等价（`_scope_item_key == ep.full_path`）/ 显式钩子驱动对象型 scope 条目 / 缺省解析器拒未知类型；`TestEngineDomainFreedom` 加 `assert "full_path" not in src`（B6a-2 域无关钉死）；`filter_to_scope` 测试改注入 `dict_scope_key`

## 门禁证据（同 B6a-1，R5 规则）
1. **EngineEvent golden 重放 diff=0**：4 轨迹对 B6a-0 golden（14:04 录制，provenance 内嵌）逐事件相等——scope 泛化零行为变化
2. **correctness 15 项**：test_truncation_recovery 全过
3. **全量绿**：**564 passed**（561 + 3 新增）；mypy strict 0 错误（65 文件）；ruff check/format 全过；`tasks validate --strict` ✓；CLI ✓
4. **域无关**：引擎源码零 `.full_path`（AST/文本双断言常驻）；残留 "endpoint" 仅为注释与冻结提示词文本

## 本任务经验（规则 4）
1. "属性读取→钩子"泛化必须区分**声明键**与**解析键**两条通道：过滤走声明键（保住"无声明条目绕过配额过滤"分支），覆盖统计/去重走解析键（无声明回退 primary scope）——混用会让 scope-less 条目被配额误杀，且这类行为差异不会体现在事件序列里，只体现在产物上
2. 冻结面包括提示词文本：引擎泛化时 `build_continue_prompt` 里 "endpoint" 这类措辞是 LLM 请求内容（指纹面），不能顺手"去域化"——域无关的正确断言对象是类型引用与属性读取，不是文本字面名词
3. 克隆式测试（从宿主引擎复制 hooks 重建 observer 引擎）对钩子增字段天然免疫；但工厂缺省值若是"未知类型报错"，直连工厂的测试要显式把 scope 条目换成支持形态（str/dict），否则泛化本身没错、测试先红

---

# 阶段十三：B6a-3 续写与 re-ask 收敛完成（2026-09-04）

按 plan-d v3 B6a-3 执行：`build_continue_context(五散参)→(EngineContext)`；build_reask 产物名词参数化（默认字节级不变）。B6a 四步泛化至此收官，引擎对域类型/属性/文案三层解耦完成。

## 交付内容
- **EngineContext**（testagent/engine/truncation.py）：frozen dataclass 五字段（user_prompt/scope_items/fingerprint/label/pending）收敛续写回调；**pending 为构造时点快照**（`dict(pending)` 拷贝——循环内 `recompute_covered_pending` 原地改活跃 dict，frozen 上下文若持活引用会在钩子暂存后被就地篡改，测试实抓）；scope_items 保持不透明透传（B6a-2 语义）
- **GenericHooks.build_continue_context** 类型收敛为 `Callable[[EngineContext], str] | None`；引擎续写分支单 ctx 调用；`__all__` 导出 EngineContext
- **生成器适配**：`_build_continue_context(self, ctx)` 单参签名，行为不变（summary 提取失败回退 build_continue_prompt 分支保留）
- **re-ask 名词参数化**（pipeline/truncation_hooks.py）：`generic_reask` 增 keyword-only 四参 product/item/items/compactness_hint；**默认值字节级重建冻结文案**（TestGenericReaskEquivalence 对 `_build_reask_prompt` 三 error_type 全等断言常驻）；non_parseable 分支本无产物名词，天然通用
- **测试**（3 新增）：名词参数化替换断言（含 "test cases" 不残留）/ EngineContext 五字段到达 + 引擎以钩子输出发起下一请求（(system,user) 对拍）+ pending 快照语义 / 无钩子时内建 full-prompt 续写不变

## 门禁证据（R5 规则 + B6a-3 附加门）
1. **EngineEvent golden 重放 diff=0**：4 轨迹对 B6a-0 golden（mtime 14:04 未变，未被重录）逐事件相等
2. **correctness 15 项**全过；**truncation.py 对 TestCase/APIEndpoint 零直接引用**（grep+AST 双断言常驻，含 full_path）
3. **全量绿**：**567 passed**（564 + 3 新增）；mypy strict 0 错误（65 文件）；ruff check/format 全过；`tasks validate --strict` ✓；CLI ✓

## 执行中的问题与修正（2 个）
1. **编辑静默回滚**：3 处已报告成功的编辑（EngineContext 类、引擎调用点、生成器 import）在磁盘上丢失（疑 IDE 缓冲竞争），靠测试收集期 ImportError + grep 全面读回才发现——多文件编辑后必须逐点 grep 读回验证，再跑测试
2. **frozen ctx 持活 dict**：首版 EngineContext.pending 直传引擎活跃字典，第 2 轮吸收后原地变更，暂存断言抓到 {POST:1}≠{POST:2}——修正为构造时点 `dict(pending)` 快照并在字段注释与测试双侧钉死

## 本任务经验（规则 4）
1. "frozen dataclass = 不可变"是错觉：frozen 只禁属性重绑定，容器字段仍持活引用——凡是"构造后可能被外部循环就地修改"的 dict/list 字段，frozen 上下文必须在构造点拷贝快照，否则暂存该上下文的消费者读到的是未来状态
2. 文案参数化的字节级等价做法：新参数全部 keyword-only + 默认值精确重建冻结文案，既有"对旧实现全等"测试常驻为回归门——泛化与冻结在同一函数上共存，不需要复制两份文案
3. 编辑工具报告成功≠落盘：与 IDE 同开文件竞争时可能静默回滚；关键结构（新类/新 import/调用点）编辑后用 grep 一次性核验全部锚点再进入测试阶段，比逐个撞 ImportError 便宜

---

## 当前思路记录（存档时点：2026-09-11，v11 写作前）
- **B6a 四步全部完成**（B6a-0 观测冻结 / B6a-1 数据钩子 / B6a-2 scope 泛化 / B6a-3 续写收敛），golden 自 B6a-0 录制后从未重录，四步重放均 diff=0
- plan-e 关键路径下一步：**B6b.1 engine-backed runtime**（依赖 B6a-0..3 已齐）——runtime 加 build_engine_generate_unit，TruncationSpec 从 manifest 消费，Outcome→UnitStatus 接出；门禁：fake LLM 真引擎三轨迹单测 + golden 重放一致
- B5 线并行待启动：B5.2 tasks/gui → B5.3 矩阵+fixtures → B5.4 删除
- 注意：tests/fixtures/（含 engine_events golden）与 checkpoint/tasks CLI 命令文件仍为未跟踪状态；B6a-1/2/3 改动均未提交（工作区 diff：truncation.py/testcase_generator.py/truncation_hooks.py + 4 测试文件）
- testcase-fix-plan.md 用户已裁决：D1=A 修后再迁（用例生成链解除冻结，修后重录迁移 fixtures）、D2=strict、D3=保留并标 out_of_spec；决策已固化进方案 §6.2，方案进入实施评审
- 跨模块关联方案：plan-links-v10 + 其评审（有条件通过，P0×2 + P1×2 必修）→ 已产出干净的自包含 **output/plan-links-v11.md**（2026-09-11）

---

## 2026-09-13：Phase 2「长时间卡住」诊断与修复（kimi-k2.7-code 未注册档案）

**现象**：`generate-tests` 跑到 Phase 2（2 个 API 批次并发）后，日志停在 `is thinking` 约 5 分钟无任何输出，用户 Ctrl+C 中断且进程未干净退出（第二个 ^C 才死，抛 `threading._shutdown` 里 `ThreadPoolExecutor._python_exit` 的 join 栈）。

**结论：不是死锁**，是「超长思考 + 日志全静默 + 流式无墙钟上限」三重叠加。

**证据链**
1. Phase 1 的 reasoning 量级已异常：32,713 / 45,714 / 103,123 字符（耗时 83 / 113 / 239s，均 ~400 chars/s 稳定流），对比 09-04 qwen 基线（output/904.log）有上限时每次仅 12.8k~14.3k 字符。
2. **根因**：`kimi-k2.7-code` / `glm-5.2` 不匹配任何档案 matcher → 落 `generic-openai-compatible`（`effort_translation={}`）→ **`.env` 的 `OPENAI_REASONING_EFFORT=low` 被丢弃，请求里一个思考参数都不带**。探针实测：kimi/glm `thinking_params={}`，qwen3.8 `{'extra_body': {'thinking_budget': 4096}}`，deepseek `{'reasoning_effort': 'low'}`。
3. **观测黑洞**：thinking 周期进度是 `logger.debug`（默认 INFO 不可见，且 08-21 的修复承诺是 INFO——手册写了 INFO、代码是 DEBUG，属漂移）；`still waiting` 看门狗在 chunk 持续到达时被抑制（reasoning 稳定流 → 永不触发）。探针：2.1s 纯思考期 INFO 只有 3 行，6 行进度藏在 DEBUG。
4. **无墙钟上限**：blocking 有 `blocking_hard_timeout`，streaming 没有任何 deadline；httpx read timeout 是逐次读、对持续到达的 chunk 不生效。探针：0.2s timeout 下流跑满 2.2s 也不中断。
5. **次生放大**：`generic` 档案 `budget_shared=False` → `classify_response` 永不返回 BUDGET_EXHAUSTED（v10 短路失效）→ 空回烧满 3 次相同请求（每次重跑同样数分钟思考）；探针 sdk_calls=3（qwen/deepseek 均为 1）。
6. 退出不干净：executor worker 线程非 daemon，atexit join 卡住。

**修复**（难度评分 3.5/10 ＜6，已实施）
- `stream_hard_timeout`（settings + OpenAIClient + `_stream_completion` 每 chunk 校验）默认 `max(timeout*2,600)`，超限抛 `LLMCallTimeoutError` 进既有有界重试路径，并 close 流释放连接。
- thinking 周期进度 DEBUG→**INFO**（含 elapsed 与 reasoning 字符数），修回手册承诺的可见性。
- `OPENAI_EXTRA_BODY_JSON` 逃生舱：未注册模型的思考控制由操作者填厂商文档原文，不猜方言；JSON 非对象在启动即 ValueError。
- 未注册档案空回快失败：按带内证据（空 + length + 观察到 reasoning）抛 `LLMOutputTooLongError`，不依赖档案标志位；`stop/None` 空回仍走瞬态重试（守卫测试锁死）。

**验证**：新增 16 项测试（streaming 上限 5 / 未注册模型守卫 4 / extra_body 解析 4 / 空回快失败 3 含 2 项防回归守卫）；全量 **582 passed**（基线 566 + 16）；ruff/mypy 全绿；`output/verify_phase2_hang_fix.py` **9/9 契约检查通过**（修复前四项全 RED）。既有失败 1 项（`test_truncation_hooks.py::test_continue_hook_receives_full_context`，改动前即失败，与本次无关）。

**未做（评分 ≥6 或独立故障面）**
- 给 kimi/glm 新增档案：**7/10**，需厂商文档核实参数方言，猜错会 400；改用逃生舱。
- Ctrl+C 干净退出：**5/10**，属独立故障面（退出路径，非卡死本体），未并入本次 diff。
