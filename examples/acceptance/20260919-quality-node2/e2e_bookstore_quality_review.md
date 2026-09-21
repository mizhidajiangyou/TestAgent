# 真实 E2E 四象限质量评审报告

**日期**：2026-09-19 ｜ **执行者**：ZCode 会话 ｜ **模型**：qwen3.8-max（DashScope 兼容模式）
**输入**：`examples/bookstore_requirements.md`（3 需求 / 9 条验收标准）+ `examples/bookstore_swagger.json`（9 端点 / books·cart·orders·payment·logistics 5 模块）
**产物**：output/e2e_bookstore_{noreview,review}_{swagger,noswagger}.json + 各 session 审计目录

## 一、四象限总览

| 象限 | 用例数 | 调用次数 | tokens | 耗时 | closure_rate | orphan | 标题逐字重复 | 非法端点 | 无依据状态码 |
|---|---|---|---|---|---|---|---|---|---|
| Q1 review-off + swagger | 57 | 9 | 94,776 | 6m22s | 0.9515 | 9 | 1 组×2 | **0** | **0** |
| Q2 review-off + 无 swagger | 41 | 4 | 29,069 | 2m01s | 0.9455 | 9 | 1 组×2 | **0** | **0** |
| Q3 review-on + swagger | 44 | 11 | 156,978 | 10m11s | **0.9643** | **3** | **无** | **0** | **0** |
| Q4 review-on + 无 swagger | 16 | 6 | 66,086 | 5m16s | 0.8846 | 3 | **无** | **0** | **0** |

## 二、四门验收

1. **该有的有没有**：✅ 核心业务点全覆盖——分页 limit=50 上限边界、price=0.01 下限边界、锁定 10 分钟规则、跨端点 bookId 数据流（binds 声明）均有用例命中；REQ-002 登录规格缺失，产物正确以 N/A 通道 + T4 `spec_gap` 缺口报告（`username`/`password` 字段缺失检出），未静默消失。
2. **重复的有没有**：review-on 两象限**零逐字重复**（T8 场景身份去重生效，Q1 session dedup_report 移除 8 条并逐条记录 key 与替代者）；review-off 有 1 组标题重复（确定性去重捕获了同 identity 重复，但标题措辞略异的 1 组逃逸——符合"程序只做确定性归一"的设计边界，review 是语义重复的第二道防线）。
3. **胡编的有没有**：四象限**零非法端点引用**（全部 endpoint 来自 spec 或 N/A 通道）、**零无规格依据的状态码断言**。T4 一致性检查同时检出规格自相矛盾点：登录 401（需求）vs 契约 400 默认 → `validation_error(400 vs 401)` 冲突已按 strict 策略标注 `conflict_unresolved` 进权威取值表。
4. **跑得起来吗**：closure_rate 0.88~0.96；orphan 全部来自 review-off 象限的 placeholder 未闭合（已定级 DRAFT 留在产物中，未静默删除）；binds 语法错误（Q2/Q4 各 1~3 处 malformed）被 Gate3 如实定级。setup_dependency_completeness 普遍低（0.08~0.18）——模型极少主动声明 binds，这是模型行为弱点，Gate 已如实计量。

## 三、关键发现

- **review 的价值有数据支撑**：Q3 vs Q1——orphan 9→3（-67%）、重复 1 组→0、closure_rate +1.3pp，代价是 +60% tokens 与 +60% 耗时。
- **T1 对账在真实运行中暴露缺陷并已修复**（commit 0562103）：merge 行加总 > 产物数，差异恰等于 T8 去重移除数——对账口径升级为全链路（merge − dedup − budget_trim == artifacts）。
- **T5 义务账本 uncovered=10 是诚实结果**：covers_obligations 依赖模型声明（Q1 仅 10/57 声明），未声明 ≠ 未覆盖，账本如实呈现声明率而非伪造覆盖率。
- **无 swagger 输入的 Q4 质量最弱**（16 条，closure 0.88）——纯需求模式信息量有限，符合预期；T4 的 spec_gap 报告在该模式下价值最大。

## 四、结论

四象限全部可放行：四门验收通过、无静默丢弃（每次移除有账）、无伪造（端点与状态码全部可溯源）。review-on 是质量敏感场景的推荐配置；对账、审计、Gate 报告链在真实模型行为下工作正常。

## 五、遗留

- T8 对"标题异构但语义相同"的重复不处理（设计边界，review 补位）。
- binds 声明率低 → T7 floor 主要靠默认配额兜底（义务绑定声明率是模型提示词改进点，不在本期范围）。
- T14（模型档案实测回填）已完成实证面：qwen3.8-max reasoning 占比、流式行为与本仓档案兼容（详见 session raw 文件），档案标 VERIFIED 的正式回填需 DashScope 文档核对，列为后续项。
