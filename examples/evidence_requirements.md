# 账户与实名认证

系统采用基于 JWT 的鉴权机制。用户注册后须完成实名认证（KYC）方可创建存证。角色分为三级：普通用户（user）、审核员（auditor）、管理员（admin）。

验收标准:
- 邮箱全局唯一，重复注册返回 409，错误详情为 {"field":"email","code":"DUPLICATE"}
- 密码长度 8-64 位，且须同时包含字母与数字；弱密码返回 400 并给出字段级错误
- 登录成功返回 200 与 accessToken、refreshToken（均为 Bearer JWT），令牌有效期 2 小时
- 同一账号连续登录失败 5 次后锁定 10 分钟，锁定期间登录返回 423 Locked
- 实名认证需提交真实姓名、证件类型（枚举 id_card / passport / business_license）、证件号；证件号须通过格式校验，校验失败返回 400
- 已认证用户重复提交认证请求返回 409
- 未实名认证用户调用创建存证接口返回 403

# 存证创建

用户提交待存证内容（上传文件或纯文本），系统自动计算 SHA-256 哈希并生成存证记录，初始状态为 pending（待锚定）。

验收标准:
- 支持两种提交方式：multipart 上传文件（单文件 ≤ 50MB）或 application/json 提交文本（text 字段）
- 存证标题 title 长度 1-200 字符；evidenceType 枚举：document / contract / image / audio / video / other
- 系统自动计算 contentHash（SHA-256，64 位 hex）；同一 contentHash 已存在时返回 409，错误详情携带已有 evidenceId（error.details.evidenceId）
- 创建成功返回 201，响应包含 evidenceId（格式 EVD- 后接 12 位大写字母与数字）、contentHash、status=pending、createdAt
- 未登录返回 401；已登录但未实名认证返回 403

# 存证查询与下载

用户可分页查询本人（及被显式授权）的存证列表，查看详情并下载存证原文。

验收标准:
- 列表支持 page / limit（limit 上限 100，默认 20），可按 status、evidenceType、keyword 筛选，按 createdAt 降序
- 详情接口仅存证所有者或 admin 可访问，越权返回 403；不存在的存证返回 404
- 下载原文接口仅 owner / admin 可访问，返回文件二进制流；越权返回 403；不存在返回 404
- 存证状态枚举：pending / anchored / revoked

# 区块链锚定

系统将存证的 contentHash 批量打包锚定到区块链，锚定完成后状态由 pending 流转为 anchored，并记录链上交易信息。

验收标准:
- 手动锚定仅 owner 可触发，未登录返回 401，非 owner 返回 403
- 已 anchored 的存证重复锚定返回 409（error.details.reason="ALREADY_ANCHORED"）
- 锚定中状态为 anchoring（pending → anchoring → anchored）
- 锚定成功返回 200，响应包含 txHash（0x 开头 66 位 hex）、blockHeight（整数）、chainName、anchoredAt
- 查询锚定状态接口公开可读（供第三方核验），不存在返回 404

# 存证验证

任何人均可验证某内容是否曾被存证、以及某存证的真伪（公开接口，无需登录）。

验收标准:
- 文件/哈希验证 POST /verify/hash：提交文件或 contentHash，返回 matched（bool）、evidenceId、contentHash、status
- contentHash 必须为 64 位 hex 字符串，格式错误返回 400
- 存证真伪验证 GET /verify/{evidenceId}：返回存证元数据、链上锚定信息（txHash / blockHeight / anchoredAt）及服务端当前重算哈希与原始哈希的一致性标记 verified（bool）
- 不存在的 evidenceId 返回 404；存在但哈希重算不一致时返回 200 且 verified=false

# 存证证书（出证）

用户可为已锚定（anchored）的存证申请存证证书（PDF / JSON 报告）。证书包含存证元数据、内容哈希、链上锚定信息、可信时间戳与签发机构。

验收标准:
- 仅 anchored 状态存证可生成证书，pending 状态返回 409（error.details.reason="NOT_ANCHORED"）
- 一个 evidenceId 仅对应一份证书，重复生成返回 200 并直接返回已有证书
- 生成成功返回 201，响应包含 certificateId（格式 CERT- 后接 12 位大写字母与数字）、issuedAt、format（pdf / json）
- 下载证书接口仅 owner / admin 可访问，不存在返回 404

# 审计日志

系统记录所有存证相关操作（创建 / 锚定 / 验证 / 下载 / 出证 / 撤销），管理员可分页审计查询。

验收标准:
- 仅 admin 角色可查询，非 admin 返回 403
- 支持按 action（枚举 create / anchor / verify / download / certify / revoke）、actorId、时间范围（startAt / endAt）筛选
- 列表支持 page / limit（limit 上限 200，默认 50），按 createdAt 降序
- 单条记录包含 logId、action、actorId、evidenceId、ip、createdAt
