# Honey 容器内完整权限与容器外隔离契约

## 目的

DRadar 用 Honey 驱动不同模型完成同一类 benchmark。模型能力评测必须排除 Harness
工具审批、文件权限和子代理开关造成的人为降分。因此，Honey 的统一原则是：

> 模型及其子代理在一次性任务容器内获得完整、无人值守的编码权限；可信边界位于
> Docker、挂载、网络出口和凭证生命周期，而不位于模型可见的工具审批层。

本契约适用于当前七个 Honey：Codex、Claude Code、DeepSeek Harness（DSH）、ZCode、
Kimi Code、Google Antigravity 和 Grok。以后接入任何新 Honey，设计评审、实现、测试和金丝雀都必须逐项
检查本文，不能只证明“CLI 能启动”。

## 容器内必须放行的能力

每个 Honey 必须满足：

1. 使用该 CLI 官方提供的全自动/全权限模式；不得在 headless 运行中等待人工批准，也
   不得把一次工具拒绝静默转换成成功退出。
2. 模型可以读写 `/app`、执行 shell、运行测试、创建和应用补丁、使用临时目录，并使用
   CLI 原生的上下文压缩、后台任务、todo/goal 等本地能力。
3. CLI 支持子代理时，根代理必须能创建、管理和接收子代理结果；子代理继承同一工作区
   权限和无人值守策略。不得仅因适配器白名单而隐藏 Agent、Task、subagent、fork、
   swarm、send/list/interrupt 等原生协作工具。
4. 不设置 Harness 内部的子代理数量上限来替代资源隔离。CPU、内存、进程数、总任务
   并发和额度控制属于容器/调度层；若以后需要资源上限，应在外层统一实现并公开记录。
5. 不加载宿主用户的 Skills、MCP、浏览器会话或个人配置。它们不是编码权限的一部分，
   会引入不可复现能力。CLI 自带且版本固定的本地工具可以使用。

## 容器外必须守住的边界

“容器内完整权限”只有在下面的外层边界全部成立时才允许上线：

1. 每题使用独立、可销毁的 Pier Docker 容器；容器看不到 Docker socket、宿主根目录、
   SSH agent、云凭证目录和其他任务工作区。
2. 只挂载完成题目所需的 `/app`、受控日志/产物目录及该 provider 的最小凭证状态；禁止
   把宿主 HOME 或通用密钥目录整体挂入。
3. 网络默认拒绝，只允许模型 provider、认证刷新及固定运行时下载所需的精确域名。
   Honey 内的 Web、URL、MCP 或 shell 即使获得批准，也不能突破这一出口策略。
4. 任务结束、取消、超时和异常退出都必须销毁容器并清理临时凭证、HOME、运行时配置和
   子代理状态；checkpoint 只保存明确白名单中的会话/工作区状态。
5. 补丁、trajectory、stderr、provider usage 和 checkpoint 在离开容器前必须执行凭证
   扫描与脱敏。无法证明安全的产物失败关闭，不得为了保住分数绕过密钥门。

## 凭证生命周期

Codex 的做法是基准：宿主先校验私有凭证，按题把它放入容器内的 provider 专用临时
目录，模型 CLI 使用隔离 HOME，退出时兜底删除临时凭证和 HOME。其他 Honey 应遵循
同样的生命周期，并在官方客户端协议允许时进一步收窄：

- API key 优先通过 `0600` 临时文件注入；provider 读入内存后立即 unlink，不能放在
  argv、普通环境变量、日志或模型提示里。
- OAuth 客户端若必须在运行期间读取并轮换凭证，可以在容器私有目录中保留到本题结束，
  但不得暴露宿主其他账号状态；轮换结果回写前必须验证结构、权限和所有者。
- Honey 可以在一次性容器内以 root 运行，但任何可写宿主 bind mount 都必须在每次官方
  CLI 命令结束后，以不跟随链接、不过文件系统边界的方式恢复为宿主调用者所有；宿主随后
  读取的日志目录也必须恢复私有权限。清理失败不得把一次成功命令伪装成 provider 失败，
  原始失败也不得被清理异常覆盖。
- OAuth 客户端生成的日志快捷链接等便利文件不得直接放宽凭证树的无链接约束。适配器只能
  在严格校验前按固定路径、不跟随目标地删除已审查的运行时链接；其他路径、特殊文件和
  越界链接仍然失败关闭，并用真实客户端目录结构做回归测试。
- 宿主机与任务/验证容器的代理地址属于不同运行环境。通用适配只读取用户显式配置的容器
  代理接口，不得把某个站点的私有地址、端口、节点或路由固化为其他用户的默认配置。
- 凭证存在于容器期间，安全性依赖一次性容器、最小挂载、出口白名单和产物脱敏；不得再
  用会阻断正常编码工具或子代理的内层审批规则“保护”凭证。

## 当前七个 Honey 的合规映射

| Honey | 容器内模式 | 子代理策略 | 凭证策略 | 外层网络 |
| --- | --- | --- | --- | --- |
| Codex | `--dangerously-bypass-approvals-and-sandbox` | 使用 Codex 原生多代理能力 | 临时 `CODEX_HOME` 与 `auth.json`，退出清理 | OpenAI/ChatGPT 精确域名 |
| Claude Code | `--permission-mode=bypassPermissions` + `--safe-mode` 等价环境开关 | 保留内置 Agent/后台任务；safe mode 只禁用外来定制 | `claude setup-token` 的订阅 OAuth 存于宿主 `0600` 私有文件，仅注入一次性 provider 进程，退出清理 | `api.anthropic.com` |
| DSH | `DSH_PERMISSION_MODE=danger-full-access`（审批 `never`） | 启用原生 spawn/fork/control/report/workflow | key 转成私有 credentials 文档，读入后 unlink | `api.deepseek.com` |
| ZCode | Protocol `mode=yolo`，不传工具 allow/deny 列表 | 保留原生 `Agent` 与任务工具 | key 在 session 建立后、首个模型工具前 unlink | `open.bigmodel.cn`、`zcode.z.ai` |
| Kimi Code | `--auto` | 保留 Agent/AgentSwarm，不设适配器并发上限 | 独立 KIMI_CODE_HOME，OAuth 私有回写，退出清理 | `auth.kimi.com`、`api.kimi.com` |
| Antigravity | `--dangerously-skip-permissions`，不启用内层 terminal sandbox | 权限模式传递到原生子代理 | 独立 `.gemini` OAuth 树，私有权限与运行后复核 | Google OAuth/Antigravity 精确域名 |
| Grok | `--auto-approve`，保留编码工具 | 当前官方 CLI 不提供子代理工具，不做适配器伪造 | 独立 `.grok` OAuth 树，使用官方共享锁，退出复核 | xAI/Grok OAuth 与模型精确域名 |

## 新 Honey 接入门禁

接入 PR 必须在描述中链接本文，并明确列出下列每一项的代码位置、测试和金丝雀证据。
适配器的运行配置版本必须在权限契约变化时递增；服务端应先兼容新旧精确 runtime tuple，
再发布新 CLI，不能让在途旧任务被误拒绝。当前运行还必须上报统一的
`honey_execution_security_profile`、`honey_inner_permission_mode`、
`honey_child_agent_access` 和 `honey_outer_isolation`，使服务端能够验签而不是只相信文档。

官方 CLI 升级同样适用上述顺序：先核对官方来源、精确版本和各平台摘要，再比较命令行、
结构化流、退出码、会话/凭证布局、模型与 effort 映射、usage 账本和 Patch 语义；服务端先
加入新旧 runtime tuple 的并行验签，客户端再发布。若官方包在版本号不变时重打包（例如
桌面应用内嵌 CLI），必须按新摘要重新做协议和金丝雀验证，不能只凭 `--version` 放行。

新适配器合并前必须提供自动化证据和一次真实金丝雀：

- [ ] 固定并校验 CLI 版本/二进制摘要，记录 Harness、provider、模型和 reasoning 档位。
- [ ] 明确官方无人值守全权限开关；初始化事件或协议回读能证明该开关真正生效。
- [ ] 在模型付费请求前验证 `/app` 可读写、shell/测试可执行、临时文件可创建和删除。
- [ ] CLI 支持子代理时，金丝雀至少完成一次子代理创建、工作区读取和结果回传；子代理可用
      工具不得窄于同类根代理工具。
- [ ] 自动化测试证明适配器没有私有 tool allowlist/denylist 隐藏编码或协作工具。
- [ ] Docker 测试证明无 Docker socket、无宿主 HOME/凭证目录、无跨任务工作区挂载。
- [ ] egress 测试证明 provider 域名可达而任意互联网目标不可达。
- [ ] 凭证不出现在 argv、模型环境、日志、patch、trajectory、checkpoint 和上传请求中；
      取消/超时路径也能清理。
- [ ] root 容器写入的 OAuth/日志 bind mount 在正常、非零退出和异常路径都会恢复宿主
      ownership 与私有 mode；回收只覆盖精确白名单根，不遍历 symlink 或其他文件系统。
- [ ] 若官方 CLI 会生成 symlink 或特殊文件，测试证明只清理固定、已审查的便利路径且从不
      跟随目标；普通文件不误删，断链可恢复，其他 symlink 仍失败关闭。
- [ ] 不只看进程退出码：provider 终态、响应、用量账本和工作区 diff 必须一致。取消、错误、
      空响应或空 Patch 不能冒充成功；已发生的 token 仍需保留可审计记录。
- [ ] 真实金丝雀检查 shell、读写、测试、子代理、非空 Patch、判分、token 对账和清理；通过后
      才允许扩并发。

任何一项无法证明时，Honey 不得进入正式榜单批次。回退值、静默工具拒绝或“退出码为 0”
都不能替代证据。
