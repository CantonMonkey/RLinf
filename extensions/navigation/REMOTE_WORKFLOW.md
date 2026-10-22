# 远端开发与实验追踪

`CantonMonkey/RLinf` 的 `navigation-vln` 是共享导航开发分支。`main` 用于跟踪
上游；较大改动可从 `navigation-vln` 建立功能分支，再合回自己的仓库。

## 1. 更新前先检查

- 核对主机身份、运行进程、实际工作目录和 GPU 使用情况。
- 保存旧代码的 HEAD、分支、tracked diff、必要的 untracked 源码及逐文件摘要。
  私有配置和未提交修改先留在主机私有快照，审阅后再决定哪些代码进入公共分支。
- 正在运行的任务继续使用原目录。新建并行 checkout，不在其运行树中切分支、
  reset、覆盖文件或修改共享 Python 环境的 editable 安装指向。

## 2. 源码、部署与实验目录

推荐在项目目录内按用途分开保存：

```text
repos/RLinf-CantonMonkey/       # 干净的 navigation-vln 源码
deployments/<version>/
  runtime/                     # 独立 runtime + 已审阅的主机部署补丁
  navigation-code/             # 本次运行使用的源码包
  DEPLOYMENT.json              # 代码与部署身份
  host-config/                 # 私有配置
provenance/<snapshot>/         # 旧代码补丁、源码清单、实验索引
runs/<run-id>/                 # 日志、manifest、checkpoint 和结果
```

本扩展是 staging 源码；clone 成功不等于 `rlinf/` 已安装导航集成。
部署应根据实际执行路径应用模块映射，并核对 Python 导入来源。
主机配置、私有补丁和实验产物放在公共 checkout 外，勿随源码推送。
runtime 如保留未提交补丁，须记录并展示 dirty 状态，不用 `assume-unchanged`
隐藏差异；也可将部署保存为仅本机持有的提交，同时记录其父级 fork 提交。

## 3. 后续同步

仅在确认此 checkout 未被运行任务使用、工作区干净后执行。以下是远端 Bash 示例：

```bash
cd "<source-checkout>" || exit 1
sync_status=$(git status --porcelain) || exit 1
test -z "$sync_status" || exit 1
git fetch origin || exit 1
git switch navigation-vln || exit 1
git pull --ff-only origin navigation-vln || exit 1
git status --short --branch
git rev-parse HEAD
```

若本地分支尚未建立，首次使用 `git switch --track origin/navigation-vln`。
出现 dirty 或分叉时先审阅、提交或另建快照；不要用 reset/clean 强行同步。
`origin` 应指向自己的 fork；公共上游只用于获取更新。推送指定目标：
`git push origin navigation-vln`。各主机按其已有授权配置访问 GitHub，私钥不随部署包迁移。

## 4. 每个新实验记录什么

将下列身份写入私有 `DEPLOYMENT.json`，由新 run manifest 引用其路径与 SHA-256：

| 内容 | 最少记录 |
| --- | --- |
| fork 源码 | 仓库 URL、分支、完整 commit、是否 dirty |
| 实际 runtime | Git commit、tracked diff 文件及 SHA-256、额外源码文件清单及 SHA-256 |
| 导航源码包 | `SOURCE_STAGE.json` 路径与 SHA-256；实际配置后的源码摘要 |
| 环境 | Python 路径、关键依赖版本、实际导入路径、必要的 GPU 身份 |
| 配置与输入 | 配置、模型/checkpoint、数据 split 的路径及对应版本或摘要 |
| 输出 | 唯一 run ID、输出目录、日志与最终产物/归档索引 |

Navigation-RLT 源码整理命令见 [迁移说明](P8_HASH_AND_MIGRATION.md)。源码包验证后
若还修改了配置或源码，必须重新记录最终字节；旧 receipt 不能代表修改后的包。
仅记录 RLinf HEAD 无法覆盖 dirty runtime；tracked diff 与额外源码清单必须一起保留。

## 5. 最小验证与旧实验

先检查提交、源码摘要、依赖和实际 import 来源，再运行相关 CPU 协议/策略检查。
需要真实仿真或 GPU 时，使用独立输出目录及明确资源分配，做有界 reset/step 或
目标链路检查；确认原进程和服务仍在。会启动 Ray 的配置验证不是纯静态检查。
源码校验、CPU mock、真实渲染和训练更新分别记录，不相互替代。

旧 runs、manifest、receipt 和 checkpoint 保留原始身份，只在实验索引中补充其
代码快照、位置和关联关系。不得把历史结果重新标成新 fork 提交，也不要改写旧哈希。
今后从 fork 创建新的部署版本与 run ID，实验产物继续保存在原有存储/归档系统中。
