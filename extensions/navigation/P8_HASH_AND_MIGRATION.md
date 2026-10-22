# P8 哈希检查与迁移说明

更新：2026-10-22。当前主线是 LightNav / Navigation-RLT。

## 这三项错误是什么

首次公开导出 `3f02d5d` 中，Feature-v2 A/B 的三个单元测试都先调用同一个
readiness → causal manifest 校验，并在同一个 canonical SHA 检查处失败。
公开导出替换了 readiness 和 Feature-v2 manifest 中各 37 个部署路径字段
（4 个 runtime 路径、1 个 train 路径、32 个场景路径），却仍保留历史实验的
固定身份引用。路径也是 JSON 内容的一部分，改变路径必然改变 canonical SHA。

- **Canonical SHA**：检查 JSON 内容；单纯空白变化不影响它。
- **Raw SHA**：检查文件字节；空白和 CRLF/LF 变化也影响它。
- **Implementation SHA**：检查运行源码；配置替换也会改变源码身份。

进一步核对发现，公开配置替换改变了 `bc_readiness.py` 的字节；另外
`official_adapter.py`、`successor_action.py`、`successor_policy.py` 与 readiness
历史清单的差异在原始提交中就存在。仅恢复机器路径也不能恢复全部历史源码。
这些错误不表示模型权重损坏；相关测试在加载权重或启动 optimizer 前就已停止。

## 已修复的内容

1. 公共协议单元测试改用独立的、固定哈希的
   [测试样例](tests/fixtures/p8_public_contract/README.md)。它们保留协议、预算、
   split、阈值和模型结构，使用明确模拟的候选和 receipt 身份，并绑定当前源码。
2. 测试先检查 profile 和样例的固定哈希，再在测试作用域内注入限定的身份常量；
   不修改生产入口、哈希函数或验证函数。退出测试（包括异常退出）恢复原值。
3. fake-provider 测试现在复制真实源码并调用真实 implementation 校验；原来的
   `verify_implementation` mock 已移除。
4. 负例验证源码变化、协议变化、JSON 空白变化仍被拒绝；生产入口拒绝公共模板
   和测试样例，并且不会调用 provider 或创建结果文件。
5. 补充 Navigation-RLT 源码整理与传输校验工具，解决发布目录和运行目录不同的问题。

历史 `EXPECTED_*`、manifest、receipt/checkpoint 校验保持原值。要精确复现旧 P8，
仍需对应版本的源码、原始 manifest、真实 receipts 和 checkpoint。要开始新的 P8
实验，则必须建立新的 readiness → causal → Feature-v2 证据链。测试样例不是这条
证据链，也不能把旧实验的通过结论直接套用到新的文件上。

## 对当前训练的影响

| 场景 | 结论 |
| --- | --- |
| 已部署 A800/4090 的 LightNav / Navigation-RLT | 本次未修改远端；这些入口不依赖上述 P8 清单 |
| 将主线代码迁移到另一台机器 | 可以整理、传输并校验源码；还需配置机器和运行资产 |
| 直接运行历史 P8 Feature-v2 入口 | 公共模板仍会被历史身份校验拒绝；这是保留的历史复现边界 |

本次没有重新执行远端训练、GPU 渲染或导航 benchmark。

## 主线源码迁移

从 `extensions/navigation` 运行，目标父目录须存在，`code` 目录须不存在：

```bash
python -B -m scripts.stage_navigation_rlt_code stage \
  --destination /path/to/new-run/code
```

工具将本次主线需要的 **45 个源码文件**（26 个 integration/helper 和 19 个
`navrl` 文件）整理为运行器需要的平铺目录，生成 `SOURCE_STAGE.json`，输出
`source_receipt_sha256`。不会复制旧实验结果、模型、P8 清单或历史 receipts。
保存输出的摘要，连同整个 `code` 目录传输，在目标机器上核验：

```bash
python -B -m scripts.stage_navigation_rlt_code verify \
  --destination /path/to/new-run/code \
  --expected-receipt-sha256 <传输前保存的摘要>
```

目标机器需能访问此 fork 的 `extensions/navigation`，才能执行上述 verifier。
校验要求文件集合、大小和 SHA-256 一致；拒绝覆盖现有目录、路径穿越及符号链接。

**源码传输校验后，还需完成实际部署配置：**

- Python/依赖环境、LightNav 源码及原始权重、Habitat 场景与训练数据。
- launcher 中的项目路径、服务地址/端口、模型路径和 GPU 选择。
- storage guard 中真实的 NFS/Btrfs 挂载来源与路径；保留身份及容量校验。
- 配置完成后生成新 run manifest，分别记录实际导航代码和 RLinf runtime 的提交，
  并绑定实际源码、数据及 checkpoint。若配置改动了文件字节，传输前重新生成源码
  包；不要在旧包上修改文件再沿用旧摘要。

`SOURCE_STAGE.json` 只证明这份源码包的身份，不证明依赖、存储、训练或模型效果
已验收。`EXPORT_PROVENANCE.json` 则专门记录 `3f02d5d` 首次导入的历史快照，
后续开发由 Git 提交跟踪，不回写该快照中的旧文件哈希。

## 本次验证

七组针对性测试共 **71 项：70 通过、1 跳过**。其中原 Feature-v2 的 6 项、相邻
causal 的 16 项、readiness 的 15 项均通过；新增边界测试 6 项。跳过项仅因
Windows 当前用户缺少创建真实 symlink 的权限。源码漂移、路径穿越、目录覆盖、
存储保护、代码清单等测试均执行。7 个修改的 Python 文件通过 Ruff 检查和格式检查。
