# 离线多候选语音增强与转写终端

Raspberry Pi 5 上的本地语音演示工程。

同一段 INPUT1 语音分别经过 **raw、GTCRN、FastEnhancer-B** 三条路径，由独立 ASR 实例转写，网页展示候选文字、处理耗时和胜出原因。使用者可以纠正最近一句，保存按语种隔离的精确纠错记忆；日语后端支持动态热词。

说话中显示可修正的临时文字，停顿后把最终句子追加到对话框。“清空显示”只清浏览器显示，后台转写和纠错记忆保留；刷新后不会立即回填已清空的旧句。实现与运行成本说明见[逐步显示](docs/REALTIME_PREVIEW.md)。

## 源代码入口

| 模块 | 源文件 | 做什么 |
|---|---|---|
| 主程序与网页 | [`app/voice_demo_multicandidate.py`](app/voice_demo_multicandidate.py) | 统一分段、三路并行、HTTP 状态与纠错回调 |
| 采集与基础设施 | [`app/voice_demo.py`](app/voice_demo.py) | ALSA INPUT1、VAD、模型加载 |
| 候选选择 | [`app/candidate_selection.py`](app/candidate_selection.py) | 字面多数、文本共识、波形安全与耗时择优 |
| 人工纠错 | [`app/personalization.py`](app/personalization.py) | 按语种持久化纠错与热词 |
| 识别后端 | [`app/asr_backends.py`](app/asr_backends.py) | SenseVoice / 日语 ReazonSpeech 路由 |
| 语言状态 | [`app/language_selection.py`](app/language_selection.py) | 语言配置验证、保存与读取 |
| 增强适配 | [`app/fastenhancer_onnx.py`](app/fastenhancer_onnx.py) | 有状态 ONNX 处理、句间状态复位与对齐 |
| 设备基准 | [`app/benchmark_multicandidate.py`](app/benchmark_multicandidate.py) | 一个已有 WAV 的三路离线处理 |

```text
UMC202HD INPUT1 → ALSA → 原始输入统一 VAD 分段
                           ├─ raw ───────────────→ ASR ─┐
                           ├─ GTCRN ─────────────→ ASR ─┤
                           └─ FastEnhancer-B ────→ ASR ─┘
                             候选选择 → 本地网页 → 人工纠错记忆
```

初始归档保留已验部署版本。2026-10-31增加前缀预览和保留式对话框，复用原权重，没有重训；最终结果仍经三路处理与候选选择。

## 获取与本地检查

```bash
git clone https://github.com/youtai123-bit/edge-voice-demo-public.git
cd edge-voice-demo-public
python scripts/check_sources.py
python scripts/test_logic.py
```

公开源码可直接克隆。上述两条检查只用 Python 标准库，不加载模型、不打开声卡、不联网。`test_logic.py` 覆盖 19 项选择、语言、纠错和后端文件合同测试；不等于完整设备或神经推理验证。

## 在已准备好的 Pi 上运行

已验证环境：Pi5 / aarch64 / 约 4GB，Python **3.13.5**，NumPy **2.2.4**，ONNX Runtime **1.28.0**，sherpa-onnx **1.13.5**。完整准备步骤见 [部署说明](docs/DEPLOYMENT.md)。

```bash
# 仅检查资产存在与已记录大小；不载入权重、不进行大文件 hash。
python scripts/verify_assets.py --models "$HOME/voice_demo/models"

# 仅供没有其他服务占用声卡/8080 的已准备设备，人工启动。
DEVICE_ROOT="$HOME/voice_demo" bash scripts/run_demo_manual.sh
```

既有服务已启动时，直接使用设备网页，不重复运行启动脚本。网页用于显示与控制，音频来自 Pi 声卡 INPUT1；程序不打开扬声器播放设备。

**仓库不包含模型权重、第三方 wheel、私人录音、历史转写或个人热词。** 需要的文件名与已观察的大小见 [`MODEL_ASSET_MANIFEST.json`](MODEL_ASSET_MANIFEST.json)。新设备必须先准备模型、tokens、兼容依赖与 ALSA 工具，不能只克隆源码就视为完成部署。

## 已有设备验证

| 指标 | 原始回执中的结果 |
|---|---:|
| 测试素材 | 1 个英文 WAV，3.943 秒 |
| 三路 warm 并行 wall（原版，无预览） | 1.150 秒 |
| RTF | 0.2916 |
| 整个进程（含初始化） | 9.545 秒 |
| 基准进程峰值 RSS | 965.2 MiB |
| 三路文字 | 一致，最终选择 raw |

证据与限制见 [测试说明](docs/TESTING.md)、[设备报告](evidence/DEVICE_VALIDATION_REPORT.md)及[脱敏 JSON](evidence/ONE_EXISTING_FIXTURE_DEVICE_VALIDATION.json)。这些是 **n=1 英文素材的吞吐结果**；没有据此声称准确率、增强音质、声学端到端延迟或硬实时提升。共识分也不是校准后的 ASR 置信度。

Pi 墙钟在记录时未同步，报告使用电脑 UTC；单调时钟区间测时单列。旧研究质量门未通过，结果保留于 [`RESEARCH_NEGATIVE_RESULT.md`](evidence/RESEARCH_NEGATIVE_RESULT.md)，不能混作 Pi 成绩。

## 文档与归属

- [架构与代码地图](docs/ARCHITECTURE.md)
- [部署说明](docs/DEPLOYMENT.md)
- [测试范围与复现口径](docs/TESTING.md)
- [人机协作与归档说明](docs/DEVELOPMENT_RECORD.md)
- [上游来源与许可说明](RIGHTS_AND_ATTRIBUTION.md)
- [源码身份清单](SOURCE_PROVENANCE.json)

GTCRN、FastEnhancer、Silero VAD、SenseVoice、ReazonSpeech 与 sherpa-onnx 是上游组件。本项目定位为终端集成、候选选择和人工纠错流程，不把这些基础模型称为自研模型。AI 助手参与代码、排错与材料整理。

本仓库从 2026-10-29 的真实源码归档创建 Git 历史，没有伪造此前的逐次开发提交。项目源码尚未指定新的开源许可证，第三方组件保留各自归属与条款。

设备记录中的身份和地址已脱敏，变换范围见[证据脱敏说明](evidence/REDACTION_NOTICE.md)。
