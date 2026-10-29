# 来源与使用范围
本仓库由 GitHub 项目维护者管理。
本仓库是既有部署源码的归档快照，不伪造历史提交。AI参与代码与材料整理，详见随包开发记录。
本次不替作者新增开源许可证；参赛提交、后续公开发布的授权范围由参赛者确认。
原始增强与ASR模型不是本队原创训练。模型权重、系统依赖和真实个人转写未打包。

| 组件 | 来源 | 本工程作用 |
|---|---|---|
| GTCRN | https://github.com/Xiaobin-Rong/gtcrn | 既有增强分支 |
| FastEnhancer | https://github.com/aask1357/fastenhancer | 既有增强分支；MIT全文见FASTENHANCER_LICENSE.txt |
| SenseVoice | https://github.com/QwenAudio/SenseVoice | 非日语识别后端，当前使用sherpa-onnx导出 |
| ReazonSpeech | https://huggingface.co/reazon-research/reazonspeech-k2-v2 | 日语识别模型，官方模型卡标注Apache-2.0 |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx | 推理接口及模型导出分发 |
| Silero VAD | https://github.com/snakers4/silero-vad | 统一语音分段 |

外部页面核对日期2026-10-29；链接不等于目标权重已下载或已重新校验。
若另行分发任何权重/依赖，应带齐对应版本许可及notice，不能用本文件替代上游条款。
