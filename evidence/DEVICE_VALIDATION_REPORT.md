# 树莓派现有部署有限验证报告

当前有限设备/软件部署验证已完成；模型研究优化保持停止。本文将可用性、单素材吞吐和未验证的准确率/物理实时性分开，不恢复训练/扫参或旧Slurm任务。

## 已核实的现有设备与服务

| 项目 | 实测或读取结果 |
|---|---|
| 主机 | <DEVICE_USER>@<DEVICE_HOST>，hostname <DEVICE_HOSTNAME>，SSH既有hostkey匹配 |
| 板卡 | Raspberry Pi 5 Model B Rev1.1，aarch64，约4GB RAM |
| 连接 | Windows <CLIENT_HOSTNAME>有线直连，链路Up1Gbps |
| 服务 | http://<DEVICE_HOST>:8080 ，HTTP200、phase=listening、error为空 |
| 程序 | ${DEVICE_ROOT}/app/voice_demo_multicandidate.py；既有PID1245持续运行，未重启 |
| 音频 | UMC202HD，plughw:U192k,0；既有采集进程owner_pid1736，ALSA RUNNING |
| 硬件参数 | 48kHz、S32_LE、2通道；period1536、buffer6147 samples。这是配置，非物理往返延迟 |
| 连续采集 | 两次约0.3秒间隔的hw_ptr从470648333到470662730，已有采集链路前进 |
| 温度/节流 | 已采样温度47.2–49.4℃，throttled=0x0；不等于长期满载热稳定证明 |

当前现有模式为日语（ja），后端ReazonSpeech（日语专用）。浏览器用于显示和控制，声音来自Pi声卡INPUT1；浏览器麦克风不是已实现的输入方式。没有更改语言、记忆、增益、网络、服务或模型。

## 实际运行环境与源码身份

现有venv为 ${DEVICE_ROOT}/venv/bin/python，Python3.13.5，NumPy2.2.4，ONNX Runtime1.28.0，sherpa-onnx1.13.5。sounddevice/soundfile发行包metadata未找到不等于服务故障；当前程序使用既有ALSA输入。

7个明确源码文件共95046bytes已一次校验身份，完整SHA见RUNTIME_AND_DEPLOYMENT_METADATA.json。生产程序SHA dc134b3266d103fea6a87ecd32476bf026455fad473f932470a713fa3bc877e7；基准脚本SHA006bf312c44df91859de2693b6ceae605df4e567864268d2996d23e388df896a。已核对GTCRN/FastEnhancer/VAD/SenseVoice模型和tokens存在性/大小；237MB大权重未重新hash，因此不称本次完整权重完整性复验。

## 一次现有英文素材的真实ARM64离线验证

仅用设备原有fastenhancer_official_p232_013.wav，63095帧/16kHz/monoPCM16，音频3.943秒，SHA ce99076fdd0801d4e338e8845aa54c9e4dbb2f2691ce4fcc348d3bf8679255fe。用nice10、各ASR/增强线程1执行现有benchmark脚本，新增结果写独立device_validation_20261029目录；没有打开额外声卡、录音、扬声器播放或训练更新。

| 路径 | 增强ms | 识别ms | 路径合计ms |
|---|---:|---:|---:|
| 原始 | 0 | 503 | 503 |
| GTCRN | 751 | 397 | 1148 |
| FastEnhancer-B | 446 | 439 | 885 |

三路并行warm wall1.150秒、RTF0.2916，含模型加载等初始化的整个进程9.545秒，benchmark进程峰值RSS965.2MiB。三路文本一致并选择raw；一致性不是经过标定的ASR置信度，也未据此计算准确率。该一次英文素材吞吐不能代表中文/日文准确率、背景噪声收益、现场话筒端到端或硬实时deadline。stderr有非致命DRM/GPU设备发现提示，CPU推理成功。

该验证已有唯一BENCHMARK_ATTEMPT与完整结果，本机ONE_EXISTING_FIXTURE_DEVICE_VALIDATION.json为成功接收证据；不要因为远端记录格式问题或新heartbeat重复推理。

## 既有现场输入事件与延迟口径

最新状态已处理5个输入/VAD片段，dropped_utterances=0、error为空；只保留元数据，没有保存/展示私人转写或热词。无法据此确认是用户有意测试语句或确认文字正确。

最新界面latest_latency_ms为2988，候选raw/GTCRN/FastEnhancer分别101/215/165ms。源码定义为 (time.monotonic()-utterance.endpoint_monotonic)*1000，在三路完成后采样；不能把候选识别耗时与此字段混为一谈，更不能称声学回环延迟。没有验证ALSA XRUN全程计数、deadline miss或物理扬声器链路；ASR队列drop0不能代替这些证据。

## 时钟与剩余人工验证

Pi墙钟仍为2026-09-01，Timezone=Asia/Shanghai、NTP=yes、NTPSynchronized=no。所有报告归档时间用主机UTC/客户端AsiaShanghai，不能用Pi墙钟日志做当前时间关联；本次没有修改设备时钟，perf_counter区间测时单独记录。

若进行下一次真人转写验证：在网页选择实际要说的语种，对UMC202HD INPUT1说一段3–5秒短句，停顿至少约0.45秒，再人工核对文字。定量准确率需要预定参考稿；声学回环/物理端到端和硬实时截止需要另行明确布线、播放与测量，不应自动执行或从RTF推断。本轮有限软件/设备可用性证据已交付，这些项目保持“未验证”。

## 研究与设备结论

现有Pi部署可访问、持续采集、既有离线推理成功。先前模型研究最终评分仍为阴性：CP全部原样COPY，英文WER53.521127%、中文CER51.046699%，没有相对A0输入收益；两类结论不相互替代。模型优化已按用户要求停止，设备验证没有训练任何模型。

证据清单及SHA见VALIDATION_EVIDENCE_LOCK.json；可复现入口见REPRODUCTION_COMMANDS.md。完成当前有限验证后结束该定时跟进，不重复已完成基准或扩展新模型任务。
