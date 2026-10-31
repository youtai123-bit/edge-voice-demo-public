# 架构与代码地图

## 输入、分段与并行

`voice_demo.py` 提供 ALSA 采集和基础模型创建函数；`voice_demo_multicandidate.py` 的 `AudioPipeline` 在原始 INPUT1 上进行 VAD，结束的同一个片段放入工作队列。

`CandidateWorker` 对该片段执行三路处理：raw 直接识别，GTCRN 和 FastEnhancer-B 先增强再识别。各路使用独立 ASR 实例；并行任务全部完成后调用选择器，写入共享状态，由网页读取。

`partial_preview.py`在说话过程中对已经采到的增长前缀发起低优先级识别，复用raw识别器并串行保护调用。临时文字可以修正，最终句子再经三路确认。`transcript_view.js`负责保留式对话框和浏览器侧清空；`transcript_archive.py`只读有限的既有转写尾部恢复历史。详见[逐步显示](REALTIME_PREVIEW.md)。

设备硬件曾协商为 48kHz / S32_LE / 双通道，程序面向 16kHz 输入且只取通道0。ALSA plug 层衔接格式。这是分段后的语句级转写，不能仅凭增强网络可以逐帧运行，就把整套系统称为严格因果流式。

## 候选选择

[`candidate_selection.py`](../app/candidate_selection.py) 对文本进行 NFKC、大写与字母数字规范化。存在至少两票的字面多数时，优先从多数集合选择，再比较音频安全与耗时。没有多数时使用固定工程评分：

```text
0.67 × 平均文本相似度
+ 0.20 × 规范化字面一致比例
+ 0.10 × 波形安全分
+ 0.03 × 相对耗时加分
```

波形安全只检查有限值、过低 RMS 与削波等明显异常。三路均空返回空结果；三路可能同时出错。因此共识与安全分不能替代识别正确率或感知音质。

## 语言后端与人工纠错

- `asr_backends.py`：日语使用 ReazonSpeech Transducer，其余配置使用 SenseVoice。
- `language_selection.py`：校验与保存语言状态，切换由工作线程处理。
- `personalization.py`：按语种保存精确纠错记忆，最多200条；日语重点词作为热词注入下一条识别流。

这属于规则记忆与推理时热词偏置，不更新神经网络参数。人工修正后的结果必须与未经帮助的原始识别成绩分开统计。

## HTTP 入口

| 路由 | 功能 |
|---|---|
| `GET /` | 展示界面 |
| `GET /transcript-view.js` | 对话框的浏览器显示状态逻辑 |
| `GET /api/state` | 当前状态、候选与历史等信息 |
| `GET /api/health` | 当前状态快照；可能包含转写内容，分享前要筛选 |
| `POST /api/correction` | 保存最近一句的人工纠正 |
| `POST /api/personalization/clear` | 清空指定语种的记忆 |

网页是本地展示/控制端，不是浏览器麦克风采集入口。当前演示面向受控局域网，不应直接暴露到公网。

## 延迟口径

界面 `latest_latency_ms` 源于三路完成后的 `time.monotonic() - utterance.endpoint_monotonic`。候选合计、并行 wall、整个进程耗时与该字段具有不同起止事件。语音队列 drop0 也不能代替 ALSA XRUN 全程计数。
