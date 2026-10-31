# 测试与结果边界

## 无模型、无硬件的仓库检查

```bash
python scripts/check_sources.py
python scripts/test_logic.py
```

来源检查核对SOURCE_PROVENANCE.json所列源码身份；原始绑定与后续修改的基线SHA分别保留。还会检查Python语法和已归档证据文件身份。

逻辑检查覆盖19项选择规则、空文本、语言状态、纠错持久化/隔离和日语后端文件合同。只依赖标准库，不进行模型推理。CI执行同样范围。

`app/test_fastenhancer_onnx.py` 需要NumPy、ORT及模型；`app/test_http_callbacks.py` 依赖完整主程序导入环境。这些不是纯逻辑CI的一部分，未执行项不记通过。

## 已完成的真实设备测试

以下1.150秒等数字属于原版，无2026-10-31新增的前缀预览开销。新预览在Pi上做了独立3次前缀解码检查：输入未结束时返回临时文字，未归档临时结果。浏览器合成状态测试覆盖追加、清空、刷新、迟到结果及桌面/手机布局；用户随后确认实机话筒下逐步显示、追加与清空正常。详见[新版验证摘要](../evidence/REALTIME_PREVIEW_VALIDATION.json)，不据此更新旧准确率/吞吐数字。

素材：`fastenhancer_official_p232_013.wav`，63095帧/16kHz/单声道PCM16，3.943秒。

```text
SHA256 ce99076fdd0801d4e338e8845aa54c9e4dbb2f2691ce4fcc348d3bf8679255fe
```

| 路径 | 增强ms | 识别ms | 合计ms |
|---|---:|---:|---:|
| raw | 0 | 503 | 503 |
| GTCRN | 751 | 397 | 1148 |
| FastEnhancer-B | 446 | 439 | 885 |

三路并行warm wall1.150秒、导出RTF0.2916；含初始化整个进程9.545秒，基准进程峰值RSS965.2MiB。三路文字一致并选择raw。只有一次素材测试，不给出均值、P95或统计显著性。

来源：[脱敏JSON](../evidence/ONE_EXISTING_FIXTURE_DEVICE_VALIDATION.json)；[设备验证报告](../evidence/DEVICE_VALIDATION_REPORT.md)。

基准命令归档在 `scripts/benchmark_existing_manual.sh`，仅供将来明确安排的复现；本次GitHub整理没有重复该测试、读取设备声卡或训练模型。

## 尚未测得的指标

当前没有人工参考稿下的多语准确率、配对干净参考的PESQ/STOI/SI-SDR、主观盲听、声学往返、连续XRUN、deadline miss与长期热稳定证据。共识分不等于准确率，RTF<1不等于硬实时，输入队列drop0不等于XRUN0。

另一路native-piece研究未部署到本终端，CP672条全部原样、英文WER53.521127%/中文CER51.046699%、quality=false。该阴性研究结果仅作独立历史说明，不能当成当前Pi性能或优化收益。

设备记录中的身份和地址已脱敏，变换范围见[证据脱敏说明](../evidence/REDACTION_NOTICE.md)。
