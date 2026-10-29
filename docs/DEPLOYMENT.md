# 部署与人工启动

## 已验证环境

观察于电脑 UTC 2026-10-28，客户端归档日期 2026-10-29：Pi5 Model B Rev1.1 / aarch64 / 约4GB，Python3.13.5、NumPy2.2.4、ONNX Runtime1.28.0、sherpa-onnx1.13.5。依据见[环境回执](../evidence/RUNTIME_AND_DEPLOYMENT_METADATA.json)。

[`requirements-pi-observed.txt`](../requirements-pi-observed.txt) 记录已观察的主要版本，不是完整的传递依赖锁。没有重新安装或升级生产设备。

## 会前准备

1. 准备 ALSA 工具、UMC202HD、INPUT1 话筒及匹配线材，确认设备名。
2. 准备与目标 Python ABI / aarch64 匹配的依赖；参考版本文件，不混用 x86 wheel。
3. 准备 [`MODEL_ASSET_MANIFEST.json`](../MODEL_ASSET_MANIFEST.json) 中的增强/VAD/ASR与tokens。仓库不分发这些权重，来源与条款见[归属说明](../RIGHTS_AND_ATTRIBUTION.md)。
4. 在备用设备上完成首次模型加载与短句检查，再安排现场使用。文件存在检查不等于完整运行验收。

如果已准备完整离线 wheel 目录，可在备用设备人工安装到其独立环境，示例：

```bash
python3.13 -m venv /path/to/device/venv
/path/to/device/venv/bin/python -m pip install --no-index \
  --find-links /path/to/prepared-wheels -r requirements-pi-observed.txt
```

以上是新设备准备示例，不是已执行的安装。若 wheel 或依赖不全应先补齐，不在会场临时替换生产环境。

## 已有部署的使用方式

设备根目录由 `DEVICE_ROOT` 指定，默认使用 `$HOME/voice_demo`，包含 `venv/`、`models/`、`testdata/`。
网页端口8080，访问 `http://<DEVICE_HOST>:8080/` 时将占位符替换为实际地址。个人设备用户名、地址和主机名已省略。

```bash
python scripts/verify_assets.py --models "$HOME/voice_demo/models"
```

此命令只读文件存在性和已记录大小，不加载或反复 hash 大权重。已有服务运行时，直接打开其网页，不额外启动占用同一声卡/端口的进程。

## 在空闲备用设备人工启动

确认没有冲突服务、依赖和模型准备完毕后：

```bash
DEVICE_ROOT="$HOME/voice_demo" bash scripts/run_demo_manual.sh
```

脚本使用 `DEVICE_ROOT/venv/bin/python` 与 `DEVICE_ROOT/models`，代码来自当前仓库 `app/`，日志写入当前仓库 `logs/`。脚本不会自动停服务或安装依赖。默认尊重设备已有语言配置。

程序不打开播放设备，不输出扬声器抵消声。默认不开启保存语音片段；本地服务仍可能写转写与纠错记录，备份与分享时需排除私人内容。

## 常见定位顺序

| 现象 | 优先检查 |
|---|---|
| 网页打不开 | 已知IP与有线链路、8080现有服务 |
| 没有输入电平 | UMC连接、INPUT1、采集设备名与现有占用 |
| 缺少 `voice_demo` | 从仓库 `app/` 执行，保留基础模块同目录 |
| 缺少 ASR/ORT | 目标venv与兼容平台的依赖，不用Windows环境代替Pi |
| 找不到模型 | 根目录和manifest文件名，日语需encoder/decoder/joiner/tokens全套 |
| 时间看起来不对 | 已知设备墙钟未同步；以主机记录与单调计时口径核对 |

离线方案是本地模型配合局域网网页。备用设备与完整资产备份需要本人实际准备，仓库不能代替物理验收。
