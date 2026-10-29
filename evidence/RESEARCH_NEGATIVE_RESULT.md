# 本轮评分结束，模型优化已按用户要求停止

CPU评分恢复77770 COMPLETED0:0/38秒，gpu-a10-04，worker4CPU20GiB/SlurmAlloc8/GPU0。只修复规范化子进程对已有joint/runtime中tn的加载路径，没有安装或更改共享库；原2688条文本SHA保持8a0dd59448105440e0a8cf3ca7504d842da39262ad7b550916869d3ea3670006，没有重训、重推理或重渲染。

| -10dB standard，三配对seed合计 | CP主方法 | 原A0raw输入 | 结论 |
|---|---:|---:|---|
| 英文WER | 1368/2556 = 53.521127% | 53.521127% | 无提升 |
| 中文CER | 1902/3726 = 51.046699% | 51.046699% | 无提升 |

CP全部672/672文本逐字与输入相同，PN也672/672相同。UC有19条变化、MC有4条变化；UC英文其中一个seed明显退化，三seed合计WER119.718310%，CER/WER大于100未删除或截断。CP对UC虽然两项5%比较通过，但仅1/3seed胜且保护门失败，不能当稳定收益。112个强控制比较中5%门2通过、protected110通过、source110通过；整体quality/combined=false，numeric=true。严格英文CP1362/2556=53.286385%，中文仍1902/3726。

训练确实发生：12组各100次更新，初末参数SHA改变；加8次丢弃smoke共1208次新corrector optimizer。全部33construct/4884teacher/4848backward/2704decode/16safe-load的attempt-success和27386条父trace已本机核验，2688预测/脚本/native-word映射及2016strict case详情、112Fraction整数门核对；不重跑模型/WeText/scorer/编辑对齐，不冒称完整参数梯度或ASR识别独立复现。旧BCF硬numericfalse不变。

本轮GPU分配elapsed2069秒（numeric26+12trial2043），CPU486秒（独立45+原评分失败403+恢复38），非kernel/RTF/物理延迟。原ASR216471/nativeASR更新11162/SE4266/旧corrector906不重复记账；本新corrector1208独立唯一账本。CPUprofile123秒已另记，不重加。72份9926402正文bytes文本一次single8192接收，wirebytes未测，无数组、checkpoint、权重或音频下载。

REPORT a5595ecfc299ba9c76a9626eb2a9f16160731575fdf066a6209b5026a10b0980；CASE372710c4382486a6a90215a73dd9391bc43beb4dacab3989c3ecbf3bf9c2cb4f；RESULT_ACCEPTANCE b542bef416723375e8f9a47bf64a4892c4939e5233467aba42cac72d9e0b8085；LEDGER0981d713d094bfdf4d27be7e81005c5a58695907f708b6e606c4871b420f9f3a。所有源、权重和失败证据保留。

用户要求本轮结束后停止优化已执行：不再新训练、扫参、换decoder、seed或追加方法诊断；定时任务保持删除。后续仅使用现有树莓派进行设备检查和既有部署验证。当前设备连接需确认，旧<DEVICE_HOST>无法解析、<DEVICE_HOST>超时，已按用户补充的实体网线直连重新检查链路；尚未宣称任何Pi实测或部署成功。
