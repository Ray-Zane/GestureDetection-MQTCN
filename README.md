# GestureDetection-MQTCN

**GestureDetection-MQTCN（Memory-Query Temporal Convolutional Network）** 是基于 IPN Hand 的骨架流连续手势识别模型与端到端实时 Demo。系统从普通 RGB 视频或摄像头画面中在线提取 21 点手部骨架，使用严格因果的 TCN、有限帧记忆和事件查询同时完成手势类别识别与起止边界定位，最终输出按时间排序的连续手势事件。

## 功能

- 支持 IPN Hand 连续视频、任意本地视频和实时摄像头三种输入；
- MediaPipe Hand Landmarker 在线提取 normalized image/world landmarks；
- 135 维严格因果骨架特征与 Stateful Causal TCN 逐帧推理；
- 256 帧 Frame Memory 与每 4 帧执行一次的 Completed-event Query；
- Frame、Query 和 Fusion 三路事件记录，以 `events.fusion` 作为最终输出；
- 预览窗口、手部骨架叠加、状态面板、带标注视频保存和完整 JSON 报告；
- 长时间丢手后的特征、TCN、Memory 和 Decoder 原子重置，避免跨片段状态污染；
- Checkpoint、运行配置和 MediaPipe 模型 SHA-256 完整性校验。

## 模型结构

```text
RGB 视频 / 摄像头
        │
        ▼
MediaPipe Hand Landmarker
21 × image/world landmarks + handedness
        │
        ▼
P3 在线骨架特征（135 维）
使用 normalized image xyz：局部姿态 + 单帧运动 + 手腕位置/运动 + 尺度 + 有效性 + 左右手
        │
        ▼
Frame Encoder（135 → 128）
        │
        ▼
Stateful Causal TCN
kernel=3，dilation=1/2/4/8/16/32，感受野 127 帧
        ├───────────────────────┐
        ▼                       ▼
Frame / Boundary Heads     256-frame Frame Memory
        │                       │
        │                 6-slot Event Query
        │                  stride = 4 frames
        └───────────┬───────────┘
                    ▼
          Frame / Query Fusion
                    ▼
       连续手势事件 + 实时界面 + JSON
```

MediaPipe 同时返回 image/world landmarks，最终冻结配置选择经过宽高比修正的 normalized image xyz。P3 的 135 维输入由 `63` 维局部姿态、`63` 维一阶姿态差分、手腕位置与差分、尺度与尺度差分、当前检测有效性、连续丢手长度和 handedness 组成。模型只使用当前帧及历史状态，不读取未来帧。

## 实验结果

模型按照 IPN Hand 官方 Train/Test 划分训练和评测，不额外拆分 validation。Official Test 在训练期间只用于稳定性和质量监控；正式实验固定训练轮数并评测最后一轮权重，不根据 Test 指标选择 checkpoint，也不进行 early stopping。由于研发过程中已观察过 Test，本项目将结果称为 **Official Split Benchmark**，不将其表述为 blind holdout。

最终固定部署模型在 52 个 Official Test 视频、1,101 个真实手势事件上的结果如下：

| 指标 | 结果 |
|---|---:|
| Levenshtein Accuracy | 0.1898 |
| Event Precision @ tIoU 0.5 | 0.5091 |
| Event Recall @ tIoU 0.5 | 0.8883 |
| Event F1 @ tIoU 0.5 | 0.6473 |
| False Positives / minute | 7.9997 |
| Predicted / Ground-truth events | 1,921 / 1,101 |

Levenshtein Accuracy 衡量按时间排序的预测类别序列与真实类别序列之间的编辑误差；Event F1 要求类别一致且时间 IoU 不低于 0.5。连续识别同时存在分类、漏检、重复检测和边界误差，因此这些事件级指标比孤立手势分类准确率更有代表性。

仓库中的 `outputs/demo/video_validation.json` 是一段 Official Test 视频前 300 帧的端到端 CUDA smoke test，用来验证视频解码、MediaPipe、模型状态、事件融合和 JSON 写入能够共同工作；它不是完整测试集精度报告，也不应作为跨硬件速度基准。

## 消融实验

以下实验与最终结果采用相同的 Official Train/Test、因果约束和事件评测口径。除明确标记为三 seed 的表格外，其余均为 seed 1234 的单 seed 受控实验。`B0`–`B4` 是研发阶段用于表示模块递进关系的历史实验编号，不是当前公开代码的文件名；公开仓库只保留最后一行对应的完整模型。

### 主模块递进

| 历史编号 | 方法 | 新增内容 | Lev.Acc | F1@0.5 | Precision | Recall | FP/min | Pred |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B0 | Frame MLP | 单帧控制组，无时序上下文 | -0.6885 | 0.0875 | 0.0615 | 0.1517 | 21.6324 | 2,717 |
| B1 | Causal TCN | 127 帧严格因果感受野 | -0.2389 | 0.3065 | 0.2238 | 0.4859 | 15.7365 | 2,390 |
| B2 | Stateful Causal TCN | 将 B1 等价改写为状态化逐帧执行 | -0.2380 | 0.3066 | 0.2239 | 0.4859 | 15.7280 | 2,389 |
| B3 | B2 + Event Query | Completed-event Query，无显式 Frame Memory | -0.1689 | 0.5699 | 0.4204 | 0.8847 | 11.3930 | 2,317 |
| **B4 / Final** | **B3 + Frame Memory** | **最近 256 帧显式记忆与 Frame/Query Fusion** | **0.1898** | **0.6473** | **0.5091** | **0.8883** | **7.9997** | **1,921** |

该递进实验支持以下结论：

- **因果时序建模有效**：B1 相对 B0 的 Lev.Acc 提高 `0.4496`、F1 提高 `0.2190`、FP/min 降低 `5.8959`，说明连续识别不能只依靠单帧姿态；
- **Stateful 改写不改变精度目标**：B2 复用 B1 权重，FP32 下最终事件输出等价。表中末位差异来自统一重计分，不应解释为新的精度收益；
- **Event Query 主要改善事件召回与碎片化**：B3 相对 B2 的 F1 提高 `0.2633`、FP/min 降低 `4.3350`，但 Lev.Acc 增量 `0.0690` 的 paired-video 95% CI 为 `[-0.0127, 0.1536]`，区间仍跨 0；
- **Frame Memory 是主链中最关键的新增模块**：B4 相对 B3 的 Lev.Acc 提高 `0.3588`、F1 提高 `0.0773`、FP/min 降低 `3.3933`，并减少 395 个 insertion；Lev.Acc 增量的 paired-video 95% CI 为 `[0.2834, 0.4357]`。

这里的 bootstrap CI 只反映 seed 1234 下 52 个测试视频的采样变化，不能替代不同训练随机种子带来的不确定性。

### 三随机种子一致性

为验证 Memory-Query 增益是否只来自单次训练，Stateful TCN 对照组与最终模型均运行了三个固定 seed：

| 模型 | Seed | Lev.Acc | F1@0.5 | FP/min |
|---|---:|---:|---:|---:|
| Stateful TCN | 1234 | -0.2380 | 0.3066 | 15.7280 |
| Stateful TCN | 2345 | -0.3124 | 0.2898 | 16.5933 |
| Stateful TCN | 3456 | -0.2062 | 0.2966 | 15.5583 |
| Final + Memory-Query | 1234 | **0.1898** | **0.6473** | **7.9997** |
| Final + Memory-Query | 2345 | **0.1617** | **0.6457** | **8.3476** |
| Final + Memory-Query | 3456 | -0.1235 | **0.5757** | **11.0792** |

| 三 seed 汇总 | Lev.Acc mean±std | F1@0.5 mean±std | FP/min mean±std |
|---|---:|---:|---:|
| Stateful TCN | -0.2522±0.0545 | 0.2977±0.0085 | 15.9599±0.5551 |
| Final + Memory-Query | **0.0760±0.1734** | **0.6229±0.0408** | **9.1422±1.6865** |
| Final − Stateful TCN（平均增量） | **+0.3282** | **+0.3252** | **-6.8177** |

三个 seed 的 Lev.Acc 和 F1 均提高、FP/min 均降低，说明结构增益方向一致；但最终模型的 Lev.Acc 标准差仍为 `0.1734`，且 seed 3456 的绝对值仍为负，因此不能将结果表述为训练已经高度稳定。最终 checkpoint 使用实验前固定的 seed 1234，不是根据 Test 指标事后选择的最好 seed。

### P3 特征 Profile

该组实验固定使用 Causal TCN，只改变输入构造：

| Profile | 输入内容 | Lev.Acc | F1@0.5 | FP/min |
|---|---|---:|---:|---:|
| P0 | 原始坐标 | -0.3624 | 0.2553 | 17.0429 |
| P1 | P0 + wrist center | -0.3506 | 0.2494 | 17.2211 |
| P2 | P1 + scale normalization | -0.3860 | 0.2527 | 17.6367 |
| **P3** | **P2 + local/global motion** | **-0.2389** | **0.3065** | **15.7365** |
| P4 | P3 + current-frame palm rotation | -0.2171 | 0.2823 | 15.8468 |

P3 在事件 F1 和误报之间取得最佳综合平衡。P4 的 Lev.Acc 比 P3 高 `0.0218`，但 F1 低 `0.0242`，同时会加重方向性 Throw 手势的语义混淆，因此最终配置保留 P3。

### Frame Memory 长度

该组保持 Query、训练目标、融合规则和其他参数不变，仅修改显式历史长度：

| Memory | Lev.Acc | F1@0.5 | FP/min | Start MAE | End MAE |
|---:|---:|---:|---:|---:|---:|
| 128 frames | -0.0545 | 0.4711 | 12.1056 | 26.90 | 15.11 |
| **256 frames** | **0.1898** | **0.6473** | **7.9997** | **21.05** | 13.52 |
| 384 frames | 0.1072 | 0.6263 | 8.7463 | 22.94 | **12.66** |

128 帧不足以为长手势保留完整上下文；384 帧虽然进一步改善 End MAE，但更长历史也会引入无关上下文，整体序列与事件指标低于 256 帧。因此最终模型选择 256 帧。该长度实验只有一个 seed，结论限定在当前固定协议内。

<details>
<summary><strong>更多输入、缺失处理与 Boundary 诊断</strong></summary>

#### 坐标来源与 Motion lag

| 配置 | Lev.Acc | F1@0.5 | FP/min | 观察 |
|---|---:|---:|---:|---|
| **`image_xyz + lag[1]`** | **-0.2389** | 0.3065 | 15.7365 | 最终冻结配置 |
| `image_xy + lag[1]` | -0.2770 | 0.2830 | 16.3982 | 删除 z 后三项均退化 |
| `world_local_xyz + lag[1]` | -0.2180 | 0.2980 | **15.6601** | Lev/FP 略好，但 F1 较低 |
| `image_xyz + lag[1,3,5]` | -0.2307 | **0.3088** | 15.8383 | 收益不一致，并增加 16,896 个参数 |

`world_local_xyz` 与多 lag 都没有形成跨指标的一致优势，且只有一个 seed，因此不足以替代 `image_xyz + lag[1]`。

#### 短时丢手 hold

| 最大 hold 帧数 | Lev.Acc | F1@0.5 | FP/min |
|---:|---:|---:|---:|
| 3 | -0.2579 | 0.3035 | 15.9316 |
| **5** | -0.2389 | 0.3065 | 15.7365 |
| 10 | **-0.2216** | **0.3071** | **15.5838** |

hold 10 的总体单 seed 数值略好，但按 missing-gap 分组后没有一致收益。为避免根据已观察 Test 事后调参，并减少旧姿态跨长缺失传播，最终仍固定 hold 5；第 6 个连续缺失帧触发全链路原子重置。

#### 数据增强

| 配置 | Lev.Acc | F1@0.5 | Frame Macro-F1 | FP/min |
|---|---:|---:|---:|---:|
| 全部关闭 | **-0.2334** | 0.3024 | **0.5706** | 15.7535 |
| **legacy missing-span** | -0.2389 | **0.3065** | 0.5689 | **15.7365** |
| 完整骨架增强 | -0.2561 | 0.2713 | 0.5500 | 16.2794 |

完整的 jitter、dropout、scale、translation 和 temporal-speed 组合产生负收益。最终保留 legacy missing-span，但它相对完全关闭的差异很小，不能作强泛化结论。

#### Boundary Head 单 seed 诊断

| 设置 | Lev.Acc | F1@0.5 | FP/min | Matched | Pred |
|---|---:|---:|---:|---:|---:|
| Causal TCN + Boundary Head | -0.2389 | 0.3065 | 15.7365 | 535 | 2,390 |
| Start/End logits 置中性并移除辅助损失 | **0.2997** | **0.5052** | **8.9838** | **730** | **1,789** |

该结果提示 Boundary 辅助任务与状态机终止条件可能导致错误终止和过分割。不过实验同时改变了训练损失与推理边界信号，并且只运行一个 seed，无法确认负迁移究竟来自训练还是解码器，因此它只作为后续研究诊断，不用于事后替换已冻结模型。

</details>

### 模型路径计算开销

以下结果是在相同交错负载下，从预提取的 135 维 feature 到最终事件逻辑的计时，不包含 OpenCV、MediaPipe、摄像头驱动和 UI：

| 模型路径 | Mean ms/frame | P95 ms/frame | Throughput |
|---|---:|---:|---:|
| Stateful TCN | 3.541 | 9.415 | 282.40 FPS |
| + Event Query | 4.183 | 12.756 | 239.04 FPS |
| **+ 256-frame Memory（Final）** | **4.446** | **13.328** | **224.91 FPS** |

Memory-Query 带来可测量的额外开销，但该模型路径的 P95 仍低于 30 FPS 对应的 `33.33 ms` 帧预算。完整端到端速度还取决于 MediaPipe、视频解码、显示和硬件，不能直接用本表替代真实 Demo 测量。

## 环境安装

已验证环境：

```text
Python 3.9.23
Windows 11
PyTorch FP32（CPU / CUDA）
```

建议新建独立环境：

```bash
conda create -n gesture_recognition python=3.9 -y
conda activate gesture_recognition
python -m pip install -r requirements.txt
```

如果需要 CUDA，请先根据本机 CUDA/驱动版本安装匹配的 PyTorch，再安装 `requirements.txt` 中的其余依赖。没有 CUDA 时使用 `--device cpu`，默认的 `--device auto` 会自动选择可用设备。

本项目开发环境中的 Python 路径为：

```text
C:\Users\MSI-NB\anaconda3\envs\gesture_recognition\python.exe
```

该路径只用于说明已验证环境，其他用户不需要使用相同安装位置。

## 快速开始

所有命令都应在项目根目录执行。

### 任意本地视频

```bash
python tools/realtime_demo.py --video "path/to/video.mp4" --device auto
```

### 摄像头

```bash
python tools/realtime_demo.py --camera 0 --device auto --mirror-display
```

如果摄像头 0 无法打开，可尝试 `--camera 1`。`--mirror-display` 只镜像预览和保存画面，不改变模型输入，避免左右方向手势的语义被反转。

### IPN Hand Official Test 视频

此方式需要先按“数据准备”章节建立 `data/`：

```bash
python tools/realtime_demo.py --video-id "1CM1_1_R_#217" --split test --device cuda
```

也可以直接指定对齐视频：

```bash
python tools/realtime_demo.py --video "data/raw/aligned_videos/1CM1_1_R_#217.mp4" --device cuda
```

### 无窗口运行并保存结果

```bash
python tools/realtime_demo.py \
  --video-id "1CM1_1_R_#217" \
  --split test \
  --device cuda \
  --no-display \
  --save-video "outputs/demo/annotated.mp4" \
  --output-json "outputs/demo/events.json"
```

Windows CMD 不支持反斜杠续行，可以将上面的参数写在同一行。PowerShell 也可以直接使用单行命令。

常用选项：

| 选项 | 作用 |
|---|---|
| `--device auto/cpu/cuda` | 选择推理设备 |
| `--display-width 1600 --display-height 900` | 调整预览窗口画布，默认 1280×720 |
| `--realtime-playback` | 按视频标称 FPS 播放本地视频 |
| `--max-frames N` | 只处理前 N 帧，适合 smoke test |
| `--no-display` | 不创建 GUI 窗口 |
| `--save-video PATH` | 保存带骨架和状态面板的结果视频 |
| `--output-json PATH` | 指定事件报告路径 |

窗口模式按 `q` 或 `Esc` 退出；无窗口摄像头模式按 `Ctrl+C` 退出。未指定 `--output-json` 时，报告默认写入：

```text
outputs/demo/<source>_<timestamp>/events.json
```

## 输出说明

每次运行只生成一份 JSON，统一记录模型、模型哈希、输入源、耗时、骨架有效率和三路事件：

```json
{
  "model": {
    "primary_output": "fusion",
    "parameter_count": 373408
  },
  "runtime": {
    "frames_processed": 300
  },
  "events": {
    "frame": [],
    "query": [],
    "fusion": [
      {
        "class_id": 0,
        "class_name": "Pointing with one finger",
        "start_frame": 24,
        "end_frame_exclusive": 159,
        "start_seconds": 0.8,
        "end_seconds": 5.3,
        "score": 0.9231
      }
    ]
  }
}
```

- `events.frame`：由逐帧类别和边界状态机产生；
- `events.query`：由 Completed-event Query 产生；
- `events.fusion`：结合两路证据、去重后的最终连续手势结果；
- 时间区间采用零基、左闭右开形式 `[start_frame, end_frame_exclusive)`。

模型输出的 13 个手势类别为：

| `class_id` | IPN code | Gesture |
|---:|---|---|
| 0 | B0A | Pointing with one finger |
| 1 | B0B | Pointing with two fingers |
| 2 | G01 | Click with one finger |
| 3 | G02 | Click with two fingers |
| 4 | G03 | Throw up |
| 5 | G04 | Throw down |
| 6 | G05 | Throw left |
| 7 | G06 | Throw right |
| 8 | G07 | Open twice |
| 9 | G08 | Double click with one finger |
| 10 | G09 | Double click with two fingers |
| 11 | G10 | Zoom in |
| 12 | G11 | Zoom out |

Background 只参与逐帧内部分类，不会作为最终手势事件输出。

## 数据准备与来源

### 获取 IPN Hand

请从 [IPN Hand 官方项目页](https://gibranbenitez.github.io/IPN_Hand/) 下载官方数据。对本项目而言，官方下载后的原始输入只有两个目录：`data/raw/annotations` 和 `data/raw/frames`；不需要另行准备 `raw/videos`。数据集包含 200 个连续序列、50 名受试者、约 80 万帧和 4,218 个手势实例，本项目严格沿用官方 Train/Test 划分：

| Split | Subjects | Videos | Gesture segments |
|---|---:|---:|---:|
| Train | 37 | 148 | 3,117 |
| Test | 13 | 52 | 1,101 |
| Total | 50 | 200 | 4,218 |

数据及 annotations 遵循 IPN Hand 官方许可，不随本仓库重新分发。

### 本项目中的数据目录

下面是本地完整研发目录的逻辑结构。公开仓库不会包含整个 `data/`，需要使用数据集功能的用户应自行建立同样的路径。

```text
data/                                      # 本地数据，不上传 GitHub
├── raw/
│   ├── annotations/                       # [官方输入] 标签、划分和 metadata
│   │   ├── Annot_List.txt
│   │   ├── Annot_TrainList.txt
│   │   ├── Annot_TestList.txt
│   │   ├── Video_TrainList.txt
│   │   ├── Video_TestList.txt
│   │   ├── classIdx.txt
│   │   └── metadata.csv
│   ├── frames/                            # [官方输入]
│   │   └── <video_id>/
│   │       ├── <video_id>_000001.jpg
│   │       ├── <video_id>_000002.jpg
│   │       └── ...                        # 官方 320×240 RGB frame 序列
│   └── aligned_videos/
│       └── <video_id>.mp4                 # [本项目生成] 标签对齐视频
├── manifests/
│   ├── train.json                         # [本项目生成] 官方 Train 索引
│   └── test.json                          # [本项目生成] 官方 Test 索引
└── skeleton_raw/
    ├── train/<video_id>.npz               # [本项目生成] 训练骨架
    └── test/<video_id>.npz                # [本项目生成] 测试骨架
```

### 从官方文件生成完整数据链

以下命令都在仓库根目录执行。首次开始前，至少应有：

```text
data/raw/annotations/Annot_List.txt
data/raw/annotations/Annot_TrainList.txt
data/raw/annotations/Annot_TestList.txt
data/raw/annotations/Video_TrainList.txt
data/raw/annotations/Video_TestList.txt
data/raw/annotations/classIdx.txt
data/raw/annotations/metadata.csv
data/raw/frames/<video_id>/<video_id>_000001.jpg
```

其中 `metadata.csv` 不影响视频/标签对齐，但最终 P3 配置启用了 handedness 特征；缺少该文件时可以运行数据加载，得到的却不是与发布训练相同的 135 维输入语义，因此正式复现时必须保留。

第一步，将每个官方 JPG 序列按数字帧号编码为 30 FPS 的标签对齐 MP4：

```bash
python -m tools.preprocess.build_aligned_videos --strict
```

该命令要求帧号从 1 开始且连续，编码完成后会重新解码校验帧数、FPS、分辨率和抽样图像内容。输出为：

```text
data/raw/aligned_videos/<video_id>.mp4
outputs/preprocessing/aligned_video_report.json
```

如需先验证单个视频，可执行：

```bash
python -m tools.preprocess.build_aligned_videos --video-id "1CM1_1_R_#217" --strict
```

第二步，根据官方划分、标签和刚生成的视频创建 manifest：

```bash
python -m tools.preprocess.build_manifests
```

输出为：

```text
data/manifests/train.json
data/manifests/test.json
outputs/preprocessing/dataset_audit.txt
```

脚本只有在 200 个视频与官方列表一一对应、Train/Test 无交叉、标签区间不越界、视频 FPS/帧数通过审计后才会写入正式 manifest。预期统计为 Train 148 个视频/3,117 段，Test 52 个视频/1,101 段。

第三步，使用仓库中的 `assets/hand_landmarker.task` 从对齐视频提取逐帧骨架：

```bash
python -m tools.preprocess.extract_skeleton --split train --fail-fast
python -m tools.preprocess.extract_skeleton --split test --fail-fast
```

输出为：

```text
data/skeleton_raw/train/<video_id>.npz
data/skeleton_raw/test/<video_id>.npz
outputs/preprocessing/skeleton_extraction_train.json
outputs/preprocessing/skeleton_extraction_test.json
```

骨架提取耗时最长。三个预处理工具都支持断点续跑：再次执行时会校验并复用已完成且配置一致的文件；只有希望强制重建时才使用 `--overwrite`。也可以通过重复传入 `--video-id` 只处理指定视频。不要在改变 MediaPipe 模型或置信度阈值后混用旧 NPZ，提取器会利用模型 SHA-256 和配置快照识别不一致产物。

### 为什么使用 `aligned_videos`

`data/raw/aligned_videos` **不是 IPN Hand 官方直接提供的一份额外视频包**，而是本项目由 `data/raw/frames` 本地生成的派生数据。这样做的直接原因是：

1. `data/raw/frames` 的逐帧序列与 `data/raw/annotations` 共用可靠的标注时间轴；
2. 某些非官方下载来源或自行转码得到的视频可能出现重复帧、漏帧或封装差异，不能假设其解码帧索引与 annotation 严格一一对应；
3. 如果直接使用未经审计的外部视频训练、评测或提取骨架，手势起止标签可能发生时移；
4. 因此本项目将 `frames` 作为时序真值来源，再构建可以被 OpenCV 连续解码的 MP4。

本项目生成 `aligned_videos` 时采用以下约束：

- 根据文件名末尾的数字对 JPG 进行数值排序，而不是字符串排序；
- 序列必须从 1 开始且连续，不允许缺号或重复编号；
- 每张源图只写入一次，不插帧、不重复、不裁剪、不改变顺序；
- 固定以 30 FPS、`mp4v` 编码，保持源图 320×240 分辨率；
- 编码后重新解码，检查帧数、FPS 和抽样图像内容；
- 检查每个视频的最后一个 annotation 没有超出源帧序列。

本地审计结果为 200/200 个 MP4 通过上述检查。以 `1CM1_1_R_#217` 为例，`frames` 中的 3,854 张连续 JPG 被写成 3,854 帧、30 FPS、320×240 的 MP4，最后一个标签帧仍被完整覆盖。

> 重要：凡是涉及 IPN Hand 标签的训练、离线评测、骨架提取和 `--video-id` Demo，都必须使用由官方 `frames` 构建并审计过的 `aligned_videos`；本项目的数据链不读取 `raw/videos`。

### `manifests` 如何生成

`data/manifests/train.json` 和 `test.json` 是本项目的本地索引文件，来源为：

- `Video_TrainList.txt` / `Video_TestList.txt`：确定官方视频划分；
- `Annot_TrainList.txt` / `Annot_TestList.txt`：提供手势类别和起止帧；
- `classIdx.txt`：提供官方类别编号与代码；
- `metadata.csv`：不写入 manifest JSON；加载训练数据时再按 `video_id` 关联左右手、场景和光照等 metadata；
- `aligned_videos`：提供经过核验的视频路径、FPS 和实际可解码帧数。

每条 manifest 记录包括 `video_id`、`split`、`video_path`、`fps`、`num_frames` 和 `segments`。JSON 中保留标签的 inclusive `start/end`；加载时由 `datasets/annotations.py` 统一转换为内部左闭右开区间，避免终点帧的 off-by-one 错误。项目只使用官方 Train/Test，不创建独立 validation。

### `skeleton_raw` 如何生成

`data/skeleton_raw/{train,test}/<video_id>.npz` 是从 `aligned_videos` 逐帧运行 MediaPipe Hand Landmarker 得到的本地派生骨架，不是 IPN Hand 官方文件。提取使用与 Demo 相同的 `assets/hand_landmarker.task`，VIDEO mode、单手和 0.5 的 detection/presence/tracking 阈值。

每个 NPZ 包含：

```text
image_landmarks   [T, 21, 3] float32
world_landmarks   [T, 21, 3] float32
valid_mask        [T]        bool
handedness        [T]
handedness_score  [T]        float32
frame_index       [T]        int64
timestamp_ms      [T]        int64
fps / width / height / video_id / split
model_sha256 / extraction_config_json / timing metadata
```

没有检测到手的帧仍占据一个时间位置，landmarks 置零且 `valid_mask=False`，因此骨架流、对齐视频和 annotation 始终保持相同时间轴。最终模型训练读取该目录；端到端 Demo 则从 RGB 输入实时执行同样的 MediaPipe 提取过程。

## 训练与官方 Test 复现

完整训练采用两阶段协议，配置分别位于 `configs/train_backbone.yaml` 和 `configs/train.yaml`：

1. 在官方 Train 上训练 80 epoch 的 causal TCN + Frame/Boundary Head；
2. 冻结整个 backbone 和 Frame/Boundary Head，再训练 20 epoch 的 Event Query + 256-frame Frame Memory；
3. 每一轮都可以在官方 Test 上做稳定性监控，但 checkpoint 固定使用最后一轮，Test 指标不参与权重筛选、早停或超参数选择；
4. 项目不额外拆分 validation，所有结论都应如实标注 Test 已在研发过程中被观察，不能声称它是未接触的 blind holdout。

### 1. 训练 causal backbone

先确认 `data/manifests/{train,test}.json` 和 `data/skeleton_raw/{train,test}` 已全部生成，然后执行：

```bash
python -m tools.train_backbone --run-id backbone_seed1234 --device cuda
```

没有 CUDA 时将 `cuda` 改为 `cpu`。首次运行会在 `outputs/cache/continuous/<signature>/` 生成可复用的 135 维 P3 特征与 dense frame/start/end targets；也可以只准备缓存：

```bash
python -m tools.train_backbone --prepare-only --device cpu
```

正式 backbone 训练输出为：

```text
outputs/training/backbone/backbone_seed1234/
├── checkpoints/last.pt
├── config_snapshot.yaml
├── training_log.json
├── metrics.json
├── predictions/<video_id>.json
└── baseline_report.md
```

`training_log.json` 是一份持续原子更新的 JSON，其中每个 epoch 同时包含 `train` 与 `test_monitor`；中断后使用完全相同的命令并追加 `--resume` 即可从 `last.pt` 继续。

### 2. 训练最终 Memory-Query 模型

使用第一阶段最后一轮 checkpoint：

```bash
python -m tools.train \
  --backbone-checkpoint outputs/training/backbone/backbone_seed1234/checkpoints/last.pt \
  --run-id mqtcn_seed1234 \
  --device cuda
```

Windows CMD 不支持反斜杠续行，可将命令写成一行。训练器会执行以下复现约束：

- 关闭 CUDA TF32，使用 FP32；
- 校验 P3 输入维数为 135；
- 将 backbone 输出缓存到 `outputs/cache/memory_query/<signature>/`；
- 只优化 Event Query 与 Frame Memory 参数，并在每轮保存前验证冻结 backbone 的 tensor digest 没有变化；
- 使用固定第 20 epoch，而不是依据 Test 曲线挑选 checkpoint。

输出为：

```text
outputs/training/mqtcn/mqtcn_seed1234/
├── checkpoints/last.pt
├── config_snapshot.yaml
├── training_log.json
├── metrics.json
├── predictions/<video_id>.json
└── memory_query_report.md
```

### 3. 独立运行官方 Test

对刚训练出的 checkpoint 运行完整 52-video Official Test：

```bash
python -m tools.evaluate \
  --checkpoint outputs/training/mqtcn/mqtcn_seed1234/checkpoints/last.pt \
  --device cuda \
  --output-dir outputs/evaluation/mqtcn_seed1234
```

如果只想复核仓库随附的发布权重：

```bash
python -m tools.evaluate \
  --checkpoint outputs/model/checkpoints/final.pt \
  --device cuda \
  --output-dir outputs/evaluation/released_model
```

评测输出包含 `metrics.json` 和 `predictions/<video_id>.json`。主要报告 `fusion` 路径的 Levenshtein Accuracy、Event F1@0.3/0.5/0.7、Precision/Recall、FP/min、边界误差与完成时延；同时保留 `frame_only` 和 `query_only` 结果便于诊断。该评测针对完整连续视频，不是按 GT 裁剪的孤立手势分类。

训练得到的 checkpoint 也可以直接传给端到端 Demo。显式指定 `--checkpoint` 时，Demo 不再要求它等于仓库随附发布权重的固定哈希，但仍会严格校验 state dict 与当前模型结构：

```bash
python tools/realtime_demo.py --video "path/to/video.mp4" --checkpoint outputs/training/mqtcn/mqtcn_seed1234/checkpoints/last.pt --device cuda
```

### 4. 结构 smoke test

下面的命令仅用于快速检查环境、反向传播和文件输出，不可作为正式实验指标：

```bash
python -m tools.train_backbone --run-id smoke_backbone --device cuda --smoke
python -m tools.train --backbone-checkpoint outputs/training/backbone/smoke_backbone/checkpoints/last.pt --run-id smoke_mqtcn --device cuda --smoke
python -m tools.evaluate --checkpoint outputs/model/checkpoints/final.pt --device cuda --limit-videos 2 --output-dir outputs/evaluation/smoke
```

`--smoke` 将训练缩减为 2 epoch 和少量样本；`--limit-videos` 会把协议标记为 `test_subset_smoke`，不能与 README 中的 52-video 正式结果比较。

## 项目结构

```text
GestureDetection-MQTCN/
├── assets/
│   └── hand_landmarker.task        # MediaPipe 官方 Hand Landmarker 模型资产
├── configs/
│   ├── runtime.yaml                # 发布 checkpoint 的冻结运行配置
│   ├── train_backbone.yaml         # 80-epoch causal backbone 协议
│   └── train.yaml                  # 20-epoch Memory-Query 协议
├── datasets/
│   ├── annotations.py              # inclusive/half-open 标签区间契约
│   ├── ipn_manifest.py             # IPN manifest 与类别读取
│   ├── ipn_skeleton.py             # NPZ 骨架流校验与加载
│   ├── continuous.py               # dense targets、P3 cache 与训练 clips
│   └── memory_query.py             # frozen embedding cache 与 Query batch
├── engine/
│   ├── backbone_trainer.py         # backbone loss 与训练循环
│   ├── backbone_evaluator.py       # causal full-video 评测
│   ├── model_trainer.py            # Event Query matching loss 与训练循环
│   ├── model_evaluator.py          # Frame/Query/Fusion 完整评测
│   └── json_logger.py              # Train/Test 同文件 JSON 日志
├── evaluation/                     # 分类、事件、编辑距离与连续识别指标
├── models/
│   ├── gesture_detection_mqtcn.py  # 最终模型组合入口
│   ├── frame_encoder.py            # 逐帧特征投影
│   ├── causal_tcn.py               # 因果残差 TCN
│   ├── streaming_tcn.py            # 有状态逐帧 TCN
│   ├── frame_head.py               # Frame/Boundary 输出头
│   ├── memory/frame_memory.py      # 256 帧有限记忆
│   └── query/                      # Event Query、target、matching 与边界预测
├── preprocessing/
│   ├── feature_builder.py          # 离线训练用 135 维 P3 特征
│   ├── p3_features.py              # 在线等价的增量 P3 特征
│   └── augmentation.py             # 训练期骨架特征增强
├── streaming/
│   ├── realtime_pipeline.py        # 端到端状态与丢手重置
│   ├── model_runtime.py            # TCN、Memory、Query 的在线执行
│   ├── decoder.py                  # Frame 连续事件解码
│   ├── event_tracker.py            # Frame 事件跟踪与去重
│   └── query_tracker.py            # Query 解码与 Frame/Query Fusion
├── tools/
│   ├── preprocess/
│   │   ├── build_aligned_videos.py # 官方 frames → aligned MP4
│   │   ├── build_manifests.py      # 官方 annotation → Train/Test manifest
│   │   └── extract_skeleton.py     # aligned MP4 → MediaPipe NPZ
│   ├── train_backbone.py           # 第一阶段训练入口
│   ├── train.py                    # 最终 MQ-TCN 训练入口
│   ├── evaluate.py                 # Official Test 离线评测入口
│   └── realtime_demo.py            # 视频/摄像头统一 CLI 与界面
├── tests/
│   ├── test_realtime_pipeline.py   # 最终运行时回归测试
│   └── test_training_pipeline.py   # 预处理等价性与反向传播测试
├── outputs/
│   ├── model/checkpoints/final.pt  # 最终冻结权重，由完整训练实验导出
│   ├── demo/video_validation.json  # 300 帧端到端验证样例
│   ├── cache/                      # 本地训练缓存，不上传
│   ├── training/                   # 本地训练日志与 checkpoint
│   └── evaluation/                 # 本地 Test metrics 与逐视频预测
├── data/                            # 本地 IPN Hand 与派生数据，不上传
├── requirements.txt
├── pytest.ini
└── README.md
```

`data/` 与 `backup/` 出现在结构图中是为了解释完整研发工程的来源关系；从 GitHub 克隆时看不到这两个目录属于正常现象。与数据重建、最终训练、评测和 Demo 有关的代码已经保留在公开目录中，不依赖 `backup/`。

## 测试

```bash
python -m pytest tests -q
```

当前最终回归测试覆盖：

- Query stride 的在线执行节奏；
- 1–5 帧短时丢手保持与第 6 帧长丢手原子重置；
- 长丢手重置时已确认事件的归档，以及显式 source reset 时的清空；
- 最终 checkpoint 可脱离训练代码加载；
- checkpoint 与 MediaPipe 模型哈希、模型参数量和 135 维特征契约。
- 离线/在线 P3 特征的数值等价性；
- 官方 annotation/video-list 解析；
- 最终 Frame Memory + Event Query 路径的前向、matching loss、反向传播及 backbone 冻结。

## 使用边界

- 这是连续手势识别模型，不是已经裁好边界的孤立手势分类器；
- 最终类别空间固定为 IPN Hand 的 13 种手势，其他动作通常会被视为背景，但域外场景仍可能产生误报；
- 单手检测和左右方向语义依赖未镜像的模型输入；摄像头预览可镜像，推理输入不应随意镜像；
- 历史 B0–B5 消融调度与中间 checkpoint 不包含在公开版中，但最终模型的预处理、训练、Official Test 评测和 Demo 均可独立执行；
- 官方输入只有 `data/raw/annotations` 与 `data/raw/frames`，涉及标签的工作必须使用由 frames 构建并审计过的 `aligned_videos`。

## IPN Hand 引用

数据集主页：[The IPN Hand Dataset](https://gibranbenitez.github.io/IPN_Hand/)  
论文：[IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition](https://arxiv.org/abs/2005.02134)  
官方代码：[GibranBenitez/IPN-hand](https://github.com/GibranBenitez/IPN-hand)

如果在研究中使用 IPN Hand，请引用原论文：

```bibtex
@inproceedings{bega2020ipnhand,
  title        = {IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
  author       = {Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
  booktitle    = {25th International Conference on Pattern Recognition (ICPR)},
  pages        = {4340--4347},
  year         = {2021},
  organization = {IEEE}
}
```

IPN Hand 数据和 annotations 采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；本仓库不重新分发数据。项目公开发布前还应在仓库根目录添加适用于本项目代码与模型权重的 `LICENSE` 文件。
