# DiffusionDrive 迁移 · 修改 1

本轮针对 `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py` 做了一次小幅修订，并完成了 head 层面的可运行性验证。

---

## 1. 代码修改

**文件**：`projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py`

### 1.1 `LossComputer.forward` 增加可选的 `target_mask`

原版（移植自 DiffusionDrive `multimodal_loss.py`）的 anchor matching 用 `dist.mean(dim=-1)`、reg loss 用 `F.l1_loss(reduction='mean')`，**不考虑 `sdc_planning_mask`**。NavSim 训练集上 8 步未来基本全有效，原作者直接忽略了 mask，但 WorldEngine 数据 pipeline 是把 mask 显式传进来的，沿用 `TrajScoringHead` 风格更稳健。

改动要点：
- `forward(...)` 新增 `target_mask=None` 参数，形状 `(B, T)`，1 表示有效。
- Anchor matching：`dist` 按 `target_mask` 加权求平均，分母为 `target_mask.sum(dim=-1).clamp_min(1.0)`。
- Reg loss：用 masked sum / (masked_count × dim) 替代 `F.l1_loss`，避免无效步把 L1 拉高。
- `target_mask is None` 时数值等价于原版（全 1 mask → 普通均值），**默认行为不变**。

### 1.2 `DiffusionPlanningHead.loss` 透传 mask

- 从 `sdc_planning_mask[:, 0, :, 0]` 取出 `(B, T)` mask（与 `traj_scoring_head.py:255` 一致）。
- 若 GT 是 40 步（兼容路径），mask 同步做 `[:, 4::5]` 切片。
- 每层 DiT 的 `LossComputer(...)` 调用都把 mask 传下去。

---

## 2. 验证

### 2.1 Config 解析

`mmcv.Config.fromfile('projects/AlgEngine/configs/navformer/e2e_diffusiondrive.py')` 解析通过，`planning_head` 字段全部正确读出。

### 2.2 Head smoke test（真实资产）

环境：torch 2.0.1+cu118、mmcv 1.6.2、mmdet 2.25.3、diffusers 已装。

- 真实 anchor `data/alg_engine/kmeans_navsim_traj_20.npy` 形状 `(20, 8, 2)`。
- 真实 vocab `data/alg_engine/test_8192_kmeans.npy` 形状 `(8192, 40, 3)`。
- `HEADS.build(cfg.model.planning_head)` 构建成功，head 参数量 **9.13 M**。

| 检查项 | 结果 |
|---|---|
| Train forward 输出 keys/形状 | ✅ `trajectory (B,40,3)`、`selected_indices (B,)`、`trajectory_8 (B,8,3)`、`poses_reg_list`、`poses_cls_list`、`plan_anchor_expanded (B,20,8,2)` |
| Train loss（真实 anchor，随机 GT） | ✅ `loss.diff_cls=0.86`、`loss.diff_reg=22.4`，非 NaN |
| `backward()` 后非零梯度参数数 | ✅ **152 / 152**（所有子模块都收到梯度，包括 cross_bev / cross_agent / cross_ego / time_modulation / task_decoder / plan_anchor_encoder / time_mlp / status_embed / query_embedding / keyval_embedding / query_tf_decoder） |
| Test forward（DDIM 2 步） | ✅ `selected_indices` 落在 `[0, 8192)` 内 |
| NavFormer 评估契约 `trajectory[:, 4::5] == trajectory_8` | ✅ 完全相等，detector `forward_test` 的 `[4::5]` 切片能拿回真实 8 步 |
| 40 步 GT + 末 4 步 mask 边界 | ✅ `loss.diff_reg` 从 15.09 → 9.51，mask 正确生效 |

---

## 3. 已知风险（送外部机器跑训练时观察）

以下两点属于**只有真实数据才能验证**的语义正确性问题，不是本地能修的 bug：

1. **BEV 坐标轴方向**：`GridSampleCrossBEVAttention` 内部 `normalized_trajectory = normalized_trajectory[..., [1, 0]]` 沿用 DiffusionDrive 原版。NavFormer 的 BEV 200×200 与 DiffusionDrive backbone 输出可能存在 90° / 镜像差。
   - **观察方法**：训练若干 step 后看 `out['trajectory'][..., 0].mean()` 是否为正（车辆前进方向应为 +x）。若分数明显低于 VADv2，先怀疑这一行。

2. **`norm_odo_xy` 硬编码范围**：`(x∈[-1.2, 55.7], y∈[-20, 26])` 来自 DiffusionDrive 在 NavSim 训练集的统计。要求 `data/alg_engine/kmeans_navsim_traj_20.npy` 必须出自相同分布的 NavSim 训练集 K-means。
   - **确认方法**：核对 anchor `.npy` 的来源；若分布偏离，加噪去噪空间会与 anchor 不一致，cls loss 会震荡。

---

## 4. 外部机器训练前的依赖

- `pip install diffusers>=0.21`
- 设置环境变量 `WORLDENGINE_ROOT` 指向项目根，否则 config 找不到 anchor / vocab。
- 训练入口与启动命令复用 `e2e_vadv2.py` 的，仅替换 config 路径为 `projects/AlgEngine/configs/navformer/e2e_diffusiondrive.py`。
