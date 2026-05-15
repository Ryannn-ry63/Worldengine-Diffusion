# 迁移 DiffusionDrive 到 WorldEngine NavFormer 框架

## Context

WorldEngine 后训练框架已经验证了 HydraMDP（`e2e_hydramdp.py` + `traj_scoring_head.py`）和 VADv2（`e2e_vadv2.py` + 同一个 `TrajScoringHead`）两种规划算法。现在要迁移第三个算法 **DiffusionDrive** 的 planning head（DiT-style 扩散解码器），按照前两者的范式：

- 新增 `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py`
- 新增 `projects/AlgEngine/configs/navformer/e2e_diffusiondrive.py`
- 在 `dense_heads/__init__.py` 中注册导出

源代码位于 `DiffusionDrive-main/`，**该目录最终会被删除**，所以新 head 必须**完全 self-contained**——把 DiffusionDrive 用到的所有子模块直接 inline 进新文件，**不要 import 任何 `navsim.*`**。

**核心原则（来自用户反馈）：尽量保持 DiffusionDrive 原样**——`CustomTransformerDecoderLayer` 中的 `cross_bev_attention / cross_agent_attention / cross_ego_attention / modulation` 全部保留，不做结构性删除。

---

## 关键发现（来自源码核对）

### 1) NavFormer 的 `planning_head.forward` 签名（`navformer.py:723-730 / 800-807`）

```python
forward(bev_embed, command, sdc_planning_past, sdc_status,
        sdc_planning_mask_past, gt_pre_command_sdc)
```

- **没有** GT 未来轨迹（GT 只在 `loss()` 阶段才传入 `sdc_planning`）
- **没有** agent queries / sdc queries（detector 在 test path 下根本不跑 detection head，`forward_track_test` 只返回 `bev_embed` 和 `bev_pos`，见 `navformer.py:899-903`）

因此，扩散训练的"加噪→去噪→匹配 GT"必须**拆成两步**：
- `forward()` 跑加噪 + 堆叠解码器，收集 `poses_reg_list / poses_cls_list`
- `loss()` 拿到 GT 后再做 anchor-matching + focal cls + L1 reg

### 2) BEV feature 形状

NavFormer 的 `bev_embed` 是 `(H*W, B, C) = (40000, B, 256)`（BEVFormer 输出），DiffusionDrive 的 `GridSampleCrossBEVAttention` 期望 `(B, C, H, W)`。需要在 head 入口 reshape：

```python
B = bev_embed.shape[1]
bev_feature = bev_embed.permute(1, 2, 0).contiguous().view(B, self.d_model, self.bev_h, self.bev_w)
```

`bev_h = bev_w = 200`，从 config 传入。

### 3) **agent queries / ego query 的来源（关键设计点，已按用户反馈修订）**

**保留原版 DiT 块结构不变**，把 DiffusionDrive 原本在 `V2TransfuserModel.forward` 里做的"query 准备"步骤 inline 到新 head 内部：

| DiffusionDrive 原作（`transfuser_model_v2.py`） | 在新 head 里的实现 |
|---|---|
| `_query_embedding = nn.Embedding(1 + 30, 256)` (line 37) | 同样 `nn.Embedding(num_bounding_boxes+1, d_model)`，作为 head 的可学习参数 |
| `_tf_decoder = nn.TransformerDecoder(layer, 3)` (line 76) | 同样 inline 一个 3 层 `nn.TransformerDecoder`，**仅作为 query→keyval 的抽取器** |
| `keyval = [bev_feature_flatten, status_encoding]` (line 112) | 同样：flatten BEV `(B, H*W, C)` + 拼接 status_encoding token |
| `bev_proj`：upscale 后融合两路 BEV (line 90-92, 115-123) | **简化掉**：NavFormer 输出的 `bev_embed` 已经是 200×200 高分辨率，不需要 V2 那个 upscale 拼接逻辑。直接用 `bev_embed` 当 `cross_bev_feature`，也用同一份 reshape 后的 `(B, H*W, C)` 当 keyval |
| `query_out.split([1, 30])` → `(ego_query, agents_query)` (line 128) | 完全一样 |

这样换来的好处：
- ✅ DiT 块**结构与论文完全一致**：`cross_bev_attention + cross_agent_attention + cross_ego_attention + modulation` 全部保留
- ✅ `ego_query` 是真正从 BEV 经 TF decoder 抽出来的（与论文一致），不是从 status 硬编码
- ✅ detector 端**零改动**，调用契约和 HydraMDP/VADv2 完全一致
- ✅ 没有破坏性删除 `cross_agent_attention`

### 4) `status_encoding` 仍然需要

DiT 层的 `cross_ego_attention` 用 `ego_query`（已由 TF decoder 抽出），但 `status_encoding` 还会作为 keyval 的一个额外 token（沿用 V2 原作做法 `keyval = [bev_tokens, status_token]`）。

构造方法沿用现有 `TrajScoringHead` 的 NeRF + max-mask-pool 模式（`traj_scoring_head.py:213-260`）：把 `command / sdc_planning_past / sdc_status / sdc_planning_mask_past / gt_pre_command_sdc` 编码成一个 `(B, 1, 256)` token。

### 5) BEV 坐标 & 轨迹归一化范围

- `GridSampleCrossBEVAttention`：原 V2 用 `lidar_max_x = lidar_max_y = 32`，对应 BEV 半径 32m。NavFormer 的 `point_cloud_range = [-51.2, ..., 51.2]`，**BEV 半径 51.2m**。新 head 的 config 把 `bev_range_x = bev_range_y = 51.2` 传进去，替换原代码中的 `self.config.lidar_max_x/y`。
- `norm_odo` / `denorm_odo` 的轨迹分布范围（`x∈[-1.2, 55.7]`, `y∈[-20, 26]`, `head∈[-2, 1.9]`）：WorldEngine 用同样的 OpenScene/NAVSIM 数据，沿用即可，但作为 ctor 可配置参数暴露出来。

### 6) `selected_indices` / `trajectory` 输出格式兼容

NavFormer 的 `forward_test`（`navformer.py:809-836`）做了两件事：
- `chosen_indices = plan_results['selected_indices']` → 用来索引 PDM 评分张量（形状 `(B, 8192)`，对应预计算的 8192-vocab `test_8192_kmeans.npy`）
- `pred_traj = pdm_dict['trajectory'][4::5, :2]` → 期望 `trajectory` 是 `(40, 3)` 形状（40-step），切片得 8 步

DiffusionDrive 原生输出 20-anchor × 8-step 轨迹，两者都不直接兼容。**兼容方案**：
- Head **同时加载** 8192-vocab（`test_8192_kmeans.npy`），**仅用于 test 时的"快照锚定"**——把扩散输出的 8 步轨迹 `(x,y)` 最近邻锚定到 8192 个 vocab 之一，返回该 index 作为 `selected_indices`（让 PDM 评估流程不报错）
- `trajectory` 返回扩散预测的轨迹，**沿时间维做 5 倍重复**到 `(B, 40, 3)`，使 `[4::5]` 切片恰好取回真实的 8 个 waypoint（5×8=40，索引 4,9,14,...,39 对应原 waypoint 0..7）

### 7) Loss key 命名约定

现有 head 用 `loss.imi`, `loss.noc` 等。`navformer.py:750-751` 会 `losses.update(track_losses)`，所以新 key 不能撞 tracker 命名。**用 `loss.diff_cls`, `loss.diff_reg`**，分开便于 TensorBoard 监控。

---

## 实现

### 新文件 1: `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py`

完全 self-contained，包含以下 inline 实现（移植自 `DiffusionDrive-main/navsim/agents/diffusiondrive/`）：

| 内联类 / 函数 | 移植自 |
|---|---|
| `SinusoidalPosEmb` | `modules/conditional_unet1d.py:44-56` |
| `linear_relu_ln`, `bias_init_with_prob`, `gen_sineembed_for_position` | `modules/blocks.py:8-39` |
| `GridSampleCrossBEVAttention` | `modules/blocks.py:42-109` —— 改 `self.config.lidar_max_x/y` 为 ctor 传入 `bev_range_x/y` |
| `ModulationLayer` | `transfuser_model_v2.py:229-268`（不变） |
| `DiffMotionPlanningRefinementModule` | `transfuser_model_v2.py:182-228` —— `StateSE2Index.HEADING` 替换为字面量 `2` |
| `CustomTransformerDecoderLayer`（**保留全部三种 cross-attn**） | `transfuser_model_v2.py:270-344`，**结构原样保留** |
| `CustomTransformerDecoder` | `transfuser_model_v2.py:350-380`，**结构原样保留** |
| `py_sigmoid_focal_loss`, `weight_reduce_loss`, `reduce_loss`, `LossComputer` | `modules/multimodal_loss.py:11-163` |
| `nerf_positional_encoding` | 沿用 `traj_scoring_head.py:14-56`（用于 status 编码） |

**外部依赖**：`from diffusers.schedulers import DDIMScheduler`（环境由用户自行安装）。

**主类 `DiffusionPlanningHead(nn.Module)`** —— `@HEADS.register_module()`，参考 `traj_scoring_head.py` 的装饰器风格使用 `@auto_fp16` / `@force_fp32`。

**构造参数**：

```python
def __init__(
    self,
    num_poses: int = 8,                # = planning_steps
    d_model: int = 256,
    d_ffn: int = 1024,
    num_heads: int = 8,
    dropout: float = 0.0,
    num_bounding_boxes: int = 30,       # 与 DiffusionDrive 原作一致：30 个 agent token
    num_query_decoder_layers: int = 3,  # 抽 query 用的 TF decoder 层数（V2 原作=3）
    num_anchors: int = 20,              # 扩散 anchor 个数（V2 原作=20）
    num_diff_decoder_layers: int = 2,   # DiT 块层数（V2 原作=2）
    plan_anchor_path: str = ...,        # (20, 8, 2) .npy（用户提供）
    vocab_path: str = ...,              # (8192, 40, 3) .npy（已存在）
    bev_h: int = 200,
    bev_w: int = 200,
    bev_range_x: float = 51.2,          # grid_sample 归一化半径
    bev_range_y: float = 51.2,
    odo_x_min: float = -1.2,            # norm_odo 范围（V2 原作硬编码值）
    odo_x_range: float = 56.9,
    odo_y_min: float = -20.0,
    odo_y_range: float = 46.0,
    odo_h_min: float = -2.0,
    odo_h_range: float = 3.9,
    num_train_timesteps: int = 1000,
    train_timestep_max: int = 50,
    inference_steps: int = 2,
    trunc_timesteps: int = 8,
    cls_loss_weight: float = 10.0,
    reg_loss_weight: float = 8.0,
    use_nerf: bool = True,
    **kwargs,
):
```

**内部模块**（构造时初始化）：

```python
# (a) status token —— 沿用 TrajScoringHead 风格
self.status_embed = nn.Sequential(nn.Linear(4+24+2, d_model), nn.ReLU())  # use_nerf=True 时
# (或 4+2+2→d_model，use_nerf=False 时)

# (b) 30 agent + 1 ego learnable query（V2 原作 line 37）
self._query_embedding = nn.Embedding(num_bounding_boxes + 1, d_model)

# (c) 用于把 BEV(+status) 抽成 query 的 TF decoder（V2 原作 line 68-76）
query_decoder_layer = nn.TransformerDecoderLayer(
    d_model=d_model, nhead=num_heads, dim_feedforward=d_ffn,
    dropout=dropout, batch_first=True,
)
self._query_tf_decoder = nn.TransformerDecoder(query_decoder_layer, num_query_decoder_layers)

# (d) BEV → keyval 用的位置编码(V2 原作 line 36)
self._keyval_embedding = nn.Embedding(bev_h * bev_w + 1, d_model)

# (e) 扩散调度器
self.diffusion_scheduler = DDIMScheduler(
    num_train_timesteps=num_train_timesteps,
    beta_schedule="scaled_linear", prediction_type="sample",
)

# (f) 扩散锚点（20, 8, 2）
self.plan_anchor = nn.Parameter(torch.from_numpy(np.load(plan_anchor_path)).float(), requires_grad=False)
self.plan_anchor_encoder = nn.Sequential(*linear_relu_ln(d_model, 1, 1, 512), nn.Linear(d_model, d_model))

# (g) time embedding(V2 原作 line 417-422)
self.time_mlp = nn.Sequential(SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model*4), nn.Mish(), nn.Linear(d_model*4, d_model))

# (h) DiT 块（原样保留三种 cross-attn）
self.diff_decoder = CustomTransformerDecoder(
    CustomTransformerDecoderLayer(num_poses, d_model, d_ffn, num_heads, dropout, bev_range_x, bev_range_y, num_anchors),
    num_diff_decoder_layers,
)

# (i) 8192-vocab 用于 test-time 快照
self.vocab_8192 = nn.Parameter(torch.from_numpy(np.load(vocab_path)).float(), requires_grad=False)  # (8192, 40, 3)

# (j) loss computer
self.loss_computer = LossComputer(cls_loss_weight, reg_loss_weight)
```

**核心方法**：

1. `_build_status_token(command, sdc_planning_past, sdc_status, sdc_planning_mask_past, gt_pre_command_sdc) -> (B, 1, 256)`
   —— 复用 `traj_scoring_head.py:213-260` 的 NeRF + max-mask-pool 逻辑，输出 unsqueeze(1) 后的 `(B, 1, 256)` token。

2. `_prepare_queries(bev_feature_flat, status_token) -> (ego_query, agents_query)`
   —— **完全复制 V2 原作 line 110-128 的逻辑**：
   ```python
   keyval = torch.cat([bev_feature_flat, status_token], dim=1)      # (B, H*W+1, C)
   keyval = keyval + self._keyval_embedding.weight[None]
   query = self._query_embedding.weight[None].repeat(B, 1, 1)        # (B, 31, C)
   query_out = self._query_tf_decoder(query, keyval)
   ego_query, agents_query = query_out.split([1, self.num_bounding_boxes], dim=1)
   ```

3. `forward(bev_embed, command, sdc_planning_past, sdc_status, sdc_planning_mask_past, gt_pre_command_sdc)`
   - reshape `bev_embed (HW, B, C) → bev_feature (B, C, H, W)` 和 `bev_feature_flat (B, HW, C)`
   - 构造 `status_token`
   - `ego_query, agents_query = _prepare_queries(bev_feature_flat, status_token)`
   - 根据 `self.training` 分派到 `_forward_train` 或 `_forward_test`，传入 `(bev_feature, ego_query, agents_query, status_token)`

4. `_forward_train(bev_feature, ego_query, agents_query, status_token) -> dict`
   - `plan_anchor.unsqueeze(0).repeat(B, 1, 1, 1)` → `(B, 20, 8, 2)`
   - `norm_odo` → 采样 `t ~ U[0, train_timestep_max)` → `scheduler.add_noise` → `clamp` → `denorm_odo` → `noisy_traj_points`
   - `gen_sineembed_for_position(noisy_traj_points, hidden_dim=64).flatten(-2)` → `plan_anchor_encoder` → `traj_feature (B, 20, 256)`
   - `time_embed = time_mlp(t).view(B, 1, -1)`
   - `poses_reg_list, poses_cls_list = diff_decoder(traj_feature, noisy_traj_points, bev_feature, (H,W), agents_query, ego_query, time_embed, status_token)`
   - 最后一层 `argmax(poses_cls)` → gather → `best_reg_8 (B, 8, 3)`
   - **`selected_indices`**：把 `best_reg_8[..., :2]` 与 `vocab_8192[None, :, 4::5, :2]` 计算 L2 距离 → `argmin` → `(B,)`
   - **`trajectory`**：`best_reg_8.unsqueeze(2).expand(-1, -1, 5, -1).reshape(B, 40, 3)`（时间维 5x 重复）
   - 返回：

     ```python
     {
       'trajectory': trajectory_40,           # (B, 40, 3) for detector
       'selected_indices': selected_indices,  # (B,) into 8192 vocab
       'trajectory_8': best_reg_8,            # (B, 8, 3) 原扩散输出
       # 给 loss() 用的中间量：
       'poses_reg_list': poses_reg_list,      # list[layer] of (B, 20, 8, 3)
       'poses_cls_list': poses_cls_list,      # list[layer] of (B, 20)
       'plan_anchor_expanded': plan_anchor_B, # (B, 20, 8, 2)
     }
     ```

5. `_forward_test(bev_feature, ego_query, agents_query, status_token) -> dict`
   - DDIM 采样 loop（移植 `transfuser_model_v2.py:504-558`）
   - 最后做 8192-vocab 快照 + 5x 重复
   - 返回 `{trajectory, selected_indices, trajectory_8}`

6. `@force_fp32 loss(result, gt_pdm_score=None, sdc_planning=None, sdc_planning_mask=None, il_target=None, il_target_mask=None)`
   - `target_traj = sdc_planning[:, 0]`（`(B, 8, 3)`）
   - 对每一层调用 `LossComputer(poses_reg, poses_cls, target_traj, plan_anchor_expanded)`，分别累加 `cls` 和 `reg`
   - 返回 `{'loss.diff_cls': cls_total, 'loss.diff_reg': reg_total}`（注：需稍微改造 `LossComputer.forward` 让它返回 `(cls_loss, reg_loss)` 而不是直接相加，便于分别 logging）

7. `norm_odo` / `denorm_odo` —— 用 ctor 参数化常数，替换 V2 硬编码值

**关于 `CustomTransformerDecoderLayer` 构造参数**：

V2 原作把 `config` 整体传入用来读 `tf_d_model / tf_num_head / tf_dropout / lidar_max_x/y`。inline 时改为接受显式参数：

```python
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, num_poses, d_model, d_ffn, num_heads, dropout,
                 bev_range_x, bev_range_y, num_anchors):
        super().__init__()
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            d_model, num_heads, num_points=num_poses,
            bev_range_x=bev_range_x, bev_range_y=bev_range_y, in_bev_dims=d_model,
        )
        self.cross_agent_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_ego_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        # ... ffn / norm1/2/3 / time_modulation / task_decoder 完全不变
```

`GridSampleCrossBEVAttention` 也同步改成接受 `bev_range_x/y` 显式参数（替换原 `self.config.lidar_max_x/y`）。

### 新文件 2: `projects/AlgEngine/configs/navformer/e2e_diffusiondrive.py`

基于 `e2e_vadv2.py` 整体复制，**只改 `planning_head` 字段**：

```python
planning_head=dict(
    type='DiffusionPlanningHead',
    num_poses=planning_steps,            # 8
    d_model=_dim_,                       # 256
    d_ffn=1024,                          # V2 原作 tf_d_ffn=1024
    num_heads=8,
    dropout=0.0,
    num_bounding_boxes=30,               # V2 原作一致
    num_query_decoder_layers=3,          # V2 原作 tf_num_layers=3
    num_anchors=20,
    num_diff_decoder_layers=2,
    plan_anchor_path=os.path.join(WORLDENGINE_ROOT, "data/alg_engine/kmeans_navsim_traj_20.npy"),
    vocab_path=os.path.join(WORLDENGINE_ROOT, "data/alg_engine/test_8192_kmeans.npy"),
    bev_h=bev_h_,                        # 200
    bev_w=bev_w_,                        # 200
    bev_range_x=51.2,
    bev_range_y=51.2,
    num_train_timesteps=1000,
    train_timestep_max=50,
    inference_steps=2,
    trunc_timesteps=8,
    cls_loss_weight=10.0,
    reg_loss_weight=8.0,
    use_nerf=True,
),
```

其它字段（dataset / pipeline / optimizer / lr_config / `load_from` 预训练 ckpt）**与 `e2e_vadv2.py` 完全保持一致**——data pipeline collect 的 keys 已经完全覆盖新 head 需要的所有输入。

### 修改文件: `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/__init__.py`

```python
from .traj_scoring_head import TrajScoringHead
from .traj_scoring_head_RL import TrajScoringHeadRL
from .diffusion_planning_head import DiffusionPlanningHead
```

仅 +1 行 import，触发 `@HEADS.register_module()`。

---

## 关键文件路径

**新建**：
- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py`（self-contained，无 `navsim` 依赖）
- `projects/AlgEngine/configs/navformer/e2e_diffusiondrive.py`

**修改**：
- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/__init__.py`（+1 行）

**参考（不修改）**：
- `projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py:723-836`（确认 forward / loss 调用契约——**detector 不需改动**）
- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/traj_scoring_head.py`（参考 NeRF status_embed + `@auto_fp16` / `@force_fp32` 装饰器风格）
- `DiffusionDrive-main/navsim/agents/diffusiondrive/transfuser_model_v2.py`（移植 V2 query splitter + DiT 块）
- `DiffusionDrive-main/navsim/agents/diffusiondrive/modules/{blocks,multimodal_loss,conditional_unet1d}.py`（移植辅助函数）

**用户负责准备的资产**：
- `data/alg_engine/kmeans_navsim_traj_20.npy`，shape `(20, 8, 2)` dtype float32（用户已说会放进来）
- `data/alg_engine/test_8192_kmeans.npy`，shape `(8192, 40, 3)`（已存在）
- 环境中安装 `diffusers >= 0.21`（用户自行 `pip install diffusers`）

---

## 验证步骤

按从轻到重的顺序，任何一步失败都先修复再往下走：

1. **配置解析 & 模型构建（不跑 forward）**

   ```bash
   cd F:/OneDrive/桌面/研0/WE1/WorldEngine-main
   python -c "
   import sys; sys.path.insert(0, 'projects/AlgEngine')
   import mmdet3d_plugin
   from mmcv import Config
   from mmdet.models import build_detector
   cfg = Config.fromfile('projects/AlgEngine/configs/navformer/e2e_diffusiondrive.py')
   model = build_detector(cfg.model, train_cfg=cfg.model.get('train_cfg'), test_cfg=cfg.model.get('test_cfg'))
   print('OK:', type(model.planning_head).__name__)
   "
   ```

2. **Head 单元前向 + loss 测试**（构造随机 tensor，不依赖整套数据集；用临时随机 .npy 替代真实 anchor 文件）

   ```python
   import torch, numpy as np, tempfile, os
   tmpdir = tempfile.mkdtemp()
   np.save(os.path.join(tmpdir, 'anchor.npy'), np.random.randn(20, 8, 2).astype(np.float32))
   np.save(os.path.join(tmpdir, 'vocab.npy'),  np.random.randn(8192, 40, 3).astype(np.float32))

   import mmdet3d_plugin
   from mmdet.models.builder import HEADS
   head = HEADS.build(dict(
       type='DiffusionPlanningHead',
       num_poses=8, d_model=256, d_ffn=1024, num_heads=8,
       plan_anchor_path=os.path.join(tmpdir, 'anchor.npy'),
       vocab_path=os.path.join(tmpdir, 'vocab.npy'),
       bev_h=200, bev_w=200,
   )).cuda()
   B = 2
   bev_embed = torch.randn(40000, B, 256).cuda()
   command = torch.zeros(B, dtype=torch.long).cuda()
   sdc_planning_past = torch.zeros(B, 1, 4, 3).cuda()
   sdc_status = torch.zeros(B, 3).cuda()
   sdc_planning_mask_past = torch.ones(B, 1, 4, 3).cuda()
   gt_pre_command_sdc = torch.zeros(B, 1, 4, 1, dtype=torch.long).cuda()

   head.train()
   out = head(bev_embed, command, sdc_planning_past, sdc_status, sdc_planning_mask_past, gt_pre_command_sdc)
   assert out['trajectory'].shape == (B, 40, 3)
   assert out['selected_indices'].shape == (B,)
   sdc_planning = torch.zeros(B, 1, 8, 3).cuda()
   sdc_planning_mask = torch.ones(B, 1, 8, 3).cuda()
   loss = head.loss(out, sdc_planning=sdc_planning, sdc_planning_mask=sdc_planning_mask)
   print('train loss:', loss)

   head.eval()
   out = head(bev_embed, command, sdc_planning_past, sdc_status, sdc_planning_mask_past, gt_pre_command_sdc)
   print('test out:', {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
   ```

3. **端到端少量迭代训练**

   用 WorldEngine 原本的训练入口（如 `scripts/dist_train.sh`），加 override 加速：
   ```
   --cfg-options total_epochs=1 data.train.load_interval=200
   ```
   预期：
   - 没有 shape mismatch
   - `loss.diff_cls` 和 `loss.diff_reg` 随 step 下降
   - 1 epoch 后能 dump checkpoint
   - eval 阶段 `forward_test` 返回 PDM 字典正常

4. **PDM 指标对比**

   在同一 split（`navtest`）上跑 eval，对比 HydraMDP / VADv2 / DiffusionDrive 的 score / ADE / FDE。DiffusionDrive 论文报告应略优于 VADv2。
