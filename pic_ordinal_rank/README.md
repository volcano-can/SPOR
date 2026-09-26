# ODISE 深度图 → Ordinal Rank 可视化

本目录使用仓库当前实现生成，而不是依据深度值自行猜测 validity。

## 选中的样本

- `000000322429`：置物架和陶器，具有丰富且清晰的离散前后层次。
- `000000228436`：前景自行车、中景河道、远景建筑，具有明显的连续纵深。

数据来自 `Datasets/coco/val2017` 与对应的 `depth_val2017`。

## 正确处理流程

1. 读取 RGB 和 8-bit 灰度深度图；若尺寸不同，先把深度图对齐到 RGB。
2. 创建全 1 的源 validity。这与 `odise/data/dataset_mapper.py` 完全一致；当前代码**不把深度值 0 当成 invalid**。
3. 对 RGB、depth、validity 复用同一组真实训练几何增强：
   `RandomFlip → ResizeScale → FixedSizeCrop`。
4. validity 用 segmentation/最近邻方式变换，因此 crop 的 padding 区域为 0。
5. 仅对 valid 且有限的深度值计算 tie-aware mid-rank：
   `rank = (相同值区间的左右索引中点) / (N_valid - 1)`。
6. invalid 像素的 rank 置 0；因此读取数值 rank 时必须同时读取 validity。

## 每个样本目录中的主要文件

- `overview.png`：完整 2×4 流程总览。
- `08_ordinal_rank_color.png`：变换后、valid-aware 的彩色 rank。
- `09_validity.png`：白色为 valid，黑色为 invalid/padding。
- `10_ordinal_rank_u16.png`：无损 16-bit rank，0–65535 对应 0–1。
- `ordinal_rank_float32.npy`：浮点 rank。
- `validity_bool.npy`：布尔 validity。
- `metadata.json`：数据路径、随机种子、实际 transforms、尺寸和有效比例。

源图对应的 `04_source_rank_color.png` 和 `05_source_validity.png` 也被保留。源 validity 全白是预期结果；增强后 validity 才会显示 padding。

## 复现

```bash
cd /home/chenduoyou/data/ODI/ODISE
/home/chenduoyou/anaconda3/envs/odise/bin/python \
  pic_ordinal_rank/build_ordinal_rank_visualization.py \
  --output pic_ordinal_rank \
  --ids 000000322429 000000228436 \
  --seed 20260922
```

脚本也支持省略 `--ids`，此时会根据深度熵、边缘变化和动态范围自动生成候选并选取样本。
