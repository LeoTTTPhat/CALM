# Literature baselines

This note documents experimental baselines from the two closest graph-attention radio-map papers and a volumetric 3-D CNN, complementing the 2.5-D `voxelcnn`. It records the three new baselines in `src/lit_baselines.py`. For each one it gives the original architecture, what we implemented, every deviation and the reason for it, the hyperparameters, and the measured accuracy and time. `src/test_lit_baselines.py` reproduces the numbers.

Interface: `run_lit_baseline(name, sc, tabs, train_tab, seed=0, epochs=None)` returns `{split: path-loss prediction in dB}` for every split in `tabs`. `LIT_BASELINES = ["phygat", "radiogat", "unet3d"]`. Per-fit diagnostics are stored in `lit_baselines.LAST_INFO[name]`: epochs run, validation RMSE, selected regime, fitted constants, parameter count and wall time.

## Common protocol (all three)

- **Inputs are limited to what CALM receives.** That means the standardised TR 36.777 prior (`gt.pl_prior`, antenna pattern included) and its LoS probability (`gt.plos`); the crowdsensed reports with their observed path loss, residual and age; and geometry relative to the known site positions (`gt.city.bs`, `gt.d3d`, `gt.d2d`, altitudes). No baseline reads the building map, the geometric LoS state, the shadowing field, `gt.sigma` (which depends on the true LoS) or `gt.g_ant` separately.
- **Labels.** Models train on the noisy probe reports `y_obs` / `r_obs` when the scenario provides them, which is the same target CALM uses. Otherwise they fall back to the latent `y` / `r`. Test RMSE is always computed against the latent `y`.
- **Early stopping.** 10% of the `train_tab` probe *locations* (all of their site rows) are held out, with seed-fixed selection. They are used for checkpoint selection and the few hyperparameter choices described below. Calibration and test rows are never seen before prediction. The scenario's own `val` split is not used.
- **Compute.** `torch.set_num_threads(4)`, CPU only. All grid models use the 10 m coarse grid (100 x 100 x 5 voxels per site), the same coarsening as the existing `voxelcnn`. Predictions are read back at each probe by bilinear interpolation within the probe's altitude layer.
- **Hyperparameter selection.** We used only the held-out 10%. Test RMSE was printed in the development runs but was not used to choose between configurations. The development runs are listed at the end.

---

## 1. `phygat`: query-conditioned hierarchical GAT with a kriging prior and residual (Li, Liu and Wang, arXiv:2604.17414v1)

### Original architecture (v1, 19 Apr 2026, "Physics-Aware Query-Conditioned Graph Attention Networks for Radio Map Estimation")

- **Setting.** 2-D region with transmitter-resolved, pointwise RSS estimation: one prediction per (target, transmitter) pair. Data are DeepMIMO ray tracing (Denver), with one shared model across sites.
- **Local graph.** A bounded reference scaffold is subsampled per transmitter. The K_ref = 16 nearest references to the target form a star graph. Each reference page keeps its own identity.
- **Node features.** The transmitter node is h_b = f_tx(p_b), a linear map of the coordinates. A reference node takes [p_r - p_b, y_r] through two branches (relative position, RSS), fused by a two-layer tanh MLP of width 128. The target node carries no RSS and enters only through geometry.
- **Transmitter-referenced geometric descriptors.** For the triad (t, r, b), each of the pairs (t,r), (r,b) and (t,b) contributes a distance and a bearing. Distances are warped with d/(1+d) and encoded by learnable uniform codebooks of 256 bins. Bearings use a circularly interpolated 256-bin codebook. Each component has a 16-d embedding. An optional LoS-agreement embedding uses per-link LoS flags. Fusion is an MLP of width 64 with tanh.
- **Local stage.** Single-head GATv2-style attention: e = a^T sigma(W [h_r; g_tr]), softmax over references, then z_l = sigma(W_b h_b + sum alpha W_m [h_r; g_tr]).
- **Global stage.** The K_g = 4 nearest *targets* of the same transmitter are used: mini-batch targets during training and the evaluated target set at inference. Attention is over [z_t; z_t'; g_tt], with messages W[z_t'; g_tt] and no self term. The readout state is s = [z_l; z_g].
- **Three regimes.** (i) Direct: y = f(s). (ii) **Prior-conditioned residual**: the prior y_bar is **ordinary kriging**, and the residual head e_hat = f_res(s, y_bar) is trained on e = y - y_bar. (iii) **Post-hoc gate**: y = y_bar + gamma * e_hat, with gamma = q([y_bar, e_hat, |e_hat|, delta_pri]) in [0, 1]. The gate is fitted by weighted Huber regression to gamma* = clip(e/e_hat, 0, 1), with the encoder frozen and only the training pairs used. delta_pri is "local variation descriptors of the prior" (derivative cues).
- **Training.** Adam, batch 128, 20 epochs, best-validation checkpoint, Huber or MSE loss. The original paper does not state the learning rate, Huber delta or dropout. d = 128.

**Version note.** v2 (1 Sep 2026) is retitled "Structured Edge-Aware Graph Attention Network for Transmitter-Resolved Pointwise Radio Map Estimation" (SeaGAT). It removes the global graph, the kriging-prior residual regime and the gate. It uses a direct RSS predictor with K = 8, temperature 0.7, Huber delta of 2 dB, Adam at 5e-4 and about 120 k parameters. We reproduce **v1**, which has the prior-plus-residual design. Where v1 is silent we borrow v2's values: lr 5e-4, Huber delta 2 dB, distance warp constant c = 1000 m, LeakyReLU slope 0.1.

### What we implemented

This is the full v1 pipeline in its strongest configuration: residual regime, followed by the gate.
- **Prior y_bar**: the repository's `kriging` baseline (`baselines.run_baseline("kriging")`). This is local ordinary kriging, with a fitted exponential variogram, of the reports' residual with respect to the 3GPP prior over the 12 nearest reports. The prior is then added back.
- **Reference scaffold**: all reports. Every report observes every site, so R_b is the same for all b and no subsampling is needed. The K_ref = 16 nearest are taken in the 3-D Euclidean metric. If a target coincides with a report, the report is removed (leave-one-out).
- **Reference node**: relative position (p_r - p_b)/1000 m, and an RSS branch [z-scored observed path loss, 3GPP residual/10].
- **Descriptors**: (t,r), (r,b) and (t,b), each with a distance codebook (256 bins, warp d/(1000 + d)), a circular bearing codebook (256 bins, interpolated) and an **elevation codebook** (256 bins over [-pi/2, pi/2], interpolated). Each is 16-d, giving 144-d, then an MLP (64, tanh) to d = 128.
- **Local stage** and **global stage** are as in v1: single head, LeakyReLU 0.1, tanh on z_l. K_g = 4 nearest targets of the same site *in the same table*.
- **Residual head**: one hidden layer (128, tanh) on [z_l; z_g; standardised y_bar]. The loss is Huber with delta = 2 dB on the standardised residual.
- **Gate**: an MLP (32, tanh, sigmoid output) on [y_bar, e_hat, |e_hat|, delta_pri]. delta_pri = [|grad PL_3GPP| at the target, standard deviation of the 16 reference residuals, distance to the nearest reference, mean reference distance]. It is fitted by Huber regression (delta 0.5) to gamma*, weighted by e_hat^2, with eps_e = 0.1 dB. The encoder is frozen, the gate is trained on the fit rows and early-stopped on the held-out rows. The final regime (residual, or residual plus gate) is chosen on the held-out rows.

### Deviations and reasons

1. **Kriging target.** v1 kriges raw RSS in 2-D. Raw path loss here has a 40 dB site-distance and antenna-pattern trend, so we krige the residual with respect to the 3GPP prior. This is exactly the manuscript's kriging baseline, which keeps y_bar identical to a baseline already in this repository.
2. **3-D descriptors.** v1 is 2-D. We add an elevation codebook per pair and use 3-D distances and offsets. Without these the model cannot tell the 100 m layer from the 300 m layer.
3. **RSS branch** has a second channel, the 3GPP residual of the report, so the model sees the same report residuals CALM sees. v1 uses RSS only.
4. **No LoS-agreement embedding.** It requires per-link LoS flags, which would mean using the building map.
5. **Global neighbours during training** come from the whole training table, not the 128-target mini-batch. Random mini-batches spread over 1 km^2 x 5 layers would give neighbours hundreds of metres away, unlike the inference graph. The global stage uses no labels, so the held-out rows can serve as neighbours without leakage. At inference the neighbours are the evaluated split's targets, as in v1. This is transductive but label-free.
6. **delta_pri** uses the 3GPP prior gradient and the spread of the references. v1 uses derivative cues of the prior field. Kriging at offset points for every query was not worth the cost.
7. **Regime selection** between residual and residual plus gate is made on the held-out rows. v1 reports both.
8. **Unstated values** take v2's settings: lr 5e-4, Huber delta 2 dB, c = 1000 m, slope 0.1. Patience is 10 epochs within the 20-epoch budget.
9. **No report-age input.** v1 has no temporal model, and we keep that.

### Hyperparameters
K_ref 16, K_g 4, d 128, 256 bins x 16-d per component, Adam lr 5e-4, batch 128, 20 epochs (best-validation checkpoint, patience 10), Huber delta 2 dB. Gate: Adam 1e-3, batch 512, up to 60 epochs, patience 8. About 310 k parameters.

---

## 2. `radiogat`: model-based graph plus GAT over grid nodes (Li et al., IEEE TWC 23(11):17777-17792, 2024; arXiv:2403.16397v2)

### Original architecture
- **Task.** Multi-band 2-D reconstruction within one area: predict the radio map at f_target from sparse observations at f_obv. Data are ray-traced WinProp maps over 10 OpenStreetMap areas, with 3 transmitters, 5 bands, 5 m grids and 200 m x 200 m blocks.
- **Graph.** The nodes are all non-building grid cells of a block, one graph per band. Observed nodes carry the normalised RSS and unobserved nodes carry 0. Node features are [RSS, position, f_obv, f_target].
- **Model-based component: the "radio depth".** D = T (C - alpha log10 d - beta log10 f), where **T is the building-free fraction of the cell-to-transmitter path, so it requires the building map**. Within d_th this is approximated by D_hat = beta T log10 f, summed over transmitters. **Edges are binary**: an edge exists when the distance is at most d_th = 15 m and |D_hat_i - D_hat_j| <= delta = 1 x 3 (for 3 transmitters). C, alpha and beta are hyperparameters (beta = 10). **They are not estimated from measurements.**
- **GAT.** Three GAT layers (Velickovic et al.). Layer 1 runs on the f_obv graph and layers 2-3 on the f_target graph. Heads and widths are not reported. Output is RSS per node.
- **Training.** Masked MSE on labelled nodes. Adam at lr 1e-3 (supervised) or 5e-3 (semi-supervised, 5% labelled), 200 epochs. The network predicts RSS directly.

### What we implemented
- **Nodes.** All 10 m voxels of the service volume, one graph per site: 5 x 100 x 100 = 50 k nodes. The sites are batched and share weights.
- **Model-based component, fitted from measurements.** For each site b, least squares on that site's reports gives PL_b(x) = theta0_b + theta1_b log10 d3D_b(x) + theta2_b PL_3GPP,b(x). That is 3 constants per site, and the prior regressor carries the known antenna pattern. The radio depth is D_b = PL_b, with T fixed at 1.
- **Edges.** Binary: same layer with horizontal distance at most d_th = 25 m, or the same column in an adjacent layer (50 m), plus self-loops, and in every case |D_i - D_j| <= delta = 3 dB. This is implemented as a dense masked-shift GAT over 23 offsets. Mean degree is 17.5 of 23, so 75% of candidate edges are kept.
- **Node features.** Gridded observed residual with respect to PL_b (divided by 10, 0 where unobserved), an observation mask, site-relative position (dx, dy, dz), log10 d3D, and standardised PL_b.
- **GAT.** Three layers: 4 heads x 8 (concatenated, ELU), then 4 heads x 8, then 1 head x 1. LeakyReLU 0.2 inside the attention. The output is a residual with respect to PL_b. The prediction is PL_b(probe) plus the bilinear read-out.
- **Training.** Masked MSE at the labelled training probes, which is the semi-supervised regime. Full-graph Adam, one step per epoch over all sites as in the original paper, lr 1e-2, best-validation checkpoint.

### Deviations and reasons
1. **Single band.** f_obv and f_target are dropped, and the layer-1 and layer-2/3 graphs coincide. The study has one carrier.
2. **No building map, so T = 1, and C, alpha and beta are fitted from the reports by least squares.** Using the true T would give the baseline information neither CALM nor any deployable method has. The model-based component therefore becomes a *measurement-calibrated* propagation model rather than a map-based one. It no longer encodes building blockage, which removes RadioGAT's main mechanism. Our baseline is a lower bound on what RadioGAT could achieve with a building map.
3. **3-D graph.** Vertical edges to adjacent altitude layers are added. Without them the five layers are disconnected, because the layers are 50 m apart and d_th is 15 m.
4. **d_th = 25 m on a 10 m grid**, instead of 15 m on a 5 m grid. Reports here are sparser: about 7.6% of 10 m cells per layer, against 5% of 5 m cells in the original paper. A 15 m radius on our grid would give 3 hops of 15 m and only 2-3 observed nodes in the receptive field. 25 m gives about 14. The 5 m native grid with a 15 m radius would need 29 offsets on 200 k nodes per site, which exceeds the time budget.
5. **delta = 3 dB.** The original paper's value is "1 x 3". Our depth is in dB of calibrated path loss rather than beta T log10 f, so the value cannot be transferred directly. 3 dB keeps 75% of edges and prunes edges across steep antenna-pattern gradients near the sites.
6. **Mask channel and site-relative coordinates** are added. The 0-for-unobserved convention is ambiguous when residuals can be 0. Sites differ, whereas the original paper keeps the transmitters fixed within an area.
7. **Residual target** with respect to the calibrated model, instead of direct RSS. Re-learning the 40 dB distance and antenna trend from coordinates would handicap the baseline for a trivial reason. All learned baselines in the manuscript are anchored in the same way.
8. **Learning rate 1e-2, 300 epochs.** The original paper uses 5e-3 and 200 epochs in the semi-supervised setting. At 5e-3 validation RMSE was still falling at 300 epochs. 1e-2 reached a lower validation RMSE in the same budget (5.31 against 5.38 dB). We also tried one stochastic step per site per epoch, which plateaued higher (about 5.5 dB at 60 epochs), and dropped it.
9. **Heads and width** are 4 x 8, because the original paper does not state them. 4 x 16 gained only 0.04 dB on validation at 300 epochs and was 2-3x slower, so we kept 4 x 8.
10. **No report-age input**, as in the original paper.

### Hyperparameters
Coarse 10 m grid, d_th 25 m plus vertical, delta 3 dB, 3 GAT layers (4 x 8, 4 x 8, 1), full-graph Adam lr 1e-2, up to 300 epochs, patience 25, 1 475 parameters. Fitted model constants (theta0, theta1, theta2) for highrise seed 1: theta2 is 0.89-1.03 for all seven sites. The fitted model alone gives 6.52 dB validation RMSE, against 8.0 dB for the 3GPP prior.

---

## 3. `unet3d`: volumetric 3-D U-Net

### Reference architecture
Cicek et al., "3D U-Net", MICCAI 2016. It is an encoder-decoder with two 3x3x3 conv + BN + ReLU per level, 2x2x2 max-pooling, 2x2x2 up-convolutions with skip concatenation, and 4 resolution levels. It stands in for the voxel and volumetric radio-map family, which applies 3-D convolutions over altitude x X x Y.

### What we implemented
- **Input volume per site**, with the batch running over sites and one network shared by all of them. There are 17 channels on 5 x 100 x 100 voxels (10 m horizontal, 50 m vertical):
  - the gridded 3GPP residual of the reports (mean per voxel), the measurement mask and the mean report age;
  - the cross-site mean residual, which is the same information as CALM's cross-site context;
  - prior/100, pLoS, log10 d3D, log10 d2D, sin(elevation) and altitude;
  - a 7-channel site one-hot.
- **Network.** 3 pooling levels with widths 16-32-64-128 and two 3x3x3 conv + GroupNorm + ReLU blocks per level. Pooling is **anisotropic (1,2,2)** and up-convolutions are transposed (1,2,2). The skips concatenate. A final 1x1x1 head gives the residual (x10 dB). There are 1.37 M parameters.
- **Output.** 3GPP prior plus the bilinear read-out of the residual volume at each probe.
- **Training.** MSE at the labelled training probes. AdamW at a constant lr of 2e-3 with weight decay 1e-4. Each step is one full-batch pass over the 7 site volumes. Dihedral augmentation of the horizontal plane: random x-flip, y-flip and transpose applied to all channels, with the output transformed back. Up to 400 epochs, patience 30, best-validation checkpoint.

### Deviations and reasons
1. **Anisotropic pooling.** There are only 5 altitude layers, 50 m apart against 10 m cells. Isotropic 2x2x2 pooling would collapse the altitude axis after two levels. In-plane-only pooling keeps 3x3x3 kernels at every level, so every level mixes altitude. This follows the nnU-Net rule for anisotropic voxels.
2. **GroupNorm instead of BatchNorm.** A batch holds 1-7 strongly correlated volumes from one scene, so batch statistics are unreliable and differ between training and evaluation.
3. **10 m grid with 3 levels and width 16** rather than 5 m with 4 levels and width 32. At 5 m resolution one epoch costs about 4x more, which exceeds the 10-minute budget. The coarsening matches the existing `voxelcnn`.
4. **Supervision.** Labels exist only at sparse probe points, so the loss is evaluated at the probes through the bilinear read-out rather than on dense label volumes.
5. **Site one-hot channel and site-relative geometry channels.** One network is shared over sites, and these channels let it do that.
6. **Optimiser chosen on validation.** We kept constant lr 2e-3 with full-batch steps (validation 4.30 dB). Per-site stochastic steps (4.37-4.50 dB) and cosine annealing from 3e-3 (4.59 dB) were both worse.

---

## Results: highrise, seed 1 (`python test_lit_baselines.py`)

Final run on 2026-11-15: 28 350 training rows (2 835 held out for early stopping), 31 500 test rows, 21 000 calibration rows, 7 sites, 4 torch threads. The shared machine's load average was 75-85 during the run, so wall times are pessimistic.

| Baseline | Test RMSE (dB) | Cal RMSE (dB) | Wall time | Epochs | Parameters | Notes |
|---|---|---|---|---|---|---|
| 3GPP prior | 8.013 | n/a | 0 | n/a | n/a | reference |
| kriging (= phygat prior y_bar) | 4.983 | n/a | <1 s | n/a | n/a | reference |
| **phygat** (HGAT, residual regime) | **4.523** | 4.658 | 267 s | 20 | 310 k | the gate did not improve validation RMSE in this run, so the residual regime was selected. In a development run the gate was selected and gave 4.555 |
| **radiogat** (model-based graph + GAT) | **5.071** | 5.163 | 634 s | 300 | 1.5 k | the fitted model alone gives 6.52 dB on validation |
| **unet3d** (volumetric 3-D U-Net) | **4.082** | 4.175 | 478 s | 282 (best at 252) | 1.37 M | |

Reading. All three beat the 3GPP prior by a wide margin. PhyGAT improves on its own kriging prior by 0.46 dB, consistent with the 0.1-0.5 dB gains the original paper reports for its residual regime over kriging. RadioGAT stays about 0.1 dB behind kriging. Its edge rule loses its blockage information without a building map, and three 25 m hops reach only about 75 m, whereas kriging always uses the 12 nearest reports. The 3-D U-Net is the strongest of the three and clearly better than the 2.5-D `voxelcnn` (about 4.9 dB). It remains behind the random forest (about 3.8-3.9 dB) and CALM (about 3.6-3.7 dB). RadioGAT's 634 s is over the 10-minute target only because of machine load: the identical configuration ran at about 1 s per epoch (about 5 min) when the machine was less loaded.

Reference values on highrise seed 1: random forest about 3.9 dB, `voxelcnn` (2.5-D) about 4.9 dB, kriging 4.98 dB, 3GPP prior 8.01 dB.

### Development runs (highrise seed 1, same scenario)

Configurations were chosen on the held-out validation RMSE (10% of the training probe locations). Test RMSE is listed for completeness only. Wall times were measured while the shared machine's load average was 60-90 on 28 cores, so they are pessimistic. The same RadioGAT configuration took 320 s in one run and 636 s in another under different load.

| Model | Configuration | Val RMSE | Test RMSE | Wall |
|---|---|---|---|---|
| phygat | 20 epochs, patience 8 (gate selected) | 4.255 | 4.555 | 282 s |
| phygat | 40-epoch budget, patience 8 (stopped at 19, residual selected) | 4.342 | 4.522 | 386 s |
| radiogat | full graph, 4x8, lr 5e-3, 300 epochs | 5.376 | 5.144 | 320 s |
| radiogat | full graph, 4x16, lr 5e-3, 400 epochs | 5.283 | 5.051 | 1082 s |
| radiogat | per-site steps, 4x16, lr 5e-3 (stopped by hand at 60 epochs, plateau) | about 5.51 | n/a | n/a |
| radiogat | **full graph, 4x8, lr 1e-2, 300 epochs (adopted)** | 5.313 | 5.071 | 636 s |
| unet3d | **full batch, constant lr 2e-3, 400 epochs, patience 30 (adopted; stopped at 282)** | 4.296 | 4.082 | 569 s |
| unet3d | per-site steps, lr 1e-3, 150 epochs | 4.498 | 4.282 | 333 s |
| unet3d | per-site steps, lr 2e-3, 250 epochs | 4.366 | 4.119 | 617 s |
| unet3d | full batch, cosine from lr 3e-3, 250 epochs | 4.595 | 4.336 | 454 s |

Run-to-run variation is about +-0.05 dB. CPU thread scheduling makes the PyTorch fits not bit-reproducible. For example, the two phygat runs above use identical settings for their first 19 epochs.

## Notes on the original designs

- **PhyGAT.** Query-conditioned local measurement graphs, transmitter-referenced geometric features and explicit prior-plus-residual prediction are the defining elements of **v1**. v1's prior is **ordinary kriging**, a statistical interpolator, not a propagation-physics prior. "Physics-aware" refers to the transmitter-referenced geometry. v1 also has a global target-target graph and a post-hoc gate. The current arXiv version, **v2 (SeaGAT), has dropped the prior-plus-residual and gate regimes and the global graph**, so this implementation follows v1.
- **RadioGAT.** The model-based part is **not** path-loss parameter estimation from measurements. It is a fixed-hyperparameter log-distance "radio depth" weighted by the building-free fraction of each link, computed **from the building map**. It is used only to decide *which edges exist*, not as an additive prior or as a node feature. The GAT predicts RSS directly. The method is transductive within an area (all grid nodes of a block in one graph) and is aimed at *cross-band* transfer. Our adaptation fits the model constants from measurements because the building map is unavailable, and that differs from the original paper.
