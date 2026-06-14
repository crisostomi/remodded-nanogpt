# Gene Map — modded-nanogpt speedrun (records 1–83)

## 1. The gene model

Every record in the speedrun is modeled as a **coherent prefix of genes**: record *N* renders to exactly the set of genes introduced by records `1..N`, applied in order, on top of the substrate in force at that point. A gene is one of four kinds:

- **additive** — a clean, slot-free feature that layers onto the current scaffold without removing or replacing anything (e.g. `qk_norm`, `smear`, `bigram_hash_embedding`, `value_embed_gates`). These are the cheap genes: they can usually be toggled on/off independently.
- **allele** — a mutually-exclusive variant occupying a named *slot* (`allele_group`). Selecting one allele *replaces* the previously-selected allele in that slot (e.g. the attention backend slot goes `FlexAttention → FlashAttention-3`; the orthogonalizer slot goes `Newton-Schulz → Triton-symmetric → Polar Express → paired-head-QK`).
- **substrate** — a structural rewrite of the model/optimizer/forward scaffold that all later genes attach to. Substrate genes partition history into **eras**: within an era the additive/allele genes are clean and independent; crossing a substrate boundary changes the API that genes plug into. These are the expensive genes.
- **tuning** — pure hyperparameter / schedule / layout changes with no new mechanism (LR, iterations, betas, eps, weight-decay multipliers, version bumps). Tuning genes carry the wall-clock win but add no renderable structure.

The chain is therefore: a sequence of **eras** (delimited by substrate genes), each containing a stack of additive genes plus a set of **allele slots** whose selected variant advances over time. The generator's job is: given a target record *N*, choose the era substrate active at *N*, the allele selected in each slot as of *N*, and the set of additive genes present at *N*, then emit the parameterized training script.

**Counts:** 83 records · 171 genes · 10 allele slots · ~13 era-defining substrate rewrites · 57 covered gene-rows / 114 uncovered.

---

## 2. Record table (1–83)

`C?` = covered by one of the current implementation's features (Y/partial/N). `conf` = sha-resolution confidence; **⚠ low** flags shaky sha resolutions where no commit isolates the record (see footnotes).

| # | date | sha | title | primary gene(s) | kind | C? | conf |
|---|------|-----|-------|-----------------|------|----|------|
| 1 | 2024-05-28 | 4a452a26 | llm.c baseline | llmc_gpt2_baseline_pytorch_port | substrate | – | med |
| 2 | 2024-06-06 | 3e48a54c | Tuned LR & rotary embeddings | rotary_position_embeddings | allele | N | high |
| 3 | 2024-10-04 | c26c2a18 | Introduced the Muon optimizer | orthogonalized_momentum_optimizer_muon | allele | N | high |
| 4 | 2024-10-11 | b356a1fb | Muon improvements | muon_formalization_and_update_rms_scaling | tuning | N | high |
| 5 | 2024-10-14 | 1d954e92 | Pad embeds, ReLU², zero-init, QK-norm | modern_arch_qknorm_relu2_zeroinit_padvocab | substrate | partial | **⚠ low** |
| 6 | 2024-10-18 | 7e1f20b8 | Distributed Muon overhead | distributed_muon_newton_schulz | substrate | N | high |
| 7 | 2024-10-18 | 7e1f20b8 | Upgraded PyTorch 2.5.0 | pytorch_250_runtime_upgrade | tuning | N | med |
| 8 | 2024-11-03 | dec62cd8 | Untied embedding and head | untie_embedding_and_head | additive | N | high |
| 9 | 2024-11-06 | e59ae87f | Value/embed skips, momentum warmup, softcap | value_residual_skip · logit_softcap_tanh30 | additive | partial | high |
| 10 | 2024-11-08 | 6671ed12 | Bfloat16 activations | bfloat16_activations_castedlinear | substrate | N | high |
| 11 | 2024-11-10 | ff93d20b | U-net skips & double lr | unet_skip_connections | substrate | N | high |
| 12 | 2024-11-19 | ebce774d | Dense → 64K FlexAttention | flex_attention_backend · long_context_flattened_sequence | allele+substrate | N | high |
| 13 | 2024-11-24 | 759b8fc7 | Attention window warmup | attention_window_warmup | additive | partial | high |
| 14 | 2024-12-04 | 49c3afac | Value Embeddings | token_value_embeddings | additive | partial | high |
| 15 | 2024-12-08 | 594bdc9a | U-net value embeds, code opts | unet_value_embeddings · distributed_batched_muon | additive+substrate | partial | med |
| 16 | 2024-12-10 | 0690f2c7 | Split VE, block sliding window | split_value_embeddings_module · block_sliding_window_full_partial_mask | additive | partial | high |
| 17 | 2024-12-17 | d32c7005 | Sparsify VE, improve rotary, drop attn | sparsify_value_embeddings · half_truncate_rope · drop_attn_layer_7 | additive+allele | partial | med |
| 18 | 2025-01-04 | 45544f53 | Lower softcap 30→15 | lower_logit_softcap_15 | tuning | N | high |
| 19 | 2025-01-13 | 602014e5 | FP8 head, offset logits, lr→0.1 | fp8_lm_head | additive | Y | high |
| 20 | 2025-01-16 | ebdeb374 | Merged QKV, long-short attn, scale, eps | merged_qkv_weights · long_short_sliding_window_attention | substrate+allele | N | high |
| 21 | 2025-01-26 | 01686bbc | Reduced batch size | reduce_train_seq_len_48k · retune_fp8_head_scales | tuning | partial | high |
| 22 | 2025-05-24 | 225e6902 | Faster gradient all-reduce | bucketed_grad_all_reduce | additive | N | high |
| 23 | 2025-05-25 | 378c4e30 | Overlap compute & grad comm | overlap_grad_comm_with_backward | allele | N | high |
| 24 | 2025-05-30 | afe0141e | all_reduce → reduce_scatter | reduce_scatter_sharded_optimizer · per_optimizer_weight_decay | substrate+additive | N | high |
| 25 | 2025-07-13 | 5e9f1631 | Upgrade PyTorch 2.9.0.dev | pytorch_290_upgrade · drop_autocast_cudagraph_workarounds | tuning | N | med |
| 26 | 2025-07-13 | 345b2fb9 | EoS-aligned batches, cooldown .45 | eos_aligned_batch_starts | additive | N | high |
| 27 | 2025-07-18 | 9d9dc969 | Transpose MLP matrix + Triton symmetric matmul | triton_symmetric_matmul_kernel | allele | N | **⚠ low** |
| 28 | 2025-08-23 | 9d9dc969 | Sparse attention gate | sparse_attention_gate | additive | Y | high |
| 29 | 2025-09-03 | 83868119 | FlashAttention-3, max_doc_len 2048 | flash_attention_3_backend · max_doc_len_2048_truncation | allele+additive | N | high |
| 30 | 2025-09-05 | 1d02940e | Drop first MLP layer | drop_first_last_mlp_blocks | additive | N | high |
| 31 | 2025-09-10 | 34ae835a | Dynamic YaRN train/val | dynamic_yarn_window_schedule · attn_args_rotary_refactor | additive+substrate | partial | high |
| 32 | 2025-09-11 | a96fbbd4 | Distributed opt, skip gating, bf16 | stacked_chunked_muon_comms · sigmoid_gated_unet_skip | substrate+additive | partial | high |
| 33 | 2025-09-15 | e8994818 | Async data, extend final-layer val window | async_data_prefetch_indexing · extended_final_layer_validation_window | additive | partial | high |
| 34 | 2025-09-18 | d149ec4a | Smear token embeddings | smear_token_embeddings | additive | Y | high |
| 35 | 2025-09-21 | 2fe38e24 | Drop first attn, extend long windows | drop_first_attention_layer · short_long_window_split_extend_validation | additive | partial | high |
| 36 | 2025-09-23 | 7d7952d2 | MuonCustomSizing, shared reduce-scatter | muon_custom_sizing_shared_reduce_scatter | substrate | N | high |
| 37 | 2025-09-27 | 346f4cb4 | Train-time CE in BF16 | train_time_cross_entropy_bf16 | allele | N | high |
| 38 | 2025-09-29 | a45ac737 | Polar Express orthogonalizer | polar_express_orthogonalizer | allele | Y | high |
| 39 | 2025-09-30 | 218b14d1 | Adam every other step, smaller batch | adam_every_other_step | additive | Y | high |
| 40 | 2025-10-04 | ba3e54f3 | Backout, hp tuning, lambda padding | backout_layer8_activation | additive | N | high |
| 41 | 2025-10-24 | 9f0d8ca6 | NorMuon | normuon_variance_normalization | allele | Y | high |
| 42 | 2025-10-27 | edf5cdb7 | NorMuon LR & step logic | optimizer_step_logic_rewrite · normuon_lr_correction | substrate+tuning | partial | high |
| 43 | 2025-11-10 | 29aefcfd | Cautious weight decay w/ schedule | cautious_weight_decay | additive | Y | high |
| 44 | 2025-11-16 | 80d68aff | Adam backward hooks | adam_backward_hook_grad_sync | additive | N | high |
| 45 | 2025-11-18 | 3843b662 | Refine skip arch, decay init | explicit_skip_connection_4_to_7 | substrate | N | high |
| 46 | 2025-11-29 | 25500ab7 | Batch size schedule | batch_size_schedule | additive | Y | high |
| 47 | 2025-12-10 | 960ad17d | Attn lambda × weight, fix warmup | sa_lambda0_premultiplied_to_qkv_weight | additive | N | high |
| 48 | 2025-12-11 | b72cabd2 | Speed Muon, premul lambda, reshape, lr | transposed_weight_layout_and_labeled_optimizer_groups · polar_express_split_baddbmm | substrate+tuning | partial | high |
| 49 | 2025-12-14 | 28fda1ef | Partial Key Offset | partial_key_offset · drop_layer0_flatten_residual_lambdas | additive+substrate | partial | high |
| 50 | 2025-12-18 | 49465ccd | Cautious WD on Adam params | cautious_weight_decay_adam | additive | Y | high |
| 51 | 2025-12-19 | d8377c77 | Retie embed to lm_head, fp8 scales | retie_embed_to_lm_head | substrate | partial | high |
| 52 | 2025-12-21 | 27ef9ef3 | Smooth scalars, smear lr, freeze, all-reduce | smoothed_scalars_separate_adam · freeze_scalars_during_transitions | additive | partial | high |
| 53 | 2025-12-22 | 9060ff47 | Multi-token prediction, untie at 2/3 | mtp_loss · untie_embed_at_2_3 | additive | Y | high |
| 54 | 2025-12-26 | 0bf08b08 | Asymmetric Logit Rescale | asymmetric_logit_rescale · training_manager_scaffold | allele+substrate | N | high |
| 55 | 2025-12-29 | 5a0486a2 | Gates on value embeds & skip | value_embed_gates · skip_gate | additive | Y | high |
| 56 | 2025-12-31 | 2e8e1655 | Compile Adam, fp32 buffers, gates→Adam | gates_to_adam_parameter_banks · compiled_adam_update | substrate+additive | N | high |
| 57 | 2026-01-04 | 13badcf9 | bf16 attn/mlp, mixed-prec Muon, interweave | interweave_adam_muon · mixed_precision_muon | substrate+additive | N | high |
| 58 | 2026-01-07 | 44b9dd71 | Paired Head Attention | paired_head_attention · yarn_rotary_method_refactor | additive+substrate | Y | high |
| 59 | 2026-01-10 | 647d4980 | Fused Triton linear-ReLU²-MLP | fused_linear_relu_square_mlp | allele | N | high |
| 60 | 2026-01-16 | 0a43716d | Fused softcapped MTP CE kernel | fused_softcapped_mtp_cross_entropy | allele | N | high |
| 61 | 2026-01-18 | 71af6208 | Unified optimizer, transposed LM head | unified_optimizer · transposed_lm_head | substrate | N | high |
| 62 | 2026-01-19 | 93f0e6b9 | Bigram Hash Embedding | bigram_hash_embedding · residual_bigram_injection | additive | Y | high |
| 63 | 2026-01-26 | db2512c5 | Untie Value Embeds | untie_value_embeds | additive | N | high |
| 64 | 2026-01-30 | 5d2365f9 | Tuned nonzero Attn V/O init | mimetic_vo_init | tuning | N | high |
| 65 | 2026-01-30 | c3b38161 | Group VE into single parameter | value_embeds_single_parameter_fusion | substrate | partial | high |
| 66 | 2026-01-31 | fea4873e | Torch 2.10 | torch_210_upgrade_and_schedule_retune | tuning | N | med |
| 67 | 2026-01-31 | 7e09edba | Tune softcap kernels, fuse fp8 in LM head | fp8_lm_head_fused_into_cross_entropy | additive | partial | high |
| 68 | 2026-01-31 | 3eb3a4d8 | Move bigram hash to GPU | bigram_hash_on_gpu_remove_h2d | additive | Y | high |
| 69 | 2026-02-02 | 866e243c | Kernel optimizations | triton_kernel_memory_and_softcap_optimizations | tuning | N | high |
| 70 | 2026-02-03 | eb1b86aa | Tune VE layout & ve_gates | value_embed_gate_on_ve_and_layout_init_tuning | tuning | partial | high |
| 71 | 2026-02-06 | b893aeda | Sparse bigram grad comms, CPU loading | sparse_bigram_gradient_comms | additive | Y | high |
| 72 | 2026-02-10 | 82179cb4 | Min lr up, max_seq_len schedule | max_seq_len_short_sequence_curriculum | additive | Y | high |
| 73 | 2026-02-12 | 699e93c8 | Partitioned Hyperconnections | partitioned_hyperconnections_two_lane_residual | substrate | partial | high |
| 74 | 2026-02-16 | 20dd1582 | Flattened forward, transpose kernels | flattened_gpt_forward · fused_nesterov_momentum_polar_express | substrate+additive | partial | high |
| 75 | 2026-02-23 | e3f74f29 | Cross-entropy kernel opts | cross_entropy_backward_kernel_optimization | additive | N | high |
| 76 | 2026-02-28 | ea75fb47 | Reuse backward transpose kernel | reuse_transpose_copy_in_ce_backward | additive | N | high |
| 77 | 2026-03-06 | 81730c30 | Replace HC with single saved activation | single_saved_activation_replaces_partitioned_hc | substrate | N | high |
| 78 | 2026-03-22 | 5845e941 | Tighten fa3 max_num_docs | tailored_fa3_max_num_docs | additive | partial | high |
| 79 | 2026-04-04 | e1c7bf5d | Fuse CE fwd/bwd CUDA kernel | fused_cuda_cross_entropy_fwd_bwd | allele | N | high |
| 80 | 2026-04-08 | 2c702746 | Muon QK orthogonalize per head-pair | paired_head_muon_qk_groups | allele | Y | high |
| 81 | 2026-04-22 | 0d28c5a7 | MUDD Skip Connections | mudd_last_two_layers | additive | Y | high |
| 82 | 2026-04-29 | 25ec3536 | Learnable XSA | learnable_xsa | additive | Y | high |
| 83 | 2026-05-20 | 2d218013 | Sign Trick on Bigram Embed | bigram_sign_trick · bigram_dim_192 · residual_slice_bigram_injection | additive | Y | high |

### Low-confidence / shaky sha resolutions (flagged)

- **#5 (1d954e92)** — *history-rebase artifact.* No commit byte-matches the embedded ModernArch log (best 0.97). The chosen commit already folds in distributed Muon (which the README attributes to #6). The canonical record-5 script is the embedded log, not any in-history blob.
- **#27 (9d9dc969)** — *no isolating commit.* The Triton symmetric-matmul kernel was first committed **bundled inside** #28's "sparse attention gate" commit. The authoritative record-27 state is the `record.txt` artifact; the sha is the earliest commit that embeds #27 as its base layer.
- **#7 (7e1f20b8)**, **#25 (5e9f1631)**, **#66 (fea4873e)** — *environment/version bumps* with no unique git diff isolating them (med confidence). #7's script is byte-identical to #6; #25's delta is autocast-removal cleanup; #66's version pin lands in a separate later commit.
- **#15 (594bdc9a)**, **#17 (d32c7005)** — *batched/rebased commits* (med). The published logs are refactored forks that never byte-match a committed train script; sha is the canonical end-of-record state pinned by config markers + ancestry.

---

## 3. Allele groups (mutually-exclusive slots)

Each slot holds exactly one active variant at any record. Listed chronologically by the record that introduced each variant.

| Slot | Variants (record → variant) |
|------|------------------------------|
| **position_embedding** | #1 learned absolute (wpe, substrate baseline) → **#2 rotary_position_embeddings** |
| **rotary** (RoPE freq scheme) | #2–16 standard RoPE → **#17 half_truncate_rope** |
| **optimizer** (core) | #1 single AdamW → **#3 orthogonalized_momentum_optimizer_muon** → #4 Muon formalization (tuning) → **#15 distributed_batched_muon** (substrate) → **#41 normuon_variance_normalization** |
| **orthogonalizer** (Newton-Schulz inner kernel) | #3 NS5 → #11 ns_iteration_rewrite (tuning) → **#27 triton_symmetric_matmul_kernel** → **#38 polar_express_orthogonalizer** → #48 split_baddbmm (tuning) → #74 fused_nesterov (additive) → **#80 paired_head_muon_qk_groups** — ✅ **implemented as a 2-member `allele_group="orthogonalizer"` slot**: `polar_express` (default) \| `newton_schulz` (NS5 coefficients on the E15 fused scaffold; numeric-validation-pending). The other slot points (triton_symmetric, paired-head-QK) remain TODO. |
| **attention_backend** | #1 SDPA → **#12 flex_attention_backend** → **#29 flash_attention_3_backend** → #38 FA3-via-kernels-hub (sourcing only) → #83 fa3_kernel_backend_bump (version pin) |
| **attention_window** | #12 fixed 1024 sliding → #13 window warmup → #16 block full/partial → **#20 long_short_sliding_window_attention** → #29 discrete ws_schedule → #31 dynamic YaRN → #35 ws_short/ws_long split |
| **grad_allreduce** | #22 bucketed_grad_all_reduce → **#23 overlap_grad_comm_with_backward** → (subsumed by #24 reduce_scatter substrate) |
| **logit_softcap** | #9 tanh30 → #18 tanh15 → (sigmoid form) → **#54 asymmetric_logit_rescale** (23·σ((x+5)/7.5)) |
| **cross_entropy** (kernel/precision) | float32 CE → **#37 train_time_cross_entropy_bf16** → **#60 fused_softcapped_mtp_cross_entropy** (Triton) → **#79 fused_cuda_cross_entropy_fwd_bwd** (raw CUDA) |
| **mlp_kernel** | F.linear×2 + relu² → **#59 fused_linear_relu_square_mlp** (Triton) |

**Consistency note:** the `optimizer` slot mixes `allele` (#3, #41) and `substrate` (#15) kinds because two of its transitions are also scaffold rewrites — both the *what* (orthogonalize-then-normalize) and the *how* (distributed comms layout) change. Treat `optimizer` as a slot whose variant *and* substrate co-advance. The `orthogonalizer` slot is the clean inner-kernel sub-slot of the optimizer and is where most isolated allele swaps happen.

---

## 4. Substrate / eras

Substrate genes are structural rewrites; each opens an **era** in which the additive/allele genes are clean. The generator should branch on era first, then apply that era's gene stack. Eras below are bounded by record numbers.

| Era | Records | Opening substrate | What changed (scaffold) |
|-----|---------|--------------------|--------------------------|
| **E0 Baseline** | 1–4 | #1 llmc_gpt2_baseline | PyTorch GPT-2 port; single AdamW; learned pos-emb; Muon introduced as a slot variant (no scaffold change yet) |
| **E1 ModernArch** | 5–9 | #5 modern_arch_qknorm_relu2 | split QKV, QK-norm, ReLU², zero-init, padded vocab, head_dim 128 |
| **E2 DistMuon** | 6–9 | #6 distributed_muon_newton_schulz | Muon step becomes a distributed gather/scatter over GPUs |
| **E3 bf16 + U-net** | 10–11 | #10 bfloat16_activations_castedlinear; #11 unet_skip_connections | bf16 model via CastedLinear; encoder/decoder U-net skips |
| **E4 Long-context FlexAttn** | 12–19 | #12 long_context_flattened_sequence (+ flex_attention) | single 64K flattened sequence; BlockMask machinery; value-embed family grows here |
| **E5 Merged-QKV** | 20–23 | #20 merged_qkv_weights | fused qkv_w parameter; long-short windows; grad-comm experiments |
| **E6 Sharded optimizer (ZeRO-1)** | 24–35 | #24 reduce_scatter_sharded_optimizer | per-rank reduce_scatter/all_gather inside Muon + DistAdam; decoupled WD |
| **E6b Rotary/AttnArgs refactor** | 31–35 | #31 attn_args_rotary_refactor | free `rotary()` + shared cos/sin buffers; AttnArgs dataclass; dynamic YaRN |
| **E7 MuonCustomSizing comms** | 32–35 / 36–41 | #32 stacked_chunked_muon_comms; #36 muon_custom_sizing | stacked/chunked collectives; module-labeled param buckets; shared reduce-scatter |
| **E8 NorMuon step rewrite** | 42–47 | #42 optimizer_step_logic_rewrite | vectorized per-group step; precomputed chunk sizes; NorMuon variance state |
| **E9 Transposed layout / explicit skip / drop-layer0** | 45 / 48 / 49 | #45 explicit_skip_4→7; #48 transposed_weight_layout; #49 drop_layer0_flatten | row-major weights; labeled Adam groups; explicit 4→7 skip; 12→11 layers, flattened residual lambdas |
| **E10 TrainingManager + retie** | 51 / 54 | #51 retie_embed_to_lm_head; #54 training_manager_scaffold | single tied embed=lm_head; all schedules/opts consolidated in TrainingManager + ForwardScheduleConfig |
| **E11 Adam param-banks / interweave** | 56–57 | #56 gates_to_adam_parameter_banks; #57 interweave_adam_muon | gates in GPT-level Adam banks; pipelined Muon/Adam step (p1/p2/p3) |
| **E12 Unified optimizer + transposed head** | 61–64 | #61 unified_optimizer (+ transposed_lm_head) | NorMuon+DistAdam merged into NorMuonAndAdam with per-param ParamConfig; lm_head stored (model_dim, vocab) |
| **E13 VE single-param fusion** | 65–72 | #65 value_embeds_single_parameter_fusion | 5 VE tables → one fused nn.Parameter; bigram family matures |
| **E14 Partitioned hyperconnections** | 73–76 | #73 two_lane_residual; #74 flattened_gpt_forward | 2-lane residual stream (lane0/lane1, parallel_start=7); Block/MLP classes removed, forward inlined |
| **E15 Single-saved-activation** | 77–83 | #77 single_saved_activation | revert HC to single residual stream + one saved activation; later additive genes (MUDD, XSA, sign-trick) layer here |

The **current implementation sits in E15** (its substrate is the flattened-forward, single-saved-activation scaffold), which is why the newest records (#77–83) are cheap and the old eras are expensive.

---

## 5. Coverage

The current implementation has **24 features**; **22 of them map to genes** via `covered_by_current` (the other 2 are infra/plumbing features not tied to a single record gene). Coverage is measured at the **gene-row** level: a gene-row is *covered* if its `covered_by_current` points at one of the current features.

> **Retired into the curriculum dimension:** `yarn_window_schedule`,
> `batch_size_schedule`, `max_seq_len_schedule` were dropped as standalone
> features (they were no-op tags). The **8 gene-rows** they covered — the
> batch/seq-len/window ramps of records **13,31,33,35 / 39,46 / 72,73** — are now
> covered by the searchable `TRAINING_STAGES` curriculum dimension
> (`search.candidate_space.curriculum_sweep`), not a named gene. They remain
> counted in "covered" below.

- **Total gene-rows:** 171
- **Covered gene-rows:** 58 (50 via features + 8 via the curriculum dimension)
- **Uncovered gene-rows:** 113

### The 21 covered features (gene-row count each) + the curriculum dimension

| Current feature | Genes it covers | Records |
|------------------|-----------------|---------|
| value_embeds | 8 | 9,14,15,16,17,65 |
| fp8_lm_head | 6 | 19,21,51,67 |
| _curriculum dimension_ (`TRAINING_STAGES`) | 8 | 13,31,33,35,39,46,72,73 |
| cautious_weight_decay | 4 | 43,50,53 |
| bigram_hash | 4 | 62,68,71 |
| polar_express | 3 | 38,48,74 |
| newton_schulz | 1 | 3 |
| normuon | 3 | 41,48,78 |
| skip_gate | 2 | 32,55 |
| smear | 2 | 34,52 |
| untie_embed_at_2_3 | 2 | 53,65 |
| value_embed_gates | 2 | 55,70 |
| residual_slice_bigram_injection | 2 | 62,83 |
| paired_head_attention | 2 | 58,80 |
| sparse_attention_gate | 1 | 28 |
| adam_every_other_step | 1 | 39 |
| mtp_loss | 1 | 53 |
| sparse_bigram_comms | 1 | 71 |
| mudd_last_layers | 1 | 81 |
| xsa | 1 | 82 |
| bigram_sign_trick | 1 | 83 |
| bigram_vocab_15x | 1 | 83 |
| bigram_dim_192 | 1 | 83 |

### The gap (114 uncovered gene-rows)

The gap is dominated by: every **substrate** rewrite (~26 gene-rows — none are "features", they're scaffolds), every **tuning** gene (LR/iters/betas/version bumps — intentionally not features), and the **early-era alleles/additives** the current code simply predates (rotary, the whole Muon→NorMuon optimizer lineage's earlier stages, FlexAttention, the merged-QKV/long-short window machinery, untie/retie embedding flips, the grad-comm evolution #22–24, backout #40, the CE/MLP/transpose Triton kernels #59,60,75,76,79, etc.). These are not bugs in coverage — they are records that only become renderable once the generator can *select an era* and *re-introduce an old allele/substrate*, which the current single-substrate implementation cannot do.

---

## 6. Recommended implementation order (newest → oldest, era by era)

Goal: make all 83 records renderable from one gene-parameterized generator. Strategy is **anchor at the current substrate (E15) and walk backward**, because the current code already *is* the E15 substrate with most of its additive genes. Each phase below is ordered by effort. "Cheap" = additive toggles on an existing substrate; "Expensive" = a new substrate fork or allele backend the generator must carry.

**Phase A — E15 tail (#77–83). CHEAP.**
The current code is here. These are nearly all additive genes already covered (MUDD, XSA, sign-trick, sparse-bigram, max_seq_len curriculum) plus three uncovered kernel/tuning genes (#75,76,79 CE-kernel opts, #78 max_num_docs). Wire the additive genes to on/off flags; the CE-kernel variants (#79 raw-CUDA vs #60 Triton) become a `cross_entropy` allele selector. Effort: days.

**Phase B — E14 hyperconnections (#73–76). EXPENSIVE (one substrate fork).**
Introduce the 2-lane partitioned-residual substrate as an alternate forward path; #77 is the revert back to single-lane. The generator must carry both the flattened-forward inline dispatch and the lane0/lane1 branching as a substrate flag. Most cost is the forward rewrite + post_lambdas [L,2,2] layout. Effort: ~1 week.

**Phase C — E12–E13 unified optimizer + VE fusion (#61–72). MODERATE.**
Fork the optimizer substrate: NorMuonAndAdam (unified, ParamConfig) vs the split NorMuon/DistAdam. Add the transposed-lm-head tie and the VE-single-parameter fusion as substrate variants. Bigram family is already covered; uncovered work is mostly the optimizer-merge plumbing and #59/#60 fused MLP/CE kernels as allele backends. Effort: ~1–2 weeks.

**Phase D — E10–E11 TrainingManager / retie / Adam-banks (#51–60). MODERATE-EXPENSIVE.**
This is where the embedding tie flips (retie #51, then untie-at-2/3 #53) and gates migrate into Adam banks (#56) and the optimizer pipelines (#57). Multiple co-dependent substrates; the asymmetric softcap allele (#54) and train-time-bf16 CE (#37 carried forward) live here. Effort: ~2 weeks.

**Phase E — E8–E9 NorMuon step rewrite + transposed layout + drop-layer0 (#42–50). EXPENSIVE.**
The optimizer step is fully vectorized and the weight memory layout transposes (#48); the model drops layer 0 and flattens residual lambdas (#49). These substrates ripple through every param-shape assumption. Backout (#40) and cautious-WD (#43,#50, covered) are additive on top. Effort: ~2 weeks.

**Phase F — E6–E7 sharded optimizer + custom-sizing comms + AttnArgs/YaRN (#24–41). EXPENSIVE (the comms substrate spine).**
The ZeRO-1 reduce_scatter rewrite (#24), stacked/chunked comms (#32), MuonCustomSizing (#36), and the NorMuon allele (#41) all live here, plus the AttnArgs/rotary refactor (#31) and FA3 backend (#29). This is the densest substrate region — three distinct optimizer-comms layouts must coexist as selectable variants, and the attention backend forks FlexAttn↔FA3. Highest risk. Effort: ~3 weeks.

**Phase G — E5 merged-QKV + long-short windows + early grad-comm (#20–23). MODERATE.**
Fork the attention parameterization (merged qkv_w vs split c_q/c_k/c_v) and the grad-allreduce alleles (#22 bucketed, #23 overlap). Long-short window allele (#20). Effort: ~1 week.

**Phase H — E4 long-context FlexAttention + value-embed genesis (#12–19). EXPENSIVE.**
The flattened-64K-sequence substrate and the FlexAttention backend, plus the original value-embed lineage, fp8 head (#19, covered), softcap evolution. The data loader and mask construction are a separate substrate from the FA3 varlen path. Effort: ~2 weeks.

**Phase I — E1–E3 ModernArch + DistMuon + bf16/U-net (#5–11). EXPENSIVE.**
The ModernArch structural rewrite (#5), distributed Muon (#6), bf16/CastedLinear (#10), U-net skips (#11). These are the foundational substrates; note #5's sha is a rebase artifact (use the embedded log as canonical). Effort: ~2 weeks.

**Phase J — E0 baseline + rotary + Muon introduction (#1–4). MODERATE.**
The original AdamW baseline, rotary allele (#2), and Muon's first appearance (#3,#4). Smallest scaffolds but the most distant from current code, so they need their own fully-separate generator branch (no shared substrate with E15). Effort: ~1 week.

### Honest summary of effort

- **Cheap (additive on current substrate):** Phase A only (#77–83). Almost free because the current 26 features already cover the tail.
- **Moderate:** Phases C, D, G, J — mostly optimizer-plumbing and attention-parameterization forks with bounded blast radius.
- **Expensive (new substrate/allele spines):** Phases B, E, F, H, I. The dominant cost is the **optimizer-comms substrate evolution** (E6→E7→E8→E11→E12: five distinct distributed layouts) and the **attention substrate** (dense→FlexAttn→FA3 with three window regimes). Roughly **70% of total effort lives in eras E4–E12**, even though they are only ~half the records, because that span is where every substrate rewrite clusters. The newest and oldest ends are comparatively tractable; the middle is where the generator must hold multiple incompatible scaffolds simultaneously.
