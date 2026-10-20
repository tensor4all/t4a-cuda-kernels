# CUDA performance contract migration

The migration preserves CubeCL's distinction between runtime tensor layout values and classified code specialization. The prior draft made rank and operation metadata dynamic too broadly. Keeping validated rank bounds, axis mappings and algorithm configuration as specialization permits unrolling and removes per-element mode dispatch while concrete extents, strides, offsets, starts, padding and window sizes remain runtime arguments.

Common per-axis metadata stays in by-value launch arguments, preserving the no-extra-upload property of runtime Sequence arguments. The CUDA port excludes the WebGPU/Metal baked-layout twins. Dense/view paths, tiled transpose and long-axis split reduction are required baselines, with an explicit paired benchmark gate for replacements. Fusion selection and lifetime/dependency logic remain consumer concerns; reusable source emission stays independent.

Evidence is pinned to tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c` and its CubeCL fork `11b52669f13e27bbe188f988fd696df6d989a562`. Current source takes precedence over superseded prose. The layout tests' compile-count observation is disabled in that baseline, so the native runtime must supply actual NVRTC counters to establish reuse.

The [performance preservation contract](../design/cubecl-performance-parity.md) records requirements and acceptance evidence. This update changes documentation/inventory and implementation trackers only. No kernel has been ported, and no new GPU correctness or performance result is claimed.
