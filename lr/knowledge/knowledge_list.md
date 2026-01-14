knowledge/
├── 00_scope_and_assumptions.md
├── 01_inference_overview.md
├── 02_model_architecture/
│   ├── transformer_basics.md
│   ├── attention_structure.md
│   ├── moe_structure.md
│   └── kv_cache_semantics.md
├── 03_inference_phases/
│   ├── prefill_phase.md
│   ├── decode_phase.md
│   ├── prefill_vs_decode_comparison.md
│   └── batch_and_request_semantics.md
├── 04_parallelism_primitives/
│   ├── tp.md
│   ├── pp.md
│   ├── ep.md
│   ├── sp.md
│   └── dp_in_inference.md
├── 05_parallelism_by_module/
│   ├── attention_parallelism.md
│   ├── ffn_parallelism.md
│   ├── moe_parallelism.md
│   └── embedding_and_lm_head.md
├── 06_execution_and_scheduling/
│   ├── pipeline_scheduling.md
│   ├── microbatching.md
│   ├── overlap_compute_communication.md
│   └── synchronization_points.md
├── 07_hardware_and_system_constraints/
│   ├── memory_constraints.md
│   ├── interconnect_topology.md
│   ├── latency_vs_throughput.md
│   └── accelerator_specific_notes.md
├── 08_dag_construction_primitives.md
├── 09_metrics_and_objectives.md
└── 10_common_failure_modes.md
