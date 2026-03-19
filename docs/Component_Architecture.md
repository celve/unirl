# DiffusionRL Component Architecture

> Auto-generated from codebase analysis. Reflects the state after the Phase-2 refactoring
> (algorithm-owned loss, advantage, batch assembly, compute_loss_and_backward).

---

## 1. Component Overview

```mermaid
block-beta
    columns 3

    block:entry["Entry Points"]:3
        trainpy["train.py\n(sync loop)"]
        trainasync["train_async.py\n(async pipeline)"]
    end

    space:3

    block:ray["Ray Actors (Distributed Infrastructure)"]:3
        rm["RolloutManager\nray/rollout_manager.py\n~1400 lines"]
        ta["TrainingActor\nray/training_actor.py\n~1100 lines"]
        ra["RolloutActor\nray/rollout_actor.py\n~1150 lines"]
        rb["BufferActor\nray/buffer_actor.py\n~1300 lines"]
    end

    space:3

    block:algo["Algorithm Layer (Core)"]:3
        ba["BaseAlgorithm\nalgorithms/base.py\n~1055 lines"]
        grpo["GRPOAlgorithm + _GRPOLoss\nalgorithms/grpo.py\n~757 lines"]
        nft["NFTAlgorithm + _NFTLoss\nalgorithms/nft.py\n~782 lines"]
        mixgrpo["MixGRPOAlgorithm\nalgorithms/mix_grpo.py\n~120 lines"]
        norm["normalizers.py\nnormalize_grouped/global"]
    end

    space:3

    block:runtime["Runtime Layer"]:3
        te["TrainExecutor\nruntime/training/train_executor.py"]
        us["TrainingUpdateSchedule\nruntime/training/update_schedule.py"]
        rp["rollout_pipeline.py\nruntime/pipeline/ (stage helpers)"]
        ct["contracts.py\nruntime/contracts.py"]
        bk["Training Backends\nFSDP / VeOmni / Megatron"]
    end

    space:3

    block:support["Support Components"]:3
        ds["DataSource\ndata/data_source.py"]
        rs["RewardService\nreward/service.py"]
        ema["EMAManager\nutils/ema.py"]
        ws["WeightSync\nruntime/weight_sync.py"]
        samp["Samplers\nsamplers/ (FSDP, SGLang)"]
        mdl["Models\nmodels/ (Flux, SD3, Hunyuan, Mochi)"]
    end
```

---

## 2. Component Interaction — High-Level

```mermaid
graph TB
    subgraph EntryPoint["Entry Point (train.py)"]
        loop["Training Loop\nfor rollout_id in range(N)"]
    end

    subgraph RayActors["Ray Actors"]
        RM["RolloutManager"]
        TA["TrainingActor"]
        RA["RolloutActor"]
        RB["RolloutBufferActor"]
    end

    subgraph AlgoLayer["Algorithm Layer"]
        ALGO["Algorithm\n(GRPO / MixGRPO / NFT)"]
        LOSS["Loss\n(_GRPOLoss / _NFTLoss)"]
    end

    subgraph RuntimeLayer["Runtime Layer"]
        TE["TrainExecutor"]
        US["UpdateSchedule"]
        RP["rollout_pipeline"]
    end

    subgraph Support["Support"]
        DS["DataSource"]
        RS["RewardService"]
        EMA["EMAManager"]
        WS["WeightSync"]
    end

    loop -->|"1. generate_and_push()"| RM
    loop -->|"2. pop_training_data()"| RB
    loop -->|"3. train()"| TA
    loop -->|"4. sync()"| WS

    RM -->|"get prompts"| DS
    RM -->|"build_sampling_batch()"| ALGO
    RM -->|"distributed_sample()"| RP
    RP -->|"generate(request)"| RA
    RM -->|"compute_rewards()"| RS
    RM -->|"compute_advantages_with_components()"| ALGO
    RM -->|"assemble_training_batch()"| ALGO
    RM -->|"push(batch)"| RB

    TA -->|"holds"| ALGO
    TA -->|"creates"| TE
    TE -->|"iter_update_chunks()"| US
    TE -->|"compute_loss_and_backward()"| ALGO
    ALGO -->|"compute_loss()"| LOSS
    TE -->|"post_optimizer_step()"| EMA

    WS -->|"get_state_dict()"| TA
    WS -->|"update_weights()"| RM
    RM -->|"update_weights()"| RA

    ALGO -->|"get_ema_spec()"| EMA
```

---

## 3. Synchronous Training Loop — Sequence Diagram

```mermaid
sequenceDiagram
    participant T as train.py
    participant RM as RolloutManager
    participant DS as DataSource
    participant ALGO as Algorithm
    participant RP as rollout_pipeline
    participant RA as RolloutActor
    participant RS as RewardService
    participant RB as RolloutBuffer
    participant TA as TrainingActor
    participant TE as TrainExecutor
    participant EMA as EMAManager
    participant WS as WeightSync

    loop for rollout_id in range(N)
        Note over T: === PHASE 1: Rollout ===
        T->>+RM: generate_and_push(rollout_id, buffer)

        RM->>DS: next_batch(batch_size)
        DS-->>RM: Dict["prompts", "metadata"]

        RM->>ALGO: build_sampling_batch(batch, K)
        ALGO-->>RM: expanded Dict + prompt_ids, group_ids, sample_ids

        RM->>ALGO: resolve_rollout_sde_indices(scheduler, step)
        ALGO-->>RM: Optional[Set[int]] sde_indices

        RM->>RP: distributed_sample(actor_group, batch, sde_indices)
        RP->>RA: generate(RolloutRequest)
        RA-->>RP: RolloutOutput
        RP-->>RM: List[RolloutOutput]

        RM->>RP: compute_rewards(reward_service, outputs, prompts)
        RP->>RS: compute_rewards(RewardRequest)
        RS-->>RP: RewardResponse
        RP-->>RM: (rewards: Tensor, components: Dict)

        RM->>ALGO: compute_advantages_with_components(rewards, group_ids, ...)
        ALGO-->>RM: advantages: Tensor

        RM->>ALGO: assemble_training_batch(outputs, rewards, advantages, prompts, sde_indices)
        ALGO-->>RM: TrainingBatch

        RM->>RB: push(rollout_id, TrainingBatch)
        RB-->>RM: {accepted: true}
        RM-->>-T: done

        T->>RB: pop_training_data(consumer_spec)
        RB-->>T: {training_data: ObjectRef}

        Note over T: === PHASE 2: Training ===
        T->>+TA: train(rollout_id, batch_ref)

        TA->>TA: ray.get(batch_ref) → TrainingBatch
        TA->>TE: prepare_batch(batch)
        TE-->>TA: sharded + on-device batch

        TA->>+TE: execute_prepared_batch(rollout_id, batch)

        loop for chunk in update_schedule.iter_update_chunks(batch)
            TE->>TE: optimizer.zero_grad()
            TE->>ALGO: compute_loss_and_backward(model, chunk, grad_accum_size)

            Note over ALGO: GRPO: iterate SDE timesteps<br/>NFT: iterate sampled timesteps
            ALGO->>ALGO: loss_fn.compute_loss(model, data)
            ALGO->>ALGO: loss.backward()
            ALGO-->>TE: (total_loss, metrics, ...)

            TE->>TE: clip_grad_norm()
            TE->>TE: optimizer.step()
            TE->>TE: lr_scheduler.step()
            TE->>EMA: post_optimizer_step(model)
        end

        TE-->>-TA: aggregated metrics
        TA-->>-T: metrics

        Note over T: === PHASE 3: Weight Sync ===
        T->>WS: sync(rollout_id)
        WS->>TA: get_state_dict()
        TA-->>WS: state_dict
        WS->>RM: update_weights(state_dict)
        RM->>RA: update_weights(state_dict)

        Note over T: === PHASE 4: Eval (periodic) ===
        T->>RM: eval(rollout_id)
        RM->>RA: generate(eval_request)
        RA-->>RM: RolloutOutput
        RM->>RS: compute_rewards(...)
        RS-->>RM: eval_rewards
        RM-->>T: eval_metrics
    end
```

---

## 4. Interface Objects — Class Diagram

```mermaid
classDiagram
    direction TB

    class RolloutRequest {
        +prompts: List~str~
        +num_inference_steps: int
        +guidance_scale: float
        +height: int
        +width: int
        +num_frames: int
        +sde_indices: Optional~Set~int~~
        +return_trajectories: bool
        +return_log_probs: bool
        +latents: Optional~Tensor~
        +kwargs: Dict
        +slice_prompts(start, end) RolloutRequest
    }

    class RolloutOutput {
        +latents: Tensor
        +timesteps: Tensor
        +trajectories: Optional~Tensor~
        +log_probs: Optional~LogProbData~
        +embeddings: Optional~PromptEmbeddings~
        +decoded_images: Optional~List~
        +step_indices: Optional~Tensor~
        +metadata: Dict
        +batch_size: int
        +sde_indices: Set~int~
        +to_device(device) RolloutOutput
        +validate_contract()
    }

    class LogProbData {
        +data: Dict~int_Tensor~
        +sde_indices: Set~int~
        +slice(start, end) LogProbData
        +reindex(indices) LogProbData
        +to_device(device) LogProbData
    }

    class PromptEmbeddings {
        +prompt_embeds: Tensor
        +pooled_prompt_embeds: Optional~Tensor~
        +encoder_attention_mask: Optional~Tensor~
        +negative_prompt_embeds: Optional~Tensor~
        +negative_pooled_prompt_embeds: Optional~Tensor~
        +text_ids: Optional~Tensor~
        +image_ids: Optional~Tensor~
        +slice(start, end) PromptEmbeddings
        +reindex(indices) PromptEmbeddings
        +to_dict() Dict
    }

    class BackwardTrainingBatch {
        +trajectories: Tensor
        +log_probs: LogProbData
        +timesteps: Tensor
        +advantages: Tensor
        +embeddings: PromptEmbeddings
        +rewards: Optional~Tensor~
        +prompts: Optional~List~str~~
        +prompt_ids: Optional~List~str~~
        +sample_ids: Optional~List~str~~
        +group_ids: Optional~List~str~~
        +step_indices: Optional~Tensor~
        +target_sde_indices: Optional~Set~int~~
        +batch_size: int
        +get_timestep_data_by_step(step_idx) TimestepData
        +slice(start, end) BackwardTrainingBatch
        +shuffle(indices) BackwardTrainingBatch
        +validate()
    }

    class ForwardTrainingBatch {
        +clean_latents: Tensor
        +advantages: Tensor
        +embeddings: PromptEmbeddings
        +rewards: Optional~Tensor~
        +prompts: Optional~List~str~~
        +prompt_ids: Optional~List~str~~
        +sample_ids: Optional~List~str~~
        +group_ids: Optional~List~str~~
        +timesteps: Optional~Tensor~
        +batch_size: int
        +slice(start, end) ForwardTrainingBatch
        +shuffle(indices) ForwardTrainingBatch
        +validate()
    }

    class TimestepData {
        +latents: Tensor
        +next_latents: Tensor
        +log_prob: Optional~Tensor~
        +sigma: Tensor
        +sigma_next: Tensor
        +timestep_idx: int
        +sigmas: Optional~Tensor~
        +to_device(device) TimestepData
    }

    class RewardRequest {
        +images: Optional~List~
        +videos: Optional~List~Tensor~~
        +prompts: List~str~
        +prompt_ids: Optional~List~str~~
        +sample_ids: Optional~List~str~~
        +group_ids: Optional~List~str~~
        +metadata: Optional~List~Dict~~
        +batch_size: int
    }

    class RewardResponse {
        +rewards: List~float~
        +reward_components: Dict~str_List~float~~
        +successes: List~bool~
        +compute_time: float
        +batch_size: int
    }

    class SamplingRequirements {
        +requires_trajectory: bool
        +requires_log_prob: bool
        +requires_embeddings: bool
        +extras: Dict
        +sde_ratio: float
        +requires_clean_latents: bool
        +is_trajectory_based: bool
        +is_forward_process: bool
        +is_mixed_sampling: bool
    }

    class EMASpec {
        +enable_eval_ema: bool
        +eval_decay: float
        +eval_update_interval: int
        +reference_mode: str
        +reference_decay: float
        +reference_decay_type: str
        +old_adapter_name: str
        +new_adapter_name: str
    }

    RolloutOutput --> LogProbData
    RolloutOutput --> PromptEmbeddings
    BackwardTrainingBatch --> LogProbData
    BackwardTrainingBatch --> PromptEmbeddings
    ForwardTrainingBatch --> PromptEmbeddings
    BackwardTrainingBatch --> TimestepData : get_timestep_data_by_step
```

---

## 5. Algorithm Hierarchy — Class Diagram

```mermaid
classDiagram
    direction TB

    class BaseAlgorithm {
        <<abstract>>
        +loss_fn: _Loss
        +clip_range: float
        +kl_coef: float
        +adv_normalization: str
        +samples_per_prompt: int
        +epsilon: float
        +clip_max: float
        +use_global_std: bool
        +trimmed_ratio: float
        +declared_requirements()$ Dict
        +from_config(config)$ BaseAlgorithm
        +from_args(args)$ BaseAlgorithm (optional convenience wrapper)
        +get_sampling_requirements()* SamplingRequirements
        +compute_advantages(rewards, group_ids) Tensor
        +compute_advantages_with_components(...) Tensor
        +build_sampling_batch(batch, K) Tuple
        +assemble_training_batch(...) TrainingBatch
        +compute_loss_and_backward(...)* Tuple
        +get_ema_spec() EMASpec
        +get_filtered_training_indices(sde_indices, n) Set
        +resolve_rollout_sde_indices(scheduler, step) Set
        +iter_sampling_request_batches(...) List
    }

    class GRPOAlgorithm {
        +_loss_cls = _GRPOLoss
        +clip_schedule: str
        +use_kl_penalty: bool
        +ratio_reg_coef: float
        +eta: float
        +sde_type: str
        +skip_last_timestep: bool
        +skip_initial_timesteps: int
        +model_type: str
        +_forward_plugin: ForwardPlugin
        +compute_loss(model, td, adv, emb, ...) Tuple
        +compute_loss_and_backward(model, batch, ...) Tuple
        +compute_log_prob(pred, sample, ...) Tuple
        +get_clip_range(progress) float
    }

    class _GRPOLoss {
        +algorithm: GRPOAlgorithm
        +declared_requirements()$ Dict
        +compute_loss(model, td, adv, emb, ...) Tuple
    }

    class MixGRPOAlgorithm {
        +sde_ratio: float
        +window_training: bool
        +_current_sde_indices: Set
        +get_sde_indices(num_steps) Set
        +set_sde_indices(indices)
        +get_training_indices(num_steps) Set
    }

    class NFTAlgorithm {
        +_loss_cls = _NFTLoss
        +beta: float
        +adv_clip_max: float
        +adv_mode: str
        +shift: float
        +use_ema: bool
        +ema_decay: float
        +old_adapter_name: str
        +new_adapter_name: str
        +_forward_plugin: ForwardPlugin
        +compute_loss(model, batch, ...) Tuple
        +compute_loss_and_backward(model, batch, ...) Tuple
        +forward_diffusion(x0, t, noise) Tuple
        +get_old_prediction(model, kwargs) Tensor
        +get_ref_prediction(model, kwargs) Tensor
        +process_advantages(advantages) Tensor
    }

    class _NFTLoss {
        +algorithm: NFTAlgorithm
        +declared_requirements()$ Dict
        +compute_loss(model, batch, ...) Tuple
    }

    BaseAlgorithm <|-- GRPOAlgorithm
    GRPOAlgorithm <|-- MixGRPOAlgorithm
    BaseAlgorithm <|-- NFTAlgorithm
    GRPOAlgorithm *-- _GRPOLoss : loss_fn
    NFTAlgorithm *-- _NFTLoss : loss_fn
```

---

## 6. Component Responsibilities

```mermaid
graph LR
    subgraph A["Algorithm (base.py, grpo.py, nft.py)"]
        A1["Declare sampling requirements\n(declared_requirements, get_sampling_requirements)"]
        A2["K-expand prompts + generate IDs\n(build_sampling_batch)"]
        A3["Compute advantages from rewards\n(compute_advantages, compute_advantages_with_components)"]
        A4["Assemble training batch\n(assemble_training_batch → _assemble_backward/forward_batch)"]
        A5["Compute loss + call backward\n(compute_loss_and_backward → loss_fn.compute_loss)"]
        A6["Declare EMA policy\n(get_ema_spec → EMASpec)"]
        A7["Filter training timesteps\n(get_filtered_training_indices, get_training_indices)"]
        A8["Resolve SDE indices for rollout\n(resolve_rollout_sde_indices)"]
    end

    subgraph B["RolloutManager (rollout_manager.py)"]
        B1["Orchestrate rollout pipeline\n(generate_and_push)"]
        B2["Fetch prompts from DataSource"]
        B3["Dispatch sampling to actors\n(via rollout_pipeline.distributed_sample)"]
        B4["Trigger reward computation\n(via rollout_pipeline.compute_rewards)"]
        B5["Delegate advantage to Algorithm"]
        B6["Delegate batch assembly to Algorithm"]
        B7["Push result to RolloutBuffer"]
        B8["Manage weight updates to RolloutActors"]
        B9["Run evaluation rollouts"]
    end

    subgraph C["TrainingActor (training_actor.py)"]
        C1["Load model, optimizer, scheduler"]
        C2["Create Algorithm instance (from_config)"]
        C3["Create EMAManager (from Algorithm.get_ema_spec)"]
        C4["Create TrainExecutor"]
        C5["Execute training step\n(train → executor.execute_prepared_batch)"]
        C6["Optional: local sampling\n(generate for direct-sampling mode)"]
        C7["Save/load checkpoints"]
    end

    subgraph D["TrainExecutor (train_executor.py)"]
        D1["Shard batch by DP rank\n(_shard_batch_by_rank)"]
        D2["Move batch to device\n(prepare_batch)"]
        D3["Drive update schedule loop\n(execute_prepared_batch)"]
        D4["Call algorithm.compute_loss_and_backward"]
        D5["Clip gradients, optimizer.step, lr_scheduler.step"]
        D6["Trigger EMA update\n(ema_manager.post_optimizer_step)"]
        D7["Shuffle samples before training"]
    end

    subgraph E["BufferActor (buffer_actor.py)"]
        E1["Apply filter plugins on push\n(FiniteTensor, RewardRange, MinSamples)"]
        E2["Partition batch for DP"]
        E3["Queue / group management"]
        E4["Pop ready batches for training"]
    end

    subgraph F["EMAManager (ema.py)"]
        F1["Maintain eval EMA weights"]
        F2["Maintain reference EMA (NFT old policy)"]
        F3["Swap EMA weights for eval"]
        F4["Restore original weights after eval"]
    end
```

---

## 7. Interface Contracts Between Components

```mermaid
graph TD
    subgraph Interfaces["Data Contracts"]
        I1["Dict with prompts\n(DataSource → RolloutManager)"]
        I2["SamplingRequirements\n(Algorithm → RolloutManager → contracts.py)"]
        I3["Expanded Dict\n+ prompt_ids, group_ids, sample_ids\n(Algorithm.build_sampling_batch → RolloutManager)"]
        I4["RolloutRequest\n(RolloutManager → RolloutActor)"]
        I5["RolloutOutput\n(RolloutActor → RolloutManager)"]
        I6["RewardRequest\n(RolloutManager → RewardService)"]
        I7["RewardResponse\n(RewardService → RolloutManager)"]
        I8["Tensor rewards + group_ids\n(RolloutManager → Algorithm.compute_advantages)"]
        I9["TrainingBatch\n(Algorithm.assemble → RolloutBuffer → TrainingActor)"]
        I10["TrainingUpdateChunk\n(UpdateSchedule → TrainExecutor → Algorithm)"]
        I11["TimestepData\n(BackwardTrainingBatch → _GRPOLoss)"]
        I12["EMASpec\n(Algorithm.get_ema_spec → EMAManager)"]
    end

    DS["DataSource"] -->|I1| RM["RolloutManager"]
    ALGO["Algorithm"] -->|I2| RM
    ALGO -->|I3| RM
    RM -->|I4| RA["RolloutActor"]
    RA -->|I5| RM
    RM -->|I6| RS["RewardService"]
    RS -->|I7| RM
    RM -->|I8| ALGO
    ALGO -->|I9| RB["RolloutBuffer"]
    RB -->|I9| TA["TrainingActor"]
    US["UpdateSchedule"] -->|I10| TE["TrainExecutor"]
    TE -->|I10| ALGO
    BTB["BackwardTrainingBatch"] -->|I11| LOSS["_GRPOLoss"]
    ALGO -->|I12| EMA["EMAManager"]
```

---

## 8. Training Step — Internal Call Chain

```mermaid
flowchart TD
    TA["TrainingActor.train(rollout_id, batch_ref)"] --> GET["ray.get(batch_ref) → TrainingBatch"]
    GET --> BUILD["_build_train_executor() → TrainExecutor"]
    BUILD --> PREP["executor.prepare_batch(batch)\n• shard by DP rank\n• move to device\n• validate"]
    PREP --> REPLAY{"BackwardBatch?\nreplay_log_probs?"}
    REPLAY -->|yes| REPLAYDO["_maybe_replay_old_log_probs(batch)"]
    REPLAY -->|no| EXEC
    REPLAYDO --> EXEC

    EXEC["executor.execute_prepared_batch(rollout_id, batch)"]
    EXEC --> SHUFFLE{"shuffle_samples?"}
    SHUFFLE -->|yes| DOSHUFFLE["shuffle_batch(batch, rollout_id)\n• deterministic permutation"]
    SHUFFLE -->|no| LOOP
    DOSHUFFLE --> LOOP

    LOOP["for chunk in update_schedule.iter_update_chunks(batch)"]
    LOOP --> ZERO["optimizer.zero_grad()"]
    ZERO --> ALGOFW["algorithm.compute_loss_and_backward(\n  model, chunk, grad_accum_size\n)"]

    subgraph AlgoInternal["Algorithm Internal (compute_loss_and_backward)"]
        direction TB
        PLAN["resolve_gradient_accumulation_plan\n→ mini_batch slices"]
        PLAN --> MBLOOP["for (start, end) in mini_batches"]
        MBLOOP --> TSLOOP["for t_idx in valid_step_indices"]
        TSLOOP --> GETDATA["batch.get_timestep_data_by_step(t_idx)\n→ TimestepData"]
        GETDATA --> LOSSCALL["loss_fn.compute_loss(\n  model, timestep_data, advantages, embeddings\n)"]
        LOSSCALL --> BACKWARD["(loss / scale).backward()"]
        BACKWARD --> TSLOOP
    end

    ALGOFW --> AlgoInternal
    AlgoInternal --> CLIP["clip_grad_norm()"]
    CLIP --> OPTSTEP["optimizer.step()"]
    OPTSTEP --> LRSTEP["lr_scheduler.step()"]
    LRSTEP --> EMASTEP["ema_manager.post_optimizer_step(model)"]
    EMASTEP --> LOOP
```

---

## 9. Rollout Pipeline — Internal Stages

```mermaid
flowchart TD
    START["RolloutManager._generate_training_data()"] --> STAGE0

    STAGE0["STAGE 0: _prepare_batch(data_source)\n→ Dict with prompts, metadata"]
    STAGE0 --> EXPAND

    EXPAND["Algorithm.build_sampling_batch(batch, K)\n• K-repeat prompts (prompt-major)\n• Generate prompt_ids, group_ids, sample_ids\n• Repeat metadata and latents"]
    EXPAND --> SDE

    SDE["Algorithm.resolve_rollout_sde_indices(scheduler, step)\n→ Set[int] SDE indices for MixGRPO\n→ None for pure GRPO/NFT"]
    SDE --> STAGE1

    STAGE1["STAGE 1: rollout_pipeline.distributed_sample()\n• Build RolloutRequest from batch\n• actor_group.generate(request)\n→ List of RolloutOutput"]
    STAGE1 --> VALIDATE

    VALIDATE["_validate_sampler_outputs()\n• Check contract (trajectory, log_prob, embeddings)\n• Attach missing embeddings if needed"]
    VALIDATE --> STAGE2

    STAGE2["STAGE 2: rollout_pipeline.compute_rewards()\n• Check precomputed_rewards in metadata\n• Extract images/videos from outputs\n• Build RewardRequest\n• reward_service.compute_rewards()\n→ rewards: Tensor, components: Dict"]
    STAGE2 --> STAGE3

    STAGE3["STAGE 3: Algorithm.compute_advantages_with_components()\n• If reward: compute_advantages(rewards, group_ids)\n• If advantage: per-component advantages + weighted sum\n• Uses normalize_grouped() or normalize_global()\n→ advantages: Tensor"]
    STAGE3 --> STAGE4

    STAGE4["STAGE 4: Algorithm.assemble_training_batch()\n• GRPO → _assemble_backward_batch() → BackwardTrainingBatch\n  - Concatenate trajectories, merge log_probs, validate timesteps\n  - Apply get_filtered_training_indices (skip_last_timestep, frozen_init)\n• NFT → _assemble_forward_batch() → ForwardTrainingBatch\n  - Concatenate clean_latents, merge embeddings\n→ TrainingBatch"]
    STAGE4 --> IDENTITY

    IDENTITY["_attach_batch_identities()\n• Set batch.prompt_ids, sample_ids, group_ids\n  from sampling batch metadata"]
    IDENTITY --> PUSH

    PUSH["RolloutBuffer.push()\n• Plugin chain: FiniteTensorFilter → RewardRangeFilter → MinSamplesGuard\n• Partition for DP if configured\n• ray.put()"]
```

---

## 10. Component × Method Matrix

```mermaid
graph LR
    subgraph Called["Methods Called ON Algorithm"]
        direction TB
        M1["get_sampling_requirements()"]
        M2["build_sampling_batch(batch, K)"]
        M3["resolve_rollout_sde_indices(scheduler, step)"]
        M4["iter_sampling_request_batches(batch, ...)"]
        M5["compute_advantages_with_components(rewards, ...)"]
        M6["assemble_training_batch(outputs, rewards, adv, ...)"]
        M7["compute_loss_and_backward(model, batch, ...)"]
        M8["get_ema_spec()"]
        M9["declared_requirements()"]
        M10["from_config(config) (+ optional from_args(args) wrapper)"]
        M11["get_sampler_validation_config(args)"]
    end

    subgraph Callers["Called BY"]
        direction TB
        C_RM["RolloutManager"]
        C_TA["TrainingActor"]
        C_TE["TrainExecutor"]
        C_CT["contracts.py"]
    end

    C_RM --> M1
    C_RM --> M2
    C_RM --> M3
    C_RM --> M4
    C_RM --> M5
    C_RM --> M6
    C_RM --> M11

    C_TE --> M7

    C_TA --> M8
    C_TA --> M10

    C_CT --> M9
```

---

## 11. Weight Sync Flow

```mermaid
sequenceDiagram
    participant T as train.py
    participant WS as WeightSync
    participant TG as TrainingGroup
    participant TA as TrainingActor
    participant RM as RolloutManager
    participant RA as RolloutActor

    T->>WS: sync(rollout_id)

    alt IPC mode
        WS->>TG: sync_weights_to_rollout_ipc()
        TG->>TA: get state_dict (rank 0)
        TA-->>TG: state_dict ObjectRef
        TG-->>WS: ObjectRef
        WS->>RM: update_weights(ObjectRef)
        RM->>RA: update_weights(ObjectRef)
    else Checkpoint mode
        WS->>TG: export_weights_to_path(path)
        TG->>TA: save state_dict to path
        WS->>RM: update_weights_from_path(path)
        RM->>RA: update_weights_from_path(path)
    else NCCL mode
        WS->>TG: sync_weights_to_rollout_nccl()
        Note over TG,RA: Direct GPU-to-GPU transfer
    end

    WS-->>T: sync_result
```

---

## 12. Buffer Plugin Chain

```mermaid
flowchart LR
    INPUT["TrainingBatch\n(from RolloutManager)"]
    INPUT --> P1

    subgraph PluginChain["Plugin Chain (sequential)"]
        P1["FiniteTensorFilterPlugin\n• Drop samples with NaN/Inf\n  in rewards or advantages"]
        P1 --> P2["RewardRangeFilterPlugin\n• Filter samples outside\n  [min_reward, max_reward]"]
        P2 --> P3["MinSamplesGuardPlugin\n• Reject batch if\n  batch_size < min_samples"]
    end

    P3 --> PARTITION{"Partition for DP?"}
    PARTITION -->|yes| SPLIT["maybe_partition_training_batch()\n→ List[TrainingBatch]\none per DP rank"]
    PARTITION -->|no| SINGLE["Single TrainingBatch"]
    SPLIT --> PUT["ray.put() each partition"]
    SINGLE --> PUT2["ray.put(batch)"]
    PUT --> QUEUE["Dispatch Queue"]
    PUT2 --> QUEUE
```

---

## 13. EMA Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: TrainingActor.init()

    state Created {
        [*] --> FromSpec: EMAManager.from_model_and_spec(\nmodel, algorithm.get_ema_spec())
        FromSpec --> EvalEMA: if enable_eval_ema
        FromSpec --> RefEMA: if reference_mode != none
        FromSpec --> DualAdapter: if reference_mode == nft_old_policy\nand use_lora
    }

    Created --> Training: execute_prepared_batch()

    state Training {
        [*] --> OptimizerStep
        OptimizerStep --> PostStep: post_optimizer_step(model)
        PostStep --> UpdateEvalEMA: eval_ema.step(params)
        PostStep --> UpdateRefEMA: reference_ema.step(params)
        PostStep --> UpdateAdapter: dual_adapter_ema.update(model)
        UpdateEvalEMA --> OptimizerStep
        UpdateRefEMA --> OptimizerStep
        UpdateAdapter --> OptimizerStep
    }

    Training --> Eval: should_eval = true

    state Eval {
        [*] --> ApplyEMA: apply_eval_ema(model)\nswap params ↔ ema_params
        ApplyEMA --> RunEval: generate + compute_rewards
        RunEval --> RestoreWeights: restore_from_eval(model)\nswap back
    }

    Eval --> Training: continue training
```
