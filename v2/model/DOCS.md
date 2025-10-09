# SUPER GIANT v2 Distillation System Documentation

## Overview

This is a knowledge distillation system for training a student model (GiantGPT) using outputs from a teacher model. The system implements top-K knowledge distillation where the student learns from the teacher's probability distributions over top candidate tokens.

## Core Components

### 1. Distillation Training Files

#### `distill_Training_step.py`
- **Purpose**: Implements the knowledge distillation training step
- **Key Functions**:
  - `_kd_loss_topk()`: Computes KL divergence between teacher and student distributions
  - `train_step()`: Combined loss function with configurable distillation weight
- **Features**:
  - Temperature-scaled softmax for teacher distributions
  - Numerically stable KL divergence calculation
  - Configurable distillation weight vs cross-entropy loss
  - JAX JIT compilation for performance

#### `distill_Run_training.py`
- **Purpose**: Main training loop orchestrator
- **Key Features**:
  - Dataset loading and batching
  - Model initialization and optimizer setup
  - Checkpoint management with resume capability
  - Learning rate scheduling with warmup
  - Training progress logging
- **Configuration**: Uses Config.yml for all hyperparameters

#### `distill_prepare_dataset.py`
- **Purpose**: Data loading and preprocessing for distillation
- **Key Components**:
  - `Batch` dataclass: Structured training data container
  - `_stream_iterator()`: Memory-efficient Arrow file streaming
  - `get_data()`: Factory functions for train/val data loaders
  - `_make_single_padded_window()`: Sequence padding and shifting
- **Data Format**:
  - Loads from Arrow files containing teacher outputs
  - Handles multiple JSON formats for top-K teacher predictions
  - Supports left-padding and sequence truncation

### 2. Model Architecture

#### `GiantGPT.py`
- **Architecture**: Decoder-only transformer
- **Key Features**:
  - Embedding layer with configurable dimensions
  - Multiple transformer blocks (`TinyTransformerBlock`)
  - Support for KV caching during inference
  - Configurable dropout and activation functions

### 3. Supporting Infrastructure

#### `Generate_faster.py`
- **Purpose**: Fast JIT-compiled inference for model evaluation
- **Features**:
  - JAX JIT compilation for prefill and decode phases
  - KV cache optimization
  - Temperature and top-K sampling
  - Performance timing and metrics

#### Configuration Files
- `Config.yml`: Central configuration with distillation-specific settings
- `checkpoint_manager.py`: Checkpoint saving/loading utilities
- `Save_params.py`: Simple parameter serialization
- `checkpoint_io.py`: NPZ format checkpoint handling

## Distillation Configuration

Key distillation parameters in `Config.yml`:

```yaml
use_distillation: true           # Enable/disable distillation
distill_topk: 8                 # Number of top-K alternatives to keep
distill_temperature: 2.0        # Temperature scaling for teacher softmax
distill_weight: 0.5             # Weight for KD loss vs CE loss

# Dataset configuration
use_custom_dataset: true
dataset_path: teacher_out
teacher_answers_filename: answers.arrow
```

## Training Process

1. **Data Preparation**: Teacher generates top-K predictions stored in Arrow format
2. **Dataset Loading**: Stream Arrow files with memory-efficient batching
3. **Loss Calculation**: Combined cross-entropy and KL divergence loss
4. **Optimization**: AdamW with warmup cosine decay learning rate schedule
5. **Checkpointing**: Automatic checkpoint saving and resume capability

## Loss Function

The total loss is a weighted combination:
```
loss = (1 - distill_weight) * CE_loss + distill_weight * KD_loss
```

Where:
- `CE_loss`: Standard cross-entropy with ground truth labels
- `KD_loss`: KL divergence between teacher and student distributions over top-K tokens

## Data Format Requirements

Teacher outputs should be in Arrow format with columns:
- `student_input_ids`: Tokenized input sequences
- `student_loss_mask`: Binary mask for answer tokens
- `topk_json_per_token`: JSON strings with top-K teacher predictions

## Usage

### Starting Training
```bash
python distill_Run_training.py
```

### Resuming Training
```bash
python distill_Run_training.py --resume
```

### Model Inference
```bash
python Generate_faster.py --checkpoint model_params.pkl --prompt "Your prompt here"
```

## Dependencies

- **JAX/Flax**: Core ML framework
- **Transformers**: Tokenizer handling
- **PyArrow**: Efficient data loading
- **Optax**: Optimizers and learning rate schedules
- **OmegaConf**: Configuration management

## Performance Features

- **JIT Compilation**: All training steps compiled for GPU acceleration
- **Memory Efficiency**: Streaming data loading avoids full dataset loading
- **KV Caching**: Optimized inference with cached attention keys/values
- **Checkpoint Management**: Automatic checkpointing with resume capability

## File Dependencies Graph

```
distill_Run_training.py
├── distill_Training_step.py
├── distill_prepare_dataset.py
├── GiantGPT.py
├── Save_params.py
├── checkpoint_manager.py
└── Config.yml
    └── checkpoint_io.py
        └── Transformer_block.py
```

This distillation system enables efficient training of smaller student models by leveraging knowledge from larger teacher models, with a focus on performance, memory efficiency, and ease of use.