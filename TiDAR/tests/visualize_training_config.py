#!/usr/bin/env python3
"""
Interactive visualization of TiDAR training configuration.

Displays:
- Learning rate schedule across all stages
- Stage boundaries with token counts
- Loss coefficients per stage
- Interactive widgets for exploring different scenarios

Usage:
    python TiDAR/tests/visualize_training_config.py --config TiDAR/model/model_configs/old/sweep_1_lr1e4_a1_l0.yml
    python TiDAR/tests/visualize_training_config.py --config TiDAR/model/model_configs/old/sweep_5_lr3e5_a0p2_l0p8_t2_ctxmix.yml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, CheckButtons
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf


def load_config_simple(config_path: str, global_config_path: Optional[str] = None):
    """Load config without strict schema validation for visualization purposes."""
    model_dir = Path(config_path).resolve().parent
    project_root = Path(config_path).resolve().parents[2] if 'model_configs' in str(config_path) else Path(config_path).resolve().parent
    
    # Try to find TiDAR root
    tidar_root = PROJECT_ROOT / "TiDAR"
    
    if global_config_path:
        global_cfg_path = Path(global_config_path)
    else:
        global_cfg_path = tidar_root / "Global_Config.yml"
    
    # Load configs
    model_cfg = OmegaConf.load(config_path)
    global_cfg = OmegaConf.load(global_cfg_path) if global_cfg_path.exists() else OmegaConf.create()
    
    # Merge (model config overrides global)
    cfg = OmegaConf.merge(global_cfg, model_cfg)
    return cfg


@dataclass
class StageInfo:
    """Computed stage information for visualization."""
    name: str
    dataset: str
    seq_len: int
    epochs: int
    start_step: int
    end_step: int
    total_steps: int
    start_ratio: float
    end_ratio: float
    # Loss coefficients
    alpha: float
    beta: float
    rho: float
    chi: float
    delta: float
    # Computed metrics
    tokens_millions: float
    steps_per_epoch: int


def estimate_stage_steps(
    cfg: Any,
    batch_size: int,
    grad_accum: int,
) -> List[StageInfo]:
    """
    Estimate steps per stage based on end_ratio differences.
    
    Since we don't have actual dataset sizes, we use end_ratio as proportional
    allocation of total training.
    """
    stages = cfg.stages or []
    if not stages:
        return []
    
    # Get loss defaults
    loss_cfg = getattr(cfg.training, 'loss', None)
    default_alpha = float(getattr(loss_cfg, 'alpha', 1.0)) if loss_cfg else 1.0
    default_beta = float(getattr(loss_cfg, 'beta', 1.0)) if loss_cfg else 1.0
    default_rho = float(getattr(loss_cfg, 'rho', 0.0)) if loss_cfg else 0.0
    default_chi = float(getattr(loss_cfg, 'chi', 0.0)) if loss_cfg else 0.0
    default_delta = float(getattr(loss_cfg, 'delta', 0.0)) if loss_cfg else 0.0
    
    # For visualization, we'll assume a reasonable total step count
    # based on typical training runs. User can adjust via slider.
    # We'll compute relative proportions from end_ratios.
    
    stage_infos = []
    prev_ratio = 0.0
    
    for stage in stages:
        ratio_span = float(stage.end_ratio) - prev_ratio
        
        # Get stage-specific loss overrides
        stage_loss = getattr(stage, 'loss', None)
        
        def get_loss_val(key: str, default: float) -> float:
            if stage_loss is None:
                return default
            val = getattr(stage_loss, key, None)
            return float(val) if val is not None else default
        
        stage_infos.append(StageInfo(
            name=stage.name,
            dataset=stage.dataset,
            seq_len=int(stage.seq_len),
            epochs=int(stage.epochs),
            start_step=0,  # Will be computed later
            end_step=0,
            total_steps=0,
            start_ratio=prev_ratio,
            end_ratio=float(stage.end_ratio),
            alpha=get_loss_val('alpha', default_alpha),
            beta=get_loss_val('beta', default_beta),
            rho=get_loss_val('rho', default_rho),
            chi=get_loss_val('chi', default_chi),
            delta=get_loss_val('delta', default_delta),
            tokens_millions=0.0,
            steps_per_epoch=0,
        ))
        prev_ratio = float(stage.end_ratio)
    
    return stage_infos


def compute_steps_from_total(stages: List[StageInfo], total_steps: int) -> List[StageInfo]:
    """Compute actual step counts given total steps."""
    result = []
    current_step = 0
    
    for i, stage in enumerate(stages):
        ratio_span = stage.end_ratio - stage.start_ratio
        stage_steps = int(total_steps * ratio_span)
        
        # Ensure at least 1 step per stage
        stage_steps = max(1, stage_steps)
        
        # Adjust last stage to hit total exactly
        if i == len(stages) - 1:
            stage_steps = total_steps - current_step
        
        new_stage = StageInfo(
            name=stage.name,
            dataset=stage.dataset,
            seq_len=stage.seq_len,
            epochs=stage.epochs,
            start_step=current_step,
            end_step=current_step + stage_steps,
            total_steps=stage_steps,
            start_ratio=stage.start_ratio,
            end_ratio=stage.end_ratio,
            alpha=stage.alpha,
            beta=stage.beta,
            rho=stage.rho,
            chi=stage.chi,
            delta=stage.delta,
            tokens_millions=0.0,
            steps_per_epoch=stage_steps // max(1, stage.epochs),
        )
        result.append(new_stage)
        current_step += stage_steps
    
    return result


def warmup_cosine_schedule(
    step: np.ndarray,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> np.ndarray:
    """Compute warmup + cosine decay schedule."""
    # Warmup phase: linear ramp from 0 to peak
    warmup_lr = peak_lr * (step / max(1, warmup_steps))
    
    # Cosine decay phase
    decay_steps = total_steps - warmup_steps
    decay_progress = np.clip((step - warmup_steps) / max(1, decay_steps), 0, 1)
    cosine_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * decay_progress))
    
    # Combine: warmup where step < warmup_steps, cosine otherwise
    lr = np.where(step < warmup_steps, warmup_lr, cosine_lr)
    return lr


def create_visualization(cfg: Any, config_path: str, headless: bool = False, output_path: Optional[str] = None):
    """Create interactive matplotlib visualization."""
    
    # Extract config values
    batch_size = int(cfg.training.batch_size)
    grad_accum = int(cfg.training.gradient_accumulation)
    effective_batch = batch_size * grad_accum
    
    warmup_steps = int(cfg.optimizer.warmup_steps)
    peak_lr = float(cfg.optimizer.base_learning_rate)
    min_lr = float(cfg.optimizer.min_learning_rate)
    weight_decay = float(cfg.optimizer.weight_decay)
    grad_clip = float(cfg.optimizer.gradient_clip_norm)
    
    # Get stage info (relative proportions)
    stages_template = estimate_stage_steps(cfg, batch_size, grad_accum)
    
    # Initial total steps estimate (adjustable via slider)
    initial_total_steps = 50000
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'TiDAR Training Config: {Path(config_path).name}', fontsize=14, fontweight='bold')
    
    # Layout: 
    # Row 1: LR schedule (main plot)
    # Row 2: Loss coefficients per stage | Config summary
    # Row 3: Sliders and controls
    
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 2, 1.5, 0.5], hspace=0.35, wspace=0.3)
    
    ax_lr = fig.add_subplot(gs[0, :])  # LR schedule spans full width
    ax_loss = fig.add_subplot(gs[1, 0])  # Loss coefficients
    ax_info = fig.add_subplot(gs[1, 1])  # Config info
    ax_stages = fig.add_subplot(gs[2, :])  # Stage timeline
    
    # Slider axes
    ax_slider_total = fig.add_axes([0.15, 0.08, 0.55, 0.02])
    ax_slider_warmup = fig.add_axes([0.15, 0.04, 0.55, 0.02])
    
    # Button axes
    ax_btn_reset = fig.add_axes([0.8, 0.04, 0.08, 0.04])
    ax_btn_export = fig.add_axes([0.8, 0.08, 0.08, 0.04])
    
    # Create sliders
    slider_total = Slider(
        ax_slider_total, 'Total Steps', 1000, 200000,
        valinit=initial_total_steps, valstep=1000,
        color='steelblue'
    )
    slider_warmup = Slider(
        ax_slider_warmup, 'Warmup Steps', 0, 10000,
        valinit=warmup_steps, valstep=100,
        color='coral'
    )
    
    # Create buttons
    btn_reset = Button(ax_btn_reset, 'Reset', color='lightgray')
    btn_export = Button(ax_btn_export, 'Print Info', color='lightgreen')
    
    # Store plot elements for updating
    plot_elements = {}
    
    # Generate consistent colors for stages (once, not per update)
    num_stages = len(stages_template)
    # Use a colormap with good distinguishability
    cmap = plt.colormaps.get_cmap('tab10')
    stage_colors = [cmap(i / max(10, num_stages)) for i in range(num_stages)]
    
    def update_plots(val=None):
        """Update all plots based on slider values."""
        total_steps = int(slider_total.val)
        current_warmup = int(slider_warmup.val)
        
        # Compute stages with current total
        stages = compute_steps_from_total(stages_template, total_steps)
        
        # Generate LR curve
        steps = np.arange(0, total_steps + 1)
        lr_values = warmup_cosine_schedule(steps, current_warmup, total_steps, peak_lr, min_lr)
        
        # Clear and redraw LR plot
        ax_lr.clear()
        ax_lr.plot(steps, lr_values * 1e4, 'b-', linewidth=1.5, label='Learning Rate')
        ax_lr.axvline(x=current_warmup, color='coral', linestyle='--', alpha=0.7, label=f'Warmup end ({current_warmup})')
        ax_lr.axhline(y=min_lr * 1e4, color='gray', linestyle=':', alpha=0.5, label=f'Min LR ({min_lr:.1e})')
        
        # Add stage boundaries with consistent colors
        for i, stage in enumerate(stages):
            if i > 0:
                ax_lr.axvline(x=stage.start_step, color=stage_colors[i], linestyle='-', alpha=0.6, linewidth=2)
            
            # Shade stage region
            ax_lr.axvspan(stage.start_step, stage.end_step, alpha=0.15, color=stage_colors[i])
            
            # Label at top - use "Stage N" for cleaner charts
            mid_step = (stage.start_step + stage.end_step) / 2
            ax_lr.text(mid_step, peak_lr * 1e4 * 1.05, f'Stage {i+1}', 
                      ha='center', va='bottom', fontsize=9, fontweight='bold',
                      color=stage_colors[i])
        
        ax_lr.set_xlabel('Training Step')
        ax_lr.set_ylabel('Learning Rate (x1e-4)')
        ax_lr.set_title('Learning Rate Schedule with Stage Boundaries')
        ax_lr.legend(loc='upper right', fontsize=8)
        ax_lr.set_xlim(0, total_steps)
        ax_lr.set_ylim(0, peak_lr * 1e4 * 1.15)
        ax_lr.grid(True, alpha=0.3)
        
        # Warmup percentage annotation
        warmup_pct = 100 * current_warmup / total_steps
        ax_lr.annotate(f'Warmup: {warmup_pct:.1f}%', 
                      xy=(current_warmup, peak_lr * 1e4), 
                      xytext=(current_warmup + total_steps * 0.05, peak_lr * 1e4 * 0.9),
                      fontsize=9, color='coral',
                      arrowprops=dict(arrowstyle='->', color='coral', alpha=0.5))
        
        # Update loss coefficients bar chart
        ax_loss.clear()
        # Use "Stage N" labels with colored backgrounds
        stage_labels = [f'Stage {i+1}' for i in range(len(stages))]
        x = np.arange(len(stages))
        width = 0.15
        
        ax_loss.bar(x - 2*width, [s.alpha for s in stages], width, label='alpha (AR)', color='steelblue')
        ax_loss.bar(x - width, [s.beta for s in stages], width, label='beta (Diff)', color='coral')
        ax_loss.bar(x, [s.rho for s in stages], width, label='rho (KL_fwd)', color='green')
        ax_loss.bar(x + width, [s.chi for s in stages], width, label='chi (KL_rev)', color='purple')
        ax_loss.bar(x + 2*width, [s.delta for s in stages], width, label='delta (Hard)', color='orange')
        
        ax_loss.set_xlabel('Stage')
        ax_loss.set_ylabel('Coefficient')
        ax_loss.set_title('Loss Coefficients per Stage')
        ax_loss.set_xticks(x)
        # Color the x-axis labels to match stage colors
        ax_loss.set_xticklabels(stage_labels, fontsize=9, fontweight='bold')
        for i, tick_label in enumerate(ax_loss.get_xticklabels()):
            tick_label.set_color(stage_colors[i])
        ax_loss.legend(loc='upper right', fontsize=7)
        ax_loss.grid(True, alpha=0.3, axis='y')
        
        # Update config info panel - stage legend with colors + config summary
        ax_info.clear()
        ax_info.axis('off')
        
        # Draw stage legend with colored text (not monospace block)
        y_pos = 0.98
        ax_info.text(0.02, y_pos, "Stage Legend:", transform=ax_info.transAxes,
                    fontsize=9, fontweight='bold', verticalalignment='top')
        y_pos -= 0.06
        
        for i, stage in enumerate(stages):
            # Truncate long stage names
            name_truncated = stage.name[:20] + "..." if len(stage.name) > 20 else stage.name
            legend_line = f"Stage {i+1} - {name_truncated} - ctx {stage.seq_len} - {stage.total_steps:,} steps"
            ax_info.text(0.04, y_pos, legend_line, transform=ax_info.transAxes,
                        fontsize=8, verticalalignment='top', color=stage_colors[i],
                        fontweight='bold')
            y_pos -= 0.05
        
        y_pos -= 0.02
        
        # Config summary (simpler box)
        config_text = f"""Configuration Summary
Model: {cfg.model.num_layers}L / {cfg.model.embedding_size}d / {cfg.model.num_heads}H
Context: {cfg.model.context_length} | Draft: {cfg.tidar.draft_length}

Optimizer:
  LR: {peak_lr:.1e} -> {min_lr:.1e}
  Warmup: {current_warmup} ({warmup_pct:.1f}%)
  WD: {weight_decay} | Clip: {grad_clip}

Training:
  Batch: {batch_size} x {grad_accum} = {effective_batch}
  Total: {total_steps:,} steps"""
        
        ax_info.text(0.02, y_pos, config_text, transform=ax_info.transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # Update stage timeline with alternating label positions
        ax_stages.clear()
        for i, stage in enumerate(stages):
            rect = mpatches.FancyBboxPatch(
                (stage.start_step, 0.35), stage.total_steps, 0.3,
                boxstyle="round,pad=0.02",
                facecolor=stage_colors[i], edgecolor='black', alpha=0.7
            )
            ax_stages.add_patch(rect)
            
            # Stage number inside the box
            mid = (stage.start_step + stage.end_step) / 2
            ax_stages.text(mid, 0.5, f"S{i+1}", ha='center', va='center', fontsize=9,
                          fontweight='bold', color='white')
            
            # Detailed info alternates above/below to avoid overlap
            if i % 2 == 0:
                # Above
                info_y = 0.78
                va = 'bottom'
            else:
                # Below
                info_y = 0.22
                va = 'top'
            
            # Truncate name for timeline
            name_short = stage.name[:12] + ".." if len(stage.name) > 12 else stage.name
            info = f"{name_short}\n{stage.total_steps:,}"
            ax_stages.text(mid, info_y, info, ha='center', va=va, fontsize=7,
                          color=stage_colors[i], fontweight='bold')
            
            # LR at stage boundaries (bottom)
            lr_start = warmup_cosine_schedule(np.array([stage.start_step]), current_warmup, total_steps, peak_lr, min_lr)[0]
            ax_stages.text(stage.start_step, 0.02, f'{lr_start:.0e}', ha='center', fontsize=5, color='blue')
        
        ax_stages.set_xlim(0, total_steps)
        ax_stages.set_ylim(-0.05, 1.05)
        ax_stages.set_xlabel('Training Step')
        ax_stages.set_title('Stage Timeline')
        ax_stages.set_yticks([])
        ax_stages.grid(True, alpha=0.3, axis='x')
        
        fig.canvas.draw_idle()
    
    def reset_sliders(event):
        slider_total.reset()
        slider_warmup.reset()
    
    def print_info(event):
        total_steps = int(slider_total.val)
        current_warmup = int(slider_warmup.val)
        stages = compute_steps_from_total(stages_template, total_steps)
        
        print("\n" + "="*70)
        print(f"CONFIG: {config_path}")
        print("="*70)
        
        # Print stage legend first
        print("\nStage Legend:")
        print("-"*70)
        for i, stage in enumerate(stages):
            print(f"  Stage {i+1} - {stage.name} - ctx {stage.seq_len} - {stage.total_steps:,} steps")
        print("-"*70)
        
        print(f"\nOptimizer Settings:")
        print(f"  Peak LR: {peak_lr:.1e}")
        print(f"  Min LR: {min_lr:.1e}")
        print(f"  Warmup: {current_warmup} steps ({100*current_warmup/total_steps:.1f}%)")
        print(f"  Total steps: {total_steps:,}")
        print(f"\nStage Details:")
        print("-"*70)
        
        for i, stage in enumerate(stages):
            lr_start = warmup_cosine_schedule(np.array([stage.start_step]), current_warmup, total_steps, peak_lr, min_lr)[0]
            lr_end = warmup_cosine_schedule(np.array([stage.end_step-1]), current_warmup, total_steps, peak_lr, min_lr)[0]
            
            print(f"\n  Stage {i+1} ({stage.name}):")
            print(f"    Steps: {stage.start_step:,} -> {stage.end_step:,} ({stage.total_steps:,} total)")
            print(f"    LR range: {lr_start:.2e} -> {lr_end:.2e}")
            print(f"    seq_len: {stage.seq_len}, epochs: {stage.epochs}")
            print(f"    Loss: alpha={stage.alpha}, beta={stage.beta}, rho={stage.rho}, chi={stage.chi}, delta={stage.delta}")
        
        print("\n" + "="*70 + "\n")
    
    # Connect callbacks
    slider_total.on_changed(update_plots)
    slider_warmup.on_changed(update_plots)
    btn_reset.on_clicked(reset_sliders)
    btn_export.on_clicked(print_info)
    
    # Initial draw
    update_plots()
    
    plt.tight_layout(rect=(0, 0.12, 1, 0.96))
    
    if headless or output_path:
        out_file = output_path or f"{Path(config_path).stem}_lr_schedule.png"
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {out_file}")
        # Also print info to console in headless mode
        print_info(None)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TiDAR training configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python TiDAR/tests/visualize_training_config.py --config TiDAR/model/model_configs/old/sweep_1_lr1e4_a1_l0.yml
    python TiDAR/tests/visualize_training_config.py --config TiDAR/model/Config.yml
    python TiDAR/tests/visualize_training_config.py --config TiDAR/model/Config.yml --headless -o output.png
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to model config YAML file'
    )
    parser.add_argument(
        '--global_config', '-g',
        type=str,
        default=None,
        help='Path to global config YAML (optional, defaults to TiDAR/Global_Config.yml)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI (save to file instead)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for the visualization (PNG)'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to project root
        config_path = PROJECT_ROOT / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print(f"Loading config: {config_path}")
    cfg = load_config_simple(str(config_path), args.global_config)
    
    create_visualization(cfg, str(config_path), headless=args.headless, output_path=args.output)


if __name__ == '__main__':
    main()
