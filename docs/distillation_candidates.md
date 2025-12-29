# Distillation Student Model Candidates

## Objective
Select a lightweight pre-trained UNet to serve as the "Student" for distilling the MuseTalk "Teacher".
Target: <8s latency (requires >3x speedup).
Model Compatibility: Must match SD1.5 latent space (4x64x64) and Cross-Attention dim (768) to reuse VAE and Whisper.

## Candidates

| Model | ID | Params | Architecture Changes | Compatibility | Speedup Est. |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stable Diffusion v1.5** | `runwayml/stable-diffusion-v1-5` | 860M | Baseline | Native | 1x |
| **BK-SDM Tiny** | `nota-ai/bk-sdm-tiny` | **323M** | - Removed 4th Down/Up Block<br>- Reduced Layers per Block (2->1) | **High** (Matches channels 320/640/1280, dim 768) | **~2.0-2.5x** |
| **Segmind Tiny-SD** | `segmind/tiny-sd` | **323M** | Same as BK-SDM (Distilled from Realistic Vision) | **High** | **~2.0-2.5x** |
| **SSD-1B** | `segmind/SSD-1B` | 1.3B | SDXL based (Incompatible) | **Low** (Wrong Latent Dim) | N/A |

## Detailed Comparison

### 1. BK-SDM-Tiny (`nota-ai/bk-sdm-tiny`)
- **Origin**: "All but One: Knowledge Distillation for Compressed Diffusion Models" (CVPR 2024).
- **Structure**:
    - **In Channels**: 4 (Matches VAE)
    - **Block Channels**: `[320, 640, 1280]` (Matches SD1.5 first 3 blocks)
    - **Cross Attention**: 768 (Matches Whisper)
- **Pros**:
    - Scientific validation.
    - Clean weights.
    - Perfectly aligns with MuseTalk's input requirements.
- **Cons**:
    - Might require retraining `ReferenceNet` to match the smaller architecture (recommended for speed).

### 2. Segmind Tiny-SD (`segmind/tiny-sd`)
- **Origin**: Industry distillation (Segmind).
- **Structure**: Identical to BK-SDM-Tiny.
- **Pros**:
    - Potentially better aesthetic priors (Realistic Vision base).
- **Cons**:
    - "Unsafe" pickle serialization warning in checks (minor).

## Recommendation
**Select `nota-ai/bk-sdm-tiny`**.
It is a rigorous academic baseline, widely supported, and its architecture is a strict subset of MuseTalk, making the "surgery" to transfer weights or train from scratch straightforward.

## Implementation Plan
1.  **Student Initialization**: Load `bk-sdm-tiny` UNet.
2.  **ReferenceNet Adaptation**: Prune MuseTalk's `ReferenceNet` to match Student's structure (remove 4th block, reduce layers) OR train a new Tiny-ReferenceNet.
3.  **Distillation Training**:
    -   **Loss**: $L_{distill} = ||\epsilon_{teacher} - \epsilon_{student}||^2$ (Logit Matching).
    -   **Data**: Utilize the project's avatar dataset.
