# Model Inventory (FastWAM)

This document summarizes the train/inference models in this repository.

- FastWAM (`src/fastwam/models/wan22/fastwam.py`): base world-action model with video expert + action expert + MoT fusion.
- FastWAMJoint (`src/fastwam/models/wan22/fastwam_joint.py`): joint-attention variant where action attends to full video tokens.
- FastWAMIDM (`src/fastwam/models/wan22/fastwam_idm.py`): IDM teacher-forcing variant with an extra conditional-video branch during action denoising.
- Wan22Core (`src/fastwam/models/wan22/wan22.py`): standalone Wan2.2 TI2V core (video diffusion only).

Core submodules:
- `WanVideoDiT`: video diffusion transformer backbone.
- `ActionDiT`: action diffusion transformer backbone.
- `MoT`: mixture-of-transformers fusion over video/action experts.
- `WanVideoVAE38`: video VAE encode/decode latent space.
- `WanTextEncoder` + tokenizer: prompt encoding.
- `WanContinuousFlowMatchScheduler`: continuous flow-matching scheduler for train/infer timesteps.
