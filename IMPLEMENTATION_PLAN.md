# Bio-Mimetic "Product of Experts" — Full Implementation Plan

> **Purpose:** A comprehensive, step-by-step plan to implement the Bio-Mimetic "Product of Experts" architecture for the Brain-to-Text '25 Kaggle competition. This document is structured so that each section can be executed independently by an LLM or developer.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [New Files to Create](#2-new-files-to-create)
3. [Existing Files to Modify](#3-existing-files-to-modify)
4. [Detailed Implementation Steps](#4-detailed-implementation-steps)
5. [Verification Plan](#5-verification-plan)

---

## 1. Architecture Overview

```mermaid
flowchart LR
    subgraph Neural Data "512 Electrodes"
        A4["Area 4 (64 TC + 64 SBP)"]
        A6v["Ventral 6v (64 TC + 64 SBP)"]
        A55b["Area 55b (64 TC + 64 SBP)"]
    end

    subgraph Expert Networks
        NetA["Net A: Body Expert<br/>(7-dim output)"]
        NetB["Net B: Semantic Expert<br/>(41-dim phoneme output)"]
        NetC["Net C: Silence Gate<br/>(1-dim)"]
        NetD["Net D: EOS Gate<br/>(1-dim)"]
    end

    A4 --> NetA
    A6v --> NetA
    A6v --> NetB
    A55b --> NetB
    A4 --> NetC
    A55b --> NetD

    NetA --> Fusion["Algorithmic Fusion<br/>(Product of Experts)"]
    NetB --> Fusion
    NetC --> Gate["Silence Gate Masking"]
    Fusion --> Gate
    Gate --> WFST["WFST Decoder"]
    WFST --> LLM["LLM Rescoring"]
    LLM --> CSV["Kaggle CSV"]
```

### Electrode Index Mapping

| Area | Threshold Crossings (TC) | Spike Band Power (SBP) |
|------|--------------------------|------------------------|
| Ventral 6v | 0-63 | 256-319 |
| Area 4 | 64-127 | 320-383 |
| Area 55b | 128-191 | 384-447 |
| ~~Dorsal 6v~~ | ~~192-255~~ | ~~448-511~~ | **(NOT USED)**

### Network Input Assignments

| Network | Areas | Electrode Indices (Python) | Input Dim |
|---------|-------|---------------------------|-----------|
| **Net A** (Body) | Area 4 + Ventral 6v | `[64:128, 0:64, 320:384, 256:320]` | 256 |
| **Net B** (Semantic) | Area 55b + Ventral 6v | `[128:192, 0:64, 384:448, 256:320]` | 256 |
| **Net C** (Silence) | Area 4 only | `[64:128, 320:384]` | 128 |
| **Net D** (EOS) | Area 55b only | `[128:192, 384:448]` | 128 |

---

## 2. New Files to Create

### 2.1 `model_training/constants.py`

**Purpose:** Centralize all constants (electrode indices, phoneme mappings, body-part matrix).

```python
# FILE: model_training/constants.py
# ACTION: CREATE NEW FILE

import torch

# =============================================================================
# Phoneme Mapping (Arpabet, 41 classes: BLANK + 39 phonemes + SIL)
# =============================================================================
LOGIT_TO_PHONEME = [
    'BLANK',    # Index 0: CTC blank
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    ' | ',       # Index 40: Silence token
]
N_PHONEMES = len(LOGIT_TO_PHONEME)  # 41

# =============================================================================
# Electrode Subset Indices (0-indexed, for 512-dim input)
# =============================================================================
# Threshold Crossings (TC): 0-255, Spike Band Power (SBP): 256-511
# NOTE: We only use Ventral 6v (NOT Dorsal 6v)

# Net A: Area 4 + Ventral 6v
NET_A_TC = list(range(64, 128)) + list(range(0, 64))   # 128
NET_A_SBP = list(range(320, 384)) + list(range(256, 320))  # 128
NET_A_INDICES = NET_A_TC + NET_A_SBP  # 256 total

# Net B: Area 55b + Ventral 6v
NET_B_TC = list(range(128, 192)) + list(range(0, 64))  # 128
NET_B_SBP = list(range(384, 448)) + list(range(256, 320))  # 128
NET_B_INDICES = NET_B_TC + NET_B_SBP  # 256 total

# Net C: Area 4 only (for silence detection)
NET_C_INDICES = list(range(64, 128)) + list(range(320, 384))  # 128 total

# Net D: Area 55b only (for EOS detection)
NET_D_INDICES = list(range(128, 192)) + list(range(384, 448))  # 128 total

# =============================================================================
# Body Configuration Labels (7 articulators)
# =============================================================================
BODY_PARTS = ['lips', 'tongue_tip', 'tongue_body', 'tongue_root', 'glottis', 'velum', 'jaw']
N_BODY_PARTS = len(BODY_PARTS)  # 7

# =============================================================================
# IPA-to-Body Matrix (M): Shape [N_PHONEMES, N_BODY_PARTS] = [41, 7]
# Binary matrix: M[i, j] = 1 if phoneme i uses body part j
# Based on standard IPA phonetic features.
# =============================================================================
def build_ipa_to_body_matrix():
    """
    Builds a [41, 7] binary matrix mapping phonemes to body parts.
    Columns: [lips, tongue_tip, tongue_body, tongue_root, glottis, velum, jaw]
    """
    M = torch.zeros(N_PHONEMES, N_BODY_PARTS)
    
    # BLANK (index 0): No articulation
    # SIL (index 40): No articulation
    
    # Vowels: primarily use tongue_body, tongue_root, glottis, jaw
    vowels = {'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6,
              'EH': 11, 'ER': 12, 'EY': 13, 'IH': 17, 'IY': 18,
              'OW': 25, 'OY': 26, 'UH': 33, 'UW': 34}
    for ph, idx in vowels.items():
        M[idx, 2] = 1  # tongue_body
        M[idx, 3] = 1  # tongue_root
        M[idx, 4] = 1  # glottis (voiced)
        M[idx, 6] = 1  # jaw
    
    # Bilabial: B, P, M, W - use lips
    bilabials = {'B': 7, 'P': 27, 'M': 22, 'W': 36}
    for ph, idx in bilabials.items():
        M[idx, 0] = 1  # lips
        if ph in ['B', 'M', 'W']:
            M[idx, 4] = 1  # glottis (voiced)
    
    # Labiodental: F, V - use lips
    labiodentals = {'F': 14, 'V': 35}
    for ph, idx in labiodentals.items():
        M[idx, 0] = 1  # lips
        if ph == 'V':
            M[idx, 4] = 1  # glottis (voiced)
    
    # Dental/Alveolar: T, D, TH, DH, S, Z, N, L, R - use tongue_tip
    alveolars = {'T': 31, 'D': 9, 'TH': 32, 'DH': 10, 'S': 29, 'Z': 38,
                 'N': 23, 'L': 21, 'R': 28}
    for ph, idx in alveolars.items():
        M[idx, 1] = 1  # tongue_tip
        if ph in ['D', 'DH', 'Z', 'N', 'L', 'R']:
            M[idx, 4] = 1  # glottis (voiced)
    
    # Postalveolar/Palatal: SH, ZH, CH, JH, Y - use tongue_body
    postalveolars = {'SH': 30, 'ZH': 39, 'CH': 8, 'JH': 19, 'Y': 37}
    for ph, idx in postalveolars.items():
        M[idx, 2] = 1  # tongue_body
        if ph in ['ZH', 'JH', 'Y']:
            M[idx, 4] = 1  # glottis (voiced)
    
    # Velar: K, G, NG - use tongue_body (back) and velum
    velars = {'K': 20, 'G': 15, 'NG': 24}
    for ph, idx in velars.items():
        M[idx, 2] = 1  # tongue_body
        M[idx, 5] = 1  # velum
        if ph in ['G', 'NG']:
            M[idx, 4] = 1  # glottis (voiced)
    
    # Glottal: HH - use glottis
    M[16, 4] = 1  # HH -> glottis
    
    return M

IPA_TO_BODY_MATRIX = build_ipa_to_body_matrix()
```

---

### 2.2 `model_training/expert_models.py`

**Purpose:** Define the four small expert networks.

```python
# FILE: model_training/expert_models.py
# ACTION: CREATE NEW FILE

import torch
import torch.nn as nn
from constants import NET_A_INDICES, NET_B_INDICES, NET_C_INDICES, NET_D_INDICES, N_PHONEMES, N_BODY_PARTS


class BodyExpertNet(nn.Module):
    """
    Net A: Predicts 7-dim body configuration from Area 4 + Ventral 6v electrodes.
    Architecture: 2-layer GRU (256 units) + Linear output with Sigmoid.
    Loss: BCEWithLogitsLoss (multi-label binary classification).
    """
    def __init__(self, input_dim=256, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.indices = NET_A_INDICES
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, N_BODY_PARTS)  # 7 outputs
        
    def forward(self, x):
        # x: [B, T, input_dim]
        out, _ = self.gru(x)
        logits = self.fc(out)  # [B, T, 7]
        return logits  # Raw logits; apply sigmoid externally for inference


class SemanticExpertNet(nn.Module):
    """
    Net B: Predicts 41-dim phoneme probabilities from Area 55b + Ventral 6v electrodes.
    Architecture: 3-layer GRU (384 units) + Linear output.
    Loss: CTCLoss.
    """
    def __init__(self, input_dim=256, hidden_dim=384, n_layers=3, dropout=0.3, n_classes=N_PHONEMES):
        super().__init__()
        self.indices = NET_B_INDICES
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)  # 41 outputs
        
    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.fc(out)  # [B, T, 41]
        return logits  # Raw logits; apply log_softmax for CTC


class SilenceGateNet(nn.Module):
    """
    Net C: Predicts silence probability from Area 4 electrodes.
    Architecture: Small MLP (time-aggregated energy).
    Loss: BCEWithLogitsLoss (binary classification per timestep).
    """
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.indices = NET_C_INDICES
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 1 output: silence logit
        )
        
    def forward(self, x):
        # x: [B, T, input_dim]
        logits = self.net(x)  # [B, T, 1]
        return logits


class EOSGateNet(nn.Module):
    """
    Net D: Predicts End-of-Sentence probability from Area 55b electrodes.
    Architecture: Small MLP.
    Loss: BCEWithLogitsLoss.
    """
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.indices = NET_D_INDICES
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 1 output: EOS logit
        )
        
    def forward(self, x):
        logits = self.net(x)  # [B, T, 1]
        return logits
```

---

### 2.3 `model_training/fusion.py`

**Purpose:** Implement the Product-of-Experts fusion and gating logic.

#### Fusion Algorithm (Corrected)

The fusion works as follows:
1. **Net B** outputs phoneme probabilities: `P_phoneme[p]` for each phoneme `p` (41 classes)
2. **Net A** outputs body configuration probabilities: `P_body[b]` for each body part `b` (7 dims)
3. **IPA→Body Matrix (M):** `M[p, b] = 1` if phoneme `p` uses body part `b`
4. **Body-Derived Score:** For each phoneme `p`, compute how well Net A's body probs match that phoneme's expected body configuration:
   ```
   body_score[p] = ∏_{b where M[p,b]=1} P_body[b]
   ```
   (Product of body probs for body parts used by phoneme `p`)
5. **Final Fused Probability:**
   ```
   P_final[p] = P_phoneme[p] × body_score[p]
   ```
6. **Re-normalize** to get valid probability distribution.

```python
# FILE: model_training/fusion.py
# ACTION: CREATE NEW FILE

import torch
import torch.nn.functional as F
from constants import IPA_TO_BODY_MATRIX


def fuse_experts(p_body: torch.Tensor, p_phoneme: torch.Tensor, M: torch.Tensor = None) -> torch.Tensor:
    """
    Product of Experts fusion.

    For each phoneme p:
      1. Look up its body vector M[p] (7-dim binary mask indicating which body parts it uses)
      2. Multiply Net A's body probs element-wise with M[p], then take the product
         of the relevant body part probs (where M[p,b]=1)
      3. Multiply this body-derived score with Net B's phoneme probability

    Args:
        p_body: [B, T, 7] Body configuration probabilities (sigmoid output from Net A).
        p_phoneme: [B, T, 41] Phoneme probabilities (softmax output from Net B).
        M: [41, 7] Binary matrix mapping phonemes to body parts. Defaults to IPA_TO_BODY_MATRIX.

    Returns:
        p_fused: [B, T, 41] Fused phoneme probabilities (re-normalized).
    """
    if M is None:
        M = IPA_TO_BODY_MATRIX.to(p_body.device)
    
    # M is [41, 7], p_body is [B, T, 7]
    # For each phoneme p, we want: product of p_body[b] where M[p, b] = 1
    # 
    # Implementation:
    # - Where M[p,b] = 0, we don't want that body part to affect the score.
    # - We use: score = exp(sum(log(p_body) * M))
    # - This computes the product of selected body probs.
    #
    # To handle p_body = 0 gracefully, add small epsilon before log.
    
    log_p_body = torch.log(p_body + 1e-9)  # [B, T, 7]
    
    # For each phoneme p, sum the log-probs of the body parts it uses
    # log_p_body: [B, T, 7], M: [41, 7]
    # We want: [B, T, 41] where result[..., p] = sum over b of (log_p_body[..., b] * M[p, b])
    # This is: log_p_body @ M.T
    log_body_score = torch.matmul(log_p_body, M.T)  # [B, T, 41]
    
    # Convert back from log-space
    body_score = torch.exp(log_body_score)  # [B, T, 41]
    
    # Product of Experts: multiply phoneme prob by body-derived score
    p_fused = p_phoneme * body_score  # [B, T, 41]
    
    # Re-normalize to valid probability distribution
    p_fused = p_fused / (p_fused.sum(dim=-1, keepdim=True) + 1e-9)
    
    return p_fused


def apply_silence_gate(p_fused: torch.Tensor, silence_logits: torch.Tensor, blank_idx: int = 0) -> torch.Tensor:
    """
    Apply silence gating. If silence prob > 0.5, force BLANK token.

    Args:
        p_fused: [B, T, 41] Fused phoneme probabilities.
        silence_logits: [B, T, 1] Raw silence logits from Net C.
        blank_idx: Index of BLANK token (default 0).

    Returns:
        p_gated: [B, T, 41] Gated phoneme probabilities.
    """
    silence_prob = torch.sigmoid(silence_logits)  # [B, T, 1]
    mask = (silence_prob > 0.5).squeeze(-1)  # [B, T]
    
    p_gated = p_fused.clone()
    p_gated[mask, :] = 0.0
    p_gated[mask, blank_idx] = 1.0
    
    return p_gated


def apply_eos_gate(eos_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Detect End-of-Sentence.

    Args:
        eos_logits: [B, T, 1] Raw EOS logits from Net D.
        threshold: Probability threshold for EOS detection.

    Returns:
        eos_mask: [B, T] Boolean mask where True = EOS detected.
    """
    eos_prob = torch.sigmoid(eos_logits)  # [B, T, 1]
    eos_mask = (eos_prob > threshold).squeeze(-1)  # [B, T]
    return eos_mask
```


---

### 2.4 `model_training/expert_trainer.py`

**Purpose:** Training class for all four expert networks (modeled after `rnn_trainer.py`).

```python
# FILE: model_training/expert_trainer.py
# ACTION: CREATE NEW FILE
# SIZE: ~500 lines (similar structure to rnn_trainer.py)

# HIGH-LEVEL STRUCTURE:
# class ExpertTrainer:
#     def __init__(self, args, expert_type):
#         # expert_type: 'body', 'semantic', 'silence', 'eos'
#         # Load appropriate model from expert_models.py
#         # Set up loss function based on expert_type:
#         #   - 'body': BCEWithLogitsLoss
#         #   - 'semantic': CTCLoss
#         #   - 'silence': BCEWithLogitsLoss
#         #   - 'eos': BCEWithLogitsLoss
#
#     def slice_electrodes(self, features, expert_type):
#         # Return features[:, :, INDICES] where INDICES is from constants.py
#
#     def train(self):
#         # Training loop, similar to BrainToTextDecoder_Trainer.train()
#         # Key difference: slice electrodes before passing to model
#
#     def validation(self):
#         # Validation loop, similar to BrainToTextDecoder_Trainer.validation()

# BODY EXPERT SPECIAL HANDLING:
# - Requires body configuration labels (7-dim binary vectors per timestep)
# - Labels must be generated from phoneme labels using IPA_TO_BODY_MATRIX
# - During training: labels = IPA_TO_BODY_MATRIX[phoneme_label] for each timestep

# SEMANTIC EXPERT SPECIAL HANDLING:
# - Uses CTCLoss (same as original RNN)
# - Phoneme labels from existing dataset

# SILENCE/EOS EXPERT SPECIAL HANDLING:
# - Requires binary labels per timestep
# - Silence labels: 1 if frame should be silent (can derive from phoneme = SIL or BLANK)
# - EOS labels: 1 at final frame of utterance
```

---

### 2.5 `model_training/expert_args.yaml`

**Purpose:** Configuration file for training expert networks.

```yaml
# FILE: model_training/expert_args.yaml
# ACTION: CREATE NEW FILE

expert_type: 'semantic'  # Options: 'body', 'semantic', 'silence', 'eos'

model:
  # Body Expert (Net A)
  body:
    input_dim: 256  # Area 4 + Ventral 6v only
    hidden_dim: 256
    n_layers: 2
    dropout: 0.3

  # Semantic Expert (Net B)
  semantic:
    input_dim: 256  # Area 55b + Ventral 6v only
    hidden_dim: 384
    n_layers: 3
    dropout: 0.3
    n_classes: 41

  # Silence Gate (Net C)
  silence:
    input_dim: 128
    hidden_dim: 64

  # EOS Gate (Net D)
  eos:
    input_dim: 128
    hidden_dim: 64

# Training settings (same structure as rnn_args.yaml)
gpu_number: '0'
use_amp: true
output_dir: trained_models/expert_${expert_type}
checkpoint_dir: trained_models/expert_${expert_type}/checkpoint

num_training_batches: 60000  # Smaller models = fewer batches needed
lr_max: 0.001
lr_min: 0.0001
# ... (copy remaining training settings from rnn_args.yaml)

dataset:
  # Same as rnn_args.yaml
  dataset_dir: ../data/hdf5_data_final
  batch_size: 64
  sessions:
    - t15.2023.08.11
    # ... (copy full session list)
```

---

### 2.6 `model_training/train_experts.py`

**Purpose:** Entry point for training individual expert networks.

```python
# FILE: model_training/train_experts.py
# ACTION: CREATE NEW FILE

from omegaconf import OmegaConf
from expert_trainer import ExpertTrainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expert', type=str, required=True,
                    choices=['body', 'semantic', 'silence', 'eos'],
                    help='Which expert network to train')
args = parser.parse_args()

config = OmegaConf.load('expert_args.yaml')
config.expert_type = args.expert

trainer = ExpertTrainer(config)
metrics = trainer.train()
print(f"Training complete. Best validation metric: {metrics['best_metric']}")
```

---

## 3. Existing Files to Modify

### 3.1 `model_training/evaluate_model.py`

**Location:** [evaluate_model.py](file:///c:/Users/tardi/OneDrive/Documents/Kaggle/Brain2Text/Kaggle-b2t/model_training/evaluate_model.py)

| Line(s) | Current Code | New Code | Description |
|---------|--------------|----------|-------------|
| 12 | `from rnn_model import GRUDecoder` | `from expert_models import BodyExpertNet, SemanticExpertNet, SilenceGateNet, EOSGateNet` | Import expert models |
| 12+ | (add) | `from fusion import fuse_experts, apply_silence_gate` | Import fusion functions |
| 12+ | (add) | `from constants import NET_A_INDICES, NET_B_INDICES, NET_C_INDICES, IPA_TO_BODY_MATRIX` | Import constants |
| 59-72 | Load single `GRUDecoder` model | Load all 4 expert models from separate checkpoints | See detailed code below |
| 116-127 | `runSingleDecodingStep(...)` → single RNN forward | Call all 4 experts, fuse, gate | Core inference change |

**Detailed Changes for Lines 59-72 (Load Models):**

```python
# REPLACE this block:
model = GRUDecoder(...)
checkpoint = torch.load(...)
model.load_state_dict(checkpoint['model_state_dict'])

# WITH:
# Load expert models
body_model = BodyExpertNet()
semantic_model = SemanticExpertNet()
silence_model = SilenceGateNet()
eos_model = EOSGateNet()

body_ckpt = torch.load(args.body_model_path)
semantic_ckpt = torch.load(args.semantic_model_path)
silence_ckpt = torch.load(args.silence_model_path)
eos_ckpt = torch.load(args.eos_model_path)

body_model.load_state_dict(body_ckpt['model_state_dict'])
semantic_model.load_state_dict(semantic_ckpt['model_state_dict'])
silence_model.load_state_dict(silence_ckpt['model_state_dict'])
eos_model.load_state_dict(eos_ckpt['model_state_dict'])

# Move to device
body_model.to(device).eval()
semantic_model.to(device).eval()
silence_model.to(device).eval()
eos_model.to(device).eval()

# Pre-compute the IPA-to-Body matrix on device
M = IPA_TO_BODY_MATRIX.to(device)
```

**Detailed Changes for Lines ~116-127 (Inference Step):**

```python
# REPLACE runSingleDecodingStep() call with:

def runExpertInference(neural_input, device, body_model, semantic_model, silence_model, M):
    """
    Run all expert networks and fuse outputs.
    """
    with torch.no_grad():
        # Slice electrodes
        x_body = neural_input[:, :, NET_A_INDICES]
        x_semantic = neural_input[:, :, NET_B_INDICES]
        x_silence = neural_input[:, :, NET_C_INDICES]
        
        # Forward pass through experts
        body_logits = body_model(x_body)           # [B, T, 7]
        semantic_logits = semantic_model(x_semantic)  # [B, T, 41]
        silence_logits = silence_model(x_silence)  # [B, T, 1]
        
        # Convert to probabilities
        p_body = torch.sigmoid(body_logits)
        p_phoneme = torch.softmax(semantic_logits, dim=-1)
        
        # Fuse
        p_fused = fuse_experts(p_body, p_phoneme, M)
        
        # Apply silence gate
        p_gated = apply_silence_gate(p_fused, silence_logits)
        
        # Convert to log-probs for WFST decoder
        log_probs = torch.log(p_gated + 1e-9)
        
    return log_probs.float().cpu().numpy()
```

---

### 3.2 `model_training/evaluate_model.py` — CLI Arguments

**Add new arguments (around line 20-28):**

```python
# ADD these argument definitions:
parser.add_argument('--body_model_path', type=str, required=True,
                    help='Path to trained Body Expert checkpoint')
parser.add_argument('--semantic_model_path', type=str, required=True,
                    help='Path to trained Semantic Expert checkpoint')
parser.add_argument('--silence_model_path', type=str, required=True,
                    help='Path to trained Silence Gate checkpoint')
parser.add_argument('--eos_model_path', type=str, required=True,
                    help='Path to trained EOS Gate checkpoint')
```

---

### 3.3 `model_training/dataset.py`

**Location:** [dataset.py](file:///c:/Users/tardi/OneDrive/Documents/Kaggle/Brain2Text/Kaggle-b2t/model_training/dataset.py)

| Line(s) | Change Type | Description |
|---------|-------------|-------------|
| 100-159 | MODIFY `__getitem__` | Add optional return of body labels (derived from phoneme labels) |
| 16+ | ADD import | `from constants import IPA_TO_BODY_MATRIX` |

**Key Addition:** When training the Body Expert, we need to generate body labels from phoneme labels.

```python
# In __getitem__, add:
if self.return_body_labels:
    # seq_class_ids contains phoneme indices
    # body_labels[t] = IPA_TO_BODY_MATRIX[phoneme_index[t]]
    body_labels = IPA_TO_BODY_MATRIX[seq_class_ids]  # [seq_len, 7]
    batch['body_labels'] = body_labels
```

---

### 3.4 `model_training/evaluate_model_helpers.py`

**Location:** [evaluate_model_helpers.py](file:///c:/Users/tardi/OneDrive/Documents/Kaggle/Brain2Text/Kaggle-b2t/model_training/evaluate_model_helpers.py)

| Line(s) | Change Type | Description |
|---------|-------------|-------------|
| N/A | ADD function | `runExpertInference()` (alternative: put in evaluate_model.py directly) |

No other changes needed—the Redis communication and LM helper functions remain unchanged.

---

### 3.5 `language_model/language-model-standalone.py` (Optional)

**Location:** [language-model-standalone.py](file:///c:/Users/tardi/OneDrive/Documents/Kaggle/Brain2Text/Kaggle-b2t/language_model/language-model-standalone.py)

| Line(s) | Change Type | Description |
|---------|-------------|-------------|
| 93-123 | OPTIONAL | Add option to use `distilgpt2` or `OPT-1.3b` instead of `OPT-6.7b` for faster rescoring |

```python
# Add to argparser:
parser.add_argument('--opt_model', type=str, default='facebook/opt-6.7b',
                    help='HuggingFace model name for LLM rescoring')

# Modify build_opt() call:
lm, lm_tokenizer = build_opt(model_name=args.opt_model, ...)
```

---

## 4. Detailed Implementation Steps

Execute these steps **in order**:

### Step 1: Create Constants File
```bash
# File: model_training/constants.py
# Action: CREATE
# Copy the code from Section 2.1
```

### Step 2: Create Expert Models File
```bash
# File: model_training/expert_models.py
# Action: CREATE
# Copy the code from Section 2.2
```

### Step 3: Create Fusion Logic File
```bash
# File: model_training/fusion.py
# Action: CREATE
# Copy the code from Section 2.3
```

### Step 4: Create Expert Trainer
```bash
# File: model_training/expert_trainer.py
# Action: CREATE
# Implement based on skeleton in Section 2.4
# Model after rnn_trainer.py but:
#   - Add electrode slicing
#   - Handle 4 different expert types
#   - Use appropriate loss functions
```

### Step 5: Create Expert Config
```bash
# File: model_training/expert_args.yaml
# Action: CREATE
# Copy the code from Section 2.5
```

### Step 6: Create Training Entry Point
```bash
# File: model_training/train_experts.py
# Action: CREATE
# Copy the code from Section 2.6
```

### Step 7: Modify Dataset
```bash
# File: model_training/dataset.py
# Action: MODIFY
# Add body label generation (see Section 3.3)
```

### Step 8: Modify Evaluate Model
```bash
# File: model_training/evaluate_model.py
# Action: MODIFY
# Replace RNN loading with expert loading (see Section 3.1)
# Replace inference step with fusion pipeline (see Section 3.1)
# Add new CLI arguments (see Section 3.2)
```

### Step 9: Train All Experts
```bash
# Train each expert network:
python train_experts.py --expert body
python train_experts.py --expert semantic
python train_experts.py --expert silence
python train_experts.py --expert eos
```

### Step 10: Run Inference
```bash
python evaluate_model.py \
    --body_model_path trained_models/expert_body/checkpoint/best_checkpoint \
    --semantic_model_path trained_models/expert_semantic/checkpoint/best_checkpoint \
    --silence_model_path trained_models/expert_silence/checkpoint/best_checkpoint \
    --eos_model_path trained_models/expert_eos/checkpoint/best_checkpoint \
    --data_dir ../data/hdf5_data_final \
    --eval_type test
```

---

## 5. Verification Plan

### 5.1 Unit Tests

| Test | File | Description |
|------|------|-------------|
| Test electrode slicing | `test_constants.py` | Verify indices are correct and non-overlapping where expected |
| Test IPA-to-Body matrix | `test_constants.py` | Verify matrix shape is [41, 7], all values are 0 or 1 |
| Test fusion function | `test_fusion.py` | Verify output shape, probabilities sum to 1 |
| Test silence gate | `test_fusion.py` | Verify BLANK token is set when silence prob > 0.5 |

### 5.2 Integration Tests

1. **Smoke Test:** Run `train_experts.py --expert semantic` for 100 batches. Verify loss decreases.
2. **Inference Test:** Run `evaluate_model.py` with dummy models. Verify CSV output format is correct.
3. **Full Pipeline Test:** Start Redis + LM, run full evaluation, verify WER is computed.

### 5.3 Performance Baselines

| Model | Expected Val PER | Training Time |
|-------|------------------|---------------|
| Original RNN | ~10% | ~3.5 hrs |
| Net B (Semantic) alone | ~12-15% | ~1.5 hrs |
| Full Fusion | ~9-11% (TBD) | N/A |

---

## Summary Checklist

| # | File | Action | Status |
|---|------|--------|--------|
| 1 | `model_training/constants.py` | CREATE | ⬜ |
| 2 | `model_training/expert_models.py` | CREATE | ⬜ |
| 3 | `model_training/fusion.py` | CREATE | ⬜ |
| 4 | `model_training/expert_trainer.py` | CREATE | ⬜ |
| 5 | `model_training/expert_args.yaml` | CREATE | ⬜ |
| 6 | `model_training/train_experts.py` | CREATE | ⬜ |
| 7 | `model_training/dataset.py` | MODIFY | ⬜ |
| 8 | `model_training/evaluate_model.py` | MODIFY | ⬜ |
| 9 | `model_training/evaluate_model_helpers.py` | OPTIONAL MODIFY | ⬜ |
| 10 | `language_model/language-model-standalone.py` | OPTIONAL MODIFY | ⬜ |
