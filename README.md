# CIMD: Cognitive Inspired Multi-perspective Descriptions

[ICASSP 2026] **CIMD: COGNITIVE INSPIRED MULTI-PERSPECTIVE DESCRIPTIONS FOR EEG-IMAGE ALIGNMENT**
---
## Abstract

Retrieving images from electroencephalogram (EEG) signals is crucial for advancing brain-computer interfaces (BCIs). Existing methods often use frozen pre-trained visual encoders to align EEG features with a generic visual space. However, such alignment fails to fully capture EEG-specific structure and the difficulty of bridging the modality gap between abstract EEG and visual stimuli. To address these challenges, we propose Cognitive Inspired Multi-perspective Descriptions (CIMD) for fine-grained alignment. Specifically, we first design the Cognitive-guided Multi-perspective Descriptions (CMD), which exploits EEG-related cognitive characteristics to construct global, color, and emotion questions, using a large multimodal model to generate cognition-inspired textual descriptions that act as semantic bridges and guide alignment between EEG and image. Second, to further achieve fine-grained alignment, we propose the Cognitive-driven Multimodal Alignment (CMA), which leverages these descriptions to fine-tune a large pre-trained model and construct a shared embedding space that unifies EEG, image, and text features. Finally, to preserve generalization while enhancing adaptability to cross-subject variability, we introduce a regularization strategy that constrains excessive drift during fine-tuning. Extensive experiments on the Things-EEG dataset demonstrate that CIMD achieves state-of-the-art performance on the challenging 200-way zero-shot retrieval task.

CIMD achieves **state-of-the-art performance** on:
- Things-EEG
---
