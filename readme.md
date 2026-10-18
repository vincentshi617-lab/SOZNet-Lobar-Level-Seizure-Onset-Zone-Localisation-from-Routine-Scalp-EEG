Overview

SOZNet is a research prototype for non-invasive, lobar-level localisation of the seizure onset zone (SOZ) from routine 18-channel scalp EEG. It is trained on expert-annotated clinical labels and predicts multi-label probabilities over six anatomical regions:

Left Frontal (LF)
Right Frontal (RF)
Left Temporo-Parietal (LT/TP)
Right Temporo-Parietal (RT/TP)
Left Occipital (LO)
Right Occipital (RO)

The project is intended to support—not replace—epileptologist judgement during presurgical evaluation. It is not a seizure detector, not a diagnostic device, and not validated for prospective clinical deployment.


Key Features

Routine low-density EEG: designed for standard 18-channel longitudinal bipolar montage, not high-density or invasive recordings.

Multi-scale temporal front-end: parallel convolutions with kernel sizes {16, 32, 64, 128} to capture brief spikes, sharp waves, and slow-wave complexes.
Factorised spatiotemporal Transformer: separates temporal attention (waveform evolution) from spatial attention (electrode topography / propagation), directly motivated by volume conduction.

Hemispheric mutual exclusivity penalty: soft anatomical prior during training, with a hard session-level unilateral constraint at inference.

Class-imbalance handling: multi-label Focal Loss with positive weighting for rare occipital regions.

Online augmentation: Gaussian noise, amplitude scaling, random temporal cropping, and hemispheric channel swap.

Model Soup + heterogeneous ensemble: three M1 variants, one M2, one M3, fused by soft voting.

Clinical-style evaluation: patient-level splits, annotation-sparse holdout, cross-institutional paediatric external set, unlabelled random windows, healthy controls, and runtime profiling.
