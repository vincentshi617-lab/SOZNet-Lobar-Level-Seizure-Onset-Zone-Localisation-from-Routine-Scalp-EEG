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


Method Pipeline

<img width="8352" height="908" alt="deepseek_mermaid_20261018_4a2e0e" src="https://github.com/user-attachments/assets/acec0509-0356-4e1a-bc36-ab2ad84bf290" />


Model Variants

Model	Role	Key design	Params
M1 / SOZNet	Main model	Factorised temporal + spatial Transformer	~0.8M

M2	Ablation / ensemble member	Single-axis Transformer with content-based pooling	~3.5M

M3	Ablation / ensemble member	Deeper single-axis Transformer	~11M

Ensemble	Final research model	3× M1 soup + M2 soup + M3 soup, soft voting	5 constituent models

The paper’s main finding is that the factorised M1 matches or outperforms much larger single-axis variants, supporting architectural inductive bias over raw parameter count.


Reported Results

These are retrospective research results from the associated paper. They are not prospective clinical validation.


Setting	                                   Metric	        Result
Internal test, patient-level	          Exact Match	      88.46%
Best generic baseline	                  Exact Match	       ≤50%
Annotation-sparse holdout	              Exact Match	      82.76%
BCH external paediatric set	            Exact Match	      84.62%
Unlabelled random windows, single M1	  Exact Match	      66.67%
Healthy controls	                      False positives	  0 / 108
24-hour recording, 256 Hz	              End-to-end runtime 4.26 min
Model inference	                        Runtime	           <8 s
