Overview

SOZNet is a research prototype for non-invasive, lobar-level localisation of the seizure onset zone (SOZ) from routine 18-channel scalp EEG. It is trained on expert-annotated clinical labels and predicts multi-label probabilities over six anatomical regions:

Left Frontal (LF)
Right Frontal (RF)
Left Temporo-Parietal (LT/TP)
Right Temporo-Parietal (RT/TP)
Left Occipital (LO)
Right Occipital (RO)

<img width="943" height="429" alt="Screenshot 2026-08-20 at 1 31 29 pm" src="https://github.com/user-attachments/assets/857e867d-bcb1-49ca-bb84-5d85a1bcee9f" />

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

<img width="2100" height="900" alt="组合模型内部测试" src="https://github.com/user-attachments/assets/bf2c7b7e-88b1-4fb5-9771-839c5a3a9a33" />
<img width="5220" height="2010" alt="beautiful_soz_pipeline" src="https://github.com/user-attachments/assets/b913aaa8-16fb-428a-b60e-d272b0bd66a5" />



Model Variants

Model	Role	Key design	Params
M1 / SOZNet	Main model	Factorised temporal + spatial Transformer	~0.8M

M2	Ablation / ensemble member	Single-axis Transformer with content-based pooling	~3.5M

M3	Ablation / ensemble member	Deeper single-axis Transformer	~11M

Ensemble	Final research model	3× M1 soup + M2 soup + M3 soup, soft voting	5 constituent models

The paper’s main finding is that the factorised M1 matches or outperforms much larger single-axis variants, supporting architectural inductive bias over raw parameter count.


Architecture

M1
<img width="1800" height="1300" alt="模型1架构 copy" src="https://github.com/user-attachments/assets/3521d77a-d04b-4a2b-9353-330dbc6d4c10" />

M2&3
<img width="4800" height="3600" alt="模型2，3架构" src="https://github.com/user-attachments/assets/c958ed5f-a45c-405c-9c49-15070fe3350f" />



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

Statistics

Performance (ensemble):
<img width="2100" height="900" alt="组合模型内部测试" src="https://github.com/user-attachments/assets/3192271a-4e03-44d4-ba3a-4cdf73c70c23" />

Holdout test:
<img width="2100" height="900" alt="holdout" src="https://github.com/user-attachments/assets/0a2280bb-52b0-4b25-a45d-c182d57439a9" />

External test:
<img width="1400" height="600" alt="外部测试" src="https://github.com/user-attachments/assets/cc918cf5-b416-4f91-9bb1-57c21ebc3005" />

Compare with other model
<img width="2200" height="1040" alt="模型对比" src="https://github.com/user-attachments/assets/09e65f4e-1445-4ae2-a630-e35237013e41" />

Model can generate probability of seizure onset zone overtime and allow the user to back trace the onset time:

<img width="1339" height="436" alt="Screenshot 2026-07-14 at 4 02 04 pm" src="https://github.com/user-attachments/assets/c3d8c91d-b9c4-4266-a8d8-51f31873103f" />

<img width="1296" height="486" alt="Screenshot 2026-07-14 at 4 02 18 pm" src="https://github.com/user-attachments/assets/6bcfbc19-a565-413d-b8a5-74a2aba2a412" />

