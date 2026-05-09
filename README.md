# Nexvara Early Warning
### Per-Individual Physiological Anomaly Detection · Preprint 2026

> *A neonatal patient's baseline is not the population average. It is their own.*

---

## What this is

Standard clinical monitoring compares a patient's signals against population thresholds — the same alarm limits for every infant, every adult, every ICU patient. This works until it doesn't: a preterm infant with an unusually low baseline heart rate triggers constant false alarms, while a patient whose "normal" sits at the edge of population norms deteriorates silently within range.

**Nexvara Early Warning** takes a different approach. A lightweight autoencoder is trained on each individual patient's own clean baseline — the first hours of stable signal after admission. From that point forward, the model scores every new window against *that patient's* learned normal. Deviation from the individual baseline, not the population mean, is the anomaly signal.

The result: the model knows this infant, not infants in general.

---

## Results

Validated across four independent clinical datasets spanning the full lifespan:

| Dataset | Population | Modality | Patients detected | Mean separation | Lead time |
|---|---|---|---|---|---|
| Helsinki Neonatal EEG | Neonatal | EEG | 10/10 | 3.35σ | ~10s before seizure |
| PICS Preterm | Preterm infant | HR / RR | 9/9 | 3.37σ | **30s** (p<0.0001, n=581) |
| CHB-MIT Paediatric EEG | Paediatric | EEG | 9/9 | **7.22σ** | p=0.022 @ −10s |
| MIMIC-IV Adult ICU | Adult ICU | HR / SpO₂ / RR | 32/37 | 3.66σ | **~2 hours** (p=0.008) |

**vs population model baseline: 0.83σ** — a 7.2× improvement in separation on CHB-MIT.

Cross-subject control: applying one patient's model to another patient's data collapses separation by 622×, confirming the per-individual specificity of the approach.

---

## Clinical significance

- **30 seconds** before bradycardia/apnea onset in preterm infants — enough time to pre-position staff and prepare intervention
- **~2 hours** before vasopressor initiation in adult ICU — a window that does not exist in current monitoring
- **Zero false positives** on the individual's own stable baseline in the majority of patients
- Identifies two distinct deterioration phenotypes in MIMIC-IV: Gradual (detectable ~2h ahead) and Acute (rapid onset, different clinical pathway)

---

## Architecture

```
Patient admitted
      │
      ▼
┌─────────────────────────┐
│  Baseline window        │  First stable hours of signal
│  (individual-specific)  │
└────────────┬────────────┘
             │  train
             ▼
┌─────────────────────────┐
│  Per-individual         │  Lightweight convolutional
│  autoencoder            │  autoencoder, ~50k parameters
└────────────┬────────────┘
             │  reconstruct
             ▼
┌─────────────────────────┐
│  Anomaly score          │  Reconstruction error vs
│  (continuous)           │  individual baseline distribution
└────────────┬────────────┘
             │
             ▼
        Early warning
```

One model per patient. No shared weights between patients. No population reference required after training.

---

## Demo

Interactive clinical dashboard — switch between datasets, patients, and signal views.

```bash
pip install streamlit matplotlib numpy pandas
streamlit run demo/nexvara_demo.py
```

Shows:
- Live signal + anomaly score with lead time window highlighted
- Per-individual vs population model comparison (false alarm rate contrast)
- Cross-dataset results summary with real preprint numbers

---

## Repository structure

```
nexvara-early-warning/
├── README.md
├── requirements.txt
├── demo/
│   └── nexvara_demo.py        # Streamlit clinical dashboard
├── figures/                   # Key result figures from preprint
│   ├── separation_comparison.png
│   ├── lead_time_pics.png
│   └── population_vs_individual.png
└── notebooks/
    └── per_individual_baseline.ipynb   # Core method walkthrough
```

---

## Preprint

> **Per-Individual Generative Modelling for Physiological Early Warning Across the Lifespan**
> Nexvara Research, 2026
> MedRxiv: *[DOI link — update on posting]*

---

## Datasets used

- [Helsinki University Hospital Neonatal EEG](https://zenodo.org/record/4940267)
- [PICS Preterm Infant Cardiorespiratory](https://physionet.org/content/pics/1.0.0/)
- [CHB-MIT Scalp EEG](https://physionet.org/content/chbmit/1.0.0/)
- [MIMIC-IV Clinical Database](https://physionet.org/content/mimiciv/2.2/)

All datasets are publicly available via PhysioNet or Zenodo. No patient data is included in this repository.

---

## Contact

**Nexvara Research**
For research collaboration enquiries: [your email]

*Registered in Hong Kong · Research focus: per-individual physiological AI*

---

## Citation

If you use this work, please cite:

```bibtex
@article{nexvara2026,
  title   = {Per-Individual Generative Modelling for Physiological Early Warning Across the Lifespan},
  author  = {Noelle Xiao},
  journal = {medRxiv},
  year    = {2026},
  doi     = {[DOI — update on posting]}
}
```

---

<sub>This repository accompanies a preprint that has not yet undergone peer review. Results should be interpreted accordingly.</sub>
