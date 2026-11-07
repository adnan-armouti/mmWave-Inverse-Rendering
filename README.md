# mmIR: Frequency-Space Inverse Rendering for 3D Millimeter-Wave Radar ADC Synthesis

> Code release for the ECCV 2026 paper.
> Project page: <https://mmwave-inverse-rendering.github.io>
> Paper: <https://doi.org/10.1007/978-3-032-37359-5_9>

## Abstract

High-resolution 3D radar data is scarce. Commodity mmWave sensors use small antenna
arrays that limit angular resolution to several degrees, and existing datasets provide
only 2D range–azimuth maps or sparse point clouds rather than raw analog-to-digital
converter (ADC) signals. Hardware scaling is expensive, synthetic-aperture scanning is
impractical at fleet scale, and learned synthesis methods are bottlenecked by the very
data shortage they aim to address.

We present mmIR, an open-source differentiable frequency-modulated continuous-wave
(FMCW) radar inverse renderer that fits a physics-based forward model to real captures
and re-renders from dense virtual apertures to synthesize high-resolution 3D radar data.
Because radar resolution is too coarse to recover geometry directly, mmIR performs
LiDAR-assisted inverse rendering: using LiDAR-derived meshes as a geometric scaffold,
mmIR optimizes per-vertex International Telecommunication Union (ITU) physics
materials, vertex normals, and antenna beam patterns through end-to-end automatic
differentiation of a phase-coherent multiple-input multiple-output (MIMO) forward
model with multi-bounce propagation, polarization, and free-space diffraction.

On seven outdoor and six indoor ColoRadar scenes, mmIR achieves 0.914 mean Pearson
correlation on range–azimuth maps versus 0.307 for Sionna-RT. Scenes trained on a
cascaded imaging radar transfer to a co-located single-chip radar without re-training
(0.554 correlation), and dense virtual arrays (100×100 elements) produce single-frame
3D occupancy validated against LiDAR.

---

## Requirements

- Linux, an NVIDIA GPU with CUDA (tested on an RTX 4090), Python 3.12.
- PyTorch 2.7.1 (cu118), Mitsuba 3.7.1, Dr.Jit 1.2.0, slangtorch 1.3.11, and the
  pinned packages in [environment.yml](environment.yml) / [requirements.txt](requirements.txt).
- CuPy (LiDAR alignment) and FAISS (preprocessing) come as CUDA 12 pip wheels
  (`cupy-cuda12x`, `faiss-gpu-cu12`); the environment installs them.

```bash
conda env create -f environment.yml
conda activate mmir
```

Preprocessing raw ColoRadar also needs screened Poisson surface reconstruction under
`third_party/` (training and evaluation do not):

```bash
git clone https://github.com/mkazhdan/PoissonRecon.git third_party/PoissonRecon
cd third_party/PoissonRecon
git checkout eea22ab701d0067e36a2de33f0251c0d0511ac69
make poissonrecon surfacetrimmer COMPILER=gcc CFLAGS+=' -DBIG_DATA'
cd ../..
```

---

## Data

The release ships no data. [data/README.md](data/README.md) describes the layout. In short:

1. Download the [ColoRadar](https://arpg.github.io/coloradar/) dataset in KITTI format
   (sequences `2_28_2021_outdoors_run0`, `run1`, `run2`).
2. Preprocess each scene: LiDAR point cloud and mesh, radar configs, ADC and LiDAR frames.

   ```bash
   DATASET=/path/to/coloRadar/raw/kitti/2_28_2021_outdoors_run   # the sequence number is appended
   python -m mmir.preprocessing all --seq 0 --frame 135 --num-radar-frames 1 \
       --dataset-dir $DATASET --calib-path mmir/preprocessing/calib --out-root data \
       --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
       --normals-radius 0.1 --remove-behind-radar
   ```

3. The cascaded-radar alignments used in the paper ship in
   [assets/alignment/](assets/alignment/), and training and evaluation use them by
   default. To regenerate them instead:

   ```bash
   python -m mmir.preprocessing.alignment.cascaded_alignment --data-root data --output-root output/alignment
   ```

   For each scene this runs a LiDAR-based 4-DOF alignment
   (`cascaded_frame_<F>_aligned_gpu.json`) and a renderer-based 2-DOF alignment
   (`cascaded_frame_<F>_aligned_2dof.json`), both written to `data/<scene>/configs/`; pass
   `--alignment data` to training and evaluation to use them. Each scene trains on the
   paper's choice between the two (`ALIGNMENT_RESULTS` in
   [cascaded_alignment.py](mmir/preprocessing/alignment/cascaded_alignment.py)).
   Regenerated configs can differ from the shipped ones by about 1 mm in antenna position
   (GPU numerics), which is about a third of a wavelength at 77 GHz. On `seq_0_frame_135`
   this lowered the training correlation from 0.920 to 0.915 to 0.917. The single-chip radar
   is aligned during the transfer evaluation.

The paper's seven outdoor scenes are

```
seq_0_frame_135   seq_0_frame_390   seq_1_frame_185   seq_1_frame_438
seq_2_frame_105   seq_2_frame_160   seq_2_frame_300
```

---

## Training

```bash
python train.py --scene seq_0_frame_135
```

This trains with the paper's settings for the outdoor scenes (500 iterations; 11 to 14
minutes per scene on one RTX 4090) and writes `best_materials.npz`, `best_normals.npz`,
`best_patterns.npz`, `best_pose.npz`, `best_metrics.json` and `training_history.json` to
`output/training/<scene>/`. Data is read from `data/<scene>/`; use `--data_root` to point
elsewhere, and `--alignment data` to train on regenerated alignments. `--config` takes a
JSON file whose keys override fields of `TrainingConfig` in
[mmir/training/config.py](mmir/training/config.py). For example, the paper's indoor scenes
used upper bounds on three material parameters:

```json
{"eps_imag_raw_upper": 4.6, "sigma_h_raw_upper": -2.3, "thickness_raw_upper": -0.7}
```

---

## Evaluation

```bash
# Training-view RA metrics (paper Table 1)
python -m mmir.evaluation training-ra

# Transfer to the single-chip radar (paper Table 2); aligns the single-chip radar first
python -m mmir.evaluation radar-transfer

# 3D occupancy from a dense 100x100 virtual array vs. LiDAR (paper Table 3); the ColoRadar
# point-cloud baseline is read from the raw dataset
python -m mmir.evaluation occupancy --coloradar-dataset-dir /path/to/coloRadar/raw/kitti/2_28_2021_outdoors_run
```

All three evaluate the seven outdoor scenes by default (`--scenes` selects some), read
trained scenes from `output/training/` (`--training-root`) and data from `data/`
(`--data-root`), and write metrics and a LaTeX table to `output/evaluation/`.

---

## Results

Results of this code on the seven outdoor scenes (mean ± standard deviation over scenes).
Higher is better for correlation, PSNR, SSIM, precision, recall, F1 and accuracy; lower is
better for RMSE and R-CD. The training-view numbers match the paper's Table 1. Transfer and
3D occupancy evaluate each scene at the radar pose it was trained with, which changes some
scenes relative to the paper's Tables 2 and 3.

| Evaluation | RA correlation | RA PSNR | RA SSIM | RA RMSE |
|---|---|---|---|---|
| Training view (cascade) | 0.919 ± 0.030 | 37.6 ± 2.4 | 0.902 ± 0.049 | 0.0137 ± 0.0038 |
| Transfer to single-chip radar | 0.571 ± 0.099 | 23.7 ± 2.2 | 0.468 ± 0.093 | 0.0675 ± 0.0167 |

3D occupancy against LiDAR (match threshold 0.5 m; R-CD is the relative Chamfer distance):

| Method | Precision | Recall | F1 | Accuracy | RMSE | R-CD |
|---|---|---|---|---|---|---|
| mmIR (dense 100×100 virtual array) | 0.627 ± 0.145 | 0.283 ± 0.106 | 0.383 ± 0.124 | 0.381 ± 0.118 | 1.755 ± 0.498 | 0.140 ± 0.040 |
| ColoRadar cascade point cloud | 0.282 ± 0.130 | 0.090 ± 0.045 | 0.132 ± 0.059 | 0.098 ± 0.047 | 2.328 ± 0.563 | 0.181 ± 0.050 |

---

## Repository structure

```
README.md
LICENSE                     # Cornell University terms
environment.yml
requirements.txt
train.py                    # train one scene
assets/antenna_pattern/     # measured antenna patterns (MMWCAS cascade, IWR1443 single chip)
assets/alignment/           # aligned cascade radar configurations used in the paper
data/README.md              # data layout
mmir/
  training/                 # configuration, parameters, forward/backward pass, training loop
  renderer/                 # differentiable FMCW renderer: BSDFs, integrators, specular paths, diffraction
  sensor/                   # radar configuration and antenna element patterns
  data/                     # ADC to range-azimuth processing
  losses/                   # RA loss and its gradients
  evaluation/               # training-RA, radar-transfer and 3D occupancy evaluation
  preprocessing/            # ColoRadar preprocessing and radar-LiDAR alignment
third_party/                # PoissonRecon (cloned, not included)
```

---

## License

See [LICENSE](LICENSE).

```
Copyright © 2026 by Cornell University.

All Rights Reserved.

Permission to use, copy, modify and distribute any part of “mmIR”
(“WORK”) and its associated copyrights for educational, research and
non-profit purposes, without fee, and without a written agreement is hereby
granted, provided that the above copyright notice, this paragraph and the
following three paragraphs shall be displayed in all copies or substantial
portions of the Work.

Those desiring to incorporate WORK into commercial products or use
WORK and its associated copyrights for commercial purposes should
contact the Center for Technology Licensing (“CTL”) at Cornell University
at 395 Pine Tree Road, Suite 310, Ithaca, NY 14850; email:
ctl-connect@cornell.edu; Tel: 607-254-4698; FAX: 607-254-5454 for a
commercial license. Inquiries should reference CTL docket number 12012.

IN NO EVENT SHALL CORNELL UNIVERSITY BE LIABLE TO ANY
PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF WORK AND ITS ASSOCIATED
COPYRIGHTS, EVEN IF CORNELL UNIVERSITY MAY HAVE BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE WORK PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND
CORNELL UNIVERSITY HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS. CORNELL UNIVERSITY MAKES NO
REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY
KIND, EITHER IMPLIED OR EXPRESS, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF
WORK AND ITS ASSOCIATED COPYRIGHTS WILL NOT INFRINGE
ANY PATENT, TRADEMARK OR OTHER RIGHTS.
```

## Citation

```bibtex
@inproceedings{mmIR2026,
  title={mmIR: Frequency-Space Inverse Rendering for 3D Millimeter-Wave Radar ADC Synthesis},
  author={Armouti, Adnan and Gao, Yixuan and Nandakumar, Rajalakshmi},
  booktitle={Computer Vision -- ECCV 2026},
  series={Lecture Notes in Computer Science},
  pages={148--166},
  year={2026},
  publisher={Springer},
  doi={10.1007/978-3-032-37359-5_9}
}
```
