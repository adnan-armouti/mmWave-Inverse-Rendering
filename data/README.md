# Data

Nothing is shipped here. Each scene lives in `data/<scene>/`, named `seq_<s>_frame_<F>`
after the ColoRadar outdoor sequence `2_28_2021_outdoors_run<s>` and its center cascade
frame `F`. The preprocessing and alignment commands in the main README produce:

```
data/
  <scene>/
    scene/pcl.npy                                  LiDAR point cloud with normals
    scene/mesh.ply                                 Poisson mesh used for training
    configs/cascaded_frame_<f>.json                cascade radar configuration per frame
    configs/cascaded_frame_<F>_aligned_gpu.json    cascade configuration, LiDAR-based 4-DOF alignment
    configs/cascaded_frame_<F>_aligned_2dof.json   cascade configuration, renderer-based 2-DOF alignment
    configs/single_chip_frame_<f>.json             single-chip radar configuration per frame
    radar/cascaded_frame_<f>.npy                   cascade ADC, complex
    radar/single_chip_frame_<f>.npy                single-chip ADC, complex
    lidar/lidar_frame_<f>.npy                      LiDAR frames
```

Training reads `scene/mesh.ply`, `radar/cascaded_frame_<F>.npy` and the aligned cascade
configuration the paper chose for the scene (`_aligned_gpu` or `_aligned_2dof`, listed in
`ALIGNMENT_RESULTS` in `mmir/preprocessing/alignment/cascaded_alignment.py`). By default
that configuration comes from `assets/alignment/<scene>/`, the files used for the paper;
`--alignment data` reads the regenerated ones in `configs/` instead. The
radar-transfer evaluation also uses the single-chip files and writes the aligned
single-chip configuration to its output folder. The occupancy evaluation also uses the
LiDAR frames and reads the ColoRadar point-cloud baseline from the raw dataset
(`--coloradar-dataset-dir`).
