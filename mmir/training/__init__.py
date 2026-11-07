"""mmIR training. Importing the package sets the Dr.Jit flags and the Mitsuba variant."""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import drjit as dr
dr.set_flag(dr.JitFlag.VCallOptimize, False)
dr.set_flag(dr.JitFlag.LoopOptimize, False)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
