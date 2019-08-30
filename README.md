# TaylorFlow

The TaylorFlow modules performs Gravitational Wave calculations within TensorFlow 1.14

There are two main functions within the package:

Waveform- this module calculates TaylorF2 waveform models up to 3.5PN order in the frequency domain. These object are tensor outputs but can easily be coverted in NumPY arrays (demoed in tutiorial)

overlap - This modules calculates comparison functions such as the inner product, match and relative log likelihood, examples of the usage of these functions are given in the tutorial section.
