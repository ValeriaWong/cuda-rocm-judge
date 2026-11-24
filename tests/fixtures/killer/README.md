# killer fixture

Synthetic Tier C/D fixture: a `.cu` kernel using `__shfl_down_sync`
(warp shuffle, hard) plus a Python import of `nvdiffrast`
(NVIDIA-only, no ROCm equivalent). Expect Tier D with file:line
evidence for both.
