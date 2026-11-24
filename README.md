# cuda-rocm-judge

A small, honest judge that answers one question: **can this CUDA wheel /
repo be ported to ROCm?**

It reads source and tells you which tier the project falls in, with
`file:line` evidence for every claim. It does not convert anything.

## What it does NOT do

- It does **not** convert CUDA to HIP/ROCm. There is no converter here.
- It does **not** build anything, run a GPU, or install anything.
- It does **not** submit PRs or post anywhere.
- A Tier B verdict does **not** mean the port will compile. A real ROCm
  build (hipcc + ROCm PyTorch) is still required, and the report says so
  on every run.

## How to run

Stdlib only, Python 3.8+. No pip installs.

```bash
# a local source directory
python judge.py /path/to/some-cuda-repo

# a wheel file (binary-only wheels are judged, not unpacked-and-built)
python judge.py some-package-1.0-cp312-cp312-linux_x86_64.whl

# a GitHub repo URL (shallow clone to a temp dir, nothing committed)
python judge.py https://github.com/runjie-yan/mip-gaussian-l1c

# machine-readable output
python judge.py /path/to/repo --json
```

Exit code is 0 for any verdict, 2 for usage errors (missing path, bad
file type, failed clone).

## Tiers

| Tier | Meaning | Score |
| --- | --- | --- |
| A | Pure PyTorch / pure Python. No `.cu`, no custom extension. Trivial: swap in ROCm PyTorch, `torch.cuda` API is unchanged. | 90-100 |
| B | HIPIFY-able. Custom kernels, but only patterns HIPIFY covers plus small manual fixes. | 60-89 |
| C | Hard / manual rewrite. Warp shuffles, PTX asm, thrust, cuDNN-style rewrites. Days to weeks. | 30-59 |
| D | Cannot / not worth it. Binary-only wheel, or an NVIDIA-only dependency (nvdiffrast, kaolin, TensorRT) with no ROCm equivalent. | 0-29 |

The score is a readiness hint, not a measurement. The tier is decided
by rules first: any no-equivalent dependency forces Tier D, any hard
killer forces at least Tier C, no matter the score.

## What it checks

With file:line evidence, in code files only (docs are not scanned --
prose mentioning CUDA is not evidence; vendored `third_party/` and
`.git` are skipped):

- **Hard killers**: `__shfl_*`, `__ballot_*`, `__frcp_rz`, inline PTX
  asm, `__ldg`, thrust, CUTLASS, direct `cudaMalloc`/stream calls,
  Windows-only paths.
- **No-equivalent libs**: nvdiffrast, kaolin, TensorRT.
- **Convertible libs**: CUB -> hipCUB/rocPRIM, cooperative_groups ->
  hip, cuBLAS -> rocBLAS, cuFFT -> rocFFT, cuSPARSE -> rocSPARSE,
  cuSOLVER -> rocSOLVER, cuRAND -> rocRAND, NCCL -> RCCL,
  cuDNN -> MIOpen (manual -- MIOpen has no cuDNN marshalling layer).
- **Easy signals**: pure Python / pure PyTorch, `torch.cuda` only.
- **Build signals**: `CUDAExtension` / `load_inline`, nvcc mentions,
  `TORCH_CUDA_ARCH_LIST` (re-target to `gfx1100` / `gfx942`).
- **Wheels**: a `.whl` with prebuilt `.so` and no native source
  (`.cu`/`.cpp`/`.h`) is Tier D binary-only. A lone `__init__.py`
  next to a `.so` does not count as source -- real binary wheels
  ship Python wrappers around a `.so`.

Every report ends with a per-tier "what to do next" checklist and a
list of unconfirmed items (real build, runtime precision compare,
target gfx arch, performance).

## Reference example: mip-gaussian-l1c (Tier B)

Upstream `runjie-yan/mip-gaussian-l1c` @ `b44209d`. The part that
needed porting is only `submodules/diff-gaussian-rasterization/`
(~2560 lines, 12 files). It has no `__frcp_rz` / `__shfl_*` / PTX /
thrust / `__ldg`; only CUB (`DeviceScan` / `DeviceRadixSort`) and
`cooperative_groups`, both covered by HIP. The real port (report in
`~/workspace/digital-pet-research/mip-gaussian-l1c-rocm-port-20261119.md`)
compiled clean, smoke passed, and the mock precision compare passed
on rel_l2 / cosine. Run the judge on that submodule:

```bash
python judge.py ~/workspace/digital-pet-research/mip-gaussian-rocm/upstream/submodules/diff-gaussian-rasterization/
# verdict: Tier B -- HIPIFY-able, readiness score: 80 / 100
```

Scope note, learned the honest way: the full repo root also contains
`submodules/simple-knn`, which uses `thrust::device_vector`. Judged
on the repo root, the tool says Tier C (score 50) because of that
thrust use. The Tier B verdict above is for the rasterizer submodule
only -- the unit that was actually ported. If you judge a whole repo,
judge each CUDA extension separately.

## Known limitations

- Static source reading only. It cannot see runtime behaviour, closed
  binaries, or whether a HIPIFY translation actually compiles.
- `float atomicAdd` reduction order is nondeterministic, so precision
  compares must use rel_l2 / cosine, not `max_abs_diff` alone -- the
  tool flags `atomicAdd` use for this reason.
- `cudaMalloc`/stream use is flagged hard because it needs hand
  review, not because it is untranslatable; HIPIFY covers the common
  cases.
- A `.cu` file that only wraps a library call can look worse than it
  is; read the evidence lines, not just the tier.
