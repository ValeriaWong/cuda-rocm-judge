# Tests

Run on 2026-11-24, Python 3.12.3, `cuda-rocm-judge 0.1.0`.
All three required runs are recorded verbatim below (commands first,
then complete stdout). Exit code was 0 in all three.

## Test 1 -- mip-gaussian upstream source: must land Tier B

Scope note: the judged unit is the submodule that was actually
ported, `upstream/submodules/diff-gaussian-rasterization/`
(run-jie-yan/mip-gaussian-l1c @ b44209d), not the whole repo root.
The root also contains `submodules/simple-knn` with thrust use --
see "Extra check 2" below, where the judge correctly says Tier C
for the root.

```bash
python judge.py ~/workspace/digital-pet-research/mip-gaussian-rocm/upstream/submodules/diff-gaussian-rasterization/
```

```
cuda-rocm-judge 0.1.0 -- /home/hatch/workspace/digital-pet-research/mip-gaussian-rocm/upstream/submodules/diff-gaussian-rasterization/
kind: source-dir
verdict: Tier B -- HIPIFY-able -- convertible with HIPIFY + small manual fixes
readiness score: 80 / 100

scanned: 6 file(s) (4 .cu, 2 .py, 2887 lines), 1 prebuilt .so

findings (file:line evidence):
  [info] atomic_add -- 10 hit(s): atomicAdd (float): HIP supports it, but note that float atomicAdd reduction order is nondeterministic, so precision compares must use rel_l2 / cosine, not max_abs_diff alone.
      cuda_rasterizer/backward.cu:582: atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
      cuda_rasterizer/backward.cu:586: atomicAdd(&l1_output[global_id], l1_diff);
      cuda_rasterizer/backward.cu:608: atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
      cuda_rasterizer/backward.cu:609: atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
      cuda_rasterizer/backward.cu:613: atomicAdd(&dL_dmean2D[global_id].z, abs_dL_dmean2D);
      ... and 5 more
  [conv] coop_groups -- 9 hit(s): cooperative_groups -> hip/hip_cooperative_groups.h. HIPIFY translates the header; cg::reduce needs checking (often unused).
      cuda_rasterizer/backward.cu:14: #include <cooperative_groups.h>
      cuda_rasterizer/backward.cu:15: #include <cooperative_groups/reduce.h>
      cuda_rasterizer/backward.cu:16: namespace cg = cooperative_groups;
      cuda_rasterizer/forward.cu:14: #include <cooperative_groups.h>
      cuda_rasterizer/forward.cu:15: #include <cooperative_groups/reduce.h>
      ... and 4 more
  [conv] cub -- 6 hit(s): CUB -> hipCUB / rocPRIM. HIPIFY translates cub:: to hipcub::.
      cuda_rasterizer/rasterizer_impl.cu:20: #include <cub/cub.cuh>
      cuda_rasterizer/rasterizer_impl.cu:21: #include <cub/device/device_radix_sort.cuh>
      cuda_rasterizer/rasterizer_impl.cu:166: cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
      cuda_rasterizer/rasterizer_impl.cu:189: cub::DeviceRadixSort::SortPairs(
      cuda_rasterizer/rasterizer_impl.cu:281: CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
      ... and 1 more
  [info] cuda_ext_build -- 2 hit(s): PyTorch CUDA extension build: on ROCm PyTorch, CUDAExtension builds with hipcc automatically -- keep it, do not replace it.
      setup.py:13: from torch.utils.cpp_extension import CUDAExtension, BuildExtension
      setup.py:21: CUDAExtension(
  [info] nvcc -- 2 hit(s): Build mentions nvcc / CUDA arch: arch must be re-targeted, e.g. TORCH_CUDA_ARCH_LIST=gfx1100 (W7900D) / gfx942 (MI300X).
      CMakeLists.txt:33: set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")
      setup.py:29: extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
  [info] torch_cuda -- 1 hit(s): torch.cuda API use: unchanged on ROCm PyTorch (device type is still kCUDA under ROCm).
      rasterize_points.cu:74: torch::Device device(torch::kCUDA);

note: Prebuilt .so file(s) present alongside source (1). Treated as a build artifact, not as binary-only -- but do not ship it; rebuild from source on ROCm.

what to do next (Tier B):
  - Freeze the upstream commit and keep a pristine copy for diffing.
  - Run hipexamine / hipify --examine first, then hipify-perl (or hipify_torch for PyTorch extensions) file by file into a new directory -- never hipify in place with the same input/output name.
  - Fix the usual leftovers by hand: dead HIPIFY includes, CHECK_CUDA -> CHECK_HIP, build script arch flags (TORCH_CUDA_ARCH_LIST=gfx1100 / gfx942).
  - Build on a real ROCm machine (hipcc + ROCm PyTorch) -- Tier B does NOT mean it will compile.
  - Verify precision with a mock/synthetic compare (fixed seed, same inputs both sides), using rel_l2 / cosine, not max_abs_diff alone -- float atomicAdd order is nondeterministic.

unconfirmed (this judge cannot verify these):
  - Needs a real ROCm build (hipcc + ROCm PyTorch) to confirm it compiles.
  - Needs a runtime precision compare on a real AMD GPU; this judge only reads source.
  - Target gfx arch (e.g. gfx1100 vs gfx942) and ROCm / PyTorch versions are not verified by this tool.
  - Performance parity is not judged here; a port that compiles can still be slower until tuned.

Tier B does NOT mean the port will compile. A real ROCm build is still required.
```

Result: **PASS** -- Tier B, score 80, with the expected CUB and
cooperative_groups findings and no hard killers. One real-world
wrinkle surfaced: the upstream tree contains a stray prebuilt `.so`
(a build artifact, not in pristine upstream). The judge notes it as
a build artifact rather than calling the tree binary-only, which is
the correct call when source is present alongside it.

## Test 2 -- tiny synthetic pure-PyTorch project: must land Tier A

Fixture: `tests/fixtures/pure_torch/` (`model.py`, a `nn.Linear`
model using only `torch` / `torch.cuda`).

```bash
python judge.py tests/fixtures/pure_torch/
```

```
cuda-rocm-judge 0.1.0 -- tests/fixtures/pure_torch/
kind: source-dir
verdict: Tier A -- pure PyTorch / pure Python -- trivial, swap in ROCm PyTorch
readiness score: 100 / 100

scanned: 1 file(s) (0 .cu, 1 .py, 22 lines)

findings (file:line evidence):
  [info] torch_cuda -- 2 hit(s): torch.cuda API use: unchanged on ROCm PyTorch (device type is still kCUDA under ROCm).
      model.py:3: No custom CUDA kernels, no extension build. Only torch + torch.cuda.
      model.py:19: dev = torch.device(device if torch.cuda.is_available() else "cpu")

what to do next (Tier A):
  - Install ROCm PyTorch for your GPU arch and run the existing tests unchanged; torch.cuda code needs no source change.
  - Re-check any pinned torch / torchvision versions against the ROCm release matrix.
  - Still run one real ROCm smoke test -- Tier A is a source judgement, not a runtime guarantee.

unconfirmed (this judge cannot verify these):
  - Needs a real ROCm build (hipcc + ROCm PyTorch) to confirm it compiles.
  - Needs a runtime precision compare on a real AMD GPU; this judge only reads source.
  - Target gfx arch (e.g. gfx1100 vs gfx942) and ROCm / PyTorch versions are not verified by this tool.
  - Performance parity is not judged here; a port that compiles can still be slower until tuned.

Tier B does NOT mean the port will compile. A real ROCm build is still required.
```

Result: **PASS** -- Tier A, score 100.

Bug caught by this test (fixed before final run): the first version
scanned `README.md`, whose prose mentioned `CUDAExtension` and
`torch.cuda`, and misjudged this fixture as Tier B (score 89).
Docs are now excluded from scanning -- prose mentioning CUDA is not
code evidence.

## Test 3 -- tiny synthetic killer project: must land Tier C/D

Fixture: `tests/fixtures/killer/` -- `killer_kernel.cu` using
`__shfl_down_sync`, plus `wrapper.py` importing `nvdiffrast`.

```bash
python judge.py tests/fixtures/killer/
```

```
cuda-rocm-judge 0.1.0 -- tests/fixtures/killer/
kind: source-dir
verdict: Tier D -- cannot / not worth it (binary-only or NVIDIA-only deps)
readiness score: 29 / 100

scanned: 2 file(s) (1 .cu, 1 .py, 23 lines)

findings (file:line evidence):
  [info] cuda_ext_build -- 2 hit(s): PyTorch CUDA extension build: on ROCm PyTorch, CUDAExtension builds with hipcc automatically -- keep it, do not replace it.
      wrapper.py:5: from torch.utils.cpp_extension import load_inline
      wrapper.py:11: return load_inline(name="killer", cuda_sources=[CUDA_SRC],
  [noequiv] nvdiffrast -- 2 hit(s): nvdiffrast: NVIDIA-only differentiable rasterizer, no ROCm equivalent. The whole feature must be replaced, not translated.
      wrapper.py:1: """Synthetic killer fixture: depends on nvdiffrast (NVIDIA-only)."""
      wrapper.py:2: import nvdiffrast.torch as dr  # noqa: F401  (no ROCm equivalent)
  [hard] shfl -- 2 hit(s): __shfl_* warp shuffle: warp-size dependent (32 vs AMD wavefront 64), must be rewritten by hand, not by HIPIFY.
      killer_kernel.cu:8: v += __shfl_down_sync(0xffffffff, v, 16);
      killer_kernel.cu:9: v += __shfl_down_sync(0xffffffff, v, 8);

what to do next (Tier D):
  - Do not start a port. Either get the source (if it is a binary wheel, the source repo is the real input), or replace the NVIDIA-only dependency with a portable alternative.
  - If the blocker is one optional feature, consider porting the rest and leaving that feature CUDA-only / disabled on ROCm.
  - Re-run this judge on the source repo, not on the wheel, to see the real tier.

unconfirmed (this judge cannot verify these):
  - Needs a real ROCm build (hipcc + ROCm PyTorch) to confirm it compiles.
  - Needs a runtime precision compare on a real AMD GPU; this judge only reads source.
  - Target gfx arch (e.g. gfx1100 vs gfx942) and ROCm / PyTorch versions are not verified by this tool.
  - Performance parity is not judged here; a port that compiles can still be slower until tuned.

Tier B does NOT mean the port will compile. A real ROCm build is still required.
```

Result: **PASS** -- Tier D, score 29, with file:line evidence for
both the `__shfl_down_sync` kernel and the `nvdiffrast` import.

## Extra checks (not required, run for honesty)

- **Binary-only wheel**: a synthetic `.whl` containing only an
  empty `__init__.py` and a fake prebuilt `.so` -> Tier D, score 10,
  kind `wheel-binary-only`. Bug caught: the first version counted
  any `.py` in the wheel as "source", so this wheel was misjudged
  Tier A. "Source" now means native source (`.cu`/`.cpp`/`.h`);
  a lone `__init__.py` next to a `.so` does not count.
- **`--json`**: `python judge.py tests/fixtures/pure_torch/ --json`
  parses as valid JSON, tier A, score 100.
- **Repo root (extra)**: judging the whole mip-gaussian-l1c repo
  root gives Tier C, score 50 -- correctly flagged by the `thrust`
  use in `submodules/simple-knn/simple_knn.cu` (`thrust::device_vector`).
  That is why Test 1 scopes to the rasterizer submodule.
- **GitHub URL**: `python judge.py
  https://github.com/runjie-yan/mip-gaussian-l1c` shallow-clones and
  gives the same Tier C / 50 as the local repo root run, confirming
  the URL path works.

## Summary

| Test | Expected | Got | Result |
| --- | --- | --- | --- |
| 1. mip-gaussian rasterizer submodule | Tier B, CUB + coop_groups | Tier B, 80 | PASS |
| 2. synthetic pure-PyTorch fixture | Tier A | Tier A, 100 | PASS |
| 3. synthetic killer fixture | Tier C/D with evidence | Tier D, 29 | PASS |
