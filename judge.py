#!/usr/bin/env python3
"""cuda-rocm-judge: judge whether a CUDA wheel / repo can be ported to ROCm.

This tool JUDGES ONLY. It does not convert code, does not build anything,
does not submit PRs, and does not touch a GPU. A Tier B verdict means
"looks convertible on paper" -- a real ROCm build (hipcc + ROCm PyTorch)
is still required to confirm anything.

Usage:
    python judge.py <path | .whl file | GitHub URL> [--json]

Verdicts (tiers):
    A  pure PyTorch / pure Python  -- trivial, swap in ROCm PyTorch
    B  HIPIFY-able                 -- convertible with HIPIFY + small fixes
    C  hard / manual rewrite       -- needs hand porting of kernels or libs
    D  cannot / not worth it       -- binary-only, or NVIDIA-only deps
       with no ROCm equivalent

Reference example for Tier B: runjie-yan/mip-gaussian-l1c @ b44209d,
submodule diff-gaussian-rasterization (~2560 lines, 12 files). It has no
__frcp_rz / __shfl_* / PTX / thrust / __ldg; only CUB (DeviceScan /
DeviceRadixSort) and cooperative_groups, both of which HIP covers.
Its real port compiled clean, smoke passed, and the mock precision
compare passed on rel_l2 / cosine -- see README.md.

Stdlib only. No pip installs, no network except an optional shallow
`git clone` of a GitHub URL you hand it.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field

VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
# Each rule: (id, severity, regex, human explanation)
# severity: "hard"    -> at least Tier C, needs manual rewrite
#           "noequiv" -> Tier D, no ROCm equivalent exists
#           "conv"    -> convertible, HIP has an equivalent (small work)
#           "info"    -> informational signal, no penalty
#
# Patterns are matched per line, in text files only. Binary files (.so,
# .dll, images) are never grepped; they are counted separately.

HARD_RULES = [
    ("shfl", "hard", r"__shfl_(?:up|down|xor|sync)",
     "__shfl_* warp shuffle: warp-size dependent (32 vs AMD wavefront 64), "
     "must be rewritten by hand, not by HIPIFY."),
    ("ballot", "hard", r"__ballot_(?:sync)?|__any_sync|__all_sync",
     "__ballot_*/__any_sync/__all_sync warp vote: warp-size dependent, "
     "must be rewritten by hand."),
    ("frcp_rz", "hard", r"__frcp_rz",
     "__frcp_rz: CUDA-only fast reciprocal, no direct HIP equivalent."),
    ("ptx", "hard", r"asm\s*(?:volatile)?\s*\(|__nv_",
     "Inline PTX assembly / __nv_ intrinsics: HIP has no PTX, must be "
     "rewritten by hand."),
    ("ldg", "hard", r"__ldg\s*\(",
     "__ldg read-only load intrinsic: CUDA-only, must be rewritten."),
    ("thrust", "hard", r"thrust::|#include\s*<thrust/",
     "thrust use: HIPIFY does not translate it; port to rocThrust by hand. "
     "(thrust::device_vector / thrust::sort are the common deep uses.)"),
    ("cutlass", "hard", r"CUTLASS|cutlass/",
     "CUTLASS: NVIDIA template library, no ROCm equivalent; needs a "
     "rocWMMA / hand-written replacement."),
    ("cuda_malloc_stream", "hard", r"cudaMalloc\s*\(|cudaStream(?:Create|Synchronize)|cudaMemcpyAsync",
     "Direct CUDA driver/runtime calls (cudaMalloc / streams): HIPIFY "
     "covers the common ones, but direct stream management usually needs "
     "hand checking. Flagged hard because it is a manual-review item, "
     "not because it is untranslatable."),
    ("windows_only", "hard", r"#include\s*<windows\.h>|_WIN32",
     "Windows-only code path: ROCm targets Linux; a Windows-only CUDA "
     "path cannot be ported as-is."),
]

NOEQUIV_RULES = [
    ("nvdiffrast", "noequiv", r"nvdiffrast|import\s+nvdiffrast",
     "nvdiffrast: NVIDIA-only differentiable rasterizer, no ROCm "
     "equivalent. The whole feature must be replaced, not translated."),
    ("kaolin", "noequiv", r"\bkaolin\b",
     "kaolin: NVIDIA 3D library, CUDA-only, no ROCm equivalent."),
    ("tensorrt", "noequiv", r"tensorrt|NvInfer|nvinfer",
     "TensorRT: NVIDIA-only inference runtime, no ROCm equivalent; "
     "replace with e.g. ONNX Runtime / MIGraphX, a different project."),
]

CONV_RULES = [
    ("cub", "conv", r"cub::|#include\s*<cub/",
     "CUB -> hipCUB / rocPRIM. HIPIFY translates cub:: to hipcub::."),
    ("coop_groups", "conv", r"cooperative_groups",
     "cooperative_groups -> hip/hip_cooperative_groups.h. HIPIFY "
     "translates the header; cg::reduce needs checking (often unused)."),
    ("cublas", "conv", r"cublas|cuBLAS",
     "cuBLAS -> hipBLAS / rocBLAS."),
    ("cufft", "conv", r"cufft|cuFFT",
     "cuFFT -> hipFFT / rocFFT."),
    ("cusparse", "conv", r"cusparse|cuSPARSE",
     "cuSPARSE -> hipSPARSE / rocSPARSE."),
    ("cusolver", "conv", r"cusolver|cuSOLVER",
     "cuSOLVER -> hipSOLVER / rocSOLVER."),
    ("curand", "conv", r"curand|cuRAND",
     "cuRAND -> hipRAND / rocRAND."),
    ("nccl", "conv", r"\bnccl\b|NCCL",
     "NCCL -> RCCL (drop-in replacement, same API)."),
    ("cudnn", "conv", r"cudnn|cuDNN",
     "cuDNN -> MIOpen, MANUAL: MIOpen has no marshalling layer for "
     "cuDNN, so this is a rewrite, not a HIPIFY translation. "
     "Counted as convertible-with-work, not as a killer."),
]

INFO_RULES = [
    ("torch_cuda", "info", r"torch\.cuda|torch::kCUDA",
     "torch.cuda API use: unchanged on ROCm PyTorch (device type is "
     "still kCUDA under ROCm)."),
    ("cuda_ext_build", "info", r"CUDAExtension|load_inline|cpp_extension",
     "PyTorch CUDA extension build: on ROCm PyTorch, CUDAExtension "
     "builds with hipcc automatically -- keep it, do not replace it."),
    ("nvcc", "info", r"nvcc|CUDA_ARCHITECTURES|TORCH_CUDA_ARCH_LIST",
     "Build mentions nvcc / CUDA arch: arch must be re-targeted, "
     "e.g. TORCH_CUDA_ARCH_LIST=gfx1100 (W7900D) / gfx942 (MI300X)."),
    ("already_hip", "info", r"hip/hip_runtime|__HIP__|__HIP_PLATFORM_AMD__",
     "Already contains HIP code: part of this tree may already be "
     "ported. CUDA findings below still apply to the rest."),
    ("atomic_add", "info", r"atomicAdd\s*\(",
     "atomicAdd (float): HIP supports it, but note that float "
     "atomicAdd reduction order is nondeterministic, so precision "
     "compares must use rel_l2 / cosine, not max_abs_diff alone."),
]

ALL_RULES = HARD_RULES + NOEQUIV_RULES + CONV_RULES + INFO_RULES

# Directories never scanned (vendored code, docs, VCS, caches).
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "third_party", "vendor",
    "build", "dist", ".eggs", "docs", "doc",
}
# File extensions scanned as text.
SCAN_EXTS = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc", ".txt",
             ".toml", ".cfg", ".cmake"}
SCAN_NAMES = {"setup.py", "CMakeLists.txt", "requirements.txt",
              "pyproject.toml", "setup.cfg", "environment.yml",
              "environment.yaml", "Makefile"}
# NOTE: README / *.md docs are deliberately NOT scanned. Prose that merely
# mentions CUDA (e.g. "no CUDA kernels") is not code evidence; scanning
# docs caused a pure-PyTorch fixture to be misjudged as Tier B.

TIERS = {
    "A": "pure PyTorch / pure Python -- trivial, swap in ROCm PyTorch",
    "B": "HIPIFY-able -- convertible with HIPIFY + small manual fixes",
    "C": "hard / manual rewrite needed",
    "D": "cannot / not worth it (binary-only or NVIDIA-only deps)",
}

NEXT_STEPS = {
    "A": [
        "Install ROCm PyTorch for your GPU arch and run the existing "
        "tests unchanged; torch.cuda code needs no source change.",
        "Re-check any pinned torch / torchvision versions against the "
        "ROCm release matrix.",
        "Still run one real ROCm smoke test -- Tier A is a source "
        "judgement, not a runtime guarantee.",
    ],
    "B": [
        "Freeze the upstream commit and keep a pristine copy for diffing.",
        "Run hipexamine / hipify --examine first, then hipify-perl (or "
        "hipify_torch for PyTorch extensions) file by file into a new "
        "directory -- never hipify in place with the same input/output name.",
        "Fix the usual leftovers by hand: dead HIPIFY includes, "
        "CHECK_CUDA -> CHECK_HIP, build script arch flags "
        "(TORCH_CUDA_ARCH_LIST=gfx1100 / gfx942).",
        "Build on a real ROCm machine (hipcc + ROCm PyTorch) -- Tier B "
        "does NOT mean it will compile.",
        "Verify precision with a mock/synthetic compare (fixed seed, "
        "same inputs both sides), using rel_l2 / cosine, not "
        "max_abs_diff alone -- float atomicAdd order is nondeterministic.",
    ],
    "C": [
        "Decide per hard finding whether to rewrite the kernel by hand, "
        "replace the library (e.g. cuDNN -> MIOpen), or drop the feature.",
        "Prototype the single hardest kernel first; if it cannot be "
        "rewritten in a day, re-scope before touching the rest.",
        "Budget days to weeks, not hours -- this is a porting project, "
        "not a HIPIFY run.",
        "Then follow the Tier B checklist for the remaining code.",
    ],
    "D": [
        "Do not start a port. Either get the source (if it is a binary "
        "wheel, the source repo is the real input), or replace the "
        "NVIDIA-only dependency with a portable alternative.",
        "If the blocker is one optional feature, consider porting the "
        "rest and leaving that feature CUDA-only / disabled on ROCm.",
        "Re-run this judge on the source repo, not on the wheel, to see "
        "the real tier.",
    ],
}

UNCONFIRMED_ALWAYS = [
    "Needs a real ROCm build (hipcc + ROCm PyTorch) to confirm it compiles.",
    "Needs a runtime precision compare on a real AMD GPU; this judge "
    "only reads source.",
    "Target gfx arch (e.g. gfx1100 vs gfx942) and ROCm / PyTorch versions "
    "are not verified by this tool.",
    "Performance parity is not judged here; a port that compiles can "
    "still be slower until tuned.",
]


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    file: str
    line: int
    snippet: str


@dataclass
class Report:
    target: str
    kind: str
    tier: str = "A"
    score: int = 100
    findings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "tool": "cuda-rocm-judge",
            "version": VERSION,
            "target": self.target,
            "kind": self.kind,
            "tier": self.tier,
            "tier_label": TIERS[self.tier],
            "score": self.score,
            "counts": self.counts,
            "findings": [
                {"rule": f.rule, "severity": f.severity,
                 "message": f.message, "file": f.file, "line": f.line,
                 "snippet": f.snippet}
                for f in self.findings
            ],
            "next_steps": NEXT_STEPS[self.tier],
            "unconfirmed": UNCONFIRMED_ALWAYS,
            "notes": self.notes,
            "disclaimer": ("Tier B does NOT mean the port will compile. "
                           "A real ROCm build is still required."),
        }


def is_binary(path, chunk=8192):
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(chunk)
    except OSError:
        return True


def iter_source_files(root):
    """Yield text source files under root, skipping vendored/doc/VCS dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not d.startswith(".")]
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            _, ext = os.path.splitext(name)
            if ext.lower() in SCAN_EXTS or name in SCAN_NAMES:
                yield path


def scan_directory(root, report):
    """Scan a source tree; fill report.counts and report.findings."""
    cu_files, py_files, total_lines = [], [], 0
    so_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not d.startswith(".")]
        for name in filenames:
            if name.endswith((".so", ".dll", ".dylib")):
                so_files.append(os.path.relpath(
                    os.path.join(dirpath, name), root))
    for path in iter_source_files(root):
        rel = os.path.relpath(path, root)
        if path.endswith((".cu", ".cuh")):
            cu_files.append(rel)
        if path.endswith(".py"):
            py_files.append(rel)
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as fh:
                lines = fh.readlines()
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: counted via .so list, not grepped
        total_lines += len(lines)
        for lineno, line in enumerate(lines, 1):
            for rule_id, severity, pattern, message in ALL_RULES:
                if re.search(pattern, line):
                    report.findings.append(Finding(
                        rule=rule_id, severity=severity, message=message,
                        file=rel, line=lineno,
                        snippet=line.strip()[:160]))
    report.counts = {
        "files_scanned": len(cu_files) + len(py_files),
        "cu_files": len(cu_files),
        "py_files": len(py_files),
        "total_lines_scanned": total_lines,
        "prebuilt_so_files": len(so_files),
        "so_files": so_files[:10],
    }
    if so_files and cu_files:
        report.notes.append(
            "Prebuilt .so file(s) present alongside source (%d). Treated "
            "as a build artifact, not as binary-only -- but do not ship "
            "it; rebuild from source on ROCm." % len(so_files))
    return report


def score_and_tier(report):
    """Decide tier + score from findings. Rules first, score second."""
    severities = {f.severity for f in report.findings}
    rules = {f.rule for f in report.findings}
    score = 100
    # Convertible items are small, known work: -5 per distinct rule.
    score -= 5 * len({f.rule for f in report.findings
                      if f.severity == "conv"})
    # Hard killers: -15 per distinct rule.
    score -= 15 * len({f.rule for f in report.findings
                       if f.severity == "hard"})
    # No-equivalent libs: -40 per distinct rule.
    score -= 40 * len({f.rule for f in report.findings
                       if f.severity == "noequiv"})
    # Custom CUDA files are work in themselves.
    n_cu = report.counts.get("cu_files", 0)
    if n_cu:
        score -= min(10, 3 * n_cu)
    # Binary-only handled by caller (kind == "wheel-binary-only").
    score = max(0, min(100, score))

    if report.kind == "wheel-binary-only":
        tier, score = "D", min(score, 10)
    elif "noequiv" in severities:
        tier, score = "D", min(score, 29)
    elif "hard" in severities:
        tier, score = "C", min(score, 59)
    elif n_cu == 0 and not any(
            f.rule == "cuda_ext_build" for f in report.findings):
        tier, score = "A", max(score, 90)
    else:
        tier, score = "B", max(60, min(score, 89))
    report.tier, report.score = tier, score
    return report


def analyse_path(path, label):
    report = Report(target=label, kind="source-dir")
    scan_directory(path, report)
    return score_and_tier(report)


def analyse_wheel(path, label):
    """Inspect a .whl (a zip). Binary-only -> D; else scan its contents."""
    tmp = tempfile.mkdtemp(prefix="rocm-judge-whl-")
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            # "Source" here means native CUDA/C++ source that HIPIFY could
            # translate. A lone __init__.py next to a .so does NOT count:
            # real binary wheels (e.g. torch_scatter) ship Python wrappers
            # around a prebuilt .so, and that .py is not portable source.
            has_source = any(n.endswith((".cu", ".cuh", ".cpp", ".h",
                                         ".hpp")) for n in names)
            has_so = any(n.endswith((".so", ".dll", ".dylib"))
                         for n in names)
            zf.extractall(tmp)
        if has_so and not has_source:
            report = Report(target=label, kind="wheel-binary-only")
            report.counts = {"files_scanned": 0, "cu_files": 0,
                             "py_files": 0, "total_lines_scanned": 0,
                             "wheel_entries": len(names),
                             "prebuilt_so_files": sum(
                                 1 for n in names
                                 if n.endswith((".so", ".dll"))),
                             "source_files": 0}
            report.notes.append(
                "Wheel contains prebuilt .so binaries and no source. "
                "HIPIFY translates source; it cannot translate a binary. "
                "Judge the source repo instead.")
            return score_and_tier(report)
        report = Report(target=label,
                        kind="wheel-with-source" if has_source else "wheel")
        scan_directory(tmp, report)
        if has_so:
            report.notes.append(
                "Wheel also ships prebuilt .so file(s); they cannot be "
                "reused on ROCm -- rebuild from source.")
        return score_and_tier(report)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def analyse_github(url, label):
    """Shallow-clone a GitHub URL to a temp dir and analyse it."""
    tmp = tempfile.mkdtemp(prefix="rocm-judge-repo-")
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", url, tmp + "/repo"],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError("git clone failed: %s"
                               % (proc.stderr.strip() or "unknown error"))
        report = analyse_path(tmp + "/repo", label)
        report.kind = "github-repo"
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def render_human(report):
    out = []
    out.append("cuda-rocm-judge %s -- %s" % (VERSION, report.target))
    out.append("kind: %s" % report.kind)
    out.append("verdict: Tier %s -- %s" % (report.tier, TIERS[report.tier]))
    out.append("readiness score: %d / 100" % report.score)
    out.append("")
    c = report.counts
    out.append("scanned: %s file(s) (%s .cu, %s .py, %s lines)%s" % (
        c.get("files_scanned", "?"), c.get("cu_files", "?"),
        c.get("py_files", "?"), c.get("total_lines_scanned", "?"),
        (", %s prebuilt .so" % c.get("prebuilt_so_files")
         if c.get("prebuilt_so_files") else "")))
    # Group findings by rule, show evidence (max 5 lines per rule).
    by_rule = {}
    for f in report.findings:
        by_rule.setdefault((f.rule, f.severity, f.message), []).append(f)
    if by_rule:
        out.append("")
        out.append("findings (file:line evidence):")
        for (rule, sev, msg), items in sorted(by_rule.items()):
            out.append("  [%s] %s -- %d hit(s): %s" % (sev, rule,
                                                        len(items), msg))
            for f in items[:5]:
                out.append("      %s:%d: %s" % (f.file, f.line, f.snippet))
            if len(items) > 5:
                out.append("      ... and %d more" % (len(items) - 5))
    else:
        out.append("")
        out.append("findings: none -- no CUDA constructs found.")
    for note in report.notes:
        out.append("")
        out.append("note: " + note)
    out.append("")
    out.append("what to do next (Tier %s):" % report.tier)
    for step in NEXT_STEPS[report.tier]:
        out.append("  - " + step)
    out.append("")
    out.append("unconfirmed (this judge cannot verify these):")
    for item in UNCONFIRMED_ALWAYS:
        out.append("  - " + item)
    out.append("")
    out.append("Tier B does NOT mean the port will compile. "
               "A real ROCm build is still required.")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Judge whether a CUDA wheel / repo can be ported to "
                    "ROCm. Judges only -- does not convert or build.")
    parser.add_argument("target",
                        help="local source dir, .whl file, or GitHub URL")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable JSON instead of text")
    args = parser.parse_args(argv)

    target = args.target
    try:
        if re.match(r"^https?://(www\.)?github\.com/", target):
            report = analyse_github(target, target)
        elif target.endswith(".whl") and os.path.isfile(target):
            report = analyse_wheel(target, target)
        elif os.path.isdir(target):
            report = analyse_path(target, target)
        elif os.path.isfile(target):
            print("error: %s is a file but not a .whl; give a directory, "
                  "a .whl, or a GitHub URL." % target, file=sys.stderr)
            return 2
        else:
            print("error: no such path: %s" % target, file=sys.stderr)
            return 2
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_human(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
