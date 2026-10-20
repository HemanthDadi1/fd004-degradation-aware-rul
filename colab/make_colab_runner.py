"""Build RUN_ON_COLAB.ipynb: runs the packaged FD004 notebooks on a Colab GPU, resumable, and zips the results."""
import json, sys

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md(r"""
# FD004 Paper 2 runs on Colab (T4 GPU)

Runs notebooks 27–36 (single-direction LSTM without attention at 30/60 cycles; the 45- and 90-cycle sweep), five seeds each,
with the same pre-specified training rule as every other run.

**Before you start**
1. Upload `FD004_Colab.zip` to the top level of your Google Drive (**My Drive**).
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*. Approve the Google Drive access prompt.

**If Colab disconnects** (it will, after a few hours on the free tier): reconnect and *Run all* again. Finished notebooks are
skipped, and inside a notebook every seed whose model is already saved is skipped too, so no finished work is repeated.

**When it is done**, download `FD004_Colab_results.zip` from My Drive and send it back (or drop it in the `Downloads` folder)
so the results can be merged with the local runs.
""")

code(r"""
# 1. Check the GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv
import tensorflow as tf
print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
assert tf.config.list_physical_devices("GPU"), "No GPU: Runtime → Change runtime type → T4 GPU, then run again."
""")

code(r"""
# 2. Mount Drive and unpack the inputs (only the first time)
from google.colab import drive
drive.mount("/content/drive")
import os, zipfile
from pathlib import Path
DRIVE = Path("/content/drive/MyDrive")
ROOT = DRIVE / "FD004_Colab"
if not (ROOT / "CMAPSSData/train_FD004.txt").exists():
    z = DRIVE / "FD004_Colab.zip"
    assert z.exists(), "Upload FD004_Colab.zip to the top level of My Drive first."
    zipfile.ZipFile(z).extractall(DRIVE)
print("Inputs in", ROOT)
print(sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb")))
""")

code(r"""
# 3. Packages the notebooks need (Colab normally has them all)
import importlib
for mod, pip_name in [("sklearn", "scikit-learn"), ("scipy", "scipy"), ("pandas", "pandas"), ("nbconvert", "nbconvert")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        !pip -q install {pip_name}
import keras, sklearn, numpy, pandas
print("keras", keras.__version__, "| sklearn", sklearn.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
""")

code(r"""
# 4. Run every notebook in order; skip finished ones; resume interrupted ones
import subprocess, time, re
EXEC = ROOT / "executed"; EXEC.mkdir(exist_ok=True)
LOGS = ROOT / "logs"; LOGS.mkdir(exist_ok=True)
env = dict(os.environ, FD004_ROOT=str(ROOT), TF_CPP_MIN_LOG_LEVEL="3")
for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
    done = EXEC / nb.name.replace(".ipynb", "_EXECUTED.ipynb")
    if done.exists():
        print(f"skip  {nb.name} (already finished)"); continue
    t0 = time.time(); print(f"run   {nb.name} ...", flush=True)
    with open(LOGS / f"{nb.stem}.log", "w") as log:
        r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb), "--output", done.name,
                            "--output-dir", str(EXEC), "--ExecutePreprocessor.timeout=-1"], env=env, stdout=log, stderr=subprocess.STDOUT)
    status = "ok" if r.returncode == 0 else f"FAILED (see logs/{nb.stem}.log)"
    print(f"      {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
print("All notebooks processed.")
""")

code(r"""
# 5. Zip the results for download
import shutil
keep = {"FD004_DS_B_Results", "FD004_DS_A_Rerun_Results"}   # inputs, not results
out = DRIVE / "FD004_Colab_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d in sorted(ROOT.glob("FD004_*_Results")):
        if d.name in keep:
            continue
        for f in d.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))
    for f in list(EXEC.glob("*.ipynb")) + list(LOGS.glob("*.log")):
        z.write(f, f.relative_to(ROOT))
print("Wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")
for d in sorted(ROOT.glob("FD004_*_Results")):
    if d.name not in keep:
        n = len(list((d / "histories").glob("*.csv"))) if (d / "histories").exists() else 0
        print(f"  {d.name}: {n}/5 seeds", "DONE" if (d / "manifest_sha256.json").exists() else "")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1])
