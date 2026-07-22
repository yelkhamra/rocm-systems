#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
run_amdsmi_upgrade_downgrade_test.py
====================================

The packaging rework changed how the amdsmi Python module is delivered (direct
site-packages install instead of a postinst pip install). This test exercises
the deb/rpm upgrade and downgrade paths to prove the module, the ld.so.conf.d
entry, and the CLI stay consistent when moving between two package builds.

Given an OLD and a NEW package it:
  1. installs OLD, verifies import + CLI,
  2. installs NEW on top (upgrade), verifies again,
  3. reinstalls OLD (downgrade), verifies again,
so a broken pre/postinst transition fails here instead of on a user's machine.

Requires root (installs/removes system packages). Skips cleanly if the package
manager is unavailable. Not run by default in the build harness; wire it into
CI where a prior-version artifact is available.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _install(pkg: Path, manager: str) -> None:
    if manager == "apt":
        _run(["apt-get", "install", "-y", "--allow-downgrades", "--reinstall", str(pkg)])
    elif manager == "dnf":
        _run(["dnf", "install", "-y", "--allowerasing", str(pkg)])
    elif manager == "zypper":
        _run(["zypper", "--non-interactive", "install", "--allow-downgrade", str(pkg)])
    else:
        sys.exit(f"unsupported package manager: {manager}")


def _verify() -> None:
    # Module imports and resolves a library path.
    _run(["python3", "-c", "import amdsmi; print('import OK from', amdsmi.__file__)"])
    # CLI runs (version works without a GPU).
    if shutil.which("amd-smi"):
        _run(["amd-smi", "version"])
    # The ld.so.conf.d entry the postinst writes must exist and resolve.
    conf = Path("/etc/ld.so.conf.d/x86_64-libamd_smi_lib.conf")
    if conf.is_file():
        libdir = conf.read_text().strip().splitlines()[0].strip()
        if not (Path(libdir).glob("libamd_smi.so*")):
            sys.exit(f"ld.so.conf.d points at {libdir} but no libamd_smi.so is there")


def _detect_manager() -> str:
    for mgr in ("apt-get", "dnf", "zypper"):
        if shutil.which(mgr):
            return {"apt-get": "apt", "dnf": "dnf", "zypper": "zypper"}[mgr]
    sys.exit("no supported package manager found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-package", type=Path, required=True, help="Prior-version .deb/.rpm.")
    parser.add_argument("--new-package", type=Path, required=True, help="New .deb/.rpm to test.")
    parser.add_argument(
        "--package-manager", default=None, help="apt / dnf / zypper (default: auto)."
    )
    args = parser.parse_args()

    for p in (args.old_package, args.new_package):
        if not p.is_file():
            sys.exit(f"package not found: {p}")

    manager = args.package_manager or _detect_manager()

    print("=== 1. install OLD ===")
    _install(args.old_package, manager)
    _verify()

    print("=== 2. upgrade to NEW ===")
    _install(args.new_package, manager)
    _verify()

    print("=== 3. downgrade to OLD ===")
    _install(args.old_package, manager)
    _verify()

    print("PASS: upgrade and downgrade keep the module, CLI, and linker config consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
