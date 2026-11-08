# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Out-of-repo test bootstrap: aliases the operator package as
``fla.ops.decay_max_scan`` so the tests run against the drop-in layout
before the PR lands. Inside the FLA repository this file is unnecessary.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OP = ROOT / 'ops' / 'decay_max_scan'


def _load(name, path, pkg=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if pkg:
        mod.__package__ = pkg
    spec.loader.exec_module(mod)
    return mod


if 'fla.ops.decay_max_scan' not in sys.modules:
    _load('fla.ops.decay_max_scan.naive', OP / 'naive.py', 'fla.ops.decay_max_scan')
    _load('fla.ops.decay_max_scan.chunk', OP / 'chunk.py', 'fla.ops.decay_max_scan')
    _load('fla.ops.decay_max_scan.fused_recurrent', OP / 'fused_recurrent.py', 'fla.ops.decay_max_scan')
    name, path = 'fla.ops.decay_max_scan', OP / '__init__.py'
    spec = importlib.util.spec_from_file_location(name, path)
    spec.submodule_search_locations = [str(OP)]
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    mod.__package__ = name
    spec.loader.exec_module(mod)
