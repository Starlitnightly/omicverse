"""Regression tests for OmicVerse Web kernel namespace handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_kernel_service_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "omicverse_web" / "services" / "kernel_service.py"
    spec = importlib.util.spec_from_file_location("ov_web_kernel_service_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cleanup_executor(executor):
    if executor.kernel_manager is None:
        return
    try:
        executor.kernel_manager.shutdown_kernel(now=True)
    except TypeError:
        executor.kernel_manager.shutdown_kernel()


class TestWebKernelService(unittest.TestCase):

    def test_build_kernel_namespace_seeds_ipython_history_keys(self):
        mod = _load_kernel_service_module()

        ns = mod.build_kernel_namespace()

        self.assertIn('_oh', ns)
        self.assertIsInstance(ns['_oh'], dict)
        self.assertIn('Out', ns)
        self.assertIs(ns['Out'], ns['_oh'])
        self.assertIn('_ih', ns)
        self.assertIn('In', ns)

    def test_custom_namespace_expression_does_not_fail_displayhook(self):
        mod = _load_kernel_service_module()

        executor = mod.InProcessKernelExecutor()
        self.addCleanup(lambda: _cleanup_executor(executor))
        ns = mod.build_kernel_namespace()
        ns['single_ad_ref'] = {'cells': 3}

        result = executor.execute('single_ad_ref', user_ns=ns, timeout=30)

        self.assertIsNone(result['error'])
        self.assertEqual(result['result'], {'cells': 3})
        self.assertIn('_oh', ns)
        self.assertTrue(any(value == {'cells': 3} for value in ns['_oh'].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
