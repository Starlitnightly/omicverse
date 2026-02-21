"""
Kernel Service - IPython Kernel Management
===========================================
Manages in-process IPython kernel execution for OmicVerse web application.
"""

import io
import sys
import base64
import logging
import traceback
import threading
import time
import uuid
import asyncio
import json
from contextlib import redirect_stdout, redirect_stderr

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Execution state tracking for interrupt support
execution_state = {
    'is_executing': False,
    'interrupt_requested': False,
    'execution_id': None,
    'start_time': None
}
execution_state_lock = threading.Lock()

# Interrupt configuration
# Set to False if trace function causes performance issues
ENABLE_TRACE_INTERRUPT = False  # Disabled by default due to performance issues


class InProcessKernelExecutor:
    """Lightweight in-process ipykernel executor for shared state."""

    def __init__(self, kernel_lock=None):
        self.kernel_manager = None
        self.shell = None
        self.kernel_lock = kernel_lock or threading.Lock()

    def _ensure_kernel(self):
        if self.kernel_manager is not None:
            return
        try:
            from ipykernel.inprocess import InProcessKernelManager
        except ImportError as exc:
            raise RuntimeError('ipykernel is required for code execution') from exc
        self.kernel_manager = InProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.shell = self.kernel_manager.kernel.shell
        plt.switch_backend('Agg')
        self.shell.user_ns.update({
            'sc': sc,
            'pd': pd,
            'np': np,
            'plt': plt,
        })

    def restart(self):
        with self.kernel_lock:
            if self.kernel_manager is not None:
                try:
                    self.kernel_manager.shutdown_kernel(now=True)
                except Exception:
                    pass
            self.kernel_manager = None
            self.shell = None
            self._ensure_kernel()

    def sync_adata(self, adata):
        self._ensure_kernel()
        # Expose as 'odata' only — keeps 'adata' free for user-defined variables
        self.shell.user_ns['odata'] = adata

    def execute(self, code, adata=None, user_ns=None, timeout=300, stdout=None, stderr=None):
        """Execute code with interrupt support.

        Args:
            code: Python code to execute
            adata: AnnData object to inject into namespace
            user_ns: User namespace to use for execution
            timeout: Maximum execution time in seconds (default 300s/5min)
            stdout: Optional custom stdout stream (for streaming output)
            stderr: Optional custom stderr stream (for streaming output)

        Returns:
            Dictionary with output, error, result, figures, and adata

        Raises:
            KeyboardInterrupt: When execution is interrupted by user
        """
        self._ensure_kernel()
        execution_id = str(uuid.uuid4())
        timeout_timer = None
        check_interrupt_handler = None

        with self.kernel_lock:
            # Set execution state
            with execution_state_lock:
                execution_state['is_executing'] = True
                execution_state['interrupt_requested'] = False
                execution_state['execution_id'] = execution_id
                execution_state['start_time'] = time.time()

            # Setup timeout handler
            if timeout and timeout > 0:
                def force_interrupt():
                    with execution_state_lock:
                        if execution_state['execution_id'] == execution_id:
                            execution_state['interrupt_requested'] = True
                            logging.warning(f"Execution timeout reached for {execution_id}")

                timeout_timer = threading.Timer(timeout, force_interrupt)
                timeout_timer.start()

            # Define interrupt check hook (must accept info parameter from IPython)
            def check_interrupt(info):
                with execution_state_lock:
                    if execution_state['interrupt_requested'] and \
                       execution_state['execution_id'] == execution_id:
                        raise KeyboardInterrupt("Execution interrupted by user")

            # Register pre_run_cell event to check for interrupts
            check_interrupt_handler = check_interrupt
            self.shell.events.register('pre_run_cell', check_interrupt_handler)

            # Install trace function for fine-grained interrupt checking (if enabled)
            # This allows interrupting code inside loops but may impact performance
            # NOTE: Currently disabled by default due to performance issues
            if ENABLE_TRACE_INTERRUPT:
                trace_counter = [0]
                def trace_interrupt(frame, event, arg):
                    # Only trace 'line' events to reduce overhead
                    if event != 'line':
                        return trace_interrupt

                    # Only check every 100 lines to reduce lock contention
                    trace_counter[0] += 1
                    if trace_counter[0] % 100 != 0:
                        return trace_interrupt

                    # Check interrupt flag
                    with execution_state_lock:
                        if execution_state['interrupt_requested'] and \
                           execution_state['execution_id'] == execution_id:
                            raise KeyboardInterrupt("Execution interrupted by user")
                    return trace_interrupt

                # Set trace function
                sys.settrace(trace_interrupt)

            try:
                original_ns = self.shell.user_ns
                if user_ns is not None:
                    self.shell.user_ns = user_ns
                if adata is not None:
                    self.shell.user_ns['odata'] = adata

                # Use provided streams or create new buffers
                stdout_buf = stdout if stdout is not None else io.StringIO()
                stderr_buf = stderr if stderr is not None else io.StringIO()

                before_figs = set(plt.get_fignums())
                try:
                    result = None
                    error_msg = None
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        result = self.shell.run_cell(code, store_history=True)
                    if result.error_before_exec or result.error_in_exec:
                        err = result.error_before_exec or result.error_in_exec
                        error_msg = ''.join(traceback.format_exception(err.__class__, err, err.__traceback__))
                    output = stdout_buf.getvalue()
                    stderr_output = stderr_buf.getvalue()

                    figures = []
                    after_figs = set(plt.get_fignums())
                    new_figs = [num for num in after_figs if num not in before_figs]
                    for fig_num in new_figs:
                        fig = plt.figure(fig_num)
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight')
                        figures.append(base64.b64encode(buf.getvalue()).decode('ascii'))
                        plt.close(fig)

                    last_result = result.result if result else None
                    adata_value = self.shell.user_ns.get('adata')
                    return {
                        'output': output,
                        'stderr': stderr_output,
                        'error': error_msg,
                        'result': last_result,
                        'figures': figures,
                        'adata': adata_value
                    }
                finally:
                    if user_ns is not None:
                        self.shell.user_ns = original_ns
            finally:
                # Cleanup: cancel timeout, unregister hook, clear trace, reset state
                # Clear trace function first (safe even if not set)
                if ENABLE_TRACE_INTERRUPT:
                    sys.settrace(None)

                if timeout_timer is not None:
                    timeout_timer.cancel()
                if check_interrupt_handler is not None:
                    try:
                        self.shell.events.unregister('pre_run_cell', check_interrupt_handler)
                    except Exception as e:
                        logging.warning(f"Failed to unregister interrupt handler: {e}")
                with execution_state_lock:
                    execution_state['is_executing'] = False
                    execution_state['execution_id'] = None
                    execution_state['start_time'] = None


def normalize_kernel_id(kernel_id):
    """Normalize kernel ID to standard format."""
    if not kernel_id:
        return 'default.ipynb'
    return str(kernel_id)


def build_kernel_namespace(include_adata=False, current_adata=None):
    """Build a fresh kernel namespace with standard imports.

    Args:
        include_adata: Whether to include adata in namespace
        current_adata: AnnData object to include if include_adata is True

    Returns:
        Dictionary with standard namespace objects
    """
    namespace = {
        'sc': sc,
        'pd': pd,
        'np': np,
        'plt': plt,
    }
    if include_adata and current_adata is not None:
        namespace['adata'] = current_adata
    return namespace


def reset_kernel_namespace(kernel_id, kernel_executor, kernel_sessions, current_adata=None):
    """Reset kernel namespace to fresh state.

    Args:
        kernel_id: Kernel identifier
        kernel_executor: InProcessKernelExecutor instance
        kernel_sessions: Dictionary of kernel sessions
        current_adata: Current AnnData object
    """
    kernel_id = normalize_kernel_id(kernel_id)
    if kernel_id == 'default.ipynb':
        kernel_executor.restart()
        if current_adata is not None:
            kernel_executor.sync_adata(current_adata)
        return
    kernel_sessions[kernel_id] = {
        'user_ns': build_kernel_namespace()
    }


def get_kernel_context(kernel_id, kernel_executor, kernel_sessions):
    """Get executor and namespace for a kernel.

    Args:
        kernel_id: Kernel identifier
        kernel_executor: InProcessKernelExecutor instance
        kernel_sessions: Dictionary of kernel sessions

    Returns:
        Tuple of (executor, namespace)
    """
    kernel_id = normalize_kernel_id(kernel_id)
    if kernel_id == 'default.ipynb':
        kernel_executor._ensure_kernel()
        return kernel_executor, kernel_executor.shell.user_ns
    session = kernel_sessions.get(kernel_id)
    if session is None:
        session = {
            'user_ns': build_kernel_namespace()
        }
        kernel_sessions[kernel_id] = session
    return kernel_executor, session['user_ns']


def get_execution_state():
    """Get current execution state (thread-safe).

    Returns:
        Dictionary with execution state snapshot
    """
    with execution_state_lock:
        return {
            'is_executing': execution_state['is_executing'],
            'interrupt_requested': execution_state['interrupt_requested'],
            'execution_id': execution_state['execution_id'],
            'start_time': execution_state['start_time']
        }


def request_interrupt():
    """Request interrupt of current execution (thread-safe).

    Returns:
        Dictionary with interrupt request result
    """
    with execution_state_lock:
        if not execution_state['is_executing']:
            return {
                'success': False,
                'message': 'No code is currently executing'
            }
        execution_state['interrupt_requested'] = True
        return {
            'success': True,
            'execution_id': execution_state['execution_id'],
            'message': 'Interrupt requested'
        }
