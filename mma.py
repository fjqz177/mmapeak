"""
JAX-based matrix multiply throughput benchmark.

The script mirrors the mmapeak timing pattern in Python/JAX to avoid CUDA
compilation overhead while keeping a comparable measurement loop.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Iterable, List, Tuple

import jax
import jax.numpy as jnp
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

DEFAULT_TARGET_TIME = 3.0
DEFAULT_MATMUL_SIZE = 4096


def print_heading(title: str, separator: str = "-") -> None:
    """
    Print a centered heading line for readability.

    Parameters
    ----------
    title : str
        Title to display.
    separator : str
        Single-character separator used on both sides.
    """

    left = separator * 20
    right = separator * 20
    print(f"\n{left} {title} {right}")


def build_matmul(dtype: jnp.dtype, size: int) -> Tuple[Callable[[], jnp.ndarray], int]:
    """
    Create a JIT-compiled matrix multiplication closure with fixed operands.

    Parameters
    ----------
    dtype : jnp.dtype
        Data type for operands.
    size : int
        Matrix dimension (square matrices of shape (size, size)).

    Returns
    -------
    callable
        Zero-argument callable executing the matmul and returning the result.
    int
        Effective floating-point operations per call (2 * size^3).
    """

    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    if jnp.issubdtype(dtype, jnp.integer):
        a = jax.random.randint(key_a, (size, size), -2, 3, dtype=dtype)
        b = jax.random.randint(key_b, (size, size), -2, 3, dtype=dtype)
    else:
        a = jax.random.normal(key_a, (size, size), dtype=dtype)
        b = jax.random.normal(key_b, (size, size), dtype=dtype)

    @jax.jit
    def _matmul(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x @ y

    def _run() -> jnp.ndarray:
        return _matmul(a, b)

    # Trigger compilation before benchmarking.
    _ = _run().block_until_ready()
    ops = 2 * size * size * size
    return _run, ops


def benchmark(op: Callable[[], jnp.ndarray], ops_per_call: int, target_time: float) -> Tuple[float, float]:
    """
    Benchmark a callable using a two-phase timing loop (calibrate then measure).

    Parameters
    ----------
    op : callable
        Zero-argument callable to benchmark. Must return a JAX array.
    ops_per_call : int
        Floating-point operations executed per call.
    target_time : float
        Target wall-clock time for measurement in seconds.

    Returns
    -------
    float
        Total elapsed time during the measurement phase in milliseconds.
    float
        Throughput in TFLOP/s.
    """

    n_loop = 8

    def _run_loops(count: int) -> float:
        start = time.perf_counter()
        for _ in range(count):
            _ = op().block_until_ready()
        end = time.perf_counter()
        return end - start

    calibrate = _run_loops(n_loop)
    n_loop = max(int(target_time / calibrate * n_loop), 1)

    measured = _run_loops(n_loop)
    t_ms = measured * 1e3
    tflops = (ops_per_call * n_loop) / measured / 1e12
    return t_ms, tflops


def run_group(title: str, tests: Iterable[Tuple[str, jnp.dtype]], size: int, target_time: float) -> None:
    """
    Execute and print a group of benchmarks.

    Parameters
    ----------
    title : str
        Group title.
    tests : Iterable[Tuple[str, jnp.dtype]]
        Iterable of (label, dtype) pairs.
    size : int
        Matrix dimension.
    target_time : float
        Target wall time per benchmark.
    """

    print_heading(title, "=" if "INT" in title else "-")
    for label, dtype in tests:
        op, ops = build_matmul(dtype, size)
        t_ms, tflops = benchmark(op, ops, target_time)
        print(f"{label}: {t_ms:.1f} ms {tflops:.1f} T(fl)ops")


def main() -> None:
    """
    Entry point for the JAX mmapeak benchmark.
    """

    parser = argparse.ArgumentParser(description="JAX mmapeak-style matmul benchmark")
    parser.add_argument(
        "-t",
        "--target-time",
        type=float,
        default=DEFAULT_TARGET_TIME,
        help=f"target time per benchmark in seconds (default: {DEFAULT_TARGET_TIME})",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=DEFAULT_MATMUL_SIZE,
        help=f"matrix dimension for square matmul (default: {DEFAULT_MATMUL_SIZE})",
    )
    args = parser.parse_args()

    float_tests: List[Tuple[str, jnp.dtype]] = [
        ("mm_jax_fp16", jnp.float16),
        ("mm_jax_bf16", jnp.bfloat16),
        ("mm_jax_fp32", jnp.float32),
        ("mm_jax_fp64", jnp.float64),
    ]
    int_tests: List[Tuple[str, jnp.dtype]] = [
        ("mm_jax_int8", jnp.int8),
        ("mm_jax_int32", jnp.int32),
    ]

    print(f"Using JAX platform: {jax.default_backend()}")
    print(f"Matrix size: {args.size}x{args.size}, target time: {args.target_time:.1f}s")

    run_group("INT", int_tests, args.size, args.target_time)
    run_group("FLOAT", float_tests, args.size, args.target_time)


if __name__ == "__main__":
    main()
