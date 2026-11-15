#!/usr/bin/env python3
"""Local bundle tool: make cases for the FHERMA harness, verify the answers.

Speaks the on-disk protocol the generated main.cpp reads and writes:

    <root>/manifest.json            {"B":..,"N":..,"W":..,"L":..,"cases":..}
    <root>/point/q.bin              L little-endian u32 limbs
    <root>/cases/000000/a.bin       B*N*L u32 limbs, b.bin likewise
    <root>/out/000000/c.bin         written by the solution
    <root>/out/results.json         written by the solution

Examples
--------
    python tools/bundle.py make /tmp/p --point '{"B":1,"N":1024,"W":128}' --seeds 0-7
    ./build/solution /tmp/p
    python tools/bundle.py verify /tmp/p

The oracle here is independent of the C++: Python integers, schoolbook for
small N and a per-coefficient direct sum (a spot check) for large N.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

# ------------------------------------------------------------------ number theory


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def derive_q(N: int, W: int) -> int:
    """The largest prime below 2^W that is 1 mod 2N, as the specification says.

    Deliberately identical to `fherma_lang.emit.testing.ntt_prime`, which is the
    helper the platform's own bundle template hands out, so local cases use the
    same q the runner will.
    """
    step = 2 * N
    q = ((1 << W) - 2) // step * step + 1
    while q > 1:
        if is_prime(q):
            return q
        q -= step
    raise SystemExit(f"no prime below 2^{W} is 1 mod {step}")


def negacyclic(a: list[int], b: list[int], q: int) -> list[int]:
    """Exact a*b mod (X^N + 1, q), schoolbook. Small N only."""
    n = len(a)
    c = [0] * n
    for i, ai in enumerate(a):
        if not ai:
            continue
        for j, bj in enumerate(b):
            if not bj:
                continue
            k = i + j
            if k < n:
                c[k] += ai * bj
            else:
                c[k - n] -= ai * bj
    return [v % q for v in c]


def coefficient(a: list[int], b: list[int], q: int, k: int) -> int:
    """One coefficient of a*b mod (X^N+1, q), by the definition. Any N."""
    n = len(a)
    total = 0
    # i + i' = k
    lo = max(0, k - (n - 1))
    for i in range(lo, k + 1):
        total += a[i] * b[k - i]
    # i + i' = k + N, subtracted
    lo = max(0, k + n - (n - 1))
    for i in range(lo, n):
        j = k + n - i
        if 0 <= j < n:
            total -= a[i] * b[j]
    return total % q


def root_of_unity(q: int, N: int, seed: int) -> int:
    """An element z with z^N = -1 (mod q); q = 1 (mod 2N) guarantees 2N of them."""
    rng = random.Random(seed)
    while True:
        g = rng.randrange(2, q - 1)
        z = pow(g, (q - 1) // (2 * N), q)
        if pow(z, N, q) == q - 1:
            return z


def algebraic_check(a: list[int], b: list[int], c: list[int], q: int, N: int, seed: int) -> bool:
    """Whole-polynomial check: evaluate everything at a 2N-th root of unity.

    x -> x^N = -1 is a ring homomorphism Z_q[X]/(X^N+1) -> Z_q, so A(z)B(z) must
    equal C(z). One field element per polynomial, and a wrong coefficient
    anywhere breaks it with probability 1 - N/q.
    """
    z = root_of_unity(q, N, seed)

    def ev(poly: list[int]) -> int:
        v = 0
        for coeff in reversed(poly):
            v = (v * z + coeff) % q
        return v

    return ev(a) * ev(b) % q == ev(c)


# ------------------------------------------------------------------------ limbs


def limbs_of(v: int, L: int) -> bytes:
    return v.to_bytes(4 * L, "little")


def read_coeffs(path: Path, N: int, L: int, B: int) -> list[list[int]]:
    raw = path.read_bytes()
    expected = B * N * L * 4
    if len(raw) != expected:
        raise SystemExit(f"{path}: {len(raw)} bytes, expected {expected}")
    out = []
    width = 4 * L
    for j in range(B):
        base = j * N * width
        out.append(
            [int.from_bytes(raw[base + i * width : base + (i + 1) * width], "little") for i in range(N)]
        )
    return out


def write_poly(path: Path, vals: list[int], L: int) -> None:
    width = 4 * L
    buf = bytearray(len(vals) * width)
    for i, v in enumerate(vals):
        buf[i * width : (i + 1) * width] = v.to_bytes(width, "little")
    path.write_bytes(buf)


# ----------------------------------------------------------------------- make


def cmd_make(args: argparse.Namespace) -> int:
    if args.point_file:
        point = json.loads(Path(args.point_file).read_text())
    else:
        point = json.loads(args.point)
    N, W = int(point["N"]), int(point["W"])
    B = int(point.get("B", 1))
    L = (W + 31) // 32
    q = derive_q(N, W)

    root = Path(args.directory)
    (root / "point").mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(parents=True, exist_ok=True)

    (root / "manifest.json").write_text(
        json.dumps({"B": B, "N": N, "W": W, "L": L, "cases": args.cases}) + "\n"
    )
    (root / "point" / "q.bin").write_bytes(limbs_of(q, L))

    lo, hi = (0, args.cases - 1)
    if args.seeds:
        lo_s, hi_s = args.seeds.split("-")
        lo, hi = int(lo_s), int(hi_s)

    for i in range(args.cases):
        seed = lo + i if hi - lo + 1 == args.cases else lo + i
        rng = random.Random((seed << 8) ^ (N * 131) ^ W)
        case = root / "cases" / f"{i:06d}"
        case.mkdir(parents=True, exist_ok=True)
        for name in ("a", "b"):
            vals = [rng.randrange(q) for _ in range(B * N)]
            width = 4 * L
            buf = bytearray(len(vals) * width)
            for k, v in enumerate(vals):
                buf[k * width : (k + 1) * width] = v.to_bytes(width, "little")
            (case / f"{name}.bin").write_bytes(bytes(buf))

    print(f"made {args.cases} case(s): B={B} N={N} W={W} L={L}")
    print(f"q = {q}")
    print(f"q is {q.bit_length()} bits, top limb {q >> (32 * (L - 1)):#x}")
    return 0


# --------------------------------------------------------------------- verify


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.directory)
    manifest = json.loads((root / "manifest.json").read_text())
    N, W, L, B = manifest["N"], manifest["W"], manifest["L"], manifest["B"]
    cases = manifest["cases"]
    q = int.from_bytes((root / "point" / "q.bin").read_bytes(), "little")

    results_path = root / "out" / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        for row in results["cases"]:
            if row["status"] != "ok":
                print(f"  case {row['i']}: {row['status']} {row.get('note', '')}")
    else:
        print("  (no out/results.json — solution has not run)")

    full = N <= args.full_limit
    failures = 0
    for i in range(cases):
        case_dir = root / "cases" / f"{i:06d}"
        ans_path = root / "out" / f"{i:06d}" / "c.bin"
        if not ans_path.exists():
            print(f"case {i:06d}: MISSING {ans_path}")
            failures += 1
            continue
        a = read_coeffs(case_dir / "a.bin", N, L, B)
        b = read_coeffs(case_dir / "b.bin", N, L, B)
        got = read_coeffs(ans_path, N, L, B)

        bad = 0
        checked = 0
        for j in range(B):
            # every returned coefficient must be canonical
            for k in range(N):
                if got[j][k] >= q:
                    bad += 1
                    if bad <= 3:
                        print(f"case {i:06d} pair {j} coeff {k}: {got[j][k]} is not canonical (< q)")
            if full:
                want = negacyclic(a[j], b[j], q)
                for k in range(N):
                    checked += 1
                    if got[j][k] != want[k]:
                        bad += 1
                        if bad <= 3:
                            print(
                                f"case {i:06d} pair {j} coeff {k}: "
                                f"got {got[j][k]}\n                   want {want[k]}"
                            )
            else:
                if args.algebraic:
                    if not algebraic_check(a[j], b[j], got[j], q, N, seed=7 * j + i):
                        bad += 1
                        if bad <= 3:
                            print(f"case {i:06d} pair {j}: algebraic check FAILED (evaluate at z)")
                rng = random.Random(1 + i * 1000 + j)
                picks = rng.sample(range(N), min(args.spot, N))
                for k in picks:
                    checked += 1
                    want = coefficient(a[j], b[j], q, k)
                    if got[j][k] != want:
                        bad += 1
                        if bad <= 3:
                            print(
                                f"case {i:06d} pair {j} coeff {k}: "
                                f"got {got[j][k]}\n                   want {want}"
                            )
        status = "ok" if bad == 0 else f"FAIL ({bad} bad of {checked} checked)"
        print(f"case {i:06d}: {status}")
        failures += 1 if bad else 0

    if failures:
        print(f"\n{failures} of {cases} case(s) failed")
        return 1
    print(f"\nall {cases} case(s) exact" + ("" if full else f" (algebraic evaluation + {args.spot} spot coefficients each)"))
    return 0


# ----------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="write a point directory with fresh cases")
    mk.add_argument("directory")
    mk.add_argument("--point", default="", help='JSON, e.g. {"B":1,"N":1024,"W":128}')
    mk.add_argument("--point-file", default="", help="file holding the point JSON")
    mk.add_argument("--cases", type=int, default=3)
    mk.add_argument("--seeds", default="", help="inclusive range, e.g. 0-7")
    mk.set_defaults(func=cmd_make)

    vf = sub.add_parser("verify", help="check out/*/c.bin against the definition")
    vf.add_argument("directory")
    vf.add_argument("--full-limit", type=int, default=2048, help="schoolbook below this N")
    vf.add_argument("--spot", type=int, default=6, help="coefficients checked when N is large")
    vf.add_argument("--algebraic", action="store_true", default=True,
                    help="also evaluate the whole polynomial at a 2N-th root of unity")
    vf.add_argument("--no-algebraic", dest="algebraic", action="store_false")
    vf.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
