"""Generate round-by-round amigurumi (single crochet) patterns from a mesh."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import trimesh


class RoundInstruction(NamedTuple):
    round_num: int
    stitch_count: int
    text: str  # e.g. "[sc 2, inc] × 6  (18 sts)"


@dataclass
class CrochetPattern:
    rounds: list[RoundInstruction]
    stitch_width_mm: float
    stitch_height_mm: float
    stitch_width_angstroms: float
    target_max_stitches: int = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_text(n_prev: int, n_curr: int, round_num: int) -> str:
    """Build human-readable instruction text for one round."""
    if n_prev == n_curr:
        if n_curr == 6:
            return f"sc × {n_curr}  ({n_curr} sts)"
        return f"sc × {n_curr}  ({n_curr} sts)"

    if n_curr > n_prev:
        # Increase round
        n_inc = n_curr - n_prev
        plain_total = n_prev - n_inc
        if n_inc == 0:
            return f"sc × {n_curr}  ({n_curr} sts)"
        if plain_total <= 0:
            # Pure increases
            return f"inc × {n_inc}  ({n_curr} sts)"
        per_group = plain_total // n_inc
        remainder = plain_total % n_inc
        base = f"[sc {per_group}, inc] × {n_inc}"
        if remainder:
            base += f", sc {remainder}"
        return f"{base}  ({n_curr} sts)"

    else:
        # Decrease round
        n_dec = n_prev - n_curr
        plain_sc = n_prev - 2 * n_dec
        if n_dec == 0:
            return f"sc × {n_curr}  ({n_curr} sts)"
        if plain_sc <= 0:
            return f"dec × {n_dec}  ({n_curr} sts)"
        per_group = plain_sc // n_dec
        remainder = plain_sc % n_dec
        base = f"[sc {per_group}, dec] × {n_dec}"
        if remainder:
            base += f", sc {remainder}"
        return f"{base}  ({n_curr} sts)"


def _expand_sequence(counts: list[int]) -> list[int]:
    """Insert intermediate rounds so no round more than doubles or halves."""
    if not counts:
        return counts
    result = [counts[0]]
    for target in counts[1:]:
        prev = result[-1]
        while True:
            if target > prev:
                max_step = prev * 2
                if target <= max_step:
                    result.append(target)
                    break
                else:
                    result.append(max_step)
                    prev = max_step
            elif target < prev:
                min_step = max(6, prev // 2)
                if target >= min_step:
                    result.append(target)
                    break
                else:
                    result.append(min_step)
                    prev = min_step
            else:
                result.append(target)
                break
    return result


def _ramp(start: int, end: int) -> list[int]:
    """Produce a feasible sequence from start to end via _expand_sequence."""
    return _expand_sequence([start, end])


def _slice_perimeter(mesh: trimesh.Trimesh, height: float,
                     principal: np.ndarray) -> float | None:
    """Return total perimeter of cross-section at given height, or None."""
    try:
        plane_origin = height * principal
        sec = mesh.section(plane_origin=plane_origin, plane_normal=principal)
        if sec is None:
            return None
        return float(sec.length)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_amigurumi_pattern(
    mesh: trimesh.Trimesh,
    stitch_width_mm: float | None = None,
    stitch_height_mm: float | None = None,
    target_max_stitches: int = 60,
    verbose: bool = False,
) -> CrochetPattern:
    """Generate an amigurumi (single-crochet) round-by-round pattern from a mesh.

    Parameters
    ----------
    mesh:
        A closed or near-closed trimesh surface (coordinates in Angstroms).
    stitch_width_mm:
        Physical stitch width in mm for display in pattern header.
        Defaults to 6.0 mm.
    stitch_height_mm:
        Physical stitch height in mm.  Defaults to stitch_width_mm.
    target_max_stitches:
        Approximate maximum stitch count at the widest round.
    verbose:
        Print progress messages.
    """
    if stitch_width_mm is None:
        stitch_width_mm = 6.0
    if stitch_height_mm is None:
        stitch_height_mm = stitch_width_mm

    # ------------------------------------------------------------------
    # Step 1 — Principal axis (PCA)
    # ------------------------------------------------------------------
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    _, eigenvectors = np.linalg.eigh(np.cov(verts.T))
    principal = eigenvectors[:, -1]   # longest axis
    heights = verts @ principal
    h_min, h_max = float(heights.min()), float(heights.max())
    total_height_A = h_max - h_min

    if verbose:
        print(f"      Principal axis: {principal}")
        print(f"      Height range: {h_min:.2f} – {h_max:.2f} Å  "
              f"(total {total_height_A:.2f} Å)")

    # ------------------------------------------------------------------
    # Step 2 — Auto-scale from max cross-section perimeter
    # ------------------------------------------------------------------
    n_presample = 50
    presample_hs = np.linspace(h_min, h_max, n_presample + 2)[1:-1]
    perimeters = [_slice_perimeter(mesh, h, principal) for h in presample_hs]
    valid_perimeters = [p for p in perimeters if p is not None and p > 0]

    if not valid_perimeters:
        raise ValueError("No valid cross-sections found; mesh may be degenerate.")

    max_perimeter_A = max(valid_perimeters)
    stitch_width_A = max_perimeter_A / target_max_stitches
    stitch_height_A = stitch_width_A  # square stitch approximation

    n_rounds_mesh = max(1, math.ceil(total_height_A / stitch_height_A))

    if verbose:
        print(f"      Max perimeter: {max_perimeter_A:.2f} Å")
        print(f"      Stitch width: {stitch_width_A:.3f} Å  "
              f"(≈ {stitch_width_mm:.1f} mm physical)")
        print(f"      Mesh rounds: {n_rounds_mesh}")

    # ------------------------------------------------------------------
    # Step 3 — Stitch count per round
    # ------------------------------------------------------------------
    slice_hs = np.linspace(h_min, h_max, n_rounds_mesh + 2)[1:-1]
    raw_counts: list[int] = []
    for h in slice_hs:
        p = _slice_perimeter(mesh, h, principal)
        if p is None or p == 0:
            raw_counts.append(0)
        else:
            raw_counts.append(max(6, round(p / stitch_width_A)))

    # Trim leading/trailing zeros
    start = 0
    while start < len(raw_counts) and raw_counts[start] == 0:
        start += 1
    end = len(raw_counts)
    while end > start and raw_counts[end - 1] == 0:
        end -= 1
    mesh_counts = raw_counts[start:end]

    if not mesh_counts:
        raise ValueError("All cross-sections returned zero stitches.")

    # Interpolate isolated interior zeros
    for i in range(1, len(mesh_counts) - 1):
        if mesh_counts[i] == 0:
            warnings.warn(
                f"Zero stitch count at interior round {i}; interpolating.",
                stacklevel=2,
            )
            mesh_counts[i] = (mesh_counts[i - 1] + mesh_counts[i + 1]) // 2

    if verbose:
        print(f"      Stitch counts (mesh body): {mesh_counts}")

    # ------------------------------------------------------------------
    # Step 4 — Expand to feasible sequence (no more than double/halve)
    # ------------------------------------------------------------------
    expanded = _expand_sequence(mesh_counts)

    # ------------------------------------------------------------------
    # Step 5 — Wrap with magic ring + closure
    # ------------------------------------------------------------------
    # Opening ramp: 6 → first count
    opening = _ramp(6, expanded[0])
    if opening[-1] == expanded[0]:
        opening = opening[:-1]  # avoid duplicate

    # Closing ramp: last count → 6
    closing = _ramp(expanded[-1], 6)
    if closing[0] == expanded[-1]:
        closing = closing[1:]  # avoid duplicate

    full_sequence = opening + expanded + closing

    # ------------------------------------------------------------------
    # Step 6 — Build RoundInstruction objects
    # ------------------------------------------------------------------
    rounds: list[RoundInstruction] = []

    # Round 1: magic ring with 6 sc
    rounds.append(RoundInstruction(
        round_num=1,
        stitch_count=6,
        text="6 sc in ring  (6 sts)",
    ))

    prev = 6
    for i, count in enumerate(full_sequence):
        rnum = i + 2
        text = _round_text(prev, count, rnum)
        rounds.append(RoundInstruction(round_num=rnum, stitch_count=count, text=text))
        prev = count

    # Final closure to 6 if not already there
    if prev != 6:
        close_seq = _ramp(prev, 6)[1:]
        for count in close_seq:
            rnum = rounds[-1].round_num + 1
            text = _round_text(prev, count, rnum)
            rounds.append(RoundInstruction(round_num=rnum, stitch_count=count, text=text))
            prev = count

    if verbose:
        print(f"      Total rounds (including open/close): {len(rounds)}")

    return CrochetPattern(
        rounds=rounds,
        stitch_width_mm=stitch_width_mm,
        stitch_height_mm=stitch_height_mm,
        stitch_width_angstroms=stitch_width_A,
        target_max_stitches=target_max_stitches,
    )


def write_amigurumi_txt(
    pattern: CrochetPattern,
    output_path: str | Path,
    verbose: bool = False,
) -> None:
    """Write a human-readable amigurumi pattern text file.

    Parameters
    ----------
    pattern:
        CrochetPattern produced by generate_amigurumi_pattern().
    output_path:
        Destination file path.
    verbose:
        Print a confirmation message after writing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Amigurumi Crochet Pattern — Protein Surface",
        "===========================================",
        f"Gauge: ~{pattern.target_max_stitches} stitches at widest circumference",
        f"       1 stitch ≈ {pattern.stitch_width_mm:.1f} mm wide, "
        f"{pattern.stitch_height_mm:.1f} mm tall",
        "Materials: worsted weight yarn, 5mm hook, fiberfill stuffing",
        "",
        "--- PATTERN ---",
        "Magic Ring",
    ]

    for r in pattern.rounds:
        lines.append(f"Round {r.round_num:<5}  {r.text}")

    lines += [
        "Fasten off, leave tail for sewing.",
        "--- END ---",
        "",
        "Stitch key:",
        "  sc    = single crochet",
        "  inc   = 2 sc in same stitch (increase)",
        "  dec   = sc2tog (single crochet 2 together, decrease)",
        "  MR    = magic ring",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if verbose:
        print(f"      Crochet pattern written → {output_path}  "
              f"({len(pattern.rounds)} rounds)")
