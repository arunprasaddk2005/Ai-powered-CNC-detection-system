"""
CNC Inspection System - Tolerance Checker Module (Micron-Accurate Edition)
==========================================================================

Improvements:
  1. UNCERTAINTY-AWARE VERDICTS
       A measurement near the tolerance boundary is flagged as "WARNING"
       when the measurement uncertainty overlaps the tolerance band.
       Verdict = "PASS" / "WARNING" / "FAIL".

  2. HOLE SPACING NOMINALS
       The config now accepts a nominal spacing value so hole-to-hole
       distances are properly checked, not just recorded.

  3. ALIGNED COORDINATES
       Hole positions are checked against their aligned (axis-relative)
       coordinates, making the check rotation-invariant.

  4. GD&T-STYLE POSITION TOLERANCE
       For each hole, the true position error
           TP = 2 × sqrt(Δx² + Δy²)
       is computed and compared against a positional tolerance zone diameter.

  5. STRUCTURED REPORT
       to_dict() now serialises uncertainty, verdict breakdown, and
       GD&T metrics.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from measurement import MeasurementResult, HoleMeasurement


# ---------------------------------------------------------------------------
# Tolerance specification
# ---------------------------------------------------------------------------

@dataclass
class ToleranceSpec:
    """Bilateral tolerance: measured must be in [nominal - minus, nominal + plus]."""
    nominal: float
    tolerance_plus: float
    tolerance_minus: float

    @property
    def lower(self) -> float:
        return self.nominal - self.tolerance_minus

    @property
    def upper(self) -> float:
        return self.nominal + self.tolerance_plus

    def deviation(self, measured: float) -> float:
        return measured - self.nominal

    def deviation_pct(self, measured: float) -> float:
        if self.nominal == 0:
            return 0.0
        return abs(self.deviation(measured)) / self.nominal * 100.0

    def verdict(self, measured: float, sigma: float = 0.0) -> str:
        """
        Returns "PASS", "WARNING", or "FAIL".
        WARNING = measured is within tolerance but within 1σ of the boundary.
        """
        dev = self.deviation(measured)
        if not (self.lower <= measured <= self.upper):
            return "FAIL"
        # Check if uncertainty band overlaps the tolerance limit
        if sigma > 0:
            if (measured - sigma) < self.lower or (measured + sigma) > self.upper:
                return "WARNING"
        return "PASS"


@dataclass
class HolePositionSpec:
    """
    GD&T true-position tolerance.
    nominal_x_mm, nominal_y_mm: expected position in aligned (part) frame.
    tolerance_zone_dia_mm:       diameter of the circular tolerance zone.
    """
    hole_id: int
    nominal_x_mm: float
    nominal_y_mm: float
    tolerance_zone_dia_mm: float


@dataclass
class ToleranceConfig:
    """Complete tolerance specification for one part type."""
    component_width:  Optional[ToleranceSpec] = None
    component_height: Optional[ToleranceSpec] = None
    hole_diameter:    Optional[ToleranceSpec] = None
    hole_spacing:     Optional[ToleranceSpec] = None   # nominal spacing mm
    hole_positions:   Optional[List[HolePositionSpec]] = None  # GD&T

    @classmethod
    def create_default(cls,
                       width_nominal: float,
                       height_nominal: float,
                       hole_diameter_nominal: float,
                       hole_spacing_nominal: float = 0.0,
                       width_tolerance: float = 0.5,
                       height_tolerance: float = 0.5,
                       hole_diameter_tolerance: float = 0.2,
                       hole_spacing_tolerance: float = 1.0,
                       ) -> 'ToleranceConfig':
        return cls(
            component_width=ToleranceSpec(
                width_nominal, width_tolerance, width_tolerance),
            component_height=ToleranceSpec(
                height_nominal, height_tolerance, height_tolerance),
            hole_diameter=ToleranceSpec(
                hole_diameter_nominal,
                hole_diameter_tolerance, hole_diameter_tolerance),
            hole_spacing=ToleranceSpec(
                hole_spacing_nominal,
                hole_spacing_tolerance, hole_spacing_tolerance)
            if hole_spacing_nominal > 0 else None,
        )


# ---------------------------------------------------------------------------
# Verdict containers
# ---------------------------------------------------------------------------

@dataclass
class InspectionVerdict:
    name:             str
    measured:         float
    nominal:          Optional[float]
    tol_plus:         Optional[float]
    tol_minus:        Optional[float]
    deviation:        Optional[float]
    deviation_pct:    Optional[float]
    sigma:            float    # measurement uncertainty (1-σ, mm)
    verdict:          str      # "PASS" / "WARNING" / "FAIL" / "NOT_CHECKED"

    def to_dict(self) -> Dict:
        return {
            'measurement':     self.name,
            'measured':        round(self.measured, 3),
            'nominal':         round(self.nominal, 3) if self.nominal is not None else None,
            'tolerance':       (f"+{self.tol_plus:.3f}/-{self.tol_minus:.3f}"
                                if self.tol_plus is not None else None),
            'deviation':       round(self.deviation, 3) if self.deviation is not None else None,
            'deviation_pct':   round(self.deviation_pct, 3) if self.deviation_pct is not None else None,
            'uncertainty_1sigma': round(self.sigma, 4),
            'verdict':         self.verdict,
        }


@dataclass
class GDTPositionVerdict:
    hole_id:          int
    true_position_mm: float    # TP = 2 × sqrt(Δx² + Δy²)
    tolerance_zone_dia: float
    dx_mm:            float
    dy_mm:            float
    verdict:          str

    def to_dict(self) -> Dict:
        return {
            'hole_id':          self.hole_id,
            'dx_mm':            round(self.dx_mm, 3),
            'dy_mm':            round(self.dy_mm, 3),
            'true_position_mm': round(self.true_position_mm, 3),
            'tolerance_zone_mm':round(self.tolerance_zone_dia, 3),
            'verdict':          self.verdict,
        }


@dataclass
class InspectionReport:
    verdicts:         List[InspectionVerdict]
    gdt_verdicts:     List[GDTPositionVerdict]
    overall_verdict:  str
    pass_count:       int
    warn_count:       int
    fail_count:       int
    not_checked_count:int

    def to_dict(self) -> Dict:
        return {
            'overall_verdict': self.overall_verdict,
            'summary': {
                'pass':        self.pass_count,
                'warning':     self.warn_count,
                'fail':        self.fail_count,
                'not_checked': self.not_checked_count,
                'total':       len(self.verdicts),
            },
            'details':     [v.to_dict() for v in self.verdicts],
            'gdt_position':[g.to_dict() for g in self.gdt_verdicts],
        }


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class ToleranceChecker:
    """
    Compare measurements against tolerances and issue PASS/WARNING/FAIL verdicts.
    """

    def __init__(self, config: ToleranceConfig):
        self.config = config

    def check(self, measurements: MeasurementResult) -> InspectionReport:
        verdicts: List[InspectionVerdict] = []
        gdt: List[GDTPositionVerdict] = []
        cfg = self.config

        # ── Component width ───────────────────────────────────────────────
        if cfg.component_width:
            verdicts.append(self._check_single(
                "Component Width (mm)",
                measurements.component_width_mm,
                cfg.component_width,
                measurements.sigma_width_mm,
            ))

        # ── Component height ──────────────────────────────────────────────
        if cfg.component_height:
            verdicts.append(self._check_single(
                "Component Height (mm)",
                measurements.component_height_mm,
                cfg.component_height,
                measurements.sigma_height_mm,
            ))

        # ── Hole diameters ────────────────────────────────────────────────
        if cfg.hole_diameter:
            for hm in measurements.hole_measurements:
                verdicts.append(self._check_single(
                    f"Hole {hm.hole_id} Diameter (mm)",
                    hm.diameter_mm,
                    cfg.hole_diameter,
                    hm.sigma_diameter_mm,
                ))

        # ── Hole spacings ─────────────────────────────────────────────────
        if cfg.hole_spacing and measurements.hole_spacings_mm:
            for i, j, spacing_mm in measurements.hole_spacings_mm[:10]:
                if cfg.hole_spacing.nominal > 0:
                    verdicts.append(self._check_single(
                        f"Hole {i}-{j} Spacing (mm)",
                        spacing_mm,
                        cfg.hole_spacing,
                        sigma=0.0,  # propagated separately if needed
                    ))
                else:
                    verdicts.append(InspectionVerdict(
                        name=f"Hole {i}-{j} Spacing (mm)",
                        measured=spacing_mm,
                        nominal=None, tol_plus=None, tol_minus=None,
                        deviation=None, deviation_pct=None,
                        sigma=0.0, verdict="NOT_CHECKED",
                    ))

        # ── GD&T true position ────────────────────────────────────────────
        if cfg.hole_positions:
            spec_map = {s.hole_id: s for s in cfg.hole_positions}
            for hm in measurements.hole_measurements:
                if hm.hole_id not in spec_map:
                    continue
                spec = spec_map[hm.hole_id]
                dx = hm.aligned_x_mm - spec.nominal_x_mm
                dy = hm.aligned_y_mm - spec.nominal_y_mm
                tp = 2.0 * math.hypot(dx, dy)
                v  = "PASS" if tp <= spec.tolerance_zone_dia else "FAIL"
                gdt.append(GDTPositionVerdict(
                    hole_id=hm.hole_id,
                    true_position_mm=tp,
                    tolerance_zone_dia=spec.tolerance_zone_dia,
                    dx_mm=dx, dy_mm=dy,
                    verdict=v,
                ))

        # ── Summary ───────────────────────────────────────────────────────
        pass_count  = sum(1 for v in verdicts if v.verdict == "PASS")
        warn_count  = sum(1 for v in verdicts if v.verdict == "WARNING")
        fail_count  = sum(1 for v in verdicts if v.verdict == "FAIL")
        nc_count    = sum(1 for v in verdicts if v.verdict == "NOT_CHECKED")
        gdt_fails   = sum(1 for g in gdt if g.verdict == "FAIL")

        if fail_count > 0 or gdt_fails > 0:
            overall = "FAIL"
        elif warn_count > 0:
            overall = "WARNING"
        elif pass_count > 0:
            overall = "PASS"
        else:
            overall = "FAIL"   # no checks performed = fail-safe

        return InspectionReport(
            verdicts=verdicts,
            gdt_verdicts=gdt,
            overall_verdict=overall,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            not_checked_count=nc_count,
        )

    @staticmethod
    def _check_single(name: str,
                       measured: float,
                       spec: ToleranceSpec,
                       sigma: float = 0.0) -> InspectionVerdict:
        v = spec.verdict(measured, sigma)
        return InspectionVerdict(
            name=name,
            measured=measured,
            nominal=spec.nominal,
            tol_plus=spec.tolerance_plus,
            tol_minus=spec.tolerance_minus,
            deviation=spec.deviation(measured),
            deviation_pct=spec.deviation_pct(measured),
            sigma=sigma,
            verdict=v,
        )