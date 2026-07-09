"""
filter.py — single-object Kalman filter, pure numpy, no I/O.

State vector:  x = [lat, lon, alt, vel_lat, vel_lon, vel_alt]
Measurement:   z = [lat, lon, alt]

Constant-velocity model — all three spatial dimensions advance linearly
with their respective velocities each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    lat: float
    lon: float
    altitude_m: float
    vel_lat: float
    vel_lon: float
    vel_alt: float


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class ObjectKalmanFilter:
    """
    3-D Kalman filter with a constant-velocity motion model.

    Parameters
    ----------
    initial_lat, initial_lon : float
        Starting position.  Velocity is initialised to zero.
    initial_alt : float
        Starting altitude above ground in metres.  Defaults to 0.0.
    dt : float
        Time step between predictions (seconds).  Defaults to 0.5 s.
    process_noise : float
        Diagonal value for the process-noise matrix Q.
    measurement_noise : float
        Diagonal value for the measurement-noise matrix R.
    """

    def __init__(
        self,
        initial_lat: float,
        initial_lon: float,
        initial_alt: float = 0.0,
        dt: float = 0.5,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-4,
    ) -> None:
        self.dt = dt

        # ── State vector x = [lat, lon, alt, vel_lat, vel_lon, vel_alt]ᵀ ───
        self.x = np.array(
            [initial_lat, initial_lon, initial_alt, 0.0, 0.0, 0.0], dtype=np.float64
        ).reshape(6, 1)

        # ── State transition matrix (constant velocity, 3 axes) ────────────
        # pos(k+1) = pos(k) + vel(k)*dt
        # vel(k+1) = vel(k)
        self.F = np.array(
            [
                [1.0, 0.0, 0.0,  dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0,  dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0,  dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # ── Measurement matrix (observes lat, lon, alt — not velocities) ───
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # ── Noise matrices ─────────────────────────────────────────────────
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        self.R = np.eye(3, dtype=np.float64) * measurement_noise

        # ── Error covariance (initially large = high uncertainty) ──────────
        self.P = np.eye(6, dtype=np.float64) * 1.0

        self._I = np.eye(6, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> KalmanState:
        """Return the current filter state as a :class:`KalmanState`."""
        return KalmanState(
            lat=float(self.x[0, 0]),
            lon=float(self.x[1, 0]),
            altitude_m=float(self.x[2, 0]),
            vel_lat=float(self.x[3, 0]),
            vel_lon=float(self.x[4, 0]),
            vel_alt=float(self.x[5, 0]),
        )

    def predict(self) -> KalmanState:
        """
        Advance the state by one time step (no measurement).

        x ← F x
        P ← F P Fᵀ + Q

        Returns the predicted :class:`KalmanState`.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, lat: float, lon: float, alt: float) -> KalmanState:
        """
        Incorporate a new 3-D position measurement.

        y = z − H x                   (innovation)
        S = H P Hᵀ + R                (innovation covariance)
        K = P Hᵀ S⁻¹                  (Kalman gain)
        x ← x + K y
        P ← (I − K H) P

        Returns the updated :class:`KalmanState`.
        """
        z = np.array([[lat], [lon], [alt]], dtype=np.float64)

        y = z - self.H @ self.x                     # innovation
        S = self.H @ self.P @ self.H.T + self.R     # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain

        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P

        return self.state
