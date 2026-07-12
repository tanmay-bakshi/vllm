"""Typed native outcomes for the NIXL transport micro-rig."""


class RigCorrectnessFailure(RuntimeError):
    """Report a validly observed transport correctness mismatch."""
