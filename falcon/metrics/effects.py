"""Direction-aware outcome effects for matched FALCON runs."""


def _oriented(value: float, higher_is_better: bool) -> float:
    return value if higher_is_better else -value


def failure_gap(
    m_ref: float, m_fail: float, higher_is_better: bool = True
) -> float:
    """Return reference performance minus failed performance."""
    return _oriented(m_ref, higher_is_better) - _oriented(
        m_fail, higher_is_better
    )


def sre(
    m_restored: float, m_fail: float, higher_is_better: bool = True
) -> float:
    """Return the stage restoration effect."""
    return _oriented(m_restored, higher_is_better) - _oriented(
        m_fail, higher_is_better
    )


def nsre(
    m_restored: float,
    m_ref: float,
    m_fail: float,
    higher_is_better: bool = True,
    min_gap: float = 1e-9,
) -> float | None:
    """Return the restoration effect normalized by the failure gap."""
    gap = failure_gap(m_ref, m_fail, higher_is_better)
    if abs(gap) < min_gap:
        return None
    return sre(m_restored, m_fail, higher_is_better) / gap


def sie(
    m_ref: float, m_injected: float, higher_is_better: bool = True
) -> float:
    """Return the stage injection effect."""
    return _oriented(m_ref, higher_is_better) - _oriented(
        m_injected, higher_is_better
    )


def nsie(
    m_ref: float,
    m_injected: float,
    m_fail: float,
    higher_is_better: bool = True,
    min_gap: float = 1e-9,
) -> float | None:
    """Return the injection effect normalized by the failure gap."""
    gap = failure_gap(m_ref, m_fail, higher_is_better)
    if abs(gap) < min_gap:
        return None
    return sie(m_ref, m_injected, higher_is_better) / gap


def bis(
    nsre_value: float | None,
    nsie_value: float | None,
    lam: float = 0.5,
) -> float | None:
    """Combine normalized restore and inject evidence."""
    if nsre_value is None or nsie_value is None:
        return None
    return (nsre_value + nsie_value) / 2 - lam * abs(
        nsre_value - nsie_value
    )


def sham_adjusted(effect: float, sham_effect: float) -> float:
    """Remove the effect measured by a sham intervention."""
    return effect - sham_effect
