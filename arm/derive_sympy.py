"""Symbolic Lagrangian derivation of the arm (PHYSICS_SPEC §5.2).

Used to verify the closed-form terms in arm/dynamics.py:

    python -m arm.derive_sympy      # prints M, C qd + G
"""
from __future__ import annotations

import sympy as sp


def derive():
    t = sp.symbols("t")
    q1, f2 = sp.Function("q1")(t), sp.Function("phi2")(t)
    (l1, l2, m1, m2, mc, lc, Ic, me, N1, Jr1, Jr2, g,
     my, dy) = sp.symbols("l1 l2 m1 m2 mc lc Ic m_e N1 Jr1 Jr2 g m_y d_y", positive=True)

    E = sp.Matrix([l1 * sp.cos(q1), l1 * sp.sin(q1)])
    a = sp.Matrix([sp.cos(f2), sp.sin(f2)])

    def ke_point(mass, pos):
        v = pos.diff(t)
        return sp.Rational(1, 2) * mass * (v.T * v)[0]

    # link 1: uniform rod about the origin + elbow mass + reflected rotor
    T = sp.Rational(1, 2) * (m1 * l1**2 / 3 + N1**2 * Jr1) * q1.diff(t)**2
    T += ke_point(me, E)
    # link 2: adapter rod (COM l2/2, I_com = m2 l2^2/12), funnel, rotor, stowed yo-yo
    T += ke_point(m2, E + a * l2 / 2) + sp.Rational(1, 2) * (m2 * l2**2 / 12) * f2.diff(t)**2
    T += ke_point(mc, E + a * lc) + sp.Rational(1, 2) * Ic * f2.diff(t)**2
    T += sp.Rational(1, 2) * Jr2 * f2.diff(t)**2
    T += ke_point(my, E + a * dy)

    def height(mass, pos):
        return mass * g * pos[1]

    V = (m1 * g * l1 / 2 * sp.sin(q1) + height(me, E) + height(m2, E + a * l2 / 2)
         + height(mc, E + a * lc) + height(my, E + a * dy))
    Lag = T - V

    qs = [q1, f2]
    qdd = sp.symbols("q1dd phi2dd")
    qd = sp.symbols("q1d phi2d")
    qv = sp.symbols("q1v phi2v")
    eqs = []
    for q in qs:
        eq = Lag.diff(q.diff(t)).diff(t) - Lag.diff(q)
        eqs.append(eq)
    subs = {}
    for i, q in enumerate(qs):
        subs[q.diff(t, 2)] = qdd[i]
    for i, q in enumerate(qs):
        subs[q.diff(t)] = qd[i]
    for i, q in enumerate(qs):
        subs[q] = qv[i]
    eqs = [sp.simplify(e.subs(subs)) for e in eqs]
    Mm = sp.Matrix([[sp.diff(e, qdd[j]) for j in range(2)] for e in eqs]).applyfunc(sp.simplify)
    h = sp.Matrix([e.subs({qdd[0]: 0, qdd[1]: 0}) for e in eqs]).applyfunc(sp.simplify)
    syms = dict(l1=l1, l2=l2, m1=m1, m2=m2, mc=mc, lc=lc, Ic=Ic, m_e=me, N1=N1,
                Jr1=Jr1, Jr2=Jr2, g=g, m_y=my, d_y=dy)
    return Mm, h, syms, qv, qd


def numeric_functions():
    """Return f(params_dict, q, qd) -> (M, h) using lambdified sympy expressions."""
    Mm, h, syms, qv, qd = derive()
    args = list(syms.values()) + list(qv) + list(qd)
    fM = sp.lambdify(args, Mm.tolist(), "math")
    fh = sp.lambdify(args, h.tolist(), "math")

    def f(vals: dict, q, qdot):
        a = [vals[k] for k in syms] + list(q) + list(qdot)
        return fM(*a), fh(*a)

    return f


if __name__ == "__main__":
    Mm, h, *_ = derive()
    sp.pprint(Mm)
    sp.pprint(h)
