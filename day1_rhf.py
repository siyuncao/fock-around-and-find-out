"""Day 1 afternoon: Restricted Hartree-Fock from scratch.
PySCF supplies the AO integrals and the reference answer; you write the SCF.
Fill the four TODOs. Each names the slide that unlocks it."""
import numpy as np
from scipy.linalg import eigh
from pyscf import gto, scf


def integrals(mol):
    S = mol.intor("int1e_ovlp")       # <chi_u|chi_v>
    T = mol.intor("int1e_kin")        # <chi_u|-1/2 nabla^2|chi_v>
    V = mol.intor("int1e_nuc")        # <chi_u|v(r)|chi_v>
    eri = mol.intor("int2e")          # (uv|ws) = int chi_u(1)chi_v(1) r12^-1 chi_w(2)chi_s(2)
    return S, T, V, eri               # NB chemists' notation: (uv|ws) = <uw|r12^-1|vs>


def density_matrix(C, nocc):
    """P_uv = 2 sum_i C_ui C_vi over the nocc doubly-occupied MOs."""
    # TODO(slide 9, slide 31): the RHF density in the AO basis.
    raise NotImplementedError


def fock_matrix(H, eri, P):
    """F = H + J - K/2  (the 1/2 is the RHF spin-summation of slide 31)."""
    # TODO(slides 16, 27): build the Coulomb and exchange matrices from P.
    #   J_uv = sum_ws (uv|ws) P_ws
    #   K_uv = sum_ws (uw|vs) P_ws
    # hint: np.einsum("uvws,ws->uv", eri, P)
    raise NotImplementedError


def energy(H, F, P, E_nuc):
    """E = 1/2 Tr[P (H + F)] + E_nuc  -- convince yourself this equals T+V+J-K."""
    # TODO(slides 15-17)
    raise NotImplementedError


def solve_roothaan_hall(F, S):
    """FC = SCe. Return (eps, C) with eps ascending."""
    # TODO(slide 32): generalised eigenproblem -> scipy.linalg.eigh(F, S)
    raise NotImplementedError


def rhf(mol, max_iter=100, tol=1e-10, verbose=True):
    S, T, V, eri = integrals(mol)
    H = T + V
    nocc = mol.nelectron // 2
    E_nuc = mol.energy_nuc()

    eps, C = solve_roothaan_hall(H, S)       # "core" guess: ignore electron repulsion
    P = density_matrix(C, nocc)
    E_old = 0.0
    for it in range(max_iter):               # slide 29 flowchart
        F = fock_matrix(H, eri, P)
        E = energy(H, F, P, E_nuc)
        eps, C = solve_roothaan_hall(F, S)
        P = density_matrix(C, nocc)
        if verbose:
            print(f"  iter {it:2d}   E = {E:.10f}   dE = {E - E_old:+.2e}")
        if abs(E - E_old) < tol:
            break
        E_old = E
    return E, eps, C


def mulliken_charges(mol, S, P):
    """Stretch goal, slide 39: q_I = Z_I - sum_{u in I} (PS)_uu."""
    # TODO(slide 39)
    raise NotImplementedError


if __name__ == "__main__":
    cases = [
        ("He", "He 0 0 0", "sto-3g"),
        ("H2", "H 0 0 0; H 0 0 0.74", "sto-3g"),
        ("H2O", "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "sto-3g"),
    ]
    for name, atom, basis in cases:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        E_ref = scf.RHF(mol).run().e_tot
        print(f"\n== {name}/{basis}   PySCF reference E = {E_ref:.10f}")
        E, eps, C = rhf(mol)
        ok = abs(E - E_ref) < 1e-8
        print(f"  yours = {E:.10f}   {'OK' if ok else 'MISMATCH'}   eps = {np.round(eps, 4)}")
