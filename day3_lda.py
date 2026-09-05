"""Day 3: Kohn-Sham DFT with Slater (LDA) exchange, built on your Day-1 SCF.
Morning: run as-is -> slide 71 (self-interaction error of H2+) and slide 70 bond lengths.
Afternoon: fill the 2 TODOs; the checkpoint is PySCF's RKS with xc='slater'."""
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft
from day1_rhf import integrals, density_matrix, solve_roothaan_hall

C_X = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)     # Dirac exchange constant, slide 63/64


# ---------------------------------------------------------------- morning: run
def sie_demo():
    """slide 71: H2+ has one electron, HF is exact, LDA is not."""
    R = np.linspace(1.0, 8.0, 15)
    E_hf, E_lda = [], []
    for r in R:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", unit="bohr", basis="aug-cc-pvdz",
                    charge=1, spin=1, verbose=0)
        E_hf.append(scf.UHF(mol).run().e_tot)
        E_lda.append(dft.UKS(mol, xc="lda,vwn").run().e_tot)
    plt.plot(R, E_hf, "k-", label="HF (exact for 1 e)")
    plt.plot(R, E_lda, "r-", label="SVWN (LDA)")
    plt.axhline(-0.5, color="grey", ls=":", label="H + H+ exact")
    plt.xlabel("r / bohr"); plt.ylabel("E / Eh"); plt.title("slide 71: self-interaction error")
    plt.legend(); plt.savefig("slide71.png", dpi=120); print("wrote slide71.png")


def bond_length_table():
    """slide 70 / 72: H2 bond length by method."""
    from scipy.optimize import minimize_scalar
    print("\nH2 equilibrium bond length / Å   (expt 0.741)")
    for label, make in [("HF", lambda m: scf.RHF(m)),
                        ("LDA", lambda m: dft.RKS(m, xc="lda,vwn")),
                        ("BLYP", lambda m: dft.RKS(m, xc="blyp")),
                        ("B3LYP", lambda m: dft.RKS(m, xc="b3lyp"))]:
        f = lambda r: make(gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="cc-pvtz", verbose=0)).run().e_tot
        print(f"  {label:6s} {minimize_scalar(f, bounds=(0.6, 0.9), method='bounded').x:.3f}")


# ---------------------------------------------------------------- afternoon: build
def slater_exchange(rho):
    """Return (eps_x, v_x) at each grid point for a closed-shell density rho = rho_a + rho_b.
    eps_x is the energy per electron so that E_x = sum_l w_l rho_l eps_x(rho_l)."""
    # TODO(slide 64, sheet 3 Q5): eps_x = -C_X rho^(1/3);  v_x = d(rho eps_x)/d rho = -(4/3) C_X rho^(1/3)
    # (check: with rho_a = rho_b = rho/2 the spin-polarised formula of sheet 3 Q5 collapses to this)
    raise NotImplementedError


def xc_matrix(mol, grids, P):
    """Quadrature evaluation of E_xc and V_xc[u,v] = <chi_u| v_xc(r) |chi_v>  (slide 61)."""
    ao = dft.numint.eval_ao(mol, grids.coords)                 # chi_u(r_l), shape (n_grid, m)
    rho = np.einsum("lu,uv,lv->l", ao, P, ao)                  # rho(r_l) = sum_uv chi_u P_uv chi_v
    eps_x, v_x = slater_exchange(rho)
    w = grids.weights
    # TODO(slide 61): E_x = sum_l w_l rho_l eps_l ;  Vxc_uv = sum_l w_l chi_u(r_l) v_l chi_v(r_l)
    raise NotImplementedError


def rks_lda(mol, max_iter=200, tol=1e-10, damp=0.5, verbose=True):
    """damp: mix the new density with the old one -- the bare iteration of slide 29 oscillates
    for H2O with LDA (try damp=0 and watch). Real codes use DIIS instead."""
    S, T, V, eri = integrals(mol)
    H = T + V
    nocc = mol.nelectron // 2
    E_nuc = mol.energy_nuc()
    grids = dft.gen_grid.Grids(mol); grids.build()

    eps, C = solve_roothaan_hall(H, S)
    P = density_matrix(C, nocc)
    E_old = 0.0
    for it in range(max_iter):                        # same loop as Day 1, K replaced by v_xc
        J = np.einsum("uvws,ws->uv", eri, P)
        E_xc, Vxc = xc_matrix(mol, grids, P)
        F = H + J + Vxc                               # slide 60: KS Fock operator
        E = np.einsum("uv,uv", P, H) + 0.5 * np.einsum("uv,uv", P, J) + E_xc + E_nuc
        eps, C = solve_roothaan_hall(F, S)
        P = (1 - damp) * density_matrix(C, nocc) + damp * P
        if verbose:
            print(f"  iter {it:2d}   E = {E:.10f}   dE = {E - E_old:+.2e}")
        if abs(E - E_old) < tol:
            break
        E_old = E
    return E, eps, C


if __name__ == "__main__":
    sie_demo()
    bond_length_table()

    for name, atom in [("He", "He 0 0 0"),
                       ("H2O", "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692")]:
        mol = gto.M(atom=atom, basis="sto-3g", verbose=0)
        E_ref = dft.RKS(mol, xc="slater").run().e_tot
        print(f"\n== {name}/STO-3G, Slater exchange   PySCF reference E = {E_ref:.8f}")
        try:
            E, eps, C = rks_lda(mol)
            print(f"  yours = {E:.8f}   {'OK' if abs(E - E_ref) < 1e-6 else 'MISMATCH'}")
        except NotImplementedError as e:
            print("  -> fill the TODOs (Day 1 functions must also be done).")
            break
