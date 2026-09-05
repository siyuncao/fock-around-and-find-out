"""Day 2: two-configuration CI for H2 in a minimal basis == sheet 2 Q8, in numbers.
Morning: run as-is (only the PySCF FCI part works). Afternoon: fill the 3 TODOs and
watch each sub-question of Q8 get checked automatically."""
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, ao2mo

ANGSTROM = 1.8897259886  # bohr per Angstrom


def h2(r_ang, basis="sto-3g"):
    return gto.M(atom=f"H 0 0 0; H 0 0 {r_ang}", basis=basis, verbose=0)


def mo_integrals(mf):
    """One- and two-electron integrals in the MO basis: h_pq and (pq|rs)."""
    C = mf.mo_coeff
    h = C.T @ mf.get_hcore() @ C
    n = C.shape[1]
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, C), n)   # (pq|rs), chemists' notation
    return h, eri


def two_config_ci(mf):
    """CI in the space {|sg sg>, |su su>}. Returns E0, c, E_HF, E_uu, K."""
    h, eri = mo_integrals(mf)
    E_nuc = mf.mol.energy_nuc()
    g, u = 0, 1                       # sigma_g = MO 0, sigma_u = MO 1

    # TODO(slide 48 line 3, or slide 31 with one doubly-occupied orbital):
    #   E_HF = 2 h_gg + (gg|gg) + E_nuc
    # TODO: same formula with sigma_u doubly occupied -> E_uu
    # TODO(sheet 2 Q8b): the off-diagonal element
    #   K = <su su| r12^-1 |sg sg> = (gu|gu) in chemists' notation
    raise NotImplementedError

    Hci = np.array([[E_HF, K], [K, E_uu]])
    E, c = np.linalg.eigh(Hci)
    return E[0], c[:, 0], E_HF, E_uu, K


if __name__ == "__main__":
    R = np.array([0.5, 0.74, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
    rows = []
    for r in R:
        mol = h2(r)
        mf = scf.RHF(mol).run()
        E_fci = fci.FCI(mf).kernel()[0]
        try:
            E0, c, E_HF, E_uu, K = two_config_ci(mf)
        except NotImplementedError:
            E0 = c = E_HF = E_uu = K = None
        rows.append((r, mf.e_tot, E_fci, E0, E_HF, E_uu, K, c))
        line = f"R={r:5.2f} Å   E_HF={mf.e_tot:.6f}   E_FCI={E_fci:.6f}   Ecorr={E_fci - mf.e_tot:+.6f}"
        if E0 is not None:
            line += f"   yours={E0:.6f}   K={K:.5f}   Delta={(E_uu - E_HF) / 2:.5f}   c={np.round(c, 3)}"
        print(line)

    if rows[0][3] is None:
        print("\nFill the TODOs in two_config_ci to unlock the Q8 checks.")
        raise SystemExit

    # ---- automatic checks of sheet 2 Q8 ------------------------------------------
    r, E_hf, E_fci, E0, E_HF, E_uu, K, c = rows[-1]
    print("\nQ8 checks at R = 10 Å:")
    print(f"  (b) 2-config CI == FCI in this basis:        {abs(E0 - E_fci) < 1e-8}")
    print(f"  (e) K tends to a constant:                    K = {K:.6f}  (compare with R=6 row)")
    print(f"  (f) E_HF and E_uu converge (Delta -> 0):       Delta = {(E_uu - E_HF) / 2:.2e}")
    print(f"  (h) |c| -> 1/sqrt(2) = 0.7071:                 c = {np.round(np.abs(c), 4)}")
    mol = h2(r)
    hA = mol.intor("int1e_kin")[0, 0] + mol.intor("int1e_nuc")[0, 0]  # <1sA|h|1sA>, h includes BOTH nuclei
    ERI_AB = mol.intor("int2e")[0, 0, 1, 1]                             # (1sA 1sA|1sB 1sB) -> 1/R
    E_H = scf.RHF(gto.M(atom="H 0 0 0", spin=1, basis="sto-3g", verbose=0)).run().e_tot
    print(f"  (g) E0 = 2<1sA|h|1sA> + (AA|BB) + E_nuc:        {E0:.6f} vs {2 * hA + ERI_AB + mol.energy_nuc():.6f}")
    print(f"      = 2 x E(H atom, STO-3G) = {2 * E_H:.6f}  -> the three 1/R terms cancel; which ones? (Q8 e,f)")

    # ---- slide 50 ------------------------------------------------------------------
    R_fine = np.linspace(0.4, 6.0, 40)
    curves = {"E_HF": [], "E_uu": [], "E0 (CI)": []}
    for r in R_fine:
        E0, _, E_HF, E_uu, _ = two_config_ci(scf.RHF(h2(r)).run())
        curves["E_HF"].append(E_HF); curves["E_uu"].append(E_uu); curves["E0 (CI)"].append(E0)
    for k, v in curves.items():
        plt.plot(R_fine * ANGSTROM, v, label=k)
    plt.xlabel("bond length / a0"); plt.ylabel("E / Eh"); plt.title("slide 50"); plt.legend()
    plt.savefig("slide50.png", dpi=120); print("wrote slide50.png")
