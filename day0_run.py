"""Day 1 morning: run first, understand later.
Reproduces slide 42 (RHF/UHF/FCI dissociation of H2), slide 38 (MO coefficients),
slide 30 (Koopmans). Nothing to fill in here -- just run it and look."""
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci

HARTREE_EV = 27.2114

# ---- slide 42: H2 dissociation, RHF vs UHF vs FCI -----------------------------
R = np.linspace(3.0, 0.4, 27)          # scan inwards: the broken-symmetry UHF solution is easy to find
E_rhf, E_uhf, E_fci = [], [], []       # at large R, then follow it as a starting guess
dm = None
for r in R:
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="cc-pvdz", verbose=0)
    rhf = scf.RHF(mol).run()
    uhf = scf.UHF(mol)
    if dm is None:
        dm = uhf.get_init_guess()
        dm[0][0, 0] += 0.5; dm[1][0, 0] -= 0.5      # alpha on atom A, beta on atom B
    uhf.kernel(dm0=dm)
    dm = uhf.make_rdm1()
    E_rhf.append(rhf.e_tot); E_uhf.append(uhf.e_tot)
    E_fci.append(fci.FCI(rhf).kernel()[0])

plt.plot(R, E_rhf, "r.-", label="RHF")
plt.plot(R, E_uhf, "b.-", label="UHF")
plt.plot(R, E_fci, "k.-", label="FCI")
plt.xlabel("r / Å"); plt.ylabel("E / Eh"); plt.title("slide 42"); plt.legend()
plt.savefig("slide42.png", dpi=120); print("wrote slide42.png")

# ---- slide 38: what an HF calculation actually outputs -------------------------
h2o = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="sto-3g", verbose=0)
mf = scf.RHF(h2o).run()
print(f"\nH2O/STO-3G  E_HF = {mf.e_tot:.6f} Eh")
print("orbital energies eps_k / Eh:", np.round(mf.mo_energy, 4))
print("MO coefficients C_uk (rows = AOs, cols = MOs):")
print(np.round(mf.mo_coeff, 3))

# ---- slide 30: Koopmans' theorem ----------------------------------------------
homo = mf.mo_energy[h2o.nelectron // 2 - 1]
print(f"\nKoopmans IP = -eps_HOMO = {-homo * HARTREE_EV:.2f} eV   (expt 12.6 eV)")
print("Q: why is HF this close despite missing correlation? -> slide 30, last line")
