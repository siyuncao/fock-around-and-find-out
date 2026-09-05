"""(Day 1) run first, understand later.
Reproduces slide 42 (RHF/UHF/FCI dissociation of H2), slide 38 (MO coefficients),
slide 30 (Koopmans). Nothing to fill in here -- just run it and look."""
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, cc, symm

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
# The slide tabulates ketene, H2C=C=O, in a minimal basis: 17 AOs over C1 C2 H3 H4 O5.
# Geometry optimised at RHF/STO-3G so the coefficients are comparable to the handout.
_a = np.radians(119.47 / 2)
_zC1, _zC2 = 1.1827, 1.1827 + 1.2997                 # C1=O5 and C1=C2 along z
_yH, _zH = 1.0750 * np.sin(_a), _zC2 + 1.0750 * np.cos(_a)
ketene = gto.M(atom=f"""C 0 0 {_zC1}
                        C 0 0 {_zC2}
                        H 0 {_yH} {_zH}
                        H 0 {-_yH} {_zH}
                        O 0 0 0""",                   # molecule in the yz plane,
               basis="sto-3g", symmetry=True, verbose=0)   # C2 axis = z, out-of-plane = x
mf_k = scf.RHF(ketene).run()
irrep = symm.label_orb_symm(ketene, ketene.irrep_name, ketene.symm_orb, mf_k.mo_coeff)
nocc_k = ketene.nelectron // 2                        # 22 electrons -> MO 11 is the HOMO

print(f"\nketene H2C=C=O / STO-3G   E_HF = {mf_k.e_tot:.6f} Eh   ({ketene.nao} AOs)")
cols = [9, 10, 11]                                    # the slide's MOs 10, 11, 12 (1-indexed)
hdr = f"{'AO no.':>6} {'Atom no.':>9} {'Atom':>5} {'AO':<9}"
pad = " " * len(hdr)
print(hdr + "".join(f"{'c_uk':>10}" for _ in cols))
print(pad[:-10] + f"{'MO no.':<9}" + "".join(f"{k + 1:>10d}" for k in cols))
print(pad[:-10] + f"{'Symmetry':<9}" + "".join(f"{'(' + irrep[k] + ')':>10}" for k in cols))
print(pad[:-10] + f"{'Energy':<9}" + "".join(f"{mf_k.mo_energy[k]:>10.5f}" for k in cols))
last = None
for u, lab in enumerate(ketene.ao_labels()):
    idx, at, ao = lab.split()[0], lab.split()[1], lab.split()[2].upper()
    shown = f"{int(idx) + 1:>9} {at:>5}" if idx != last else " " * 15   # atom named once
    last = idx
    print(f"{u + 1:6d}{shown} {ao:<9}" + "".join(f"{mf_k.mo_coeff[u, k]:>10.5f}" for k in cols))
print("The 0.00000 entries are exact: symmetry forbids mixing between irreps, so the B1")
print("column lives entirely on 2PX (out of plane, zero on both H) and B2 on 2PY, where")
print("the two symmetry-equivalent hydrogens appear as +c and -c.")

# ---- slide 30: Koopmans' theorem ----------------------------------------------
# The slide's last line claims a cancellation between two neglected effects. To see it
# you need a real basis: at STO-3G the basis error (~4 eV) swamps both of them.
h2o = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="cc-pvtz", verbose=0)
cation = gto.M(atom=h2o.atom, basis="cc-pvtz", charge=1, spin=1, verbose=0)
mf = scf.RHF(h2o).run()
mf_c = scf.UHF(cation).run()

koopmans = -mf.mo_energy[h2o.nelectron // 2 - 1] * HARTREE_EV   # frozen orbitals, no corr.
dscf = (mf_c.e_tot - mf.e_tot) * HARTREE_EV                     # orbitals relax, still no corr.
cc_n = cc.CCSD(mf).run(); cc_c = cc.UCCSD(mf_c).run()           # relaxation AND correlation
dccsdt = (cc_c.e_tot + cc_c.ccsd_t() - cc_n.e_tot - cc_n.ccsd_t()) * HARTREE_EV

print(f"\nH2O/cc-pVTZ  E_HF = {mf.e_tot:.6f} Eh")
print("first ionisation potential / eV")
print(f"  Koopmans   -eps_HOMO          {koopmans:6.2f}   neglects relaxation AND correlation")
print(f"  dSCF       E+(UHF) - E(RHF)   {dscf:6.2f}   orbitals relax, correlation still missing")
print(f"  dCCSD(T)                      {dccsdt:6.2f}   both included")
print(f"  experiment                     12.62")
print(f"-> relaxation alone lowers the IP by {koopmans - dscf:.2f} eV; correlation alone raises")
print(f"   it by {dccsdt - dscf:.2f} eV. Koopmans omits both, so the errors partly cancel and it")
print(f"   lands {koopmans - dccsdt:+.2f} eV out instead of {dscf - dccsdt:+.2f} -- slide 30, last line.")
