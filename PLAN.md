# Electronic Structure Theory — 3-day top-down build

**Goal:** by the end of day 3 you own a ~300-line quantum-chemistry code (HF → CI → DFT) that reproduces the numbers and plots in Tew's slides. Theory is learned only when the code refuses to run.

**Setup:** `pip install pyscf numpy scipy matplotlib`, then `python day0_run.py` — if the plot of slide 42 appears, you're ready.

**Rules**
- Every `TODO(...)` in the code names the slide / exercise that unlocks it. Watch or read *only* when a TODO blocks you.
- Tasks within a day are in dependency order, not time order; do them whenever.
- Checkpoints are the only definition of "done" (match to 1e-6 Eh). Stretch tasks and sheet questions are optional.

---

## Day 1 — Hartree–Fock (recordings 1–3 · slides 5–42)

Tasks
1. Run `day0_run.py`: look at the RHF/UHF/FCI dissociation plot (slide 42), the MO coefficient matrix of H₂O (slide 38), and the Koopmans IP vs experiment (slide 30).
2. Fill the 4 TODOs in `day1_rhf.py`: density matrix, Fock matrix (J − K/2), energy, Roothaan–Hall solve.
3. Checkpoint (STO-3G): He −2.807784 · H₂(0.74 Å) −1.116759 · H₂O −74.963023.
4. Stretch: Mulliken charges for the HF molecule (slide 39); sheet 1 Q10 coefficient matrix for He₂.

Theory to pull in when blocked: slides 14–17 (J and K), 27–29 (Fock operator, SCF loop), 31–32 (RHF, matrix equations).
Sheet questions that match today's code: 2 Q1(a,b), 2 Q3, 1 Q10.

---

## Day 2 — Configuration interaction (recordings 4–5 · slides 43–51)

Tasks
1. Run `day2_ci.py` as-is: PySCF FCI on H₂ from 0.5 to 10 Å; read off how the correlation energy grows with R.
2. Fill the 3 TODOs in `two_config_ci`: E_HF, E_uu, and the coupling K in the MO basis.
3. Checkpoint: the script's automatic sheet-2-Q8 checks all pass — 2-config CI == FCI, K → constant, Δ → 0, c → ±1/√2, E₀ → 2 E(H atom).
4. Look at the generated `slide50.png` and identify the covalent/ionic story in the two curves.
5. Stretch: general FCI by enumerating determinants with the Slater–Condon rules (slide 48) for He in 6-31G; count the dimension asked in sheet 2 Q6.

Theory when blocked: slides 46–48 (configurations, Slater–Condon), 50 (static vs dynamic correlation).
Sheet: 2 Q6, Q7, Q8.

---

## Day 3 — Density functional theory (recordings 6–8 · slides 53–75)

Tasks
1. Run `day3_lda.py` as-is: H₂⁺ HF vs LDA dissociation (`slide71.png`, self-interaction error) and the H₂ bond-length table across HF / LDA / BLYP / B3LYP (slides 70, 72).
2. Fill the 2 TODOs: Slater exchange ε_x, v_x; quadrature assembly of E_xc and V_xc (slide 61).
3. Checkpoint: match `dft.RKS(mol, xc='slater')` for He (−2.657312) and H₂O (−74.059956).
4. Set `damp=0` in `rks_lda` and watch H₂O oscillate — that is why real codes use DIIS.
5. Stretch: add B88 exchange (slide 69) via x = |∇ρ|/ρ^{4/3}; verify the scaling relations of sheet 3 Q8 numerically by scaling the geometry.

Theory when blocked: slides 55–57 (Hohenberg–Kohn I/II), 58–60 (Kohn–Sham equations), 61 (quadrature), 63–65 (LDA, scaling), 66–74 (Jacob's ladder).
Sheet: 3 Q1, Q4, Q5, Q9.

---

## After day 3
You have touched every slide except the proofs (HK I, sheet 3 Q2–3); watch recordings 6–7 last — they explain why the code you just wrote is allowed to exist.
