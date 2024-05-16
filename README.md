# Data creation:
1. Full "Litle Prince" parallel corpus of EN and FR [link](https://github.com/averkij/lingtrain-aligner-editor)
2. Parse EN and FR to AMR + clean up -> (FR bracketing, EN bracketig)
3. Parse EN and FR to UCCA + clean up -> (FR bracketing, EN bracketig)
4. Training and testing data split

# Training:
1. AMR FR -> EN
2. UCCA FR-> EN

# Evaluation:
1. FR AMR -> EN AMR vs gold EN AMR => accuracy #1
2. FR UCCA -> EN UCCA vs gold EN UCCA => accuracy #2
3. Compare a #1 and a #2 and conlclude smh
