# Data creation:
1. Get English, Chinese and Persian AMR annotations of "The Little Prince"
2. Clean up the data, so that annotations contain only common English labels among all three languages + bracketing structure
3. Align sentences of EN and CH and create training pairs
4. Align sentences of EN and FA and create training pairs

# Training:
1. Train a model to translate from EN -> CN
2. Train a model to translate from EN -> FA
