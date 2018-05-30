# Analysis pipeline


### Communication task


### Recognition (aka matching) task


### Sketch-3D adaptor: adapted readout from VGG layers (see pix2svg repo)

5 cross-validated 80/20 splits.

### prep for BDA to infer parameters & conduct model comparison: 3 perception (pool1, conv42, fc6) x 2 pragmatics (context-unaware, context-sensitive) x 2 production (cost, nocost) models

These are the three most critical scripts:
- preprocess_similarities.py
- prep4RSA.py
- RSA.py
