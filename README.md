## overview
Drawing is a versatile tool for communication, spanning detailed life drawings and simple whiteboard sketches. Even the same object can be drawn in many ways, depending on the context. How do people decide how to draw in order to be understood? In this project, we collected a large number of drawings in different contexts and found that people adapted their drawings accordingly, producing detailed drawings when necessary, but simpler drawings when sufficient. To explain this contextual flexibility, we developed a computational model combining the capacity to perceive the correspondence between an object and drawing with the ability to infer what information is relevant to the viewer in context. Our results suggest drawing may be so versatile because of humans' joint capacity for visual abstraction and social reasoning.

## workflow

- **Run human experiments**
  - communication task (`/experiments/draw`)
    - Input: 3D objects in context
    - Output: sketches & viewer decisions in context
  - recognition task (`/experiments/recog`) 
    - Input: sketches from communication task and 3D objects
    - Output: sketch recognizability out of context, human confusion matrix for training visual adaptor network
- **Analyze human experimental data**
  - `/analysis/preprocess_experimental_data.py`
    - Wrapper around four scripts located at `/analysis/`: `generate_refgame_dataframe.py`,`analyze_refgame_data.py`, `generate_recog_dataframe.py`, `analyze_recog_data.py`
    - Input: raw mongo database records
    - Output: tidy formatted dataframes for communication and recognition experiments, plots, statistics
- **Train visual adaptor module networks**
  - `/models/adaptor`
    - Input: human confusion matrix from recognition experiment
    - Output: five sets of adaptor weights (for each crossvalidation fold) out of three layers (pool1, conv42, fc6); for each test set, the compatibility score between each sketch and every object (ranging from 0 to 1), and list of test set examples for each crossval fold
- **Infer social reasoning module parameters** 
  - run `/analysis/prep4RSA.py` which applies necessary preprocessing before performing Bayesian data analysis
  - model comparison (`/models/RSA.py`)
    - Run `BDA-enumerate.wppl` over large grid to get exact likelihood for every set of parameters within grid
    - Use the output from above to compute Bayes Factors for various model comparisons of interest: effect of context sensitivity, effect of cost sensitivity, effect of "visual abstraction" (fc6 adaptor vs. conv42/pool1)
  - posterior predictive checks (`/models/RSA.py`)
    - Run `BDA.wppl` which uses MCMC to infer parameter posterior for each of several models of interest, for each split.
    - Run `evaluate.wppl` to get model predictions for each trial in test set (i.e., probability of each sketch category given the current context)
    - Estimate confidence intervals for various statistics of interest, e.g., probability of congruent context condition ("close sparrow" vs. "far sparrow" on a close trial where sparrow is target), cost of sketches produced in close vs. far contexts
- **Write paper**
  - `/manuscript/`
