This repository is created for the final exam project of the  course "Math Concept for Developers". The goal is to investigate a mathematical problem and create a complete project with documentation, test files and visualizations. 

PageRank Algorithm — The Billion-Dollar Eigenvector
A technical report implementing and analyzing the PageRank algorithm using NumPy. This project covers the full pipeline: Problem formulation → Python implementation → Experimental analysis.

## Reading Order
 
Follow this sequence for the full narrative of the project:
 
```
1. Problem_Formulation/
        ↓
2. Python_Implementation/
        ↓
3. Analysis_and_Experiments/
        ↓
4. src/   ← clean, runnable code only
```

---
 
## Repository Structure
 
```
PageRank_Algorithm/
│
├── Problem_Formulation/
│   └── Problem_Formulation_f.ipynb         # Presenting: context, math, and research gap.
│
├── Python_Implementation/
│   └── Web_Modelling_and_Python_Implementation.ipynb  # Showing the full PageRank implementation with explanations.
│
├── Analysis_and_Experiments/
│   ├── Analysis_and_Experiments.ipynb      # Documented experiments and results
│   ├── pagerank_implementation.py          # Implementation script (imported by experiment notebooks).
│   └── pagerank_experiments.ipynb         # Experiment code only.
│
└── src/
    ├── pagerank_implementation.py          # Clean implementation — no documentation
    └── pagerank_experiments.py            # Clean experiments — no documentation
```
 
---
 
## What This Project Covers
 
| Step | Topic |
|------|-------|
| Problem Formulation | Web graph model, Random Surfer, Adjacency matrix, Transition matrix, Eigenvector equation|
| Implementation | Power Iteration, damping factor, dangling node handling |
| Validation | Eigenvalue verification |
| Experiments | Damping factor sensitivity, convergence analysis, graph structure effects |
 
---
 
## Core Dependencies
 
```
numpy | matplotlib | jupyter
```
