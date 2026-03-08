#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


# In[ ]:


def build_transition_matrix (adjacent_matrix):
    mat_copy = adjacent_matrix.copy()  # Copy the Adjacenet Matrix for safety
    page_count = mat_copy.shape[0] # count the number of pages

    out_degree = mat_copy.sum(axis = 0) # Calculate the column sum 
    dangling_nodes = (out_degree == 0) #Check if any column sum = 0

    if np.any(dangling_nodes):     
        mat_copy[:, dangling_nodes] = 1.0 / page_count # Set entire column to 1/n (uniform distribution)
        out_degree[dangling_nodes] = 1.0  

    tran_mat = mat_copy / out_degree #Make each column a probability distribution

    #Assert column-stochastic property
    assert np.allclose(tran_mat.sum(axis=0),1.0), "Matrix is not column stochastic"
    
    return tran_mat


# In[ ]:


#Add damping function (from Issue #7 preview) and full 8-page test:
def add_damping(transition_matrix, damping_factor=0.85):
    n = transition_matrix.shape[0] 
    d = damping_factor
    e = np.ones((n, n))
    
    m_prime = d * transition_matrix + ((1 - d) / n) * e # apply the damping factor to the transition matrix 
    column_sums = m_prime.sum(axis=0)
    
    assert np.allclose(column_sums, 1.0), "Error: Damped matrix is not column-stochastic"
    print(f"Damping factor d = {d} applied successfully")
    
    return m_prime


# In[ ]:


#Compute PageRank using the Power Iteration method with Damping Factor!
def pagerank_power_iteration(m_prime, epsilon=1e-6, max_iterations=1000):
    # Get the number of pages
    page_count = m_prime.shape[0]
    x = np.ones(page_count) / page_count
    
    error_history = [] # Storage for convergence error tracking
    
    # Main iteration loop
    for iteration in range(max_iterations):
        x_new = m_prime @ x
        error = np.linalg.norm(x_new - x, ord = 1)
        
        error_history.append(error)
        
        if error < epsilon: 
            print(f"Converged in {iteration + 1} with Final error = {error:.2e}\n")
            return x_new, error_history
        
        x = x_new # update vector
    
    print (f"Did not converge in {max_iterations} iterations")
    return x, error_history


# In[ ]:


# Error Tracking and Visualizaiton
def plot_convergence (convergence_history, title = "Plot Convergence History"):

    fig = plt.figure(figsize=(10,6))
    iterations = range(1, len(convergence_history) + 1)
    plt.semilogy(iterations, convergence_history, 'b-',linewidth = 2) #Make a plot with log scaling on the y-axis

    plt.axhline(y = 1e-6, color = 'r', linestyle = '--', label = "Conv Treshold") #Add a horizontal line spanning the whole or fraction of the Axes.

    plt.xlabel('Iteration k')
    plt.ylabel('Abs Error')
    plt.title(title)
    plt.grid()
    plt.legend()

    #display the plot
    plt.show()


# In[ ]:


def plot_pagerank_scores(pagerank_vector, labels, title="PageRank Scores"):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(pagerank_vector / pagerank_vector.max())
    bars = plt.bar(labels, pagerank_vector, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, pagerank_vector):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Labels and formatting
    plt.xlabel('Page', fontsize=12, fontweight='bold')
    plt.ylabel('PageRank Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, pagerank_vector.max() * 1.15)  # Add space for labels
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# In[ ]:


#8-Page Complex Web
np.set_printoptions(precision=4, suppress=True) #floating numbers have 4 digits after .0 and no scientific format 10e-2

# Pages: A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7
A_web = np.array([
    #A,B,C,D,E,F,G,H
    [0,0,1,1,0,0,0,0], # A receives links from C, D
    [1,0,0,0,0,0,0,0], # B receives links from A
    [1,1,0,0,0,0,0,0], # C receives links from B, A
    [0,1,1,0,0,0,0,0], # D receives links from B, C
    [0,0,0,1,0,0,0,0], # E receives links from D
    [0,0,0,0,1,0,1,0], # F receives links from E, G
    [0,0,0,0,0,1,0,0], # G receives links from F (
    [0,0,0,1,0,0,0,0] # H receives links from D
])

A_web = np.array (A_web, dtype = 'float')

#Spider Traps & Dangling Nodes
page_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

print("8-Page Web Adjacency Matrix:")
print("    ", "  ".join(page_labels))
for i, label in enumerate(page_labels):
    print(f"{label}   {A_web[i]}")


# In[ ]:


#Detailed Link Structure
for j, page in enumerate(page_labels):
    outlinks_indices = np.where(A_web[:,j]== 1)[0]
    if len(outlinks_indices)> 0 :
        outlinks_labels = [page_labels[k] for k in outlinks_indices]
        print(f"Page {page} -> {list(outlinks_labels)})")
    else:
        print(f"Page {page} -> None (dangling_node)\n")
        

#Identify dangling nodes (nodes without output links)
out_degrees = A_web.sum(axis=0)
dangling_nodes = np.where(out_degrees== 0)[0]
dangling_labels = [page_labels[i] for i in dangling_nodes]

print(f"Dangling Nodes:{dangling_labels} - no outgoing links\n")


# In[ ]:


m_web = build_transition_matrix(A_web)


# In[ ]:


m_prime_web = add_damping(m_web, damping_factor=0.85)


# In[ ]:


r_web, history_web = pagerank_power_iteration(m_prime_web, epsilon=1e-6, max_iterations=1000)


# In[ ]:


print("\nPage | PageRank Score | Percentage")
print("-" * 42)
for i, (label, score) in enumerate(zip(page_labels, r_web)):
    print(f"  {label}  |    {score:.6f}    |   {score*100:.2f}%")
print("-" * 42)
print(f"Sum: {r_web.sum():.6f} (should be 1.0)")


# In[ ]:


plot_convergence(history_web, title="Power Iteration Convergence: 8-Page Web (d=0.85)")


# In[ ]:


plot_pagerank_scores(r_web, page_labels, title="PageRank Scores: 8-Page Web (d=0.85)")


# In[ ]:


#Normalizing Pricnipal Eigenvector + Validate with Power Iteration Approach
def find_and_validate(m_prime, pagerank_power_result): 
    eigenvalues, eigenvectors = np.linalg.eig(m_prime)
    
    idx_lambda_1 = np.argmin(np.abs(eigenvalues - 1.0)) #Returns indices of the smallest abs_eigenvalue => the eigenvalue closest to 1
    
    dominant_eigenvalue = eigenvalues[idx_lambda_1] #Exctract the dominat eigenvalue
    
    pagerank_eig = np.abs(eigenvectors[:,idx_lambda_1].real)
    pagerank_eig = pagerank_eig / pagerank_eig.sum() # Normalize to sum = 1. This step should make the probability distribution = Power Iteration result 

    validation_error = np.linalg.norm(
        pagerank_power_result - pagerank_eig, ord = 1 
    )

    if validation_error < 1e-3: 
        print ("Validation Passed\n")
    else:
        print ("Validation Failed\n")

    return pagerank_eig, validation_error


# In[ ]:


r_eig, error = find_and_validate(m_prime_web, r_web)


# In[ ]:


sorted_indices = np.argsort(r_web)[::-1]  # Descending order
for rank, idx in enumerate(sorted_indices, 1):
    label = page_labels[idx]
    score = r_web[idx]
    print(f"  Rank {rank}: Page {label}  -  {score:.6f}  ({score*100:.2f}%)")


# In[ ]:


def plot_validation_comparison(r_power, r_eig, labels):
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(labels)) # Bar positions
    width = 0.35
    
    bars1 = plt.bar(x - width/2, r_power, width, label='Power Iterns',
                    color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = plt.bar(x + width/2, r_eig, width, label='Eigenvalue Decomp',
                    color='coral', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Formatting
    plt.xlabel('Page', fontsize=12, fontweight='bold')
    plt.ylabel('PageRank Score', fontsize=12, fontweight='bold')
    plt.title('Validation: Power Iteration vs. Eigenvalue Decomposition', 
              fontsize = 11, fontweight='bold')
    plt.xticks(x, labels)
    plt.grid(axis='y', alpha=0.3)
    
    # highlight difference
    for i, (p, e) in enumerate(zip(r_power, r_eig)):
        diff = abs(p - e)
        if diff > 1e-5:  # Only annotate if difference is visible
            plt.text(i, max(p, e) + 0.01, f'Δ={diff:.2e}',
                    ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.show()


# In[ ]:


plot_validation_comparison(r_web, r_eig, page_labels)

