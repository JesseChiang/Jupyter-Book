#!/usr/bin/env python
# coding: utf-8

# # Generic Algorithm

# ## Discret Case

# In[1]:


import numpy as np


# In[2]:


def selection(pop, scores, k=3):
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def crossover(p1, p2, r_cross):
    if np.random.rand() < r_cross:
        pt = np.random.randint(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2


def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if np.random.rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]


def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    pop = [np.random.randint(0,2,n_bits).tolist() for _ in range(n_pop)]
    best, best_eval = None, objective(pop[0])
    
    for gen in range(n_iter):
        
        scores = []
        for i, gene in enumerate(pop):
            scores.append(objective(gene))
            if scores[-1] < best_eval:
                best_eval = scores[-1]
                best = gene
        if gen % 10 == 0:
            print(f"{best}: {best_eval}")
        
        selected = [selection(pop, scores) for _ in range(n_pop)]
        
        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
            pop = children

    return best, best_eval 


# In[3]:


objective = lambda x: -sum(x)
n_iter = 100
n_bits = 20
n_pop = 100
r_cross = 0.9
r_mut = 1. / n_bits
best, score = genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut)


# convergence criteria?

# In[4]:


def decode(bounds, n_bits, bitstring):
    decoded = []
    largest = 2**n_bits
    for i in range(len(bounds)):
        start, end = i*n_bits, (i+1)*n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = bounds[i][0] + (integer/largest)*(bounds[i][1]-bounds[i][0])
        decoded.append(value)
    return decoded


# In[5]:


def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    pop = [np.random.randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    best, best_eval = None, objective(decode(bounds, n_bits, pop[0]))
    
    for gen in range(n_iter):
        
        scores = []
        for i, gene in enumerate(pop):
            decoded = decode(bounds, n_bits, gene)
            scores.append(objective(decoded))
            if scores[-1] < best_eval:
                best_eval = scores[-1]
                best = decoded
        if gen % 10 == 0:
            print(f"{best}: {best_eval}")
        
        selected = [selection(pop, scores) for _ in range(n_pop)]
        
        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
            pop = children

    return best, best_eval 


# In[6]:


objective = lambda x: (x[0]**2+x[1]**2)
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
n_iter = 100
n_bits = 16
n_pop = 100
r_cross = 0.9
r_mut = 1./(n_bits*len(bounds))
best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)


# In[ ]:




