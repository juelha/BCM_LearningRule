
# BCM Learning Rule

## Theory

### Motivation

- BCM rule introduced to account for "striking dependence of the sharpness of orientation selectivity on the visual env" https://www.researchgate.net/publication/2308452_Effect_of_Binocular_Cortical_Misalignment_on_Ocular_Dominance_and_Orientation_Selectivity




### The Model


OG paper version:
$$
\dot{m}_j (t) = \phi(c(t))d_j(t) - \epsilon m_j(t)
$$

$$
\Theta_M = (\bar{c}/c_0)^p \bar{c}
$$

$$
\bar{c}(t) = m(t) \cdot \bar{d}
$$

- apperently BCM og uses $\phi = y (y-  \theta)$ for their results https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0044-6  

Scholarpedia version:
$$
y = \sum_i w_i x_i
$$
$$
\frac{dw_i}{dt} = y(y - \Theta_M) x_i - \epsilon w_i
$$
$$
\Theta_M = E^p [(y/y_o)]
$$

Questions:

- val for p, c_o, eps
- func for $\phi$
- calculation of activation average

Law and Cooper version:
$$
y = \sigma (\sum_i w_i x_i)
$$
$$
\theta = E [y^2]
$$
$$
\frac{dw_i}{dt} = \frac{y (y-  \theta) x_i}{\theta}
$$

The Law and Cooper form has all of the same fixed points as the Intrator and Cooper form, but speed of synaptic modification increases when the threshold is small, and decreases as the threshold increases. The practical result is that the simulation can be run with artificially high learning rates, and wild oscillations are reduced. This form has been used primarily when running simulations of networks, where the run-time of the simulation can be prohibitive. ascholarped

Squadrani et al version:
$$
z = ReLU(\sum_i w_i x_i)
$$
$$
\theta_t = \gamma \theta_{t-1} + (1-\gamma) \langle z^2 \rangle_{b_t}
$$ 

$$
\frac{dw_i}{dt} = \frac{z (z-  \theta) x_i}{\theta} 
$$


in the code with einstein notation:

$$
\Delta W = \frac{(\frac{z (z-  \theta) }{\theta} )^i_j  x^j_k}{batchsize}
$$

$$
W \leftarrow \epsilon * \Delta W 
$$

for trace:
$$
\langle \theta \rangle = \frac{1}{B} \sum_{t=1}^B \theta_t
$$ 

update theta = use a moving average of previous batch-averaged quadratic postsynaptic activities
where 

$\gamma$ : decay-memory factor 

$\langle \cdot \rangle_{b_t}$ : average over the batch of training patterns considered at the time step t




Choice of activation function $\phi$: 

- requirements are given in the paper

- apperently BCM og uses $\phi = y (y-  \theta)$ https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0044-6 

- function should be non symmetrical:  https://www.researchgate.net/publication/2308452_Effect_of_Binocular_Cortical_Misalignment_on_Ocular_Dominance_and_Orientation_Selectivity  

- "The mathematical framework of the BCM model establishes that postsynaptic activity is given by a linear combination of synaptic weights and inputs, processed by an activation function. No constraints on the form of the activation function are posed: to achieve nontrivial results, the nonlinearity could be imposed, while for the biological interpretation, the positivity is required. Historically, the classical formulation of the model uses a logistic activation function, following the trend proposed by other neuroscience applications"  https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/

- Relu performs well in deep neural network learning so they chose RELU 
https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/

- Activation function in book is a combination of sinus and sigmoid: 
<img src="figs\phi_book.png" width="450"/>
https://neuronaldynamics.epfl.ch/online/Ch19.S2.html 

- in scholarpedia article, its a sigmoid:
http://www.scholarpedia.org/article/BCM_theory


Resources:
- help: https://neuronaldynamics.epfl.ch/online/Ch19.S2.html 
- https://www.researchgate.net/publication/2308452_Effect_of_Binocular_Cortical_Misalignment_on_Ocular_Dominance_and_Orientation_Selectivity 
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/
- https://github.com/Nico-Curti/plasticity 
- https://github.com/Nico-Curti/plasticity/blob/main/plasticity/source/bcm.pyx 
- https://github.com/Nico-Curti/plasticity/blob/main/plasticity/model/bcm.py 

## Metrics

https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/

### Selectivity 

### Competitiveness 

### Memorization


## Stuff to show:

### How threshold adapts 

### weights & feature map


### Selectivity increases with learning 



### Effect of Inhibitory Synapse

show effect of inhib synapses by setting negative values to zero, selectivity should drop