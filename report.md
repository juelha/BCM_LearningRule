
# BCM Learning Rule


## The Model


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


- help: https://neuronaldynamics.epfl.ch/online/Ch19.S2.html 
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/
- https://github.com/Nico-Curti/plasticity 
- https://github.com/Nico-Curti/plasticity/blob/main/plasticity/source/bcm.pyx 
- https://github.com/Nico-Curti/plasticity/blob/main/plasticity/model/bcm.py 


## Stuff to show:

### How threshold adapts 

### weights & feature map


### Selectivity increases with learning 



### Effect of Inhibitory Synapse

show effect of inhib synapses by setting negative values to zero, selectivity should drop