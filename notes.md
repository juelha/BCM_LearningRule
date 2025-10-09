
### Original
$$
\bar{c}(t) = m(t) \cdot \bar{d}
$$
$$
\theta = (\bar{c}/c_0)^p \bar{c}
$$

$$
\dot{m}_j (t) = \phi(c(t))d_j(t) - \epsilon m_j(t)
$$
 

### Original with today's conventions 
$$
y = \sum_i w_i x_i
$$
$$
\theta = E^p [(y/y_o)]
$$
$$
\frac{dw_i}{dt} = \phi(y) x_i - \epsilon w_i
$$

### Law and Cooper 
$$
y = \sigma (\sum_i w_i x_i)
$$
$$
\theta = E [y^2]
$$
$$
\frac{dw_i}{dt} = \frac{y (y-  \theta) x_i}{\theta}
$$

### Squadrani et al version
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

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>


How to implement




####  Modification function $\phi$


It is a bit unclear which modification function was used in the original paper.
While some sources state that it was $\phi = y (y-  \theta)$ [https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0044-6 , scholarpedia], I could not confirm this.
Additionally, the modification function is also often displayed as a combination of a sinus and a sigmoid (Intrator & Cooper, 1992): 
<div align="center">
<img src="figs\phi_book.png" width="350"/>
<figcaption>Modification function according to the book Neuronal Dynamics by Gerstner et al. (2014) </figcaption>
</a></p></div>

explanation in the paper: "The response of favored patterns grows until the mean response is high enough and the state stabilizes" (p. 36) 


####  Sliding Modification Threshold $\theta$



Instead of a temporal average, using the spatial average has become popular https://www.sciencedirect.com/science/article/pii/S0893608005800036

Aside from that calculating the treshold with a moving average
$$
\theta_t = \gamma \theta_{t-1} + (1-\gamma) \langle z^2 \rangle_{b_t}
$$ 
(Squadrani et al., 2022)

and a first order low-pass filter  (Udeigwe et al., 2017) have been introduced

$$
\tau_\theta \frac{d\theta}{dt} = (v^2 - \theta) 
$$ 



#### Activation Function 

The original model does not provide an activation function for the postsynaptic activtiy. 

Research has showed that a good activation function should be:
- nonlinear "ot achieve nontrivial results" https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/
- positive for biological interpretation https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/
 
- function should be non symmetrical:  https://www.researchgate.net/publication/2308452_Effect_of_Binocular_Cortical_Misalignment_on_Ocular_Dominance_and_Orientation_Selectivity  


Often the sigmoid function is used as an activation https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/  Law and Cooper  http://www.scholarpedia.org/article/BCM_theory
 
- Relu performs well in deep neural network learning so they chose RELU  (Squadrani et al., 2022)


Resources:
- http://www.scholarpedia.org/article/BCM_theory
- https://neuronaldynamics.epfl.ch/online/Ch19.S2.html 
- https://www.researchgate.net/publication/2308452_Effect_of_Binocular_Cortical_Misalignment_on_Ocular_Dominance_and_Orientation_Selectivity 
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9141587/
- https://github.com/Nico-Curti/plasticity 
- https://github.com/Nico-Curti/plasticity/blob/main/plasticity/source/bcm.pyx 
- https://github.com/Nico-Curti/plasticity/blob/main/plasticity/model/bcm.py 