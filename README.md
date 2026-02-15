# BCM Learning Rule

> Implementation of the Bienenstock-Cooper-Munro (BCM) learning rule in PyTorch


### About
- Final Project for the course Neural Information Processing 2025
- Demonstrates how the threshold adapts based on postsynaptic activity history and how weights develop orientation selectivity
- BCM Implementation is based on the [plasticity package](https://github.com/Nico-Curti/plasticity/blob/main/plasticity/model/bcm.py), which I rewrote, translated into PyTorch and to which I added timescales for threshold and weight adaption 
- Check out the full [report](report.md) 



---
### Weights develop orientation selectivity

<div align="center">
<img src="figs\gratings.png" width="450"/>
<figcaption>8 orientations evenly spaced over 180°.  </figcaption>
</a></p></div>

<div align="center">
<img src="figs\curti_config1.png" width="450"/>
<figcaption>Weights for tau_th = 0.1 and
tau_w = 0.05  </figcaption>
</a></p></div>

<div align="center">
<img src="figs\curti_config2.png" width="450"/>
<figcaption>Weights for tau_th = 0.1 and
tau_w = 0.001 </figcaption>
</a></p></div>


---
### Threshold preventing runaway activity

The threshold is successfully regulating the activity. Looking at t=20 and t=25, the spike in activation is picked up by the threshold and the activation in the next step is reduced.  


<div align="center">
<img src="figs\mod_working.png" width="350"/>
<figcaption>The activity and threshold of a single neuron while training. </figcaption>
</a></p></div>


