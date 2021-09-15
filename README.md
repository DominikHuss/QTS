# Greedy (argmax) sequence generation
<p align="center"> 
          <img src="https://github.com/DominikHuss/QTS/blob/main/plots/train.png" width="45%" height="30%"> 
          <img src="https://github.com/DominikHuss/QTS/blob/main/plots/eval.png" width="45%" height="30%"> 
</p>

# Random samples
<p align="center"> 
          <img src="https://github.com/DominikHuss/QTS/blob/main/plots/random_train.png" width="45%" height="30%"> 
          <img src="https://github.com/DominikHuss/QTS/blob/main/plots/random_eval.png" width="45%" height="30%"> 
</p>

# \# TODO list
Not in any particular order.

- [ ] add logging, and everything that might concern it  
- [ ] add synthetic data generation
- [ ] add model checkpointing, therefore saving and loading  
- [ ] add support for cuda  
- [x] add soft cross-entropy objective AND random +- class data augmentation  
  - [ ] make it parametizable
- [x] add plotting  
- [ ] add partial mlm support for unseen data; proper masking and all  
- [ ] think about distinguishing between different sample frequencies (e.g. daily vs hourly)
- [ ] think about generalizing to different data scaling  
