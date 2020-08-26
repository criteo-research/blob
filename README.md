# BLOB: a probabilistic model for recommendation that combines organic and bandit signals

Source code for our paper "BLOB: a probabilistic model for recommendation that combines organic and bandit signals" published at KDD 2020.

## Results Reproducibility
To create a virtual Python environment that holds all the packages the code relies on, run:

    virtualenv -p python3.6 blob
    source blob/bin/activate
    pip install -r requirements.txt
    
Now, you can run the ''simulate_abtest_with_bandit.py'' file to run both experiments that generate the results in Table 3 and 4 :

Table 3 results:

    python simulate_abtest_with_bandit.py --K 20 --P 100 --num_sessions 1000 --num_sessions_organic 20000 --organic_epochs 1000 --bandit_epochs 800 --num_users_to_score 4000

Table 4 results:

    python simulate_abtest_with_bandit.py --K 20 --P 1000 --num_sessions 1000 --num_sessions_organic 20000 --organic_epochs 1000 --bandit_epochs 1200 --num_users_to_score 4000

## Acknowledgements
We're grateful to Hao-Jun Michael Shi (Northwestern University) and Dheevatsa Mudigere (Facebook) for their implementation of L-BFGS in PyTorch: https://github.com/hjmshi/PyTorch-LBFGS and to Dawen Liang (Netflix) for his implementation of the multi-vae model for collaborative filtering : https://github.com/dawenl/vae_cf. 

## Paper
If you use our code in your research, please remember to cite our paper:

    @inproceedings{SakhiKDD2020,
      author = {Sakhi, Otmane and Bonner, Stephen and Rohde, David and Vasile, Flavian},
      title = {BLOB: A Probabilistic Model for Recommendation That Combines Organic and Bandit Signals},
      year = {2020},
      publisher = {Association for Computing Machinery},
      url = {https://doi.org/10.1145/3394486.3403121},
      doi = {10.1145/3394486.3403121},
      keywords = {Bayesian inference, latent variable models, recommender systems},
      series = {KDD '20}
      }
    
    
