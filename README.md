# TBM-missing-data-handling

Temporal Belief Memory: Imputing Missing Data during RNN Training

Jin Kim, Yeo & Chi, Min. (2018). Temporal Belief Memory: Imputing Missing Data during RNN Training. 2326-2332. 
10.24963/ijcai.2018/322. 

* Abstract
We propose a bio-inspired approach named Temporal Belief Memory (TBM) for handling missing data with recurrent neural networks (RNNs). When modeling irregularly observed temporal sequences, conventional RNNs generally ignore the real-time intervals between consecutive observations. TBM is a missing value imputation method that considers the time continuity and captures latent missing patterns based on irregular real time intervals of the inputs. We evaluate our TBM approach with real-world electronic health records (EHRs) consisting of 52,919 visits and 4,224,567 events on a task of early prediction of septic shock. We compare TBM against multiple baselines including both domain experts' rules and the state-of-the-art missing data handling approach using both RNN and long-short term memory. The experimental results show that TBM outperforms all the competitive baseline approaches for the septic shock early prediction task.

