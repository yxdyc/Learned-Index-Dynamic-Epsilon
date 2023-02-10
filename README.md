# Learned Index with Dynamic $\epsilon$
Based on the theoretically derived prediction error bounds, we propose a 
mathematically-grounded learned index framework with dynamic $\epsilon$
, which is efficient and pluggable to several state-of-the-art learned index methods.
This repository is the implementation of the proposed framework, see more 
details on our [paper](https://openreview.net/pdf?id=VyZRObZ19kt).

>Index structure is a fundamental component in database and facilitates 
broad data retrieval applications. Recent learned index methods show 
superior performance by learning hidden yet useful data distribution with 
the help of machine learning, and provide a guarantee that the prediction 
error is no more than a pre-defined $\epsilon$. However, existing learned index 
methods adopt a fixed $\epsilon$ for all the learned segments, neglecting 
the diverse characteristics of different data localities. In this paper, we 
propose a mathematically-grounded learned index framework with dynamic 
$\epsilon$, which is efficient and pluggable to existing learned index 
methods. We theoretically analyze prediction error bounds that link 
$\epsilon$ with data characteristics for an illustrative learned index 
method. Under the guidance of the derived bounds, we learn how to vary $\epsilon$ 
and improve the index performance with a better space-time trade-off. Experiments 
with real-world datasets and several state-of-the-art methods demonstrate the efficiency, effectiveness and usability of the proposed framework.



# Datasets
We attach binary files of the public experimental datasets in the "data" directory, in which the keys have been de-duplicated and sorted. The keys of *IoT* dataset and *Weblogs* dataset are in the UNIX timestamp format.

# Running Experiments
We provide notebooks to run the experiments reported in our paper.
- `iot_PGM.ipynb`, `latilong_PGM.ipynb`, `lognormal_PGM.ipynb`, 
  `weblogs_PGM.ipynb` are for the experiments of [PGM](http://www.vldb.org/pvldb/vol13/p1162-ferragina.pdf) method. Here the 
  latilong indicates a small version of Map datasets, to use the larger version, please download from [ALEX-git](https://github.com/microsoft/ALEX) and replace the data path.
- the other notebooks adopt the similar naming style as `data_method.ipynb`,
  where the `method` can be `MET`, `RS`, `FT` to indicates the [MET](http://proceedings.mlr.press/v119/ferragina20a/ferragina20a.pdf), 
  [RadixSpline](https://dl.acm.org/doi/pdf/10.1145/3401071.3401659) and 
  [FITing-Tree](https://arxiv.org/pdf/1801.10207.pdf) method respectively.
- *Theorem.ipynb* is for the experiments of the validation of the 
theoretical results.

# Reference
This project adopts the Apache-2.0 License. Welcome to contributions and suggestions.

If you find our work useful for your research or development, please cite the following paper (and the respective papers of the baseline methods used):

```
@inproceedings{
  chen2023learned,
  title={Learned Index with Dynamic \${\textbackslash}epsilon\$},
  author={Daoyuan Chen and Wuchao Li and Yaliang Li and Bolin Ding and Kai   Zeng and Defu Lian and Jingren Zhou},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```
