# Unknown fault incremental learning based on shapelet prototypical network for streaming industrial signals
#### This is the source code for paper "Unknown fault incremental learning based on shapelet prototypical network for streaming industrial signals". The paper is accepted in journal named Engineering Applications of Artificial Intelligence.

> @article{WAN2025112094,
> title = {Unknown fault incremental learning based on shapelet prototypical network for streaming industrial signals},
> journal = {Engineering Applications of Artificial Intelligence},
> volume = {161},
> pages = {112094},
> year = {2025},
> issn = {0952-1976},
> doi = {https://doi.org/10.1016/j.engappai.2025.112094},
> url = {https://www.sciencedirect.com/science/article/pii/S0952197625021025},
> author = {Xiaoxue Wan and Lihui Cen and Xiaofang Chen and Yongfang Xie and Zhaohui Zeng},
> keywords = {Fault diagnosis, Unknown fault, Meta learning, Shapelet, Prototype}}

---

## 《Unknown fault incremental learning based on shapelet prototypical network for streaming industrial signals》

#### Abstract:Unknown faults represent faults that have never occurred before, they are constantly emerging due to the changing environments and operations in industrial processes. It is a challenge for existing fault diagnosis methods to continually detect unknown faults and effectively classify known faults in streaming industrial signals. This article proposes an unknown fault incremental learning method for streaming industrial signals. In this work, shapelet prototypical embedding combined with a memory distance matrix is employed to embed streaming industrial signals into a  discriminative feature space. Therefore, the category information in the signals can be extracted and is not limited by the size of the sliding window. Besides, a new training paradigm based on meta-learning by sampling simulated-incremental tasks is proposed to obtain generalizable shapelets. Moreover, based on the new training paradigm, the meta-discovery module is proposed to continually detect unknown faults, and the meta-calibrate module can calibrate all prototypes into a distinguishable space. Experiments on the simulated streaming time series, benchmark Tennessee Eastman process, and real-world aluminum electrolysis process illustrate the superiority of the proposed method in terms of accuracy and interpretability.

Required packages:
* python == 3.6
* pytorch == 1.10.2
* matplotlib
* numpy  

