# MAGIC

> **Note**
> <span style="color:blue"> Pre-processed datasets for **StreamSpot** and **Unicorn Wget** and their corresponding **log parsers** will be released as soon as possible. The current version are not ready for one-click run and is not compatible with `DGL 1.0.0`. Additional information concerning DARPA TC labeling will also be included soon.</span>

This is official code for the USENIX Security 24 paper:

**MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning**

![](./figs/model.pdf)

In this paper, we introduce MAGIC, a novel and flexible self-supervised approach for multi-granularity APT detection. MAGIC leverages masked graph representation learning to model benign system entities and behaviors, performing efficient deep feature extraction and structure abstraction on provenance graphs. By ferreting out anomalous system behaviors via outlier detection methods, MAGIC is able to perform both system entity level and batched log level detection. MAGIC is specially designed to handle concept drift with a model adaption mechanism and successfully applies to universal conditions and detection scenarios.

## Dependencies

* Python 3.8
* PyTorch 1.12.1
* DGL 1.0.0
* Scikit-learn 1.2.2

## Datasets

We use two public datasets for evaluation on *batched log level detection*: `StreamSpot` and `Unicorn Wget`.
We use the DARPA Transparent Computing Engagement 3 sub-datasets `E3-Trace`, `E3-THEIA` and `E3-CADETS` for evaluation on *system entity level detection*.
Due to the enormous size of these datasets, we include our **pre-processed** datasets in the `data/` folder. In each sub-directory under the `.data` folder, there is a `.zip` file. You need to **unzip** these `.zip` files into one `graphs.pkl` for each dataset. 

> Pre-processed datasets for **StreamSpot** and **Unicorn Wget** will be released soon.

To pre-process these datasets from scratch, do as the follows:

- **StreamSpot Dataset**
  - Download and unzip `all.tar.gz` from [StreamSpot](https://github.com/sbustreamspot/sbustreamspot-data), which includes a single data file `all.tsv`.
  - Copy `all.tsv` to `data/streamspot`
  - Run `utils/streamspot_parser.py`. This will result in 600 graph data files in the JSON format. 
    > `utils/streamspot_parser.py` will be released soon.
  - During training and evaluation, function `load_batch_level_dataset` in `utils/loaddata.py` will automatically read and label these graph and store them into the compressed data archive `graphs.pkl` for efficient data loading.
- **Unicorn Wget Dataset**
  > Will be released soon.
- **DARPA TC E3 Sub-datasets**
  - Go to [DAPRA TC Engagement 3 data release](https://github.com/darpa-i2o/Transparent-Computing).
  - Download and unzip `ta1-trace-e3-official-1.json.tar.gz` into `data/trace/`.
  - Download and unzip `ta1-theia-e3-official-6r.json.tar.gz` into `data/theia/`.
  - Download and unzip `ta1-cadets-e3-official-2.json.tar.gz` and `ta1-cadets-e3-official.json.tar.gz` into `data/cadets/`.

Meanwhile, we elaborated an alternative labeling methodology on the DARPA TC datasets in this paper (Appendix G). We also release the corresponding ground truth labels in `data/alternative_labels/` and a `labeling.pdf` file describing this labeling methodology.

> These additionaly materials will be released soon.


## Run

This is a guildline on reproducing MAGIC's evaluations. There are three options: **Quick Evaluation**, **Standard Evaluation** and **Training from Scratch**.

### Quick Evaluation

Make sure you have MAGIC's parameters saved in `checkpoints/` and KNN distances saved in `eval_result/`. Then execute `eval.py` and assign the evaluation dataset using the following command:
```
  python eval.py --dataset *your_dataset*
```
### Standard Evaluation

Standard evaluation trains the detection module from scratch, so the KNN distances saved in `eval_result/` need to be removed. MAGIC's parameters in `checkpoints/` are still needed. Execute `eval.py` with the same command to run standard evaluation:
```
  python eval.py --dataset *your_dataset*
```
### Training from Scratch

Namely, everything, including MAGIC's graph representation module and its detection module, are going to be trained from raw data. Remove model parameters from `checkpoints/` and saved KNN distances from `eval_result/` and execute `train.py` to train the graph representation module. 
```
  python train.py --dataset *your_dataset*
```
Then execute `eval.py` the same as in standard evaluation:
```
  python eval.py --dataset *your_dataset*
```
For more running options, please refer to `utils/config.py`


## Cite 
 
If you make advantage of MAGIC in your research, please cite the following in your manuscript:

> Will be released soon.
