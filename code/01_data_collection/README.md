# Projections Survey - Experiment

- Setup:

* Install Java 8
* Install Octave 4.x
* Install Anaconda for Python 3.6
* Install python packages:
```
pip install umap-learn wget keras numpy pandas scikit-learn
conda install -c conda-forge shogun

git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git
cd Multicore-TSNE/
pip install .
```

- Sanity check:

```
python test_wrappers.py
python test_projections.py
```

- Getting the datasets (it will take some time. About 20 GB of disk space will be used):

```
python get_datasets.py
```

- Viewing all datasets available:

```
python runner.py
```

- Running the projections:

```
python runner.py [-d dataset name] [-k neighbors] [-o output_dir]
```

- Adding new projections: see projections.py
- Addind new metrics: see metrics.py
