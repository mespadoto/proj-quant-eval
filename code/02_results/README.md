# Projections Survey - Results

- Concatenation of results files

```
bash 01_concat_results.sh <OUTPUT_FOLDER>
```

where OUTPUT_FOLDER is the folder containing *pq*.csv files from the experiment.

- Data consolidation

```
python 02_consolidate.py
```

- Generating heatmaps:

```
python heatmap2.py
```

- Generating Shepard diagrams

```
python shepard.py
```

- Running the time evaluation:

```
python time_eval.py
```

- Generating time evaluation plots:

```
python time_eval_plot.py
```
