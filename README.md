# AI611_FINAL
KAIST AI611 final project

The video of the best performance model.
![image](https://user-images.githubusercontent.com/45076203/122025051-faa0d480-ce03-11eb-87ef-96db2ff1a9ee.gif)

- Train

To train, use the command `python train.py NCPU`, where `NCPU` is the number
of the cpu you will use for grid search.

- Results for discussion

To get the result and analyze, use the command `python analyze.py`.
This will generate result file `result.txt` and reward plots as `pdf` format.

- Simulate

To simulate the best model, use the command `python simulate.py BEST`, where
`BEST` is the name of the best model.
