# tdt76-deep-learning
Repository for the course TDT67 - Deep Learning at NTNU


## Actions
Train network:

```bash
./main.py 
```

Train composer:
```bash
./main.py --action train_composer --composer [mz, bach, brahms, debussy] --training_data [data/training/{composer}_fs1/]

Mozart:
./main.py --action train_composer --training_data data/training/mz_fs1/ --composer mz 
./main.py --action train_composer --training_data data/training/bach_fs1/ --composer bach
./main.py --action train_composer --training_data data/training/brahms_fs1/ --composer brahms
```



Predict:
```
./main.py --action predict --checkpoint_dir [checkpoint directory] [--composer [mz, bach, brahms, debussy]]

./main.py --action predict --checkpoint_dir checkpoints_sigmoid/ --composer brahms
```