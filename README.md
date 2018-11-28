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
```

Predict:
```
./main.py --action predict --checkpoint_dir [checkpoint directory] [--composer [mz, bach, brahms, debussy]]
```