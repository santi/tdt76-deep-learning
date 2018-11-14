## [Overleaf Report](https://www.overleaf.com/16606540djgddyxgdfyd)


# Data format
## Naming
There are two training sets available:

- `train_ROC.csv`
- `test_2016`

We created false endings for the `train_ROC` set using a script (© Hlynur, 2018).

The sets used for training after being preprocessed and split into the correct format are named:

- `train_ROC_split.csv`
- `test_2016_split.csv`

A separate validation set is used for both training sets:

- `valid_TA.csv`

It is preprocessed aswell, to match the format of the training sets:

- `valid_TA_split.csv`

## Format
After preprocessing, all datasets are stored as .csv files, using commas `,` as seperators. The columns are named `InputSentence1	InputSentence2	InputSentence3	InputSentence4	InputSentence5	AnswerRightEnding
`

The first five columns each contain a sentence (string), and the last column contains an integer (0 or 1), indicating if the fifth sentence is the true ending of the story (1) or a false, generated ending (0).



# Setup on local computer

Clone repository from Github, or download source from somewhere. Make sure you are using Python 3.6.x as your Python interpreter.

```bash
cd nlu-projects/project2/
```

Install requirements from requirements.txt

```bash
pip install -r requirements.txt
```

NOTE: The default Tensorflow version is `tensorflow-gpu`. If your machine does not have a GPU, use the `tensorflow` package instead.

Download datasets from [data source](https://drive.google.com/open?id=1grdTUYEWH-2i9tinfaHDieNs5cKin7vh), unzip and move them into the correct folder:
```bash
mkdir data/
# unzip downloaded data first
cp -r /path/to/downloaded/data/data/* data/
```
NOTE: Do NOT move the train_embeddings.pkl file into data/ folder. 


Download the Skip Thoughts model from Tensorflow:
```bash
wget "http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz"
tar -xvf skip_thoughts_uni_2017_02_02.tar.gz
rm skip_thoughts_uni_2017_02_02.tar.gz
```


Move files to correct folders:
```bash
mkdir data/skipthoughts
mv skip_thoughts_uni_2017_02_02 data/skipthoughts/
mv train_encodings.pkl data/skipthoughts/
```

Download embeddings and data used for Skip Thought model:
```bash
mv /path/to/downloaded/data/train_encodings.pkl data/skipthoughts/
printf "import nltk\nnltk.download('punkt')" | python
```

Download saved checkpoints from [Google Drive](https://drive.google.com/open?id=1gsuZDXMkX2HUKS6L_Gn6CLcJCgFFgcn_):

Put the folders inside the `checkpoints/` folder in the `project2/checkpoints/` folder.




# Setup on Leonhard

`ssh` into Leonhard using your ETH credentials.

Clone repository:
```bash
git clone git@github.com:hlynurf/nlu-projects.git nlu-projects
```

Create data directory:
```bash
mkdir nlu-projects/project2/data
```

Transfer data into `nlu-projects/project2/data/`:
```
scp data/* <USERNAME>@login.leonhard.ethz.ch:/cluster/home/<USERNAME>/nlu-projects/project2/data/
```

Comment out the line "tensorflow-gpu" in requirements.txt

Install dependencies:
```bash
module load python_gpu/3.6.4
pip install --user -r requirements.txt
```


Submit jobs:
```
bsub -n 8 -W 01:00 -R "rusage[mem=3000,ngpus_excl_p=1]" <COMMAND>
```


# Running the model

The default action for the model is to start training with the given datasets.

To run the model:
```bash
python main.py
```

To generate predictions:
Place `test_nlu18_utf-18.csv` into the `data/` folder and run:
```
python split_data.py
python main.py --checkpoint_dir checkpoints/skip_2016/ --log_dir log/skip_2016/ --action predict --training_data data/test_2016_split.csv --prediction_data data/test_nlu18_split.csv --batch_size 2
```

NOTE: The model generates encodings for the training set, which might take 30+ minutes.