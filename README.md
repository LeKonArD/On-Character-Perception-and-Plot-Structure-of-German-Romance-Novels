# On character perception and plot structure of romance novels (Submission to CHR2023)
## Description
This repository is used to reproduce the experiments of our paper. <br>
The structure is as follows: <br>

### /code
plots.ipynb -> code to create the figures used in the paper using the data from data/bigframe.tsv <br>
prepare_data.ipynb -> transform the single novel prediction (data/prediction.zip) into data/bigframe.tsv <br>
Consent.py -> Training routine for consent scene detection <br>
HappyEnd.py -> Training routine for Happy End scene detection<br>
Meeting.py -> Training routine for Meeting scene detection<br>
Attraction.py -> Training routine for attraction detection <br>
Emotions_Perception.py -> Training routine for emotion and perception detection<br>
<br>

### /data:
bigframe.tsv -> all predictions of the complete corpus <br>
prediction.zip -> one file of predictions for each model and novel in the corpus <br>

## Limitations
Since we are working with material under copyright restrictions all text is deleted from the 
data. 


