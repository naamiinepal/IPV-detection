#! /usr/bin/bash

echo 'Syncing collected tweets'
rsync -av ~/IPV-Scraper/twarc_scrape/{past_tweets,realtime_tweets,all_realtime_tweets} datasets/raw/
rsync -av datasets/raw/{past_tweets,realtime_tweets,all_realtime_tweets} datasets/raw_words/

echo 'Creating a Prediction dataset'
./create_sent_prediction_dataset.py
./create_word_prediction_dataset.py

echo 'Generating Predictions'
if [[ $# -ge 1 && $1 -ge -1 ]] ; then
    echo "Overridding default gpu to $1"
    args="--gpu $1"
fi

./sent_predict.py --no-use_cache $args
./word_predict.py --no-use_cache $args

echo 'Generating Predictions with Info'
./create_detailed_sent_predictions.py
./get_aspect_spans.py
./create_detailed_word_predictions.py

echo 'Combining word and sent predictions'
./combine_word_sent_predictions.py

echo 'Populating Database'
cd ~/ipv-dashboard/server/
.venv/bin/python -m utils.pull_new_tweets
