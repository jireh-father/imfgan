#/bin/sh
nohup python -u main.py \
--poster_dir=data/poster \
--bg_dir=data/bg \
--title_dir=data/title \
--credit_dir=data/credit \
--episodes=1000000 \
--log_dir=logs &