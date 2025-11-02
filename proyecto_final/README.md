USAGE

To train the model:

python train.py --config configs/<rf/xgb>.yaml

Adjust .yaml files to enable/disable HPO or adjust hyperparameters.

To predict using a model:

python predict.py \
  --model-uri models:/energy-efficiency-regressor/BestModel-Cooling \
  --csv ../data/processed/newdata.csv \
  --out predictions.json