# transmaf

**описание проведенных экспериментов в `./problem_statement.md`*

transmaf это библиотека на основе [PyTorch](https://github.com/pytorch/pytorch) для вероятностного прогнозирования временных рядов с использованием модела Trans-MAF. В качестве бэк-энд API используется [GluonTS](https://github.com/awslabs/gluon-ts) для загрузки, трансформации и бэк-теста датасетов.

## Installation

```
$ pip install transmaf
```

## Quick start
### Imports
```python
import numpy as np
import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from transmaf import Trainer
from transmaf.model.transformer_tempflow import TransformerTempFlowEstimator
```
### Read data
```python
electricity = get_dataset("electricity_nips", regenerate=False)

# create train/test groupers
electricity_train_grouper = MultivariateGrouper(
    max_target_dim=min(2000, int(electricity.metadata.feat_static_cat[0].cardinality))
    )
electricity_test_grouper = MultivariateGrouper(
    num_test_dates=int(len(electricity.test) / len(electricity.train)), 
    max_target_dim=min(2000, int(electricity.metadata.feat_static_cat[0].cardinality))
    )

# create train/test datasets
electricity_dataset_train = list(electricity_train_grouper(electricity.train))
electricity_dataset_train *= 100 
electricity_dataset_test = electricity_test_grouper(electricity.test)
```
### Train estimator
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


estimator = TransformerTempFlowEstimator(
    input_size=744,
    target_dim=int(electricity.metadata.feat_static_cat[0].cardinality),
    prediction_length=electricity.metadata.prediction_length,
    context_length=electricity.metadata.prediction_length * 4,
    flow_type='MAF',
    dequantize=True,
    freq=electricity.metadata.freq,
    trainer=Trainer(
        device='cpu',
        epochs=14,
        learning_rate=1e-3,
        num_batches_per_epoch=100,
        batch_size=64,
    )
)

predictor = estimator.train(
    electricity_dataset_train, 
    num_workers=4
    )
```
### Prediction
```python
# init evaluator
evaluator = MultivariateEvaluator(
    quantiles=(np.arange(20)/20.0)[1:],
    target_agg_funcs={'sum': np.sum}
)

# prediction
forecast_it, ts_it = make_evaluation_predictions(
    dataset=electricity_dataset_test,
    predictor=predictor,
    num_samples=20
)

forecasts = list(forecast_it)
targets = list(ts_it)
```
### Calculate metrics
```python
agg_metric, _ = evaluator(
    targets, forecasts, num_series=len(electricity_dataset_test)
    )
```



