# Gas Sensor Calibration Model

Predicts what a certified carbon monoxide (CO) analyser would read, using signals from five low-cost gas sensors plus temperature and humidity. The idea came from the live air-quality screens on a Delta Electronics manufacturing line: cheap sensors could cover more of a factory floor if a model can turn their readings into analyser-grade estimates.

## How it works

1. **Data**: the UCI Air Quality dataset, 9,357 hourly records from sensors placed outdoors in an Italian city over a year. Each record has five sensor readings (PT08.S1 to PT08.S5), temperature, relative and absolute humidity, and certified analyser readings for several gases.
2. **Cleaning**: the dataset marks missing values as -200, so these become NaN. The analyser columns for the other gases are dropped, since they are not sensor inputs. NMHC(GT) was about 90% missing anyway (8,443 of 9,357). Rows with any gaps left are removed, leaving 7,344.
3. **Split**: 60% training, 20% validation and 20% test, in time order without shuffling, because the readings are a time series.
4. **Models**: Linear Regression, K-Nearest Neighbours, Decision Tree and Random Forest regressors, each scored on validation RMSE, MAE and R². KNN gets inputs scaled with StandardScaler, fitted on the training data only.
5. **Tuning**: Ridge Regression (Linear Regression with regularisation) over alpha values from 0.001 to 1000, KNN over odd k from 1 to 99, the Decision Tree over depths 1 to 20, and the Random Forest over tree counts and depths. Each is compared on validation RMSE.
6. **Final test**: the best model is retrained on the training and validation data together, then scored once on the test set.

## Results

### The first target looked too good

The first target was benzene, C6H6(GT), the dataset's original research target. It had a 0.98 correlation with the PT08.S2 sensor, and a decision tree scored an R² that rounded to 1.000.

A scatter plot showed benzene following an almost exact curve of PT08.S2, with nearly no scatter. That looks like target leakage, where the answer is already hidden in the inputs. The cause could not be confirmed from the dataset information, so the target was switched to CO instead of reporting a misleading score.

### CO models

| Model | Best settings | Validation RMSE |
|---|---|---:|
| Linear Regression | No tuning | 0.794 |
| Ridge Regression | alpha = 10 | 0.792 |
| K-Nearest Neighbours | k = 5 | 0.845 |
| Decision Tree | Max depth 6 | 0.759 |
| Random Forest | 300 trees, max depth 20 | 0.716 |

Random Forest won. On the test set it scored an RMSE of 0.535 mg/m³, an MAE of 0.366 mg/m³ and an R² of 0.830.

At higher concentrations it tends to underestimate, which is the dangerous direction for a CO alarm. The training data is also hourly outdoor readings, not factory data. So the notebook proposes it as cheaper, wider indicative monitoring alongside certified analysers, not a replacement for them.

## Why it reads like a report

The brief asked for a technical report in a Jupyter notebook, built around a problem inspired by an industry visit, with several models compared and tuned. That is why the code sits between sections such as the problem statement, data assumptions, limitations and references.

## Running it

```
pip install -r requirements.txt
```

Open `gas_sensor_model.ipynb` in Jupyter and run all cells. The first code cell downloads the dataset from UCI with `ucimlrepo`, so it needs an internet connection. No data file is included.

## Dataset

Air Quality by S. De Vito, UCI Machine Learning Repository, dataset 360: https://archive.ics.uci.edu/dataset/360/air+quality (DOI https://doi.org/10.24432/C59K5F). Licensed under CC BY 4.0.
