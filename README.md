# Numerai Signals Pipeline
Downloads data from Yahoo Finance, generates features, trains a model and submits the predictions.

### Running the pipeline

```sh
python launcher.py <your_properties_file>.json
```
The json properties file must contain 3 keys:
```sh
{
    "model_id": "xyz", 
    "public_id": "xyz",
    "secret_key": "xyz"
}
```
Corresponding to:
- **model_id**: Numerai ID of the model we want to submit predictions to.
- **public_id**: Our public Numerai key.
- **secret_key**: Our private Numerai key.

***Make sure your properties file is added to the .gitignore as contains sensitive data.***

### Output data
Once the pipeline finishes, there will be 3 folders with data files:
- **db_raw_downloaded**: Contains data downloaded from our source (currently Yahoo Finance). We keep it as there is an option to run the pipeline with already downloaded data. 
- **db_ml_csv**: Contains data to train, validate and predict. We can use this file to improve training or try other models outside this pipeline.
- **db_predictions**: Contains a file that will be submitted automatically to the indicated model. We can also manually upload it as a diagnostics file.

If we want to remove the data from these 3 folders, we will have to do it manually.

### Targets
We are creating our custom target but the pipeline uses the one provided by Numerai.

### Future lines
Some ideas thay can be integrated in the pipeline or implemented using data generated by it: 
- Mean Decrease Accuracy for feature selection.
- Era wise Time Series Purged Cross Validation. 
- Z-Score with some king of regularization instead of binning.
- Denoising techniques.
- Try paid sources of data to generate features based on fundamentals or improve the quality of the ones based on price and volume.