# libraries:
import os
import numpy as np
import pandas as pd
import torch
import torch.backends
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, Baseline, TemporalFusionTransformer, QuantileLoss, MAE, NaNLabelEncoder

if torch.backends.mps.is_available():
    device = "mps"
    workers = 8
    # enable cpu fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # print environment variables
    print(os.environ)
elif torch.cuda.is_available():
    device = "cuda"
    workers = 12
else:
    device = "cpu"

def preprocess():
    path = "dataset/2022.csv"
    df = pd.read_csv(path, sep=',')
    # remove columns that have only NaN values
    df = df.dropna(axis=1, how='all')

    # convert date column to weekday
    df["Weekday"] = pd.to_datetime(df["Date"]).dt.weekday
    # set the uid to "0" for all rows
    df["uid"] = '0'
    df['index'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])

    # fill nan values with interpolation
    df = df.interpolate(method='linear', limit_direction='forward')

    print(df.head(20))
    return df


def forecast(data, max_encoder_length=30, max_prediction_length=7):
    # TRAINING
    training_cutoff = data["Heart Rate Variability (ms)"].max(
    ) - max_prediction_length
    print("Training cutoff:", training_cutoff, "\n")

    training = TimeSeriesDataSet(
        data[lambda x: x.index <= training_cutoff],
        time_idx="index",
        target="Resting Heart Rate (count/min)",
        group_ids=["uid"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["uid"],
        time_varying_known_reals=["index", "Date"],
        time_varying_unknown_reals=['Heart Rate Variability (ms)', 'Active Energy (kJ)', 'Apple Exercise Time (min)', 'Apple Exercise Time (min)',
                                    'Basal Energy Burned (kJ)', 'Environmental Audio Exposure (dBASPL)', 'Flights Climbed (count)', 'Headphone Audio Exposure (dBASPL)', 'Heart Rate [Min] (count/min)', 'Heart Rate [Max] (count/min)', 'Heart Rate [Avg] (count/min)', 'Heart Rate Variability (ms)', 'Resting Heart Rate (count/min)'],
        target_normalizer=GroupNormalizer(
            groups=["uid"], transformation="softplus"
        ),  # we normalize by group
        categorical_encoders={
            # special encoder for categorical target
            "uid": NaNLabelEncoder(add_nan=True)
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True)

    # create dataloaders
    batch_size = 64
    training_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=12)
    validation_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=12)

    # debug
    """
    x, y = next(iter(training_dataloader))
    print(x['encoder_target'])
    print(x['groups'])
    print('\n')
    print(x['decoder_target'])
    """

    torch.set_float32_matmul_precision('medium')  # todo: set to 'high'
    actuals = torch.cat(
        [y for x, (y, weight) in iter(validation_dataloader)]).to(device=device)
    baseline_predictions = Baseline().predict(validation_dataloader)
    (actuals - baseline_predictions).abs().mean().item()

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=2, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=160,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=160,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4)

    trainer.fit(
        tft,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    print("Best model path:", best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # PREDICTION
    # evaluate on training data

    predictions = best_tft.predict(
        validation_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
    MAE()(predictions.output, predictions.y)

    raw_predictions = best_tft.predict(
        validation_dataloader, mode="raw", return_x=True)
    print("Raw Prediction Fields:")
    print(raw_predictions._fields)
    print('\n')
    print(raw_predictions.output.prediction.shape)

    # list of all the unique uids
    uids = data["uid"].unique()

    for uid in uids:
        fig, ax = plt.subplots(figsize=(10, 5))

        raw_prediction = best_tft.predict(
            training.filter(lambda x: (x.uid == uid)),
            mode="raw",
            return_x=True,
        )
        best_tft.plot_prediction(
            raw_prediction.x, raw_prediction.output, idx=0, ax=ax)
        # set the title to the uid
        ax.set_title(uid)
        # save the figure to /results
        fig.savefig(f"results/prediction_{uid}.png")


def visualize(data):
    # use matplotlib to visualize the data, different stocks have different colors
    plt.figure(figsize=(15, 8))
    for uid in data["uid"].unique():
        # x axis is time_idx, y axis is rtn
        plt.plot(
            data[data["uid"] == uid]["Heart Rate [Avg] (count/min)"],
            label=uid,
        )
        plt.plot(
            data[data["uid"] == uid]["Heart Rate [Max] (count/min)"],
            label=uid,
        )
        plt.plot(
            data[data["uid"] == uid]["Heart Rate [Min] (count/min)"],
            label=uid,
        )
        # add legend
        plt.legend()
    # break # plot only the first chart
    plt.title("Returns {uid}")
    # save the figure to /results
    plt.savefig('visual.png')


if __name__ == "__main__":
    data = preprocess()
    # visualize(data)
    forecast(data)
