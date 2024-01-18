# libraries:
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import xml.etree.ElementTree as ET


def preprocess(file):
    if file.endswith('.csv'):
        df = pd.read_csv(file, sep=',')
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
    elif file.endswith('.xml'):
        # create element tree object
        tree = ET.parse(file)
        print('Reading ' + file + '...')
        # for every health record, extract the attributes
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]

        health_records = pd.DataFrame(record_list)

        # proper type to dates
        for col in ['creationDate', 'startDate', 'endDate']:
            health_records[col] = pd.to_datetime(health_records[col])

        # handle SleepAnalysis records
        if 'HKCategoryTypeIdentifierSleepAnalysis' in health_records['type'].unique():
            sleep_analysis_records = health_records[health_records['type']
                                                    == 'HKCategoryTypeIdentifierSleepAnalysis']
            sleep_analysis_records['value'] = sleep_analysis_records['value'].map({
                'HKCategoryValueSleepAnalysisInBed': 0,
                'HKCategoryValueSleepAnalysisAwake': 1,
                'HKCategoryValueSleepAnalysisAsleepCore': 2,
                'HKCategoryValueSleepAnalysisAsleepREM': 3
            })
            health_records.update(sleep_analysis_records)

        # value is numeric, NaN if fails
        health_records['value'] = pd.to_numeric(
            health_records['value'], errors='coerce')
        # some records do not measure anything, just count occurences
        # filling with 1.0 (= one time) makes it easier to aggregate
        health_records['value'] = health_records['value'].fillna(1.0)

        # shorter observation names
        health_records['type'] = health_records['type'].str.replace(
            'HKQuantityTypeIdentifier', '')
        health_records['type'] = health_records['type'].str.replace(
            'HKCategoryTypeIdentifier', '')
        health_records.tail()

        # pivot the dataframe so that each type is a column the and each row is a date
        # since some types have multiple records per day, we interpolate the others
        df = health_records.pivot_table(
            index='creationDate',
            columns=['type'],
            values='value',

        ).interpolate(method='time', limit_direction='both')

        # filter out all columns that have the word "dietary" in them
        df = df[df.columns.drop(list(df.filter(regex='Dietary')))]
        # save the file as it takes a while to process
        df.to_pickle('dataset/export.pkl')
    elif file.endswith('.pkl'):
        df = pd.read_pickle(file)

    else:
        df = None
        print("File type not supported")
    return df


def forecast(data, max_encoder_length=30, max_prediction_length=7):
    # TRAINING
    training_cutoff = data["Heart Rate Variability (ms)"].max(
    ) - max_prediction_length
    print("Training cutoff:", training_cutoff, "\n")

    training = TimeSeriesDataSet(
        data[lambda x: x.index <= training_cutoff],
        time_idx="index",
        target="Heart Rate Variability (ms)",
        group_ids=["uid"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["uid"],
        time_varying_known_reals=["index", "Date"],
        time_varying_unknown_reals=[
            'Heart Rate [Avg] (count/min)', 'Body Temperature (degC)', 'Environmental Audio Exposure (dBASPL)'],
        target_normalizer=GroupNormalizer(
            groups=["uid"], transformation="count"
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
        max_epochs=10,
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


def visualize(data, features, start_date='2021-01-01', end_date='2023-12-31'):
    """Plot averaged heart rate over the course of a day for the given date range.
    """
    # filter data for the given date range
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # for each feature, calculate the average over the course of a day
    for feature in features:
        # group by the time of day and calculate the mean
        data_grouped = data[feature].groupby(data.index.time).mean()
        # convert the index to "hh:mm" format
        data_grouped.index = data_grouped.index.map(
            lambda t: f"{t.hour:02d}:{t.minute:02d}")
        # create rolling average over 5 minutes
        data_grouped = data_grouped.rolling(30).mean()

        # create a new plotly graph object
        fig = go.Figure()

        # add the data to the plot
        fig.add_trace(go.Scatter(x=data_grouped.index,
                      y=data_grouped.values, mode='lines', name=feature))

        # set the title and labels
        fig.update_layout(
            title=f'Average {feature} over the course of a day',
            xaxis_title='Time',
            yaxis_title=feature,
            template='plotly_dark'  # use the plotly_dark theme
        )

        # set the x-axis to the hours in "hh:mm" format
        fig.update_xaxes(tickvals=[f"{i:02d}:00" for i in range(24)])

        # show the plot
        fig.show()


def correlation(data):
    # create a correlation matrix from the "value" column
    corr = data.corr()

    # create a heatmap from the correlation matrix
    heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        colorscale='Viridis',
        showscale=True,
        reversescale=True,
    )
    # show the plot
    heatmap.update_layout(
        plot_bgcolor='rgb(10,10,10)',  # dark background
        paper_bgcolor='rgb(10,10,10)',  # dark background
        font_color='white',  # white text
        template='plotly_dark'  # use the plotly_dark theme
    )
    heatmap.show()


def export_to_csv(data):
    data.to_csv('dataset/export.csv', index=False)


def get_features(data):
    print('-'*50)
    print('Available features:')
    print(data.columns)
    print('-'*50)
    print(data["SleepAnalysis"].describe())
    print('-'*50)
    # print last 10 rows of SleepAnalysis column
    print(data["SleepAnalysis"].tail(10))


if __name__ == "__main__":
    data = preprocess('dataset/export.pkl')
    get_features(data)
    # export_to_csv(data)
    visualize(data, ['SleepAnalysis'],
              start_date='2023-01-01', end_date='2023-12-31')
    # correlation(data)
    # forecast(data)
    # correlation(data)
