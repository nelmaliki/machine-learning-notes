# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "tensorflow==2.19.*",
#   "tensorflow-metal==1.2.*; platform_system == 'Darwin' and platform_machine == 'arm64'",
#   "keras-nlp==0.21.1",
#   "matplotlib==3.10.5",
#   "pandas==2.3.1",
#   "seaborn==0.13.2",
#   # Tip: let TF pull a compatible numpy; don't pin unless you must.
#   "openai==1.99.1",
#   "ruff==0.12.8",
#   "pyarrow==21.0.0",
#   "anthropic==0.62.0",
# ]
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <center><img src="https://keras.io/img/logo-small.png" alt="Keras logo" width="100"><br/>
    This starter notebook is provided by the Keras team.</center>

    ## Keras NLP starter guide here: https://keras.io/guides/keras_nlp/getting_started/

    __This starter notebook uses the [BERT](https://arxiv.org/abs/1810.04805) pretrained model from KerasNLP.__

    **BERT** stands for **Bidirectional Encoder Representations from Transformers**. BERT and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models.

    The BERT family of models uses the **Transformer encoder architecture** to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.

    BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.


    ![BERT Architecture](http://miro.medium.com/v2/resize:fit:1032/0*x3vhaoJdGndvZqmL.png)


    This notebook contains complete code to fine-tune BERT to perform a **Natural Language Inferencing (NLI)** model. NLI is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related.

    Our **NLI model** will assign labels of 0, 1, or 2 (corresponding to **entailment, neutral, and contradiction**) to pairs of premises and hypotheses.

    Note that the train and test set include text in **fifteen different languages**!


    In this notebook, you will:

    - Load the Contradictory, My Dear Watson dataset
    - Explore the dataset
    - Preprocess the data
    - Load a BERT model from Keras NLP
    - Train your own model, fine-tuning BERT as part of that
    - Generate the submission file
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install -q keras-nlp --upgrade
    # (use marimo's built-in package management features instead) !pip install seaborn
    return


@app.cell
def _():
    import os

    import keras_nlp
    import matplotlib.pyplot as plt
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    import seaborn as sns
    import tensorflow as tf
    from tensorflow import keras

    print("TensorFlow version:", tf.__version__)
    print("KerasNLP version:", keras_nlp.__version__)
    return keras, keras_nlp, np, os, pd, plt, sns, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Accelerator

    Detect hardware, return appropriate distribution strategy
    """
    )
    return


@app.cell
def _(tf):
    print(tf.config.list_physical_devices("GPU"))
    try:
        # detect and init the TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        # instantiate a distribution strategy
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        print("TPU not activated")
        strategy = (
            tf.distribute.MirroredStrategy()
        )  # Works on CPU, single GPU and multiple GPUs in a single VM.

    print("replicas:", strategy.num_replicas_in_sync)
    return (strategy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Load the Contradictory, My Dear Watson dataset
    Let's have a look at all the data files

    The training set contains a premise, a hypothesis, a label (0 = entailment, 1 = neutral, 2 = contradiction), and the language of the text. For more information about what these mean and how the data is structured, check out the data page: https://www.kaggle.com/c/contradictory-my-dear-watson/data
    """
    )
    return


@app.cell
def _(os):
    DATA_DIR = "data/"

    RESULT_DICT = {0: "entailment", 1: "neutral", 2: "contradiction"}

    for dirname, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    return DATA_DIR, RESULT_DICT


@app.cell
def _(DATA_DIR, pd):
    df_train = pd.read_csv(DATA_DIR + "train.csv")
    df_train.head()
    return (df_train,)


@app.cell
def _(DATA_DIR, pd):
    df_test = pd.read_csv(DATA_DIR + "test.csv")
    df_test.head()
    return (df_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's look at some pairs of sentences.""")
    return


@app.cell
def _(df_train):
    def display_pair_of_sentence(x):
        print("Premise : " + x["premise"])
        print("Hypothesis: " + x["hypothesis"])
        print("Language: " + x["language"])
        print("Label: " + str(x["label"]))
        print()

    df_train.head(10).apply(lambda x: display_pair_of_sentence(x), axis=1)

    df_train.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Explore the dataset

    Let's look at the distribution of labels in the training set.
    """
    )
    return


@app.cell
def _(RESULT_DICT, df_train, plt, sns):
    (_f, _ax) = plt.subplots(figsize=(12, 4))
    sns.set_color_codes("pastel")
    sns.despine()
    _ax = sns.countplot(
        data=df_train, y="label", order=df_train["label"].value_counts().index
    )
    _abs_values = df_train["label"].value_counts(ascending=False)
    _rel_values = (
        df_train["label"].value_counts(ascending=False, normalize=True).values
        * 100
    )
    _lbls = [f"{p[0]} ({p[1]:.0f}%)" for p in zip(_abs_values, _rel_values)]
    _ax.bar_label(container=_ax.containers[0], labels=_lbls)
    _ax.set_yticklabels([RESULT_DICT[index] for index in _abs_values.index])
    _ax.set_title("Distribution of labels in the training set")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's look at the distribution of languages in the training set.""")
    return


@app.cell
def _(df_train, plt, sns):
    (_f, _ax) = plt.subplots(figsize=(10, 10))
    sns.set_color_codes("pastel")
    sns.despine()
    _ax = sns.countplot(
        data=df_train,
        y="language",
        order=df_train["language"].value_counts().index,
    )
    _abs_values = df_train["language"].value_counts(ascending=False)
    _rel_values = (
        df_train["language"]
        .value_counts(ascending=False, normalize=True)
        .values
        * 100
    )
    _lbls = [f"{p[0]} ({p[1]:.0f}%)" for p in zip(_abs_values, _rel_values)]
    _ax.bar_label(container=_ax.containers[0], labels=_lbls)
    _ax.set_title("Distribution of languages in the training set")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's look at the length of the sentences""")
    return


@app.cell
def _(df_train):
    df_train["premise_length"] = df_train["premise"].apply(lambda x: len(x))
    df_train["hypothesis_length"] = df_train["hypothesis"].apply(
        lambda x: len(x)
    )
    df_train[["hypothesis_length", "premise_length"]].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Preprocess the data

    Text inputs need to be transformed to numeric token ids and arranged in several Tensors before being input to BERT.

    The BertClassifier model can  be configured with a preprocessor layer, in which case it will automatically apply preprocessing to raw inputs during fit(), predict(), and evaluate(). This is done by default when creating the model with from_preset().

    Bert is only trained in English corpus. That's why people use multilingual Bert or XLM-Roberta for this competition.

    Here are some models for multi-language NLP available in Keras NLP:
    - bert_base_multi
    - deberta_v3_base_multi
    - distil_bert_base_multi
    - xlm_roberta_base_multi
    - xlm_roberta_large_multi
    """
    )
    return


@app.cell
def _(df_train, strategy):
    VALIDATION_SPLIT = 0.3
    TRAIN_SIZE = int(df_train.shape[0] * (1 - VALIDATION_SPLIT))
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    return BATCH_SIZE, TRAIN_SIZE


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here's a utility function that splits the example into an `(x, y)` tuple that is suitable for `model.fit()`.

    By default, `keras_nlp.models.BertClassifier` will tokenize and pack together raw strings using a `"[SEP]"` token during training.

    Therefore, this label splitting is all the data preparation that we need to perform.
    """
    )
    return


@app.cell
def _(BATCH_SIZE, TRAIN_SIZE, df_train, keras, tf):
    def split_labels(x, y):
        return (x[0], x[1]), y

    training_dataset = tf.data.Dataset.from_tensor_slices(
        (
            df_train[["premise", "hypothesis"]].values,
            keras.utils.to_categorical(df_train["label"], num_classes=3),
        )
    )

    train_dataset = training_dataset.take(TRAIN_SIZE)
    val_dataset = training_dataset.skip(TRAIN_SIZE)

    # Apply the preprocessor to every sample of train, val and test data using `map()`.
    # [`tf.data.AUTOTUNE`](https://www.tensorflow.org/api_docs/python/tf/data/AUTOTUNE) and `prefetch()` are options to tune performance, see
    # https://www.tensorflow.org/guide/data_performance for details.

    train_preprocessed = (
        train_dataset.map(split_labels, tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_preprocessed = (
        val_dataset.map(split_labels, tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_preprocessed, val_preprocessed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load a BERT model from Keras NLP - Train the model""")
    return


@app.cell
def _(keras, keras_nlp, strategy):
    # Load a BERT model.

    with strategy.scope():
        classifier = keras_nlp.models.BertClassifier.from_preset(
            "bert_base_multi", num_classes=3
        )

        # in distributed training, the recommendation is to scale batch size and learning rate with the numer of workers.
        classifier.compile(
            optimizer=keras.optimizers.Adam(
                1e-5 * strategy.num_replicas_in_sync
            ),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        classifier.summary()
    return (classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Train your own model - Fine-tuning BERT""")
    return


@app.cell
def _(classifier, train_preprocessed, val_preprocessed):
    EPOCHS = 3
    history = classifier.fit(
            train_preprocessed, epochs=EPOCHS, validation_data=val_preprocessed
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Generate the submission file

    Let's get the test data
    """
    )
    return


@app.cell
def _(BATCH_SIZE, classifier, df_test):
    predictions = classifier.predict(
        (df_test["premise"], df_test["hypothesis"]), batch_size=BATCH_SIZE
    )
    return (predictions,)


@app.cell
def _(df_test, np, predictions):
    submission = df_test.id.copy().to_frame()
    submission["prediction"] = np.argmax(predictions, axis=1)

    submission
    return (submission,)


@app.cell
def _(BATCH_SIZE, classifier, df_train):
    test_train_predictions = classifier.predict(
        (df_train["premise"], df_train["hypothesis"]), batch_size=BATCH_SIZE
    )
    return (test_train_predictions,)


@app.cell
def _(df_train, np, test_train_predictions):
    df_train["prediction"] = np.argmax(test_train_predictions, axis=1)
    incorrect_answers = df_train.query("prediction != label")[["id", "premise", "hypothesis", "language", "prediction", "label"]].copy()
    incorrect_answers


    return


@app.cell
def _(submission):
    submission.to_csv("submission.csv", index=False)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
