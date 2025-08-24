# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib==3.10.5",
#     "openai==1.101.0",
#     "torch==2.8.0",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # N-gram Language Model

        Self Assigned Homework from [Andrej Karpathy's makemore video](https://www.youtube.com/watch?v=PaCmpygFfXo&t=382s)

        ## Interactive N-gram Model
        Instead of just a trigram language model, I think it would be interesting to let you decide how many characters to look ahead!
        Below are a variety of parameters to adjust for the model. You can re-run the last cell to see example names.
        """
    )

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # N-gram size slider (how many characters to look at to predict the next one)
    ngram_size = mo.ui.slider(
        start=1,
        stop=8,
        value=3,
        label="Number of chars to predict next char. 1 = bigram, 2 = trigram, etc",
        show_value=True,
    )

    # Checkbox to enable cross-entropy loss
    use_cross_entropy = mo.ui.checkbox(
        value=False, label="Use cross-entropy loss"
    )

    # Regularization/smoothing strength slider
    regularization_strength = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.01,
        label="Regularization/smoothing strength",
        show_value=True,
    )

    # Amount of Training Iterations
    training_iterations = mo.ui.slider(
        start=1.0,
        stop=10000.0,
        step=1,
        value=100,
        label="Number of Training Iterations",
        show_value=True,
    )

    test_train_split = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.2,
        label="% of data to withold for testing",
        show_value=True,
    )

    mo.md(f"""
    ## Model Configuration

    {ngram_size}

    {use_cross_entropy}

    {regularization_strength}

    {training_iterations}

    {test_train_split}
    """)

    return (
        ngram_size,
        regularization_strength,
        test_train_split,
        training_iterations,
        use_cross_entropy,
    )


@app.cell(hide_code=True)
def _(ngram_size, test_train_split):
    import torch

    # Copy some code from the previous example
    # import and initialize packages
    num_preceding_chars = ngram_size.value
    names = open("names.txt", "r").read().splitlines()

    separator = "."
    chars = (
        [separator] + sorted(list(set("".join(names))))
    )  # create a set of every character, turn it into a list and use the index of each character in that list as its unique integer

    # create a lookup table from character to index
    char_lookup = {char: index for index, char in enumerate(chars)}

    def create_labels(labels):
        _input_labels, _output_labels = [], []
        for _n in train_names:
            _characters = (
                ([separator] * num_preceding_chars) + list(_n) + [separator]
            )
            for _i in range(num_preceding_chars, len(_characters)):
                _inputs = tuple(
                    [
                        char_lookup[x]
                        for x in _characters[_i - num_preceding_chars : _i]
                    ]
                )
                _output = char_lookup[_characters[_i]]
                _input_labels.append(_inputs)
                _output_labels.append(_output)
        return _input_labels, _output_labels

    # unlike the video, input_labels is now a list of tuples of characters
    train_names = names[: int(len(names) * (1 - test_train_split.value))]
    test_names = names[int(len(names) * test_train_split.value) :]
    input_labels, output_labels = create_labels(train_names)
    test_input_labels, test_output_labels = create_labels(test_names)

    return chars, input_labels, output_labels, test_output_labels, torch


@app.cell(hide_code=True)
def _(input_labels):
    # unlike before, our input into the model is a list of characters.
    # We could try Binary Encoding but we're still going to do hot encoding
    # maybe add Binary Encoding later to compare

    all_char_pairs = list(set(input_labels))
    char_pair_lookup = {t: index for index, t in enumerate(all_char_pairs)}
    return all_char_pairs, char_pair_lookup


@app.cell(hide_code=True)
def _(
    char_pair_lookup,
    input_labels,
    output_labels,
    test_output_labels,
    torch,
):
    # Speeds up some of the operations especially when ngram_size is high
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        # On Apple Silicon (M1/M2/M3...), check MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = get_device()
    # Now setup a tensor where each combination of n characters is represented by a single value in the hot encoding
    unified_input_label = [char_pair_lookup[t] for t in input_labels]
    input_labels_tensor = torch.tensor(unified_input_label, device=device)
    output_labels_tensor = torch.tensor(output_labels, device=device)
    test_unified_input_label = [char_pair_lookup[t] for t in input_labels]
    test_input_labels_tensor = torch.tensor(
        test_unified_input_label, device=device
    )
    test_output_labels_tensor = torch.tensor(test_output_labels, device=device)
    return (
        device,
        input_labels_tensor,
        output_labels_tensor,
        test_input_labels_tensor,
        test_output_labels_tensor,
    )


@app.cell(hide_code=True)
def _(
    all_char_pairs,
    chars,
    device,
    input_labels_tensor,
    output_labels_tensor,
    regularization_strength,
    test_input_labels_tensor,
    test_output_labels_tensor,
    torch,
    training_iterations,
    use_cross_entropy,
):
    num_classes = len(all_char_pairs)
    g = torch.Generator(device=device).manual_seed(2147483647)
    W = torch.randn(
        (num_classes, len(chars)),
        generator=g,
        requires_grad=True,
        device=device,
    )
    import torch.nn.functional as F

    def train_model(in_tensor, out_tensor, iterations):
        loss_data = []
        for _i in range(iterations):
            W.grad = None
            logits = W[in_tensor]
            logcount = logits.exp()
            probs = logcount / logcount.sum(1, keepdim=True)
            if use_cross_entropy.value:
                loss = F.cross_entropy(logits, out_tensor)
            else:
                loss = (
                    -probs[
                        torch.arange(len(in_tensor), device=device),
                        out_tensor,
                    ]
                    .log()
                    .mean()
                )
            loss = loss + (W**2).mean() * regularization_strength.value
            loss_data.append(loss.item())
            loss.backward()
            W.data += -100 * W.grad

        return loss_data

    loss_data = train_model(
        input_labels_tensor,
        output_labels_tensor,
        int(training_iterations.value),
    )
    test_loss = train_model(
        test_input_labels_tensor, test_output_labels_tensor, 1
    )[0]  # Get the single loss value
    final_training_loss = loss_data[-1] if loss_data else 0

    print(f"Final Training Loss: {final_training_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    return W, g, loss_data, final_training_loss, test_loss


@app.cell(hide_code=True)
def _(final_training_loss, loss_data, test_loss):
    import matplotlib.pyplot as plt

    # Create subplots for loss visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training loss over time
    ax1.plot(
        [i for i in range(len(loss_data))], loss_data, linewidth=2, color="blue"
    )
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.grid(True, alpha=0.3)

    # Plot training vs test loss comparison
    losses = [final_training_loss, test_loss]
    labels = ["Training Loss", "Test Loss"]
    colors = ["blue", "red"]

    bars = ax2.bar(labels, losses, color=colors, alpha=0.7)
    ax2.set_ylabel("Loss")
    ax2.set_title("Training vs Test Loss Comparison")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(W, chars, g, mo, torch):
    def sample():
        _out = []
        _index = 0
        while True:
            _logits = W[_index]
            _log_count = _logits.exp()
            _prob = _log_count / _log_count.sum()
            _out_index = torch.multinomial(
                _prob, num_samples=1, replacement=True, generator=g
            ).item()
            _index = _out_index
            if _index == 0:
                break
            _out += chars[_index]
        return _out

    # Generate multiple sample names for display
    sample_names = ['"' + "".join(sample()) + '"' for _ in range(10)]

    mo.md(f"""
    ## Generated Sample Names

    Here are 10 names generated by the trained model. You can rerun this cell to regenerate them.

    {", ".join(sample_names)}
    """)
    return


if __name__ == "__main__":
    app.run()
