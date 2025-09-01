# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib==3.10.5",
#     "torch==2.8.0",
# ]
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # N-gram Language Model Parameter Analysis

        This notebook explores how different parameters affect the performance of an n-gram language model.
        We'll systematically vary each parameter and visualize its impact on training and test loss.

        Disclaimer: This was originally intended to be an interactive application to run in your browser. But due to limitations involving running torch in WASM, this has become a static notebook. I used Claude to rewrite the interactive code into a linear static notebook. The original interactive notebook can be found here: https://github.com/nelmaliki/machine-learning-notes/blob/main/makemore/ngram.py

        Based on [Andrej Karpathy's makemore video](https://www.youtube.com/watch?v=PaCmpygFfXo&t=382s)
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    # Load data
    names = open("names.txt", "r").read().splitlines()
    separator = "."
    chars = [separator] + sorted(list(set("".join(names))))
    char_lookup = {char: index for index, char in enumerate(chars)}

    # Get device for computation
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = get_device()
    print(f"Using device: {device}")

    # Define ngram_sizes once for all analyses
    ngram_sizes = range(1, 8)
    return (
        F,
        char_lookup,
        chars,
        device,
        names,
        ngram_sizes,
        np,
        plt,
        separator,
        torch,
    )


@app.cell(hide_code=True)
def _(F, char_lookup, chars, device, names, separator, torch):
    def create_dataset(name_list, num_preceding_chars):
        _input_labels, _output_labels = [], []
        for _n in name_list:
            _characters = (
                ([separator] * num_preceding_chars) + list(_n) + [separator]
            )
            for _i in range(num_preceding_chars, len(_characters)):
                _inputs = tuple(
                    [
                        char_lookup[_x]
                        for _x in _characters[_i - num_preceding_chars : _i]
                    ]
                )
                _output = char_lookup[_characters[_i]]
                _input_labels.append(_inputs)
                _output_labels.append(_output)
        return _input_labels, _output_labels

    def train_model(
        num_preceding_chars,
        iterations=100,
        use_cross_entropy=False,
        reg_strength=0.01,
        test_split=0.2,
    ):
        # Split data
        _split_idx = int(len(names) * (1 - test_split))
        _train_names = names[:_split_idx]
        _test_names = names[_split_idx:]

        # Create datasets
        _train_inputs, _train_outputs = create_dataset(
            _train_names, num_preceding_chars
        )
        _test_inputs, _test_outputs = create_dataset(
            _test_names, num_preceding_chars
        )

        # Create lookup for character combinations
        _all_char_pairs = list(set(_train_inputs))
        _char_pair_lookup = {
            _t: _index for _index, _t in enumerate(_all_char_pairs)
        }

        # Convert to tensors
        _train_input_tensor = torch.tensor(
            [_char_pair_lookup.get(_t, 0) for _t in _train_inputs],
            device=device,
        )
        _train_output_tensor = torch.tensor(_train_outputs, device=device)
        _test_input_tensor = torch.tensor(
            [
                _char_pair_lookup.get(_t, 0)
                for _t in _test_inputs
                if _t in _char_pair_lookup
            ],
            device=device,
        )
        _test_output_tensor = torch.tensor(
            [
                _test_outputs[_i]
                for _i, _t in enumerate(_test_inputs)
                if _t in _char_pair_lookup
            ],
            device=device,
        )

        # Initialize weights
        _num_classes = len(_all_char_pairs)
        _g = torch.Generator(device=device).manual_seed(2147483647)
        _W = torch.randn(
            (_num_classes, len(chars)),
            generator=_g,
            requires_grad=True,
            device=device,
        )

        # Training loop
        _train_losses = []
        for _ in range(iterations):
            _W.grad = None
            _logits = _W[_train_input_tensor]

            if use_cross_entropy:
                _loss = F.cross_entropy(_logits, _train_output_tensor)
            else:
                _logcount = _logits.exp()
                _probs = _logcount / _logcount.sum(1, keepdim=True)
                _loss = (
                    -_probs[
                        torch.arange(len(_train_input_tensor), device=device),
                        _train_output_tensor,
                    ]
                    .log()
                    .mean()
                )

            _loss = _loss + (_W**2).mean() * reg_strength
            _train_losses.append(_loss.item())
            _loss.backward()
            _W.data += -100 * _W.grad

        # Calculate test loss
        with torch.no_grad():
            _test_logits = _W[_test_input_tensor]
            if use_cross_entropy:
                _test_loss = F.cross_entropy(
                    _test_logits, _test_output_tensor
                ).item()
            else:
                _test_logcount = _test_logits.exp()
                _test_probs = _test_logcount / _test_logcount.sum(
                    1, keepdim=True
                )
                _test_loss = (
                    -_test_probs[
                        torch.arange(len(_test_input_tensor), device=device),
                        _test_output_tensor,
                    ]
                    .log()
                    .mean()
                    .item()
                )

        return _train_losses, _test_loss
    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1. Effect of N-gram Size

    It appears that looking ahead 2 tokens does perform better than 1, but looking any further only has a negative impact.
    """
    )
    return


@app.cell(hide_code=True)
def _(ngram_sizes, np, plt, train_model):
    ngram_train_losses = []
    ngram_test_losses = []

    for _n in ngram_sizes:
        _train_loss, _test_loss = train_model(_n, iterations=200)
        ngram_train_losses.append(_train_loss[-1])
        ngram_test_losses.append(_test_loss)
        print(
            f"N-gram size {_n}: Train Loss = {_train_loss[-1]:.4f}, Test Loss = {_test_loss:.4f}"
        )

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _x = np.arange(len(ngram_sizes))
    _width = 0.35

    _ax1.bar(
        _x - _width / 2,
        ngram_train_losses,
        _width,
        label="Train Loss",
        color="#3498db",
        alpha=0.8,
    )
    _ax1.bar(
        _x + _width / 2,
        ngram_test_losses,
        _width,
        label="Test Loss",
        color="#e74c3c",
        alpha=0.8,
    )
    _ax1.set_xlabel("N-gram Size")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Training vs Test Loss by N-gram Size")
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels([f"{_n}-gram" for _n in ngram_sizes])
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    _ax2.plot(
        list(ngram_sizes),
        ngram_train_losses,
        "o-",
        label="Train Loss",
        linewidth=2,
        markersize=8,
        color="#3498db",
    )
    _ax2.plot(
        list(ngram_sizes),
        ngram_test_losses,
        "s-",
        label="Test Loss",
        linewidth=2,
        markersize=8,
        color="#e74c3c",
    )
    _ax2.set_xlabel("N-gram Size")
    _ax2.set_ylabel("Loss")
    _ax2.set_title("Loss Trends with N-gram Size")
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2. Loss Function Comparison: Negative Log Likelihood vs Cross-Entropy

    **Finding**: Cross-entropy loss and negative log likelihood perform similarly across different n-gram sizes due to the one-hot encoding approach used.
    """
    )
    return


@app.cell(hide_code=True)
def _(ngram_sizes, plt, train_model):
    # Store results for each loss type
    nll_train_losses = []
    nll_test_losses = []
    ce_train_losses = []
    ce_test_losses = []

    for _n in ngram_sizes:
        # Negative Log Likelihood
        _train_loss_nll, _test_loss_nll = train_model(
            _n, iterations=200, use_cross_entropy=False
        )
        nll_train_losses.append(_train_loss_nll[-1])
        nll_test_losses.append(_test_loss_nll)

        # Cross-Entropy
        _train_loss_ce, _test_loss_ce = train_model(
            _n, iterations=200, use_cross_entropy=True
        )
        ce_train_losses.append(_train_loss_ce[-1])
        ce_test_losses.append(_test_loss_ce)

        print(
            f"N-gram {_n} - NLL: Train={_train_loss_nll[-1]:.4f}, Test={_test_loss_nll:.4f} | "
            f"CE: Train={_train_loss_ce[-1]:.4f}, Test={_test_loss_ce:.4f}"
        )

    # Create line graph
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss comparison
    _ax1.plot(
        list(ngram_sizes),
        nll_train_losses,
        "o-",
        label="Negative Log Likelihood",
        linewidth=2,
        markersize=8,
        color="#9b59b6",
        alpha=0.8,
    )
    _ax1.plot(
        list(ngram_sizes),
        ce_train_losses,
        "s-",
        label="Cross-Entropy",
        linewidth=2,
        markersize=8,
        color="#1abc9c",
        alpha=0.8,
    )
    _ax1.set_xlabel("N-gram Size (Characters of Context)")
    _ax1.set_ylabel("Training Loss")
    _ax1.set_title("Training Loss: NLL vs Cross-Entropy")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    # Test loss comparison
    _ax2.plot(
        list(ngram_sizes),
        nll_test_losses,
        "o-",
        label="Negative Log Likelihood",
        linewidth=2,
        markersize=8,
        color="#9b59b6",
        alpha=0.8,
    )
    _ax2.plot(
        list(ngram_sizes),
        ce_test_losses,
        "s-",
        label="Cross-Entropy",
        linewidth=2,
        markersize=8,
        color="#1abc9c",
        alpha=0.8,
    )
    _ax2.set_xlabel("N-gram Size (Characters of Context)")
    _ax2.set_ylabel("Test Loss")
    _ax2.set_title("Test Loss: NLL vs Cross-Entropy")
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Effect of Regularization Strength Across N-gram Sizes

    Regularization didn't seem to have a positive affect. I don't think overfitting was a issue.
    """
    )
    return


@app.cell(hide_code=True)
def _(ngram_sizes, np, plt, train_model):
    reg_strengths = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]

    # Store results for each n-gram size
    reg_results_by_ngram = {}

    for _n in ngram_sizes:
        reg_train_losses = []
        reg_test_losses = []

        for _reg in reg_strengths:
            _train_loss, _test_loss = train_model(
                _n, iterations=200, reg_strength=_reg
            )
            reg_train_losses.append(_train_loss[-1])
            reg_test_losses.append(_test_loss)
            # print(f"N-gram {_n}, Reg {_reg}: Train={_train_loss[-1]:.4f}, Test={_test_loss:.4f}")

        reg_results_by_ngram[_n] = {
            "train": reg_train_losses,
            "test": reg_test_losses,
        }

    # Create visualization
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
    _axes = _axes.flatten()

    _colors = plt.cm.viridis(np.linspace(0, 1, len(reg_strengths)))

    for _idx, _n in enumerate(ngram_sizes):
        _ax = _axes[_idx]

        _ax.semilogx(
            [_r + 1e-6 for _r in reg_strengths],
            reg_results_by_ngram[_n]["train"],
            "o-",
            label="Train",
            linewidth=2,
            markersize=6,
            color="#f39c12",
            alpha=0.8,
        )
        _ax.semilogx(
            [_r + 1e-6 for _r in reg_strengths],
            reg_results_by_ngram[_n]["test"],
            "s-",
            label="Test",
            linewidth=2,
            markersize=6,
            color="#8e44ad",
            alpha=0.8,
        )

        _ax.set_xlabel("Regularization Strength")
        _ax.set_ylabel("Loss")
        _ax.set_title(f"{_n}-gram Model")
        _ax.legend(loc="best", fontsize=8)
        _ax.grid(True, alpha=0.3)

    # Hide the 8th subplot
    _axes[7].axis("off")

    plt.suptitle(
        "Regularization Effect Across N-gram Sizes", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. Training Iterations Analysis Across N-gram Sizes

    Number of iterations obviously relies on how many characters ahead we look. It appears that even the 5+ gram models taper off between 10^3 and 10^4 iterations. All models greater than 1-gram seem to converge to around 2.4 or 2.5 training loss.  
    """
    )
    return


@app.cell(hide_code=True)
def _(ngram_sizes, plt, train_model):
    iteration_counts = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    # Store results for each n-gram size
    iter_results_by_ngram = {}

    for _n in ngram_sizes:
        iter_train_losses = []
        iter_test_losses = []

        for _iters in iteration_counts:
            _train_loss, _test_loss = train_model(_n, iterations=_iters)
            iter_train_losses.append(_train_loss[-1])
            iter_test_losses.append(_test_loss)
            # print(f"N-gram {_n}, Iters {_iters}: Train={_train_loss[-1]:.4f}, Test={_test_loss:.4f}")

        iter_results_by_ngram[_n] = {
            "train": iter_train_losses,
            "test": iter_test_losses,
        }

    # Create visualization
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
    _axes = _axes.flatten()

    for _idx, _n in enumerate(ngram_sizes):
        _ax = _axes[_idx]

        _ax.plot(
            iteration_counts,
            iter_results_by_ngram[_n]["train"],
            "o-",
            label="Train",
            linewidth=2,
            markersize=6,
            color="#2ecc71",
            alpha=0.8,
        )
        _ax.plot(
            iteration_counts,
            iter_results_by_ngram[_n]["test"],
            "s-",
            label="Test",
            linewidth=2,
            markersize=6,
            color="#e67e22",
            alpha=0.8,
        )

        _ax.set_xlabel("Training Iterations")
        _ax.set_ylabel("Loss")
        _ax.set_title(f"{_n}-gram Model")
        _ax.legend(loc="best", fontsize=8)
        _ax.grid(True, alpha=0.3)
        _ax.set_xscale("log")

    # Hide the 8th subplot
    _axes[7].axis("off")

    plt.suptitle(
        "Training Iterations Effect Across N-gram Sizes", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. Train/Test Split Analysis Across N-gram Sizes

    **Finding**: A 80/20 or 70/30 split typically provides a good balance between having enough training data and reliable test metrics across all n-gram sizes.
    """
    )
    return


@app.cell(hide_code=True)
def _(ngram_sizes, plt, train_model):
    split_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Store results for each n-gram size
    split_results_by_ngram = {}

    for _n in ngram_sizes:
        split_train_losses = []
        split_test_losses = []

        for _split in split_ratios:
            _train_loss, _test_loss = train_model(
                _n, iterations=5000, test_split=_split
            )
            split_train_losses.append(_train_loss[-1])
            split_test_losses.append(_test_loss)
            # print(f"N-gram {_n}, Split {_split:.0%}: Train={_train_loss[-1]:.4f}, Test={_test_loss:.4f}")

        split_results_by_ngram[_n] = {
            "train": split_train_losses,
            "test": split_test_losses,
        }

    # Create visualization
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
    _axes = _axes.flatten()

    for _idx, _n in enumerate(ngram_sizes):
        _ax = _axes[_idx]

        _train_percentages = [(1 - _s) * 100 for _s in split_ratios]

        _ax.plot(
            _train_percentages,
            split_results_by_ngram[_n]["train"],
            "o-",
            label="Train",
            linewidth=2,
            markersize=6,
            color="#16a085",
            alpha=0.8,
        )
        _ax.plot(
            _train_percentages,
            split_results_by_ngram[_n]["test"],
            "s-",
            label="Test",
            linewidth=2,
            markersize=6,
            color="#c0392b",
            alpha=0.8,
        )

        _ax.set_xlabel("Training Data %")
        _ax.set_ylabel("Loss")
        _ax.set_title(f"{_n}-gram Model")
        _ax.legend(loc="best", fontsize=8)
        _ax.grid(True, alpha=0.3)
        _ax.invert_xaxis()  # Show from 95% to 50% training data

    # Hide the 8th subplot
    _axes[7].axis("off")

    plt.suptitle(
        "Train/Test Split Effect Across N-gram Sizes", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
