# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas==2.3.2",
# ]
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("Homework assignment from [makemore pt2](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4). Goal: to beat Andrej's loss of 2.17 of loss and read https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf")
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    import itertools
    import pandas as pd

    # Enable MPS acceleration for Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cpu" and torch.backends.mps.is_built():
        print("MPS is built but not available. Try: pip install --upgrade torch torchvision")
    elif device.type == "mps":
        print("✓ MPS acceleration enabled - training will be significantly faster!")

    words = open("names.txt", "r").read().splitlines()
    separator = "."
    chars = [separator] + sorted(list(set("".join(words))))
    char_lookup = {char: index for index, char in enumerate(chars)}

    # Split words once to ensure consistency across all experiments
    train_words = words[:int(0.8*len(words))]
    dev_words = words[int(0.8*len(words)):int(0.9*len(words))]
    test_words = words[int(0.9*len(words)):]
    print(f"Data split: {len(train_words)} train, {len(dev_words)} dev, {len(test_words)} test words")

    def make_dataset(block_size, data):
        _X, _Y = [],[]
        for _w in data:
            _context = [0] * block_size
            for _ch in _w + ".":
                _ix = char_lookup[_ch]
                _X.append(_context)
                _Y.append(_ix)
                _context = _context[1:] + [_ix]
        _X = torch.tensor(_X)
        _Y = torch.tensor(_Y)
        return _X, _Y

    def get_datasets(block_size):
        X_Train, Y_Train = make_dataset(block_size, train_words)
        X_Dev, Y_Dev = make_dataset(block_size, dev_words)
        X_Test, Y_Test = make_dataset(block_size, test_words)
        return  X_Train, Y_Train, X_Dev, Y_Dev, X_Test, Y_Test

    def calculate_loss(C, W1, B1, W2, B2, input, output, input_dimensionality, block_size):
        # Ensure input and output are on the same device as model parameters
        input = input.to(device)
        output = output.to(device)
        _emb = C[input]
        _h = torch.tanh(_emb.view(-1,block_size * input_dimensionality) @ W1 + B1)
        _logits = _h @ W2 + B2
        _loss = F.cross_entropy(_logits,  output)
        return _loss

    def custom_train_with_decay(train_input, train_output, learning_rate_per_step, input_dimensionality, block_size, hidden_layer_count, batch_size):
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((27,input_dimensionality), generator=g).to(device)
        W1 = torch.randn((block_size * input_dimensionality,hidden_layer_count), generator=g).to(device)
        B1 = torch.randn(hidden_layer_count, generator=g).to(device)
        W2 = torch.randn((hidden_layer_count,27), generator=g).to(device)
        B2 = torch.randn(27, generator=g).to(device)

        # Move training data to device
        train_input = train_input.to(device)
        train_output = train_output.to(device)

        parameters = [C, W1, B1, W2, B2]
        #print(f"Total Parameter Count = {sum(p.nelement() for p in parameters)}")
        for _p in parameters:
            _p.requires_grad = True

        loss_per_step = []
        for _i, _current_learning_rate in enumerate(learning_rate_per_step):
            #Nobody does forward/backward passes on the entire dataset every time
            #Instead they batch the data to get faster iterations
            batch = torch.randint(0, train_input.shape[0], (batch_size,) )
            #Forward Pass
            _loss = calculate_loss(C, W1, B1, W2, B2, train_input[batch], train_output[batch], input_dimensionality, block_size)
            loss_per_step.append(_loss.cpu().item())
            #Backward Pass
            for _p in parameters:
                _p.grad = None
            _loss.backward()
            for _p in parameters:
                _p.data += _current_learning_rate * _p.grad
        return loss_per_step, C, W1, B1, W2, B2


    ##Don't do this until the end with the best dev configuration:
        #Eval Loss on entire dataset
        # calculate_loss(test_input, test_output)
        # _emb = C[test_input]
        # _h = torch.tanh(_emb.view(-1,block_size * input_dimensionality) @ W1 + B1)
        # _logits = _h @ W2 + B2
        # _loss = F.cross_entropy(_logits, test_output)
        # return _loss.item()
    return (
        calculate_loss,
        custom_train_with_decay,
        get_datasets,
        itertools,
        mo,
        pd,
        plt,
    )


@app.cell
def _(custom_train_with_decay, get_datasets):
    #Sanity Check run
    _learning_rate_trial_1 = ([-.1] * 10) + ([-.01] * 10)
    _X_Train, _Y_Train, _X_Dev, _Y_Dev, _X_Test, _Y_Test = get_datasets(3)
    _loss_per_step, _C, _W1, _B1, _W2, _B2 = custom_train_with_decay(_X_Dev, _Y_Dev, _learning_rate_trial_1, 2, 3, 100, 32)
    _loss_per_step[0]
    return


@app.cell
def _(calculate_loss, custom_train_with_decay, get_datasets, itertools, pd):
    _learning_rate_trial_1 = ([-.1] * 1500) + ([-.01] * 6000)
    input_dimensionalities = [6,7, 8, 9,10]
    batch_sizes = [256]
    block_sizes = [5]
    hidden_layer_counts = [200, 500, 1000, 2000, 5000]

    results = []
    for _input_dim, _batch_size, _block_size, _hidden_count in itertools.product(
        input_dimensionalities, batch_sizes, block_sizes, hidden_layer_counts):

        X_Train, Y_Train, X_Dev, Y_Dev, X_Test, Y_Test = get_datasets(_block_size)
        loss_per_step, C, W1, B1, W2, B2 = custom_train_with_decay(
            X_Dev, Y_Dev, _learning_rate_trial_1, 
            _input_dim, _block_size, _hidden_count, _batch_size
        )
        dev_loss = calculate_loss(C, W1, B1, W2, B2, X_Dev, Y_Dev, _input_dim, _block_size)
        results.append({
            'input_dim': _input_dim,
            'batch_size': _batch_size,
            'block_size': _block_size,
            'hidden_count': _hidden_count,
            'loss_history': loss_per_step,  # Store full history
            'model_params': (C, W1, B1, W2, B2),
            'dev_loss': dev_loss.item()
        })

    df_results = pd.DataFrame(results)

    # Compute derived metrics
    df_results['final_loss'] = df_results['loss_history'].apply(lambda x: x[-1])
    df_results['min_loss'] = df_results['loss_history'].apply(lambda x: min(x))
    df_results['convergence_step'] = df_results['loss_history'].apply(lambda x: x.index(min(x)))

    return (df_results,)


@app.cell
def _(df_results, mo, plt):
    #Vibe Coded Visualizations
    # Visualization of hyperparameter search results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Final loss by block size
    ax = axes[0, 0]
    df_results.groupby('block_size')['final_loss'].agg(['min', 'std']).plot(kind='bar', y='min', yerr='std', ax=ax)
    ax.set_title('Final Loss by Block Size')
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Loss')

    # 2. Final loss by input dimensionality
    ax = axes[0, 1]
    df_results.groupby('input_dim')['final_loss'].agg(['min', 'std']).plot(kind='bar', y='min', yerr='std', ax=ax)
    ax.set_title('Final Loss by Input Dimensionality')
    ax.set_xlabel('Input Dim')
    ax.set_ylabel('Loss')

    # 3. Final loss by hidden layer count
    ax = axes[0, 2]
    df_results.groupby('hidden_count')['final_loss'].agg(['min', 'std']).plot(kind='bar', y='min', yerr='std', ax=ax)
    ax.set_title('Final Loss by Hidden Layer Count')
    ax.set_xlabel('Hidden Count')
    ax.set_ylabel('Loss')

    # 4. Heatmap of block_size vs input_dim
    ax = axes[1, 0]
    pivot = df_results.pivot_table(values='final_loss', index='block_size', columns='input_dim')
    im = ax.imshow(pivot, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Input Dim')
    ax.set_ylabel('Block Size')
    ax.set_title('Loss Heatmap: Block Size vs Input Dim')
    plt.colorbar(im, ax=ax)

    # 5. Training curves for top 5 configurations
    ax = axes[1, 1]
    top_5 = df_results.nsmallest(5, 'final_loss')
    for _idx, _row in top_5.iterrows():
        label = f"bs={_row['block_size']}, id={_row['input_dim']}, hc={_row['hidden_count']}"
        ax.plot(_row['loss_history'][::100], label=label, alpha=0.7)
    ax.set_xlabel('Step (x100)')
    ax.set_ylabel('Loss')
    ax.set_title('Top 5 Training Curves')
    ax.legend(fontsize=8)

    # 6. Dev Loss by Hidden Layer Count
    ax = axes[1,2]
    df_results.groupby('hidden_count')['dev_loss'].agg(['min']).plot(kind='bar', y='min', ax=ax)
    ax.set_title('Dev Loss by Hidden Layer Count')
    ax.set_xlabel('Hidden Count')
    ax.set_ylabel('Dev Loss')

    plt.tight_layout()
    plt.show()
    # Best configuration summary
    best_idx = df_results['dev_loss'].idxmin()
    best_config = df_results.loc[best_idx]
    mo.md(f"""
    ## Best Configuration
    - **Block Size**: {best_config['block_size']}
    - **Input Dimensionality**: {best_config['input_dim']}
    - **Hidden Layer Count**: {best_config['hidden_count']}
    - **Batch Size**: {best_config['batch_size']}
    - **Final Loss**: {best_config['final_loss']:.4f}
    - **Min Loss**: {best_config['dev_loss']:.4f}
    - **Convergence Step**: {best_config['convergence_step']}
    """)
    return


@app.cell
def _(calculate_loss, custom_train_with_decay, get_datasets, plt):
    # try the best model with less steps since it seems like steps didn't really help
    _block_size = 5
    _input_dim = 8
    _less_learning_rate = ([-0.1] * 1500) + ([-0.01] * 6000)
    _X_Train, _Y_Train, _X_Dev, _Y_Dev, final_X_Test, final_Y_Test = get_datasets(_block_size)
    #Larger block size mostly just decreases variance
    _loss_per_step, C_2, W1_2, B1_2, W2_2, B2_2 = custom_train_with_decay(
        _X_Train, _Y_Train, _less_learning_rate, _input_dim, _block_size, 500, 256
    )
    print(f"Train: {_loss_per_step[-1]}")
    plt.plot(_loss_per_step)
    plt.yscale('log')
    plt.ylabel("loss (log)")
    plt.xlabel("steps")
    _dev_loss = calculate_loss(C_2, W1_2, B1_2, W2_2, B2_2, _X_Dev, _Y_Dev, _input_dim, _block_size)
    print(f"Dev: {_dev_loss}")
    plt.show()
    return


@app.cell
def _():
    #Don't do this until the end with the best dev configuration:
    #     # Eval Loss on entire dataset
    # final_loss = calculate_loss(C_2, W1_2, B1_2, W2_2, B2_2, final_X_Test, final_Y_Test, 9, 4)
    # final_loss
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
