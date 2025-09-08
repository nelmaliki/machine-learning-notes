# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib==3.10.6",
#     "torch==2.8.0",
# ]
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    words = open("names.txt", "r").read().splitlines()
    separator = "."
    chars = [separator] + sorted(list(set("".join(words))))
    char_lookup = {char: index for index, char in enumerate(chars)}

    block_size = 3
    X, Y = [],[]
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = char_lookup[ch]
            X.append(context)
            Y.append(ix)
            print(''.join(chars[i] for i in context), '--->', chars[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return (
        F,
        X,
        Y,
        block_size,
        ch,
        char_lookup,
        chars,
        context,
        ix,
        plt,
        torch,
        words,
    )


@app.cell
def _(X, Y):
    X.shape, X.dtype, Y.shape, Y.dtype
    return


@app.cell
def _(F, chars, torch):
    _num_dimensions = 2
    c = torch.randn((len(chars), _num_dimensions))
    #These are equivalent but indexing is just faster
    print((F.one_hot(torch.tensor(5), num_classes=len(chars)).float() @ c))
    print(c[5])
    return (c,)


@app.cell
def _(c, torch):
    #Torch indexing is crazy, note the behavior of all of these
    print(c[[5, 6]])
    print(c[(5,0)])
    #First tuple contains rows, second tuple contains columns the following statements are identical
    print(c[((5,5,5,5), (0,1,0,1))])
    print(torch.tensor([c[(5,0)], c[(5,1)], c[(5,0)], c[(5,1)]]))
    return


@app.cell
def _(c, torch):
    #can also index with another tensor
    print(c[torch.tensor(5)])
    print(c[torch.tensor(5)].shape)
    #Can create a tensor from an array notice the shape changes
    print(c[torch.tensor([5])])
    print(c[torch.tensor([5])].shape)
    #Can create a tensor from a tuple notice the shape is like the first example
    print(c[torch.tensor((5))])
    print(c[torch.tensor((5))].shape)
    #Also notice the weird behavior if you add a , to the tuple, now it's like the array example
    print(c[torch.tensor((5,))])
    print(c[torch.tensor((5,))].shape)
    return


@app.cell
def _(X):
    print(X)
    return


@app.cell
def _(X, c):
    print(c[X].shape)
    print(X[13,2])
    #notice that the following are equivalent
    print(c[X][13,2])
    print(c[X[13,2]])
    print(c[1])
    return


@app.cell
def _(X, c, torch):
    _embedding  = c[X]
    print(_embedding.size())
    #this is not a useful size, if our weights are 6,100
    #instead lets concat each token together to get a 32,6 instead of 32,3,2
    print(_embedding[:, 0, :].shape)
    #second arg to cat is the dimension, we want to stack them by the second dimension (0 indexed)
    _cat_embedding = torch.cat([_embedding[:, 0, :], _embedding[:, 1, :], _embedding[:, 2, :]], 1)
    print(_cat_embedding.shape)
    #torch.unbind is the equivalent of pulling out a dimension, it's doing this part: [_embedding[:, 0, :], _embedding[:, 1, :], _embedding[:, 2, :]]
    print(len(torch.unbind(_embedding, 1)))
    #we can use this to make our embedding the right shape
    _unbind_embedding = torch.cat(torch.unbind(_embedding, 1),1)
    print(_unbind_embedding.shape)
    return


@app.cell
def _(torch):
    #There's an even better way to do this
    #torch.view is crazy magic. You can resize a tensor into an arbitarily different sized tensor assuming that it holds the same number of elements
    a = torch.arange(18)
    print(a.shape)
    print(a)
    return (a,)


@app.cell
def _(a):
    #You can split this into all kinds of other views
    print(a.view(9,2))
    return


@app.cell
def _(a):
    print(a.view(2,9))
    return


@app.cell
def _(a):
    #Doesn't really matter as long as the requested size fits 18 numbers
    print(a.view(3,3,2))
    return


@app.cell
def _(a):
    #view is much more efficient because torch stors everything as a 1 dimensional array anyway. View just manipulates the metadata for how torch should interpret it. 
    print(a.storage())
    #Every other approach involves moving or copying data around which is unecessary for us.
    return


@app.cell
def _(X, c, torch):
    w1 = torch.randn((6, 100))
    b1 = torch.randn(100)
    #-1 tells pytorch to infer the dimension
    #Check broadcasting rules to ensure that adding b1 does what we think it should do
    # ??, 100
    # __, 100
    #the 100's line up so __ becomes ?? where ?? is the number of entries in our data
    _logits = (c[X].view(-1,6) @ w1) + b1
    print(_logits.shape)
    activations = torch.tanh(_logits)
    print(activations.shape)
    return (activations,)


@app.cell
def _(activations, torch):
    w2 = torch.randn((100, 27)) #this is out output layer so it needs 100 to connect to w1 layer and 27 to output one of our expected characters
    b2 = torch.randn((27))
    #broadcasting rules again
    logits_2 = (activations @ w2) + b2
    counts = logits_2.exp()
    prob = counts / counts.sum(1, keepdim=True)
    return (prob,)


@app.cell
def _(Y, prob, torch):
    # now we compute the probability that it chose the correct answer (Y)
    #These are equivalent ways to do that
    print(torch.tensor([prob[(index, pred)] for index, pred in enumerate(Y)]))
    print(prob[torch.arange(prob.shape[0]), Y])
    _loss = -prob[torch.arange(prob.shape[0]), Y].log().mean()
    print(_loss)
    return


@app.cell
def _(F, X, Y, torch):
    # now time to make this a real trainable thing
    def train(learning_rate = -.1, steps=1000):
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((27,2), generator=g)
        W1 = torch.randn((6,100), generator=g)
        B1 = torch.randn(100, generator=g)
        W2 = torch.randn((100,27), generator=g)
        B2 = torch.randn(27, generator=g)
        parameters = [C, W1, B1, W2, B2]
        #print(f"Total Parameter Count = {sum(p.nelement() for p in parameters)}")
        for _p in parameters:
            _p.requires_grad = True

        _loss = None
        batch_size = 32
        for _ in range(steps):
            #Nobody does forward/backward passes on the entire dataset every time
            #Instead they batch the data to get faster iterations
            batch = torch.randint(0, X.shape[0], (batch_size,) )
            #Forward Pass
            _emb = C[X[batch]] # (batch_size, 3, 2)
            _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
            _logits = _h @ W2 + B2
            _loss = F.cross_entropy(_logits, Y[batch])
            #Backward Pass
            for _p in parameters:
                _p.grad = None
            _loss.backward()
            #Update
            for _p in parameters:
                _p.data += learning_rate * _p.grad
        #Eval Loss on entire dataset
        _emb = C[X] # (batch_size, 3, 2)
        _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
        _logits = _h @ W2 + B2
        _loss = F.cross_entropy(_logits, Y)
        return _loss.item()
    print(train())
    return (train,)


@app.cell
def _(train):
    #How do we know that we're getting the right learning rate?
    #Guess and Check:
    _learning_rates = [-.0001, -.001, -.01, -.1, -1, -10]
    print("\n".join([f"learning rate: {rate} loss {train(rate)}" for rate in _learning_rates]))
    #looks like -0.1 is our best bet
    return


@app.cell
def _(F, X, Y, plt, torch):
    #There's a better way to do this though, we're going to start at a low learning rate and increase it every iteration
    def train_with_variable_rate(learning_rates):
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((27,2), generator=g)
        W1 = torch.randn((6,100), generator=g)
        B1 = torch.randn(100, generator=g)
        W2 = torch.randn((100,27), generator=g)
        B2 = torch.randn(27, generator=g)
        parameters = [C, W1, B1, W2, B2]
        #print(f"Total Parameter Count = {sum(p.nelement() for p in parameters)}")
        for _p in parameters:
            _p.requires_grad = True

        _loss = None
        batch_size = 32
        _losses = []
        for _i in range(len(learning_rates)):
            #Nobody does forward/backward passes on the entire dataset every time
            #Instead they batch the data to get faster iterations
            batch = torch.randint(0, X.shape[0], (batch_size,) )
            #Forward Pass
            _emb = C[X[batch]] # (batch_size, 3, 2)
            _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
            _logits = _h @ W2 + B2
            _loss = F.cross_entropy(_logits, Y[batch])
            #Backward Pass
            for _p in parameters:
                _p.grad = None
            _loss.backward()
            #Update
            for _p in parameters:
                _p.data += learning_rates[_i] * _p.grad
            _losses.append(_loss.item())
        return _losses
    #create 1000 exponent values between -3 and 0
    _lre = torch.linspace(-3, 0, 1000)
    #10^-3 and 10^0
    _learning_rates = -10**_lre
    _losses = train_with_variable_rate(_learning_rates)
    plt.plot(_lre, _losses)
    plt.show()
    #based on this graph it looks like 0.1ish is the best learning rate after all
    return


@app.cell
def _(train):
    #Now that we know the optimal rate, lets go for many steps with that learning rate
    print(train(-.1, 100000)) #remember that the bigram only got to 2.45 loss so we're already better than that
    return


@app.cell
def _(F, X, Y, torch):
    #Learning Rate Decay: We're going decrease our learning rate after a large number of steps. The idea is that as we get closer to the local minimum, we want smaller steps to see if there are any small valleys that would otherwise be stepped over by our normal rate

    def train_with_decay(learning_rate = -.1, steps=1000, stepsAfterDecay=0):
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((27,2), generator=g)
        W1 = torch.randn((6,100), generator=g)
        B1 = torch.randn(100, generator=g)
        W2 = torch.randn((100,27), generator=g)
        B2 = torch.randn(27, generator=g)
        parameters = [C, W1, B1, W2, B2]
        #print(f"Total Parameter Count = {sum(p.nelement() for p in parameters)}")
        for _p in parameters:
            _p.requires_grad = True

        _loss = None
        batch_size = 32
        for i in range(steps + stepsAfterDecay):
            #Nobody does forward/backward passes on the entire dataset every time
            #Instead they batch the data to get faster iterations
            batch = torch.randint(0, X.shape[0], (batch_size,) )
            #Forward Pass
            _emb = C[X[batch]] # (batch_size, 3, 2)
            _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
            _logits = _h @ W2 + B2
            _loss = F.cross_entropy(_logits, Y[batch])
            #Backward Pass
            for _p in parameters:
                _p.grad = None
            _loss.backward()
            #Update
            if(i > steps):
                learning_rate /= 10
            for _p in parameters:
                _p.data += learning_rate * _p.grad

        #Eval Loss on entire dataset
        _emb = C[X] # (batch_size, 3, 2)
        _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
        _logits = _h @ W2 + B2
        _loss = F.cross_entropy(_logits, Y)
        return _loss.item()
    print(train_with_decay(-.1, 100000, 1000)) 
    #Yay we're got even lower loss!!
    return


@app.cell
def _(block_size, ch, char_lookup, context, ix, torch, words):
    #This loss really isn't useful because our model could be memorizing the data
    #It is normal to split the dataset into 3 Training Split, Validation (Dev) split, Test Split
    #usually 80%, 10%, 10%

    #The Validation(Dev) split is what we use to identify hyper-parameters (layer size, learningrate, etc)
    #Training Split is used when you're ready to go big and train
    #The test split should be used sparingly. The more you use it, the more you as the developer train yourself to the test data which should be avoided

    #Unclear, can we use the validation split for training too? Why not?

    def make_dataset(words):
        _block_size = 3
        _X, _Y = [],[]
        for _w in words:
            _context = [0] * block_size
            for _ch in _w + ".":
                _ix = char_lookup[ch]
                _X.append(context)
                _Y.append(ix)
                _context = context[1:] + [ix]
        _X = torch.tensor(_X)
        _Y = torch.tensor(_Y)
        return _X, _Y

    import random
    random.seed(42)
    random.shuffle(words)
    X_Train, Y_Train = make_dataset(words[:int(0.8*len(words))])
    X_Dev, Y_Dev = make_dataset(words[int(0.8*len(words)):int(0.9*len(words))])
    X_Test, Y_Test = make_dataset(words[int(0.9*len(words)):])
    print(len(X_Train))
    print(len(X_Dev))
    print(len(X_Test))
    return X_Dev, X_Test, X_Train, Y_Dev, Y_Test, Y_Train


@app.cell
def _(F, X_Dev, Y_Dev, plt, torch):
    def dev_train(input, output, learning_rates):
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((27,2), generator=g)
        W1 = torch.randn((6,100), generator=g)
        B1 = torch.randn(100, generator=g)
        W2 = torch.randn((100,27), generator=g)
        B2 = torch.randn(27, generator=g)
        parameters = [C, W1, B1, W2, B2]
        for _p in parameters:
            _p.requires_grad = True

        _loss = None
        batch_size = 32
        _losses = []
        for _i in range(len(learning_rates)):
            batch = torch.randint(0, input.shape[0], (batch_size,) )
            #Forward Pass
            _emb = C[input[batch]] # (batch_size, 3, 2)
            _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
            _logits = _h @ W2 + B2
            _loss = F.cross_entropy(_logits, output[batch])
            #Backward Pass
            for _p in parameters:
                _p.grad = None
            _loss.backward()
            #Update
            for _p in parameters:
                _p.data += learning_rates[_i] * _p.grad
            _losses.append(_loss.item())
        return _losses
    #create 1000 exponent values between -3 and 0
    _lre = torch.linspace(-3, 0, 10)
    #10^-3 and 10^0
    _learning_rates = -10**_lre
    _losses = dev_train(X_Dev, Y_Dev, _learning_rates)
    #kinda hard to tell but let's just use 0.1 since we know that worked well last time
    plt.plot(_lre, _losses)
    plt.show()
    return


@app.cell
def _(F, X_Test, X_Train, Y_Test, Y_Train, torch):
    def custom_train_with_decay(train_input, train_output, test_input, test_output, learning_rate = -.1, steps=1000, stepsAfterDecay=0):
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((27,2), generator=g)
        W1 = torch.randn((6,100), generator=g)
        B1 = torch.randn(100, generator=g)
        W2 = torch.randn((100,27), generator=g)
        B2 = torch.randn(27, generator=g)
        parameters = [C, W1, B1, W2, B2]
        #print(f"Total Parameter Count = {sum(p.nelement() for p in parameters)}")
        for _p in parameters:
            _p.requires_grad = True

        _loss = None
        batch_size = 32
        for i in range(steps + stepsAfterDecay):
            #Nobody does forward/backward passes on the entire dataset every time
            #Instead they batch the data to get faster iterations
            batch = torch.randint(0, train_input.shape[0], (batch_size,) )
            #Forward Pass
            _emb = C[train_input[batch]] # (batch_size, 3, 2)
            _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
            _logits = _h @ W2 + B2

            _loss = F.cross_entropy(_logits, train_output[batch])
            #Backward Pass
            for _p in parameters:
                _p.grad = None
            _loss.backward()
            #Update
            if(i == steps):
                learning_rate /= 10
            for _p in parameters:
                _p.data += learning_rate * _p.grad
        #Eval Loss on entire dataset
        _emb = C[test_input]
        _h = torch.tanh(_emb.view(-1,6) @ W1 + B1)
        _logits = _h @ W2 + B2
        _loss = F.cross_entropy(_logits, test_output)
        return _loss.item()
    _loss = custom_train_with_decay(X_Train, Y_Train, X_Test, Y_Test, -.1, 100000, 10000)
    print(f"Loss on Test Set after training on training set: {_loss}")
    return


if __name__ == "__main__":
    app.run()
