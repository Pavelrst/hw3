import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO: Create two maps as described in the docstring above.
    # It's best if you also sort the chars before assigning indices, so that
    # they're in lexical order.
    # ====== YOUR CODE: ======
    #sorted_letters = sorted(set(text))
    #idx_to_char = {}
    #for i in range (len(sorted_letters)):
    #    idx_to_char[i]=sorted_letters[i]
    #char_to_idx = dict([[v,k] for k,v in idx_to_char.items()])

    # Create chars and indicies lists
    chars = sorted(set(text))
    chars_len = len(chars)
    indices = list(range(chars_len))

    # Create the dictionaries
    char_to_idx = dict(zip(chars, indices))
    idx_to_char = dict(zip(indices, chars))
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    #beforeletters=len(text)
    #for i in range(len(chars_to_remove)):
    #    text=text.replace(chars_to_remove[i],"")
    #n_removed=beforeletters-len(text)
    #text_clean=text
    text_len = len(text)
    chars = ''.join(chars_to_remove)
    chars = '[' + chars + ']'
    #for ich in range(len(chars_to_remove)):
    #    chars = chars + chars_to_remove[ich]
    #chars = chars + ']'
    text_clean = re.sub(chars, '', text)
    n_removed = text_len - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    #result = torch.zeros([len(text),len(char_to_idx)],dtype=torch.int8)
    #for i in range (len(text)):
    #    j=char_to_idx[text[i]]
    #    result[i,j]=1

    text_len = len(text)
    dict_len = len(char_to_idx)

    text_idx = [char_to_idx[j] for j in text]
    text_idx = torch.tensor(text_idx)

    result = torch.zeros([text_len, dict_len], dtype=torch.int8)
    result.scatter_(1, text_idx.view(-1, 1), 1)
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    #letters= torch.argmax(embedded_text,dim=1)
    #let = letters.numpy()
    #result = []
    #for i in range(len(letters)):
#   #      idx = letters[i].numpy()
#   #      index = 0
#   #      index = idx[0,0]
    #    letter=idx_to_char[let[i]]
    #    result.append(letter)
    #result=''.join(result)
    result = ''
    embedded_len = embedded_text.size(0)
    _, idx = torch.max(embedded_text, 1)

    idx = idx.numpy()
    for embed in range(embedded_len):
        letter = idx_to_char.get(idx[embed])
        result = result + letter
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    #print('chars_to_labelled_samples called')
    #print("length of test is = ",len(text))
    ##embedded_text=chars_to_onehot(text,char_to_idx)
    #samples = torch.zeros([(len(text)-1)//seq_len,seq_len,len(char_to_idx)],dtype=torch.int8)
    #tensor = torch.zeros((),dtype=torch.int8)
    #labels = tensor.new_empty([(len(text)-1)//seq_len,seq_len])
    #i=0
    #indicator = 0
    #for letter in text:
    #    indicator+=1
    #    if indicator%50000 == 0:
    #        print (indicator," letters processed")
    #    embedded_letter = chars_to_onehot(letter,char_to_idx)
    #    index = torch.argmax(embedded_letter,dim=1).numpy()
    #    if ( i < len(text)-seq_len):
    #        samples[i//seq_len,i%seq_len,index]=1
    #    if ( i >= 1 and i <= len(text)-seq_len):
    #        labels[(i-1)//seq_len,(i-1)%seq_len]=char_to_idx[letter]
    #    i+=1

    #v = len(char_to_idx)
    dict_len = len(char_to_idx)
    #n_seqs = (len(text) - 1)//seq_len
    num_of_seq = (len(text) - 1)//seq_len

    #embed the text and reshape it
    embed = chars_to_onehot(text, char_to_idx)
    samples = embed[:num_of_seq*seq_len, :].view(num_of_seq, seq_len, dict_len)

    #get all indicies of next letter and reshape it
    idxs = [char_to_idx[j] for j in text]
    idxs = torch.Tensor(idxs)
    labels = idxs[1:num_of_seq*seq_len+1].view(num_of_seq, seq_len)

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    result = torch.nn.functional.softmax(y/temperature, dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().
    # ====== YOUR CODE: ======
    b_debug = False
    # prepare input to feed the GRU model
    start_tensor = torch.unsqueeze(chars_to_onehot(start_sequence, char_to_idx), 0)
    # print('start_tensor.shape', start_tensor.shape)
    y, h_state = model(start_tensor.to(dtype=torch.float))
    # print('y.shape', y.shape)

    while len(out_text) < n_chars:
        last_char_scores = torch.squeeze(y[0, -1, :])
        if b_debug:
            print('last_char_scores.shape', last_char_scores.shape)
            print('last_char_scores', last_char_scores)
        last_char_probs = hot_softmax(last_char_scores).data
        if b_debug:
            print('last_char_probs', last_char_probs)
        char_idx = torch.multinomial(last_char_probs, 1)[0].item()
        if b_debug:
            print('char_idx', char_idx)
        next_char = idx_to_char.get(char_idx)
        if b_debug:
            print('next_char', next_char)
        s_next_char = str(next_char)
        out_text = out_text + s_next_char
        if len(out_text) < n_chars:
            char_tensor = torch.unsqueeze(chars_to_onehot(s_next_char, char_to_idx), 0)
            y, h_state = model(char_tensor.to(dtype=torch.float), h_state)
        b_debug = False

    return out_text


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model.
        # To implement the affine transforms you can use either nn.Linear
        # modules (recommended) or create W and b tensor pairs directly.
        # Create these modules or tensors and save them per-layer in
        # the layer_params list.
        # Important note: You must register the created parameters so
        # they are returned from our module's parameters() function.
        # Usually this happens automatically when we assign a
        # module/tensor as an attribute in our module, but now we need
        # to do it manually since we're not assigning attributes. So:
        #   - If you use nn.Linear modules, call self.add_module() on them
        #     to register each of their parameters as part of your model.
        #   - If you use tensors directly, wrap them in nn.Parameter() and
        #     then call self.register_parameter() on them. Also make
        #     sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        print ('init called')
        # TODO: create a list of parameters of layers using n_layers.

        in_features = self.in_dim
        out_features = self.h_dim
        for layer in range(self.n_layers):

            # For first layer we have different out dimension.
            if layer > 0:
                x_in_features = self.h_dim
                h_in_features = self.h_dim
            else:
                x_in_features = self.in_dim
                h_in_features = self.h_dim

            # For each layers we need W_xz,W_hz,W_xg,W_hg,W_xr,W_hr matrices
            # z_module: W_xz,W_hz
            # r_module: W_xr,W_hr
            # g_module: W_xg,W_hg

            zx = nn.Linear(x_in_features, out_features, bias=True)
            zh = nn.Linear(h_in_features, out_features, bias=False)
            z_sigmoid = nn.Sigmoid()
            self.add_module("z_module_x_" + str(layer), zx)
            self.add_module("z_module_h_" + str(layer), zh)
            self.add_module("z_sigmoid_" + str(layer), z_sigmoid)

            rx = nn.Linear(x_in_features, out_features, bias=True)
            rh = nn.Linear(h_in_features, out_features, bias=False)
            r_sigmoid = nn.Sigmoid()
            self.add_module("r_module_x_" + str(layer), rx)
            self.add_module("r_module_h_" + str(layer), rh)
            self.add_module("r_sigmoid_" + str(layer),r_sigmoid)

            gx = nn.Linear(x_in_features, out_features, bias=True)
            gh = nn.Linear(h_in_features, out_features, bias=False)
            g_tanh = nn.Tanh()
            self.add_module("g_module_x_" + str(layer), gx)
            self.add_module("g_module_h_" + str(layer), gh)
            self.add_module("g_tanh_" + str(layer), g_tanh)

            dp_module = nn.Dropout(dropout) # self.dropout = dropout probability.
            self.add_module("dp_module_" + str(layer), dp_module)

            # Append a tuple of 6 modules to the list.
            self.layer_params.append((zx, zh, z_sigmoid, rx, rh, r_sigmoid, gx, gh, g_tanh, dp_module))

        # Define the last - output module
        out_module = nn.Linear(self.h_dim, self.out_dim)
        self.add_module("out_module", out_module)
        self.layer_params.append(out_module)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor=None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                # print ("state zeroed")
                layer_states.append(torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None



        # TODO: Implement the model's forward pass.
        # You'll need to go layer-by-layer from bottom to top (see diagram).
        # Tip: You can use torch.stack() to combine multiple tensors into a
        # single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        hidden_state = torch.zeros(batch_size, self.n_layers, self.h_dim, device=input.device)

        next_layer_input = []
        for i_layer in range(self.n_layers):
            next_layer_input.append(torch.zeros(batch_size, seq_len, self.h_dim, device=input.device))

        for layer_idx in range(self.n_layers):
            b_debug = False
            h_prev = layer_states[layer_idx]

            z_module_x, z_module_h_b, z_sig, r_module_x, r_module_h_b, r_sig, \
            g_module_x, g_module_h_b, g_tanh, dropout = self.layer_params[layer_idx]

            for t in range(seq_len):
                x = layer_input[:, t, :]

                # Cacl Z,R,G
                Wx = z_module_x.forward(x)
                Wh = z_module_h_b.forward(h_prev)
                z = z_sig.forward(Wx + Wh)
                Wx = r_module_x.forward(x)
                Wh = r_module_h_b.forward(h_prev)
                r = r_sig.forward(Wx + Wh)
                Wx = g_module_x.forward(x)
                Wh = g_module_h_b.forward(r * h_prev)
                g = g_tanh.forward(Wx + Wh)

                h = torch.mul(z, h_prev) + torch.mul(1 - z, g)
                h_prev = h

                # Give to next layer, the dropout of h.
                h_dropped = dropout.forward(h)
                next_layer_input[layer_idx][:, t, :] = h_dropped

            hidden_state[:, layer_idx, :] = h
            layer_input = next_layer_input[layer_idx]

        # Generate the final output.
        layer_output = torch.zeros(batch_size, seq_len, self.out_dim, device=input.device)
        output_module = self.layer_params[self.n_layers]
        for t in range(seq_len):
            x = layer_input[:, t, :]
            layer_output[:, t, :] = output_module.forward(x)

        # ========================
        return layer_output, hidden_state

