import torch
import torch.nn as nn
from typing import List, Tuple


class LSTM(nn.Module):
    """ Takes only the last output of the LSTM and propagates it to a FC layer """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, history_size: int, output_dim: int, hidden_dims_list: List[int]):
        """
        an MLP implementation that takes performs (Batch,Hist,Features) -> (Batch,Hist*Features) -> MLP -> (Batch,Out)
        :param input_dim: the number of features in the input, required to calculate the MLP input dim
        :param history_size: feature window, essentially, required to calculate the MLP input dim
        :param output_dim: output dim for multi-target prediction
        :param hidden_dims_list: the dimensions of the hidden layers only (without input and output)
        """
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim * history_size, hidden_dims_list[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims_list) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims_list[i], hidden_dims_list[i + 1]))
            self.hidden_layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_dims_list[-1], output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

##### seq2seq + attention #####
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_utils.ml_core.models.base_model import Model
from ml_utils.ml_core.models.registry import register_parser, register_constructor
from ml_utils.ml_core.utils import utils


class Encoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, dec_hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.embedding = nn.Linear(self.input_size, self.embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.dec_hidden_size)

    def forward(self, enc_input: torch.Tensor):
        """
        @param enc_input: [batch_size, seq_size, enc_input_size]
        @return:
        """

        embedded = self.embedding(enc_input)
        outputs, last_hidden = self.rnn(embedded)
        # output -> [batch_size, seq_size, hidden_size * num_directions] = [batch_size, seq_size, hidden_size * 2]
        # hiddens -> [num_layers * num_directions,batch_size, hidden_size] = [2, batch_size, hidden_size]
        # namely, output is all the hidden states and hiddens is the last one:
        # outputs -> [:, [[h_1->,h_L<-],[h_2->,h_(L-1)<-],...[h_L->,h_1<-]]
        fwd_last_hidden, bwd_last_hidden = last_hidden[-2, :, :], last_hidden[-1, :, :]
        hidden_cat = torch.cat((fwd_last_hidden, bwd_last_hidden), dim=1)
        # now our hidden has shape [batch_size, hidden_size * num_directions] and after FC: [batch_size, dec_hid_size]
        # we cast to dec_hidden_size as this hidden state is the initial hidden state of the decoder!
        return outputs, torch.tanh(self.fc(hidden_cat))


class Attention(nn.Module):
    def __init__(self, enc_hidden_size: int, enc_bidirectional: bool, dec_hidden_size: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hidden_size * (2 if enc_bidirectional else 1) + dec_hidden_size, dec_hidden_size)
        self.energy_weights_vec = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor):
        """
        @param dec_hidden:  [batch_size, dec_hidden_size]
        @param enc_outputs: [batch_size, seq_size, hidden_size * 2]
        @return:
        """
        batch_size, seq_size = enc_outputs.shape[0], enc_outputs.shape[1]
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, seq_size, 1)
        # repeat the hidden dim seq_size times so dec_hidden -> [batch_size, seq_size, dec_hidden_size]
        cat_hiddens = torch.cat((enc_outputs, dec_hidden), dim=2)
        # catting the hiddens so -> [batch_size, seq_size, dec_hidden_size + num_directions * enc_hidden_size]
        energy = torch.tanh(self.attn(cat_hiddens))  # [batch_size, seq_size, dec_hidden_size]
        attn_weights = self.energy_weights_vec(energy).squeeze(2)  # [batch_size, seq_size]
        return F.softmax(attn_weights, dim=1)


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, output_size: int, enc_hidden_size: int,
                 attention: Attention, enc_bidirectional: bool = True):
        super(Decoder, self).__init__()
        self.input_size = output_size  # = output size because we loop with input_(t) = output_(t-1)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.enc_hidden_size = enc_hidden_size
        self.attention = attention
        self.embedding = nn.Linear(self.input_size, embedding_size)
        enc_num_dirs = 2 if enc_bidirectional else 1
        self.rnn = nn.GRU((enc_hidden_size * enc_num_dirs) + embedding_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(enc_hidden_size * enc_num_dirs + embedding_size + hidden_size, output_size)

    def forward(self, dec_input: torch.Tensor, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor):
        """
        @param dec_input: [batch size, 1, enc_output_size]
        @param dec_hidden:  [batch size, dec_hidden_size]
        @param enc_outputs: [batch_size, seq_size , enc_hidden_dim * num_directions]
        @return:
        """
        embedded = self.embedding(dec_input).unsqueeze(1)
        attn_weights = self.attention(dec_hidden, enc_outputs).unsqueeze(1)  # [batch size,1, seq_size]
        weighted_enc_outputs = torch.bmm(attn_weights, enc_outputs)  # [batch size, 1, enc hidden dim * num directions]
        rnn_input = torch.cat((embedded, weighted_enc_outputs), dim=2)  # [batch size, 1,enc hid dim * dirs + emb size]
        dec_output, dec_hidden = self.rnn(rnn_input, dec_hidden.unsqueeze(1).permute(1, 0, 2))
        # output = [batch size, seq len, dec hidden size * n directions] = [batch_size, 1, dec hidden size]
        # hidden = [n layers * n directions, batch size, dec hid dim] = [1, batch_size, dec hidden size]
        # this also means that output == hidden (up to permute)
        prediction = self.fc_out(torch.cat((dec_output, weighted_enc_outputs, embedded), dim=2)).squeeze(
            1)  # [batch size, output dim]
        return prediction, dec_hidden.squeeze(0)


@register_constructor
class Seq2SeqAttn(Model):
    def __init__(self, input_dim: Tuple, target_lag,
                 enc_embedding_size, enc_hidden_size, enc_num_layers, enc_bidirectional,
                 dec_embedding_size, dec_hidden_size, dec_output_size
                 ):
        """

        @param input_dim: [batch_size, feature_lags, encoder_input_size]
        @param target_lag: horizon to forcast
        @param enc_embedding_size: the embedding size for the encoder
        @param enc_hidden_size: the hidden size for the encoder
        @param enc_output_size: the output size of the encoder
        @param enc_num_layers: the number of rnn cells in the encoder
        @param enc_bidirectional: True iff bidirectional
        @param dec_embedding_size: the embedding size for the decoder
        @param dec_hidden_size: the hidden size for the decoder
        @param dec_output_size: the output size for the decoder
        """
        super(Seq2SeqAttn, self).__init__()
        self.batch_size, self.feature_lags, self.input_size = input_dim
        self.target_lag = target_lag
        self.cast_input_to_dec_output = nn.Linear(input_dim[-1], dec_output_size)
        self.encoder = Encoder(self.input_size, enc_embedding_size, enc_hidden_size, dec_hidden_size, enc_num_layers,
                               enc_bidirectional)
        self.attention = Attention(enc_hidden_size, enc_bidirectional, dec_hidden_size)
        self.decoder = Decoder(dec_embedding_size, dec_hidden_size, dec_output_size, enc_hidden_size,
                               self.attention, enc_bidirectional)
        self.output_size = self.batch_size, self.target_lag, dec_output_size

    def forward(self, x):
        """
        @param x: [batch_size, src len, input size]
        @return:
        """
        outputs = []
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(x)
        input = self.cast_input_to_dec_output(x[:, -1, :])
        for t in range(self.target_lag):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # place predictions in a tensor holding predictions for each token
            outputs.append(output)
            input = output
        return torch.stack(outputs).to(x.device).permute(1, 0, 2)

    def get_output_dim(self):
        return self.output_size


@register_parser
def parse_seq2seq(params: str, to_ignore_substitutes: dict = {}):
    parsed_params, parsed_kwargs = utils.parse_params(params, to_ignore_substitutes=to_ignore_substitutes)
    return Seq2SeqAttn(*parsed_params, **parsed_kwargs)


if __name__ == '__main__':
    target_lag = 1
    enc_embedding_size = 10
    enc_hidden_size = 16
    enc_num_layers = 1
    enc_bidirectional = True
    dec_embedding_size = 10
    dec_hidden_size = 12
    dec_output_size = 1
    batch_size = 2048
    feature_lag = 480
    input_size = 7
    input_dim = batch_size, feature_lag, input_size
    x = torch.randn(batch_size, feature_lag, input_size)
    s2s = Seq2SeqAttn(input_dim, target_lag, enc_embedding_size, enc_hidden_size, enc_num_layers, enc_bidirectional,
                      dec_embedding_size, dec_hidden_size, dec_output_size)
    y = s2s(x)
    print(y)
