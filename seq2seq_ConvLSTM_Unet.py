import torch
import torch.nn as nn

from models.ConvLSTMCell import ConvLSTMCell
from models.U_Net import UNet

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan, out_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        
        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        
        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=out_chan,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))
        
        self.U_Net = UNet(n_channels=nf,
                          bilinear=False)


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
        # encoder_vector
        encoder_vector = h_t
        
        # Unet
        output_U_Net = self.U_Net(encoder_vector)
        
        # decoder
        for t in range(future_step):
            h_t2, c_t2 = self.decoder_1_convlstm(input_tensor=output_U_Net,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            output_U_Net = h_t2
            outputs += [h_t2]  # predictions
        
        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Tanh()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2)

        return outputs
