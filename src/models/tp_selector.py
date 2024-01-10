# trajectory prediction layer of TNT algorithm with Binary CE Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def masked_softmax(vector, mask, dim=-1, memory_efficient=True, mask_fill_value=-1e32):
    """
    Title    : A masked softmax module to correctly implement attention in Pytorch.
    Authors  : Bilal Khan / AllenNLP
    Papers   : ---
    Source   : https://github.com/bkkaggle/pytorch_zoo/blob/master/pytorch_zoo/utils.py
               https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
    A masked softmax module to correctly implement attention in Pytorch.
    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    Args:
        vector (torch.tensor): The tensor to softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the softmax operation.
        dim (int, optional): The dimension to softmax over.
                            Defaults to -1.
        memory_efficient (bool, optional): Whether to use a less precise, but more memory efficient implementation of masked softmax.
                                            Defaults to False.
        mask_fill_value ([type], optional): The value to fill masked values with if `memory_efficient` is `True`.
                                            Defaults to -1e32.
    Returns:
        torch.tensor: The masked softmaxed output
    """
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(-1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = F.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            result = result.masked_fill((1 - mask).bool(), 0.0)
        else:
            masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
            result = F.softmax(masked_vector, dim=dim)
            result = result.masked_fill((1 - mask).bool(), 0.0)
    return result


# MLP
class MLP(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        hidden=64,
        bias=True,
        activation="relu",
        norm="layer",
    ):
        super(MLP, self).__init__()

        # define the activation function
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "relu6":
            act_layer = nn.ReLU6
        elif activation == "leaky":
            act_layer = nn.LeakyReLU
        elif activation == "prelu":
            act_layer = nn.PReLU
        else:
            raise NotImplementedError

        # define the normalization function
        if norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise NotImplementedError

        # insert the layers
        # print("MLP TargetPred hidden:", hidden, "in_channels:", in_channel)
        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)
        self.linear2.apply(self._init_weights)

        self.norm1 = norm_layer(hidden)
        self.norm2 = norm_layer(out_channel)

        self.act1 = act_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

        self.shortcut = None
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias), norm_layer(out_channel)
            )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # print("x", x.shape)
        # print(x[0])

        out = self.linear1(x)
        # print("out", out.shape)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return self.act2(out)


class TPSelector(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64):
        """"""
        super(TPSelector, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        # print("TargetPred hidden:", self.hidden_dim, "in_channels:", self.in_channels)

        self.prob_mlp = nn.Sequential(
            MLP(in_channels + 2, hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        past_feat: torch.Tensor,
        map_feat: torch.Tensor,
        z_samp: torch.Tensor,
        tp_candidate: torch.Tensor,
        candidate_mask=None,
    ):
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in:        the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate:  the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask: the mask of valid target candidate
        :return:
        """
        # dimension must be [batch size, 1, in_channels]
        # print("past_feat:", past_feat.shape)
        # print("map_feat:", map_feat.shape)
        # print("z_samp:", z_samp.shape)
        feat_in = torch.cat([past_feat, map_feat], dim=1)
        # print("feat_in:", feat_in.shape)
        # feat_in = feat_in.unsqueeze(1)
        # print("feat_in:", feat_in.shape)
        # assert feat_in.dim() == 3, "[TNT-TPSelector]: Error input feature dimension"
        # print("tar_candidate:", tar_candidate.shape) #64, 2485, 2
        # print("tp_candidate:", tp_candidate.shape)  # 64, 2485, 2
        n, _ = tp_candidate.size()

        # stack the target candidates to the end of input feature
        # print(feat_in.shape, tp_candidate.shape)
        # feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tp_candidate], dim=2)
        feat_in_repeat = torch.cat([feat_in, tp_candidate], dim=-1)
        # print("feat_in_repeat:", feat_in_repeat.shape)  # 64, 2485, 66
        # compute probability for each candidate
        prob_tensor = self.prob_mlp(feat_in_repeat).squeeze(1)
        # print("prob_tensor:", prob_tensor.shape)
        # print("prob_tensor:", prob_tensor)

        if not isinstance(candidate_mask, torch.Tensor):
            tar_candit_prob = F.softmax(prob_tensor, dim=-1)  # [batch_size, n_tar, 1]
        else:
            tar_candit_prob = masked_softmax(
                prob_tensor, candidate_mask, dim=-1
            )  # [batch_size, n_tar, 1]
        # print(final_h.shape, final_h) #1 64 1
        # print("feat_in_repeat:", feat_in_repeat.shape) #64, 2485, 66
        # print("prob_tensor:", prob_tensor.shape) #64, 2485
        # print("tar_candit_prob:", tar_candit_prob.shape, tar_candit_prob[0][0]) #64, 2485
        # print("tar_offset_mean:", tar_offset_mean.shape, tar_offset_mean[0][0]) #64, 2485, 2
        # print("yaw_pred:", yaw_pred.shape, yaw_pred) #64, 2485, 1
        return tar_candit_prob

    def inference(
        self, feat_in: torch.Tensor, tar_candidate: torch.Tensor, candidate_mask=None
    ):
        """
        output only the M predicted propablity of the predicted target
        :param feat_in:        the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate:  tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask: the mask of valid target candidate
        :return:
        """
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in: the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask:
        :return:
        """
        return self.forward(feat_in, tar_candidate, candidate_mask)
