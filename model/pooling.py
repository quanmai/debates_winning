import torch.nn as nn

class Pooling(nn.Module):
    """ Should do pooling layer before the readout
        IDEAS: should we choose only nodes that have high attention score?
    """
    def __init__(self,):
        self.attn = nn.Linear()

    def forward(self, g, t):
        # for earch turn:
        #   + pick top-k highest attention score nodes, remove others
        #   -> new graph representation (node index)
        #   -> gnn
        # pooling be like: attn_score * (feat) -> feat
        # TODO: complete later 
        # node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        # feat = g.nodes[node_id].data[HID_FEAT_NAME]
        pass

class GlobalAttentionPooling(nn.Module):
    r"""Global Attention Pooling from `Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493>`__

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : torch.nn.Module
        A neural network that computes attention scores for each feature.
    feat_nn : torch.nn.Module, optional
        A neural network applied to each feature before combining them with attention
        scores.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import GlobalAttentionPooling
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> gate_nn = th.nn.Linear(5, 1)  # the gate layer that maps node feature to scalar
    >>> gap = GlobalAttentionPooling(gate_nn)  # create a Global Attention Pooling layer

    Case 1: Input a single graph

    >>> gap(g1, g1_node_feats)
    tensor([[0.7410, 0.6032, 0.8111, 0.5942, 0.4762]],
           grad_fn=<SegmentReduceBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats], 0)
    >>>
    >>> gap(batch_g, batch_f)
    tensor([[0.7410, 0.6032, 0.8111, 0.5942, 0.4762],
            [0.2417, 0.2743, 0.5054, 0.7356, 0.6146]],
           grad_fn=<SegmentReduceBackward>)
    Notes
    -----
    See our `GGNN example <https://github.com/dmlc/dgl/tree/master/examples/pytorch/ggnn>`_
    on how to use GatedGraphConv and GlobalAttentionPooling layer to build a Graph Neural
    Networks that can solve Soduku.
    """
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, graph, get_attention=False):
        from dgl import softmax_nodes, sum_nodes
        r"""

        Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input node feature with shape :math:`(N, D)` where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.
        get_attention : bool, optional
            Whether to return the attention values from gate_nn. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers
            to the batch size.
        torch.Tensor, optional
            The attention values of shape :math:`(N, 1)`, where :math:`N` is the number of
            nodes in the graph. This is returned only when :attr:`get_attention` is ``True``.
        """
        feat = graph.ndata['hp']
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            if get_attention:
                return readout, gate
            else:
                return readout