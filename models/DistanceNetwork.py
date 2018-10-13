##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import unittest

class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):

        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            ### support_image shape [batch_size, 64]
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            ### sum_support shape [batch_size] (64-dim features shrink into one point per sample, totally batch_size)
            ### DOUBT: directly add them together along feature dim, will this lose some info?
            support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            ### .rsqrt() -> return the reciprocal of the square-root of each of the elements of input.
            ### support_magnitude shape [batch_size]
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            ### [batch_size, 64] -> [batch_size, 1, 64] * [batch_size, 64, 1] -> [batch_size, 1] -> [batch_size]
            cosine_similarity = dot_product * support_magnitude
            ### [batch_size] * [batch_size] = [batch_size]
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        ### DOUBT the shape of returned tensor
        ### [sequence_length, batch_size]
        return similarities

class DistanceNetworkTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward(self):
        pass

if __name__ == '__main__':
    unittest.main()
