##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch.nn as nn
import unittest

### based on DistanceNetwork, whose output [sequence_length, num_classes]
class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):

        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        ### softmax_similarities.unsqueeze(1) -> [sequence_length, 1, batch_size]
        ### .bmm(support_set_y) -> [sequence_length, 1, num_classes]
        ### .squeeze() -> [sequence_length, num_classes]
        return preds

class AttentionalClassifyTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_forward(self):
        pass

if __name__ == '__main__':

    unittest.main()
