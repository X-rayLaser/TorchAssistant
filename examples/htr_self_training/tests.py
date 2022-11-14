from unittest import TestCase
import unittest
from models import HybridAttention
from torch import nn
import torch


class AttentionTests(TestCase):
    def test_convolve(self):
        k = 10
        kernel = 5
        conv = nn.Conv1d(1, k, kernel_size=kernel, padding='same')

        v = torch.zeros(5, 20)
        self.assertEqual((5, 10, 20), HybridAttention.convolve(conv, v).shape)

        v1 = torch.randn(1, 20)
        v2 = torch.randn(1, 20)
        a1 = HybridAttention.convolve(conv, v1)
        a2 = HybridAttention.convolve(conv, v2)

        self.assertEqual((1, k, 20), a1.shape)
        self.assertEqual((1, k, 20), a2.shape)

        a = HybridAttention.convolve(conv, torch.stack([v1[0], v2[0]]))
        self.assertEqual((2, k, 20), a.shape)

        self.assertTrue(torch.allclose(a[0], a1))
        self.assertTrue(torch.allclose(a[1], a2))

    def test_repeat_state(self):
        t = torch.tensor([
            [1, 2, 3],
            [10, 20, 30]
        ])
        t = t.unsqueeze(0)
        res = HybridAttention.repeat_state(t, 4)
        expected = torch.tensor([
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[10, 20, 30], [10, 20, 30], [10, 20, 30], [10, 20, 30]],
        ])
        self.assertTrue(torch.allclose(expected, res))
        self.assertEqual((2, 4, 3), res.shape)

    def test_concat_features(self):
        t1 = torch.tensor([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ])

        t2 = torch.tensor([
            [[10, 20], [40, 50]],
            [[70, 80], [100, 110]]
        ])

        v = HybridAttention.concat_features(t1, t2)
        self.assertEqual((2, 2, 5), v.shape)

        expected = torch.tensor([
            [[1, 2, 3, 10, 20], [4, 5, 6, 40, 50]],
            [[7, 8, 9, 70, 80], [10, 11, 12, 100, 110]]
        ])
        self.assertTrue(torch.allclose(expected, v))

    def test_smoothing(self):
        v = torch.tensor([[1, 2, 3], [4, 5, 6]])

        sigmoid_values = torch.sigmoid(v)
        sum1, sum2 = sigmoid_values[0].sum(), sigmoid_values[1].sum()
        expected1 = sigmoid_values[0] / sum1
        expected2 = sigmoid_values[1] / sum2

        expected = torch.stack([expected1, expected2])

        smoothed = HybridAttention.smoothing(v)

        self.assertEqual((2, 3), smoothed.shape)

        self.assertTrue(torch.allclose(expected, smoothed))

    def test_forward(self):
        embedding_size = 128
        decoder_state_dim = 50
        batch_size = 5
        num_embeddings = 16

        attention = HybridAttention(embedding_size=embedding_size,
                                    decoder_state_dim=decoder_state_dim,
                                    hidden_size=32, filters=4, kernel_size=3)

        decoder_state = torch.zeros(1, batch_size, decoder_state_dim)
        embeddings = torch.ones(batch_size, num_embeddings, embedding_size)
        attention_weights = attention.initial_attention_weights(
            batch_size, num_embeddings, device=torch.device("cpu")
        )
        new_weights = attention(decoder_state, embeddings, attention_weights)
        self.assertEqual((batch_size, num_embeddings), new_weights.shape)

        context = attention.compute_context_vectors(new_weights, embeddings)
        self.assertEqual((batch_size, embedding_size), context.shape)


if __name__ == '__main__':
    unittest.main()
