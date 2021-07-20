import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def load_model(feature_dim, code_length, num_classes, num_prototypes):
    """
    Load CNN model.

    Args
        code_length(int): Hashing code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    resnet34 = models.resnet34(pretrained=True)
    model = Resnet34(resnet34, feature_dim, code_length, num_classes, num_prototypes)

    return model


class Resnet34(nn.Module):
    def __init__(self, origin_model, feature_dim=2000, code_length=64, num_classes=100, num_prototypes=100):
        super(Resnet34, self).__init__()
        # self.dynamic_meta_embedding = dynamic_meta_embedding
        self.feature_dim = feature_dim
        self.code_length = code_length
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        self.features = nn.Sequential(*list(origin_model.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

        self.fc_hallucinator = nn.Linear(feature_dim, num_prototypes)
        self.fc_selector = nn.Linear(feature_dim, feature_dim)
        self.attention = nn.Softmax(dim=1)

        self.hash_layer = nn.Linear(feature_dim, code_length)
        # self.scale = nn.Parameter(torch.FloatTensor(np.ones([1, code_length])), requires_grad=True)
        # self.scale = nn.Parameter(torch.FloatTensor(np.ones(1)), requires_grad=True)

        self.classifier = nn.Linear(code_length, num_classes)
        self.assignments = nn.Softmax(dim=1)

    def forward(self, x, dynamic_meta_embedding, prototypes):
        # generate the feature
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)

        # storing direct feature
        direct_feature = x

        if dynamic_meta_embedding:
            # visual memory: consisted of prototypes, each of which represents a center of one semantic structure.
            # visual_memory = prototypes, sized by [num_prototypes, feature_dim].
            if prototypes.size(0) != self.num_prototypes or prototypes.size(1) != self.feature_dim:
                print(prototypes.size(0))
                print(prototypes.size(1))
                print(prototypes.size(0) != self.num_prototypes)
                print(prototypes.size(1) != self.feature_dim)
                print('prototypes error')
                return

            # computing memory_feature by querying and associating visual memory (prototypes)
            attention = self.fc_hallucinator(x)
            attention = self.attention(attention)
            memory_feature = torch.matmul(attention, prototypes)

            # computing concept selector
            concept_selector = self.fc_selector(x)
            concept_selector = nn.Tanh()(concept_selector)

            # infused feature
            x = direct_feature + concept_selector * memory_feature

            # generate hashing
            x = self.hash_layer(x)
            # hash_codes = nn.Tanh()(self.scale.repeat(x.size(0), 1) * x)
            # hash_codes = nn.Tanh()(self.scale * x)
            hash_codes = nn.Tanh()(x)

            # class assignments
            assignments = self.classifier(hash_codes)
            assignments = self.assignments(assignments)

        else:
            # generate hashing
            x = self.hash_layer(x)
            # hash_codes = nn.Tanh()(self.scale.repeat(x.size(0), 1) * x)
            # hash_codes = nn.Tanh()(self.scale * x)
            hash_codes = nn.Tanh()(x)

            # class assignments
            assignments = self.classifier(hash_codes)
            # assignments = self.assignments(assignments)

        return hash_codes, assignments, direct_feature
