import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    3-block CNN for binary knee OA classification.

    Named access points:
      model.conv1 / conv2 / conv3     individual convolutional blocks
      model.gradcam_layer             ReLU after last conv (GradCAM hook target)
      model.latent(x)                 128-dim feature vector before dropout/classifier
      BaselineCNN.load(path, device)  load weights, return eval-mode model

    Hook support:
      model.attach_hooks(['conv1', 'conv2', 'conv3'])  capture forward activations
      model.activations                                 dict of captured outputs
      model.remove_hooks()                              clean up
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128×128 → 64×64
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64×64 → 32×32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # index 2 — GradCAM target
            nn.MaxPool2d(2),  # 32×32 → 16×16
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 16 * 16, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, 1)

        self.activations: dict = {}
        self._hooks: list = []

    @property
    def gradcam_layer(self):
        """ReLU after the last conv block — recommended GradCAM hook target."""
        return self.conv3[2]

    def latent(self, x):
        """128-dim feature vector before dropout and classifier (latent space input)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.relu(self.fc(self.flatten(x)))

    def forward(self, x):
        return self.classifier(self.dropout(self.latent(x)))

    def attach_hooks(self, layer_names=None):
        """
        Register forward hooks to capture output activations of named layers.

        layer_names: list of attribute names to hook, e.g. ['conv1', 'conv2', 'conv3'].
                     Defaults to all three conv blocks.

        After a forward pass, captured tensors are in model.activations[name].
        Call remove_hooks() when done to avoid memory leaks.
        """
        if layer_names is None:
            layer_names = ["conv1", "conv2", "conv3"]

        self.remove_hooks()  # clear any existing hooks first

        for name in layer_names:
            layer = getattr(self, name)
            hook = layer.register_forward_hook(
                lambda *args, n=name: self.activations.update({n: args[2].detach()})
            )
            self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered forward hooks and clear captured activations."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.activations.clear()

    @classmethod
    def load(cls, path, device, dropout_rate=0.5):
        """Load from a .pth file and return an eval-mode model."""
        model = cls(dropout_rate=dropout_rate).to(device)

        sd = torch.load(path, map_location=device)

        # legacy key remapping for weights saved before the conv1/conv2/conv3 rename
        # (conv_block / fc_block → conv1 / conv2 / conv3 / fc / classifier)
        # keep commented out; uncomment if loading pre-rename .pth files
        # legacy = {
        #     "conv_block.0": "conv1.0",
        #     "conv_block.1": "conv1.1",
        #     "conv_block.4": "conv2.0",
        #     "conv_block.5": "conv2.1",
        #     "conv_block.8": "conv3.0",
        #     "conv_block.9": "conv3.1",
        #     "fc_block.1": "fc",
        #     "fc_block.4": "classifier",
        # }
        # remapped = {}
        # for k, v in sd.items():
        #     for old, new in legacy.items():
        #         if k.startswith(old):
        #             k = k.replace(old, new, 1)
        #             break
        #     remapped[k] = v
        # sd = remapped

        model.load_state_dict(sd)
        model.eval()
        return model
