import torch
import torch.nn as nn


class BestModel(nn.Module):
    """
    Dynamically built CNN for binary knee OA classification.

    Named access points:
      model.conv1 / conv2 / … / convN    individual convolutional blocks
      model.gradcam_layer                 ReLU after last conv (GradCAM hook target)
      model.latent(x)                     128-dim feature vector before dropout/classifier
      Best_Model.load(path, device, ...)  load weights, return eval-mode model

    Hook support:
      model.attach_hooks(['conv1', 'conv2', ...])  capture forward activations
      model.activations                             dict of captured outputs
      model.remove_hooks()                          clean up
    """

    def __init__(
        self,
        num_conv_layers=3,
        base_filters=64,
        kernel_size=5,
        dropout_rate=0.3,
        use_1x1=True,
    ):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        padding = kernel_size // 2

        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = base_filters * (2**i)
            setattr(
                self,
                f"conv{i + 1}",
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ),
            )
            in_channels = out_channels

        # Optional 1×1 projection to 2048 channels (matches ResNet50 feature dimensions)
        if use_1x1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, 2048, kernel_size=1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
            )
            fc_in = 2048
        else:
            self.proj = None
            fc_in = in_channels

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_in, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, 1)

        self.activations: dict = {}
        self._hooks: list = []

    @property
    def gradcam_layer(self):
        """ReLU after the last conv block — recommended GradCAM hook target."""
        return getattr(self, f"conv{self.num_conv_layers}")[2]

    def latent(self, x):
        """128-dim feature vector before dropout and classifier (latent space input)."""
        for i in range(1, self.num_conv_layers + 1):
            x = getattr(self, f"conv{i}")(x)
        if self.proj is not None:
            x = self.proj(x)
        x = self.pool(x)
        return self.relu(self.fc(self.flatten(x)))

    def forward(self, x):
        return self.classifier(self.dropout(self.latent(x)))

    def attach_hooks(self, layer_names=None):
        """
        Register forward hooks to capture output activations of named layers.

        layer_names: list of attribute names, e.g. ['conv1', 'conv2'].
                     Defaults to all conv blocks.

        After a forward pass, captured tensors are in model.activations[name].
        Call remove_hooks() when done to avoid memory leaks.
        """
        if layer_names is None:
            layer_names = [f"conv{i + 1}" for i in range(self.num_conv_layers)]
        self.remove_hooks()
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
    def load(
        cls,
        path,
        device,
        num_conv_layers=3,
        base_filters=64,
        kernel_size=5,
        dropout_rate=0.3,
        use_1x1=True,
    ):
        """Load from a .pth file and return an eval-mode model."""
        model = cls(
            num_conv_layers, base_filters, kernel_size, dropout_rate, use_1x1
        ).to(device)
        sd = torch.load(path, map_location=device)

        # remap old conv_block / fc_block keys to the new named-block structure
        if any(k.startswith("conv_block.") or k.startswith("fc_block.") for k in sd):
            mapping = {}
            for i in range(num_conv_layers):
                base_idx = i * 4
                mapping[f"conv_block.{base_idx}"] = f"conv{i + 1}.0"  # Conv2d
                mapping[f"conv_block.{base_idx + 1}"] = f"conv{i + 1}.1"  # BatchNorm2d
            mapping["fc_block.1"] = "fc"
            mapping["fc_block.4"] = "classifier"
            remapped = {}
            for k, v in sd.items():
                new_k = k
                for old, new in mapping.items():
                    if k.startswith(old + "."):
                        new_k = new + k[len(old) :]
                        break
                remapped[new_k] = v
            sd = remapped

        model.load_state_dict(sd)
        model.eval()
        return model


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
