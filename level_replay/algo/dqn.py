from __future__ import division

import math

import torch
from torch import nn
from torch.nn import functional as F


def apply_init_(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)
        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ImpalaCNN(nn.Module):
    """
    Arguments:
    ----------
    num_inputs : `int`
        Number of channels in the input image.
    """

    def __init__(self, num_inputs, channels=[16, 32, 32]):  # noqa: B006
        super(ImpalaCNN, self).__init__()

        # define Impala CNN
        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.flatten = Flatten()
        self.relu = nn.ReLU()

        # initialise all conv modules
        apply_init_(self.modules())

        # put model into train mode
        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = list()

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(ResidualBlock(out_channels))
        layers.append(ResidualBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        return x


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class RainbowDQN(nn.Module):
    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        self.features = ImpalaCNN(args.state_dim[0])
        self.conv_output_size = 2048
        self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        value = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        advantage = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        value, advantage = value.view(-1, 1, self.atoms), advantage.view(-1, self.action_space, self.atoms)
        q = value + advantage - advantage.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension

        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if "fc" in name:
                module.reset_noise()


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.action_space = action_space

        self.features = ImpalaCNN(args.state_dim[0])
        if args.state_dim != (3, 64, 64):
            example_state = torch.randn((1,) + args.state_dim)
            self.conv_output_size = self.features(example_state).shape[1]
        else:
            self.conv_output_size = 2048
        self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc_h_a = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc_z_v = nn.Linear(args.hidden_size, 1)
        self.fc_z_a = nn.Linear(args.hidden_size, action_space)

    def forward(self, x, log=False):
        x = self.features(x)

        x = x.view(-1, self.conv_output_size)
        value = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        advantage = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        value, advantage = (
            value.view(
                -1,
                1,
            ),
            advantage.view(-1, self.action_space),
        )
        q = value + advantage - advantage.mean(1, keepdim=True)  # Combine streams
        return q


class TwoNetworkDQN(nn.Module):
    def __init__(self, args, action_space):
        super(TwoNetworkDQN, self).__init__()
        self.action_space = action_space

        self.value_features = ImpalaCNN(args.state_dim[0])
        self.advantage_features = ImpalaCNN(args.state_dim[0])
        self.conv_output_size = 2048
        self.value_fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
        self.advantage_fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
        self.value_fc2 = nn.Linear(args.hidden_size, 1)
        self.advantage_fc2 = nn.Linear(args.hidden_size, action_space)

    def forward(self, x, log=False):
        value_x = self.value_features(x)
        advantage_x = self.advantage_features(x)
        value = self.value_fc2(F.relu(self.value_fc1(value_x)))
        advantage = self.advantage_fc2(F.relu(self.advantage_fc2(advantage_x)))
        value, advantage = (
            value.view(
                -1,
                1,
            ),
            advantage.view(-1, self.action_space),
        )
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q
