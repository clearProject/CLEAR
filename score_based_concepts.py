import torch
from torch import nn as nn
import torch.autograd as autograd


# Define the model
class SimpleScoreNet(nn.Module):
    """
    Define the score model
    """
    def __init__(self, input_size, hidden1, hidden2):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(input_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, input_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x


def sliced_score_estimation(score_net, samples, n_particles=1):
    """
    Estimate the sliced score matching loss
    """
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def train_score_model(attributes, attributes_loader, img_loader):
    """
    Train the score model
    """
    print("Training the score model")
    model = SimpleScoreNet(attributes.shape[1], hidden1=1024, hidden2=1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    att_loader = attributes_loader
    # Train the model
    epochs = 1000
    model = model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for batch in att_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, loss1, loss2 = sliced_score_estimation(model, batch, n_particles=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        for batch in img_loader:
            s, t = batch[0], batch[1]
            s = s.float().cuda()
            optimizer.zero_grad()
            loss, loss1, loss2 = sliced_score_estimation(model, s, n_particles=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss}')
    return model


def sample_points(score_net, eps=0.5, t=10, num_samples=32):
    """
    Sample points from the score-based model.
    The formula for sampling is given by:
    x_t = x_{t-1} + (eps/2) * score(x_{t-1}) + sqrt(eps) * z_t
    :param score_net: The score model
    :param eps: The step size
    :param t: The number of steps
    :param num_samples: The number of samples to generate
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # generate a random gaussian start
    all_samples = []
    for i in range(num_samples):
        sample = torch.randn(1, score_net.linear1.in_features).to(device)
        for _ in range(t):
            sample = sample + (eps / 2) * score_net(sample)
            sample = sample + torch.randn_like(sample) * (eps ** 0.5)
        all_samples.append(sample)
    all_samples = torch.cat(all_samples, dim=0)
    return all_samples


def sample_from_known_points(samples, score_net, eps=1, t=10):
    """
    Sample points from the score-based model. samples are the starting points and are a tensor of shape (num_samples, dim)
    The formula for sampling is given by:
    x_t = x_{t-1} + (eps/2) * score(x_{t-1}) + sqrt(eps) * z_t
    :param score_net: The score model
    :param eps: The step size
    :param t: The number of steps
    :param samples: The starting points
    :return:
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(t):
        samples = samples + (eps / 2) * score_net(samples)
        samples = samples + torch.randn_like(samples) * (eps ** 0.5)
    return samples
