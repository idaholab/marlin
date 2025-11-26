import torch
import torch.nn as nn


class GibbsEnergy(nn.Module):
    def __init__(self, E: torch.Tensor, c0_a: torch.Tensor, c0_b: torch.Tensor):
        """
        Initialize the GibbsEnergy model.

        Args:
            E (torch.Tensor): Energy parameter.
            c0_a (torch.Tensor): Concentration parameter a.
            c0_b (torch.Tensor): Concentration parameter b.
        """
        super(GibbsEnergy, self).__init__()
        self.register_buffer("E", nn.Parameter(E))
        self.register_buffer("c0_a", nn.Parameter(c0_a))
        self.register_buffer("c0_b", nn.Parameter(c0_b))

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the GibbsEnergy model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 2).

        Returns:
            torch.Tensor: Computed Gibbs energy.
        """
        h_eta = x[:, 0]
        c = x[:, 1]

        c_a = c + (1 - h_eta)*(self.c0_a - self.c0_b)
        c_b = c - h_eta*(self.c0_a - self.c0_b)

        return 0.5 * self.E * (h_eta * torch.square(c_a - self.c0_a)
                               + (1-h_eta) * torch.square(c_b - self.c0_b))


def main():
    # Initialize the model with specific values
    G_torch = GibbsEnergy(torch.tensor(
        [2.0]), torch.tensor([0.3]), torch.tensor([0.7]))

    # Set the model to evaluation mode
    G_torch.eval()

    # Sample input tensor
    x = torch.tensor([[0.0, 0.3], [1.0, 0.7]])

    try:
        # Trace the model with the sample input
        scripted_model = torch.jit.trace(G_torch, x)

        # Save the traced model
        scripted_model.save('torch_NN_gibbs_model.pt')
        print("Model saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
