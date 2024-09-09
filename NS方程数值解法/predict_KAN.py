import torch
import torch.nn as nn

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def fit(self, train_a, train_u, epochs, batch_size, learning_rate, checkpoint):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            permutation = torch.randperm(train_a.size(0))
            for i in range(0, train_a.size(0), batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i + batch_size]
                batch_a = train_a[indices]
                batch_u = train_u[indices]

                outputs = self(batch_a)
                loss = criterion(outputs, batch_u)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                torch.save({
                    'model_state_dict': self.state_dict(),
                }, checkpoint)
                with open('training_log.txt', 'a') as f:
                    f.write(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}\n")

    def predict(self, test_a, test_u, checkpoint, slice_indices):
        try:
            checkpoint_data = torch.load(checkpoint, map_location=lambda storage, loc: storage, weights_only=True)
            self.load_state_dict(checkpoint_data['model_state_dict'])
        except Exception as e:
            print(f"Failed to load checkpoint from '{checkpoint}': {e}")
            print("Making predictions based on current model weights.")

        self.eval()
        with torch.no_grad():
            pred_a = self(test_a.reshape(test_a.size(0), -1))
            pred_u = self(test_u.reshape(test_u.size(0), -1))
        return pred_a, pred_u

    def save_predictions(self, pred_a, pred_u, test_a, test_u):
        scipy.io.savemat('predictions.mat', {'pred_a': pred_a.cpu().numpy(), 'pred_u': pred_u.cpu().numpy(), 'test_a': test_a.cpu().numpy(), 'test_u': test_u.cpu().numpy()})
