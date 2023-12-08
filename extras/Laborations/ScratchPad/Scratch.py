# %%
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

RANDOM_SEED = 42
n_samples = 10000

# Generate test data
X, y = make_moons(n_samples,
                  noise=0.08,
                  random_state=42)
print(X[:10])
print(y[:10])
# Print untrained data
plt.figure(figsize=(20, 10))
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlGn)


# %%
X.dtype, y.dtype

# %%
# Convert to correct dtype (float32)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X.dtype, y.dtype

# %%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=RANDOM_SEED)

# %%
# Define learning model


class MoonModelBinary(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_layers):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=out_feat)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

# %%


def MoonTrainingLoopBinary(X_train,
                           X_test,
                           y_train,
                           y_test,
                           learning_model,
                           epochs,
                           learning_rate,
                           interactive=False):
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(learning_model.parameters(), lr=learning_rate)
    acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

    for epoch in range(epochs):
        learning_model.train()

        y_logits = learning_model(X_train).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train)
        acc = acc_fn(y_preds, y_train.int())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        learning_model.eval()
        with torch.inference_mode():
            # 1. Forward pass (to get the logits)
            test_logits = learning_model(X_test).squeeze()
            # Turn the test logits into prediction labels
            test_preds = torch.round(torch.sigmoid(test_logits))
            # 2. Caculate the test loss/acc
            test_loss = loss_fn(test_logits, y_test)
            test_acc = acc_fn(test_preds, y_test.int())

        if epoch % (epochs/10) == 0 and interactive == True:
            print(
                f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")

# %%


def plot_decision_boundary(model, X, y):

    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(
        np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlGn, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlGn)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# %%
# Create the learning model (Neural network)
model = MoonModelBinary(in_feat=2, out_feat=1, hidden_layers=10)
# Train the model
MoonTrainingLoopBinary(X_train=X_train,
                       X_test=X_test,
                       y_train=y_train,
                       y_test=y_test,
                       learning_model=model,
                       epochs=1000,
                       learning_rate=0.1,
                       interactive=True)

# %%
# Display results
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
