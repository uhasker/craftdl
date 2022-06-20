from craftdl import (
    circles_dataset,
    train_test_split,
    LinearClassificationNet,
    plot_labels,
    plot_labels_with_decision_surface,
    plot_losses,
)

if __name__ == "__main__":
    # Obtain the dataset
    inputs, targets = circles_dataset(rs=[1, 3], ns=[50, 50])

    # Create a train-test split
    (inputs_train, targets_train), (inputs_test, targets_test) = train_test_split(
        inputs, targets
    )

    # Plot the train and test datasets
    plot_labels(inputs_train, targets_train)
    plot_labels(inputs_test, targets_test)

    # Create and fit a LinearClassificationNet
    model = LinearClassificationNet(2, [6], 1)
    losses = model.fit(inputs_train, targets_train, 100, lr=1.0)
    print(f"final loss on train set={losses[-1]}")

    # Plot the losses
    plot_losses(losses)

    # Get the accuracy on the test dataset
    accuracy = model.accuracy(inputs_test, targets_test)
    print(f"accuracy on test set={accuracy}")

    # Get the predictions on the test dataset
    predictions_test = model.predict(inputs_test)

    # Show the decision boundary
    x1_grid, x2_grid, predictions_grid = model.predict_grid(-4, 4, -4, 4)
    plot_labels_with_decision_surface(
        inputs_test, targets_test, x1_grid, x2_grid, predictions_grid
    )
