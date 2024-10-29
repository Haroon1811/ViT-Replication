# Plotting loss curves :
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
def plot_loss_curves(results: Dict[str, List[float]]):
  """Plot training curves of a results dictionary"""
  # Get the training and testing loss values from the results dictionary
  train_loss = results["train_loss"]
  test_loss = results["test_loss"]

  # Get the values of accuracy from the results dictionary(train and test)
  train_accuracy = results["train_accuracy"]
  test_accuracy = results["test_accuracy"]

  # Figure out the number of epochs :
  epochs = range(len(results["train_loss"]))

  # Setup a plot
  plt.figure(figsize=(18,9))

  # PLot the loss
  plt.subplot(1,2,1)       # one row, two columns, first index
  plt.plot(epochs, train_loss, label="Train_loss")
  plt.plot(epochs, test_loss, label="Test_loss")
  plt.title("LOSS")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()


  # PLot the accuracy
  plt.subplot(1,2,2)     # one row, two columns, second index
  plt.plot(epochs, train_accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="Test_accuracy")
  plt.title("ACCURACY")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend();
