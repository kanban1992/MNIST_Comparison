import mnist_loader
import network2
import matplotlib.pyplot as plt


N_epochs=5
batch_size=10
eta=3.0

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30,30,30, 10], cost=network2.QuadraticCost)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy= net.SGD(training_data, N_epochs, batch_size, eta,lmbda = 0.0,evaluation_data=validation_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,monitor_training_accuracy=True,monitor_training_cost=True)




#plotting

plt.figure(1)
plt.title("Costfunction of Training-data")
plt.xlabel("epochs")
plt.ylabel("cost function")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,training_cost)
plt.savefig("cost_on_training_data.png")

plt.figure(2)
plt.title("correct classidied numbers (out of 10000)")
plt.xlabel("epochs")
plt.ylabel("evaluation accuracy")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,evaluation_accuracy)
plt.savefig("evaluation_accuracy.png")


