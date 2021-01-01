import numpy as np
import matplotlib.pyplot as plt

# Read 
loss_history_full_file = open("training/loss_history_full.txt", "r")
loss_history_full = loss_history_full_file.read().split('\n')
loss_history_full.pop()
new_loss_history_full = []
for item in loss_history_full:
    new_loss_history_full.append(float(item))

loss_history_last_file = open("training/loss_history_last.txt", "r")
loss_history_last = loss_history_last_file.read().split('\n')
loss_history_last.pop()
new_loss_history_last = []
for item in loss_history_last:
    new_loss_history_last.append(float(item))

acc_history_full_file = open("training/acc_history_full.txt", "r")
acc_history_full = acc_history_full_file.read().split('\n')
acc_history_full.pop()
new_acc_history_full = []
for item in acc_history_full:
    new_acc_history_full.append(float(item))

acc_history_last_file = open("training/acc_history_last.txt", "r")
acc_history_last = acc_history_last_file.read().split('\n')
acc_history_last.pop()
new_acc_history_last = []
for item in acc_history_last:
    new_acc_history_last.append(float(item))

# Plot acc
plt.plot(range(len(new_acc_history_full)), new_acc_history_full, 'r+', label='full layer training')
plt.plot(range(len(new_acc_history_last)), new_acc_history_last, 'b+', label ='last layer training')
plt.title('Training accuracy')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('batch')
plt.show()


# Plot loss
plt.plot(range(len(new_loss_history_full)), new_loss_history_full, 'r+', label='full layer training')
plt.plot(range(len(new_loss_history_last)), new_loss_history_last, 'b+', label ='last layer training')
plt.title('Training loss')
plt.legend()
plt.ylabel('loss')
plt.xlabel('batch')
plt.show()
