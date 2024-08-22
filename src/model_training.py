from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.regularizers import l2

def get_untrained_custom_model(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],x_train=0):
    model = Sequential([
    Dense(6, activation='relu', input_shape=(x_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    # Dense(32, activation='relu'),
    # Dense(32, activation='relu'),
    # Dense(64, activation='relu'),
    # Dropout(0.25),
    # Dense(128, activation='relu'),
    # Dense(64, activation='relu'),

    # Dropout(0.5),
    Dense(1, activation='sigmoid')
])
    model.compile(optimizer=optimizer, loss=loss , metrics=metrics)
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model



def train_model(model, x_train, y_train, epochs, x_val, y_val, batch_size):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    return history



def plot_training_history(history):
    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    # Saving the plots
    plt.savefig('../results/training_validation_loss_and_accuracy.png')
    
    # Display the plot
    plt.show()

