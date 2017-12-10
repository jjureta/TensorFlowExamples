import matplotlib.pyplot as plt

# display an image from the given set
def display_digit(data_set, num, prediction = -1):
    label = data_set.labels[num].argmax(axis=0)
    image = data_set.images[num].reshape([28,28])
    plt.title('Example: %d  Label: %d Prediction: %d' % (num, label, prediction))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()