import matplotlib.pyplot as plt

# display an image from the given set
def display_digit(data_set, num, prediction = -1):
    label = data_set.labels[num].argmax(axis=0)
    image = data_set.images[num].reshape([28,28])
    plt.title('Example: %d  Label: %d Prediction: %d' % (num, label, prediction))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

# display an image from the given set
def display_digit_v1(data_set, num, prediction = -1):
    label = data_set[1][num]
    image = data_set[0][num]
    plt.title('Example: %d  Label: %d Prediction: %d' % (num, label, prediction))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

# display an image from the given set
def display_digit_v2(images, labels, num, prediction = -1):
    label = labels[num]
    image = images[num]
    plt.title('Example: %d  Label: %d Prediction: %d' % (num, label, prediction))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def unfold_labels(labels):
    return labels.argmax(axis=1)
