import matplotlib.pyplot as plt
import numpy as np

def display_digit(index, image, label, prediction):
    image = image.reshape([28,28])
    title = ['Sample:%d' % (index),
             'Label: %d' % (label),
             'Prediction: %d' % (prediction)]
        
    plt.title('\n'.join(title))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))

# display an image from the given set
def display_errors(images, labels, predictions, grid = -1):
    labels = labels.argmax(axis=1)
    
    errors = np.where(predictions != labels)[0]
    
    number_of_errors = errors.shape[0]
    
    if grid == -1:
        random_error_indx = errors[np.random.randint(errors.size, size=1)[0]]
        prediction = predictions[random_error_indx]
        
        label = labels[random_error_indx]
        image = images[random_error_indx].reshape([28,28])
        plt.title('Example: %d Label: %d Prediction: %d' % (random_error_indx, label, prediction))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    else:
        grid = grid if number_of_errors > grid else number_of_errors
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.25, wspace=2)
        for index in range(1, grid + 1):
            random_error_indx = errors[np.random.randint(errors.size, size=1)[0]]
            
            plt.subplot(2, 4, index)
            display_digit(random_error_indx,
                          images[random_error_indx],
                          labels[random_error_indx],
                          predictions[random_error_indx])

    plt.show()


def unfold_labels(labels):
    return labels.argmax(axis=1)
