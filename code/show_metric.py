import json
import os
import matplotlib.pyplot as plt
import argparse

def main(config):
    path = 'output/' + config.name    

    files = os.listdir(path)
    
    for f in files:
        if f.endswith('.json'):

            with open(os.path.join(path,f), 'r') as fp:
                d = json.load(fp)

            acc = d['accuracy']
            test_acc = d['test_accuracy']
            loss = d['loss']
            test_loss = d['test_loss']

            max_test_acc = max(test_acc)
            max_test_acc_ind = test_acc.index(max_test_acc)+1

            min_test_loss = min(test_loss)
            min_test_loss_ind = test_loss.index(min_test_loss)+1

            plt.subplot(1,2,1)
            plt.title('Min Test Loss:{:.3f} at epoch {}'.format(min_test_loss, min_test_loss_ind))
            plt.plot(range(len(loss)), loss)
            plt.plot(range(len(test_loss)), test_loss)
            plt.legend(['Training Loss', 'Test Loss'])

            plt.subplot(1,2,2)
            plt.title('Max Test Accuracy:{:.3f} at epoch {}'.format(max_test_acc, max_test_acc_ind))
            plt.plot(range(len(acc)), acc)
            plt.plot(range(len(test_acc)), test_acc)
            plt.legend(['Training Accuracy', 'Test Accuracy'])
            plt.show()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default="debug", help="Name of the run")

    config = parser.parse_args()

    # Train the model
    main(config)