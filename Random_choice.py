# randomly select some raw of a matrix. To much datapoints is inuseful for symbolic regression, so we select patial data.

import numpy as np

def random_choice(data,number):
    index = np.arange(0,len(data))
    choice = np.random.choice(index,number)
    return data[choice]