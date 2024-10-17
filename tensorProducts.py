import numpy as np
import random
# Pauli-Z Matrix
sigma_z = np.array([[1, 0],
                    [0, -1]])

I_2 = np.eye(2)

tensor_product = np.kron(sigma_z, I_2)
print(tensor_product)




