import numpy as np

# Location of for the local and global maxima
b1max1, b2max1 =  2.0, 1.2 # Global maximum
b1max2, b2max2 = -0.5, 2.0 # Local maximum

# # Function to plot the likelihood function
def likelihood_function(b1, b2):
    likelihood = 0.5 * np.exp(-((b1 - b1max1)**2 + (b2 - b2max1)**2)) + 0.4 * np.exp(-((b1 - b1max2)**2 + (b2 - b2max2)**2))
    return likelihood


# Function to plot the likelihood function
# def likelihood_function(b1, b2):
#     likelihood = -((1-b1)**2 + 100*(b2-b1**2)**6)
#     return likelihood

# b1max1, b2max1 = 1,1 
# b1max2, b2max2 = -1,-1


def gradient_ascent(start_point, stepsize, tolX, max_iter):
    
    # Gradient of the likelihood function
    def gradient(b1, b2):
        grad_b1 = -2 * (b1 - b1max1) * np.exp(-((b1 - b1max1)**2 + (b2 - b2max1)**2)) - (b1 - b1max2) * np.exp(-((b1 - b1max2)**2 + (b2 - b2max2)**2))
        grad_b2 = -2 * (b2 - b2max1) * np.exp(-((b1 - b1max1)**2 + (b2 - b2max1)**2)) - (b2 - b2max2) * np.exp(-((b1 - b1max2)**2 + (b2 - b2max2)**2))
        return grad_b1, grad_b2

    b1,b2 = start_point
    points = [(b1, b2)]
    for i in range(max_iter):
        grad_b1, grad_b2 = gradient(b1, b2)
        rmse_grad = np.sqrt(grad_b1**2 + grad_b2**2)
        if rmse_grad < tolX:
            print(f'Converged in {i} iterations. RMSE of gradient at convergence: {rmse_grad:.2e}')
            break
        else:
            b1 += stepsize * grad_b1
            b2 += stepsize * grad_b2
            points.append((b1, b2))
    if i == max_iter - 1:
        print(f'Warning: Maximum number of iterations ({max_iter}) reached.')
    points = np.array(points)
    return points