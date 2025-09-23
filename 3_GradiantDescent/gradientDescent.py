import numpy as np

def gradiant_descent(x, y):
    m_curr = b_curr = 0
    iterations= 10000
    n= len(x)
    learning_rate= 0.001
    epsilon=1e-6 #this is used to stop if the cost is not improving significantly which means if it increase we should stop
    prev_cost = float('inf') #initialise with a random large number
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr #our example is about the MSE od the simple linear regression
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)]) #one calcule le MSE
        
         # Stop if the cost is not improving significantly (if it increase)
        if abs(prev_cost - cost) < epsilon:
            print(f"Converged at iteration {i}")
            break
        prev_cost = cost
        
        m_deriv = -(2/n)*sum(x*(y-y_predicted)) #la dérivé de MSE par apport a m
        b_deriv = -(2/n)*sum(y-y_predicted)
        
        m_curr = m_curr - learning_rate * m_deriv #this is the rule of the algo
        b_curr = b_curr - learning_rate * b_deriv
        print("m {}, b {},cost {}, iteration {}".format(m_curr, b_curr,cost, i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradiant_descent(x, y)