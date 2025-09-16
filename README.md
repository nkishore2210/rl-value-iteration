# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## POLICY ITERATION ALGORITHM
The policy iteration algorithm is a method for finding the optimal policy in a Markov Decision Process (MDP). It alternates between policy evaluation (finding the value function for a fixed policy) and policy improvement (updating the policy using the new value function).

## VALUE ITERATION ALGORITHM
- Value iteration is a method of computing an optimal MDP policy  and its value.
- It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
- The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

The algorithm is as follows:
1. Initialize the value function `V(s)` arbitrarily for all states `s`.
2. Repeat until convergence:
    - Initialize aaction-value function `Q(s, a)` arbitrarily for all states `s` and actions `a`.
    - For all the states s and all the action a of every state:
        - Update the action-value function `Q(s, a)` using the Bellman equation.
        - Take the value function `V(s)` to be the maximum of `Q(s, a)` over all actions `a`.
        - Check if the maximum difference between `Old V` and `new V` is less than `theta`, where theta is a **small positive number** that determines the **accuracy of estimation**.
3. If the maximum difference between Old V and new V is greater than theta, then
    - Update the value function `V` with the **maximum action-value** from `Q`.
    - Go to **step 2**.
4. The optimal policy can be constructed by taking the **argmax** of the action-value function `Q(s, a)` over all actions `a`.
5. Return the optimal policy and the optimal value function.

## VALUE ITERATION FUNCTION
### Name: KISHORE N
### Register Number: 212222240049
```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, s_prime, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[s_prime] * (1.0 - done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
        pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi
```

## OUTPUT:

### POLICY:
<img width="632" height="169" alt="image" src="https://github.com/user-attachments/assets/a922bd33-66ff-461a-b5b5-def05a2f55a7" />

### SUCCESS RATE:
<img width="733" height="58" alt="image" src="https://github.com/user-attachments/assets/3185b3e0-7e23-46c4-aa80-bba4ad1bc42e" />

### STATE VALUE FUNCTION:
<img width="538" height="132" alt="image" src="https://github.com/user-attachments/assets/af3b4a55-7dd2-40c1-84be-89d91bba9ce7" />

## RESULT:
Thus we successfully developed Python program to find the optimal policy for the given MDP using the value iteration algorithm.
