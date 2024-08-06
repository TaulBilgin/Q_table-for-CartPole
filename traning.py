import gymnasium as gym
import numpy as np
import pickle


def q_table_line(state, Cart_Position, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity):
    Cart_Position_dqn = np.digitize(state[0], Cart_Position)
    Cart_Velocity_dqn = np.digitize(state[1], Cart_Velocity)
    Pole_Angle_dqn = np.digitize(state[2], Pole_Angle)
    Pole_Angular_Velocity_dqn = np.digitize(state[3], Pole_Angular_Velocity)

    return Cart_Position_dqn, Cart_Velocity_dqn, Pole_Angle_dqn, Pole_Angular_Velocity_dqn
    

def train():
    env = gym.make('CartPole-v1')

    Cart_Position = np.linspace(-4.8, 4.8, 10)  # Between -4.8 and 4.8	
    Cart_Velocity = np.linspace(-3.40, 3.40, 10) # Between -3.40 and 3.40
    Pole_Angle = np.linspace(-0.418, 0.418, 10)  # Between -0.418 ana 0.418
    Pole_Angular_Velocity = np.linspace(-3.40, 3.40, 10) # Between -3.40 and 3.40

    """
    The code doesn't work like : Cart_Velocity = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
    you must put manual value of env.observation_space.low[0] and env.observation_space.high[0]
    """
    q_table = np.zeros((len(Cart_Position)+1, len(Cart_Velocity)+1, len(Pole_Angle)+1, len(Pole_Angular_Velocity)+1, env.action_space.n))

    gamma = 0.99
    run = 0

    for i in range(1000):
        now_state = env.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0
          # Increment the episode counte

        # Play one episode
        while not done and step < 10000 :
            q_Cart_Position, q_Cart_Velocity, q_Pole_Angle, q_Pole_Angular_Velocity = q_table_line(now_state, Cart_Position, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)
            action = np.argmax(q_table[q_Cart_Position, q_Cart_Velocity, q_Pole_Angle, q_Pole_Angular_Velocity, :])
            step += 1
            # Take action and observe result
            new_state, reward, done, truncated, _ = env.step(action)

            new_q_Cart_Position, new_q_Cart_Velocity, new_q_Pole_Angle, new_q_Pole_Angular_Velocity = q_table_line(new_state, Cart_Position, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)

            # Update Q-value using Bellman equation
            if done:
                q_table[q_Cart_Position, q_Cart_Velocity, q_Pole_Angle, q_Pole_Angular_Velocity, action] = -10
            else:
                q_table[q_Cart_Position, q_Cart_Velocity, q_Pole_Angle, q_Pole_Angular_Velocity, action] = reward + gamma * np.max(q_table[new_q_Cart_Position, new_q_Cart_Velocity, new_q_Pole_Angle, new_q_Pole_Angular_Velocity])
            
            now_state = new_state

        run += 1
        print(step, run)
    return q_table

        
q_table = train()
f = open('cartpole.pkl','wb')
pickle.dump(q_table, f)
f.close()
