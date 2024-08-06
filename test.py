import gymnasium as gym
import numpy as np
import pickle

def q_table_line(state, Cart_Position, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity):
    Cart_Position_dqn = np.digitize(state[0], Cart_Position)
    Cart_Velocity_dqn = np.digitize(state[1], Cart_Velocity)
    Pole_Angle_dqn = np.digitize(state[2], Pole_Angle)
    Pole_Angular_Velocity_dqn = np.digitize(state[3], Pole_Angular_Velocity)

    return Cart_Position_dqn, Cart_Velocity_dqn, Pole_Angle_dqn, Pole_Angular_Velocity_dqn

f = open('cartpole.pkl', 'rb')
q_table = pickle.load(f)
f.close()

env = gym.make('CartPole-v1', render_mode="human")
Cart_Position = np.linspace(-4.8, 4.8, 10)  # Between -4.8 and 4.8	
Cart_Velocity = np.linspace(-3.40, 3.40, 10) # Between -3.40 and 3.40
Pole_Angle = np.linspace(-0.418, 0.418, 10)  # Between -0.418 ana 0.418
Pole_Angular_Velocity = np.linspace(-3.40, 3.40, 10) # Between -3.40 and 3.40

for i in range(1000):
    now_state = env.reset()[0]  # Reset environment and get initial state
    done = False  # Flag to check if the episode is finished
    step = 0
    # Play one episode
    while not done and step < 10000 :
        q_Cart_Position, q_Cart_Velocity, q_Pole_Angle, q_Pole_Angular_Velocity = q_table_line(now_state, Cart_Position, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity)

        action = np.argmax(q_table[q_Cart_Position, q_Cart_Velocity, q_Pole_Angle, q_Pole_Angular_Velocity, :])
        step += 1
        # Take action and observe result
        new_state, reward, done, truncated, _ = env.step(action)
        
        now_state = new_state

    print(step)
    
