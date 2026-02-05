import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, s):
        return self.net(s)

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Outputs Q(s, a) for all a
        )
        
    def forward(self, s):
        return self.net(s)

def train_prefix_rl(env, d_off, iterations=100, n_samples=32, eta=0.1):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 1. Initialize policy and critic
    pi = PolicyNet(state_dim, action_dim)
    critic = CriticNet(state_dim, action_dim)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    # Store policies for the final mixture return
    policy_history = []

    for t in range(iterations):
        batch_states, batch_actions, batch_rewards = [], [], []
        
        # 4. Collection loop
        for _ in range(n_samples):
            # 5. Sample prefixed problem from D_off
            # d_off is assumed to be a list of (state, action) pairs
            idx = np.random.randint(len(d_off))
            s_h, a_off_h = d_off[idx]
            
            # 6. Action selection with 50/50 probability
            if np.random.rand() < 0.5:
                a_h = a_off_h
            else:
                s_tensor = torch.FloatTensor(s_h).unsqueeze(0)
                probs = pi(s_tensor)
                a_h = torch.multinomial(probs, 1).item()
            
            # 7. Execute rollout from step h+1
            # Note: This requires an env that supports setting state (reset_to_state)
            total_reward = rollout(env, pi, s_h, a_h)
            
            batch_states.append(s_h)
            batch_actions.append(a_h)
            batch_rewards.append(total_reward)

        # 9. Critic Fit (Regression Oracle)
        states_t = torch.FloatTensor(np.array(batch_states))
        actions_t = torch.LongTensor(np.array(batch_actions)).unsqueeze(1)
        rewards_t = torch.FloatTensor(np.array(batch_rewards)).unsqueeze(1)
        
        # Standard MSE Loss: (Q(s,a) - r)^2
        q_values = critic(states_t).gather(1, actions_t)
        critic_loss = nn.MSELoss()(q_values, rewards_t)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # 11. Natural Policy Update (Mirror Ascent)
        # We update the policy weights to match the Mirror Ascent derivation
        with torch.no_grad():
            full_q_values = critic(states_t) # Q(s, .) for all actions
            old_probs = pi(states_t)
            
            # The closed form update for KL-constrained Mirror Ascent:
            # pi_new = pi_old * exp(eta * Q) / Z
            new_probs = old_probs * torch.exp(eta * full_q_values)
            new_probs /= new_probs.sum(dim=-1, keepdim=True)
            
        # Update policy network via a simple supervised step to match new_probs
        update_policy_weights(pi, states_t, new_probs)
        
        policy_history.append(pi.state_dict())
        print(f"Iteration {t}: Critic Loss {critic_loss.item():.4f}")

    return policy_history

def rollout(env, policy, start_state, first_action):
    """Mocks a rollout from a specific state and action."""
    obs = env.reset_to_state(start_state) # Custom env method
    obs, reward, done, _ = env.step(first_action)
    total_r = reward
    
    while not done:
        with torch.no_grad():
            a = torch.multinomial(policy(torch.FloatTensor(obs).unsqueeze(0)), 1).item()
        obs, reward, done, _ = env.step(a)
        total_r += reward
    return total_r

def update_policy_weights(pi, states, target_probs):
    """Helper to update policy network to match the mirror ascent target."""
    optimizer = optim.Adam(pi.parameters(), lr=1e-3)
    for _ in range(10): # Small inner loop to fit the new distribution
        current_probs = pi(states)
        loss = nn.KLDivLoss(reduction='batchmean')(current_probs.log(), target_probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



"""
Algorithm 1 PrefixRL with Natural Policy Gradients
Require: Base policy π0, off-policy data Doff , horizon H, iterations T , step size η, Q function class F .
1: Initialize the iterative algorithm with base policy: π1 ← π0.
2: for t = 1,...,T do
3: Initialize dataset Dt ← {}.
4: for i = 1...n do
5: Sample (sh,aoffh ) uniformly across state-action pairs in Doff . ▷ sample prefixed problem
6: ah ← aoffh with probability 1/2 and ∼ πt(· | sh) otherwise.
7: Execute πt(· | sh,ah) from step h+1 through H to obtain the full trace with reward r.
8: Dt ← Dt ∪(sh,ah,r).
9: Critic fit (regression oracle):
10: ˆQt ← argminf ∈FP(s,a,r)∈Dt (f (s,a)−r)2.
11: Natural policy update (mirror ascent): ▷ performed state-wise
12: πt+1(· | s) ← argminp ⟨− ˆQt(s,·),p⟩+ 1η KL(p∥πt(· | s)).
13: end for
14: end for
15: return  ̄πT ← 1TPT
t=1πt. ▷ return mixture policy
"""