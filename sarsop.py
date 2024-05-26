from model import create_scenario
from collections import defaultdict, deque

class BeliefNode:
    def __init__(self, belief):
        self.belief = belief
        self.children = defaultdict(dict) # action to ObservationNode
        self.parent_observation = -1
        self.parent = None

# class ObservationNode:
#     def __init__(self, parent_action):
#         self.children = {} # observation to BeliefNode
#         self.parent = None
#         self.parent_action = parent_action

class Alphavector:
    
    def __init__(self, values, action):
        self.values = values
        self.action = action

    def get_product(self, belief):
        val = 0
        for state, prob in belief.items():
            val += self.values[state] * prob
        return val

class SARSOP:
    def __init__(self, pomdp, gamma = 0.95):
        self.bins = [] # for predicting the value of belief
        self.pomdp = pomdp
        self.gamma = gamma
        
    def get_lower_bound(self, belief, vectors):
        L = float("inf")
        for vector in vectors:
            val = 0
            for state, prob in belief.items():
                val += vector.values[state] * prob
            L = min(L, val)
        return L

    def get_upper_bound(self, belief, vectors):
        U = float("-inf")
        for vector in vectors:
            val = 0
            for state, prob in belief.items():
                val += vector.values[state] * prob
            U = max(U, val)
        return U

    def get_reward_belief_action(self, belief, actionIndex):
        reward = 0
        for state, prob in belief.items():
            reward += prob * self.pomdp.state_action_reward_map[state][actionIndex]
        return reward
    
    def expand_belief_action(self, belief, actionIndex):
        """Given a belief, an action, explore all possible observations and updated beliefs"""
        print("_belief_", belief, actionIndex)
        obs = defaultdict(dict)
        for state, prob in belief.items():
            for nxt_state, nxt_prob in self.pomdp.robot_state_action_map[state][actionIndex].items():
                observation = self.pomdp.state_observation_map[nxt_state]
                obs[observation][nxt_state] = obs[observation].get(nxt_state, 0) + prob * nxt_prob
        print(obs)
        return obs

    def predict(self, belief, V_upper_dash, V_lower_dash): #TODO
        return V_upper_dash

    def sample(self, root, vectors, epi = 0):
        L = self.get_lower_bound(root.belief, vectors)
        U = L + epi
        self.samplepoints(root, vectors, root.belief, L, U, epi, 1)

    
    def samplepoints(self, root, vectors, belief, L, U, epi, t):
        gamma = self.gamma
        print(self.pomdp.end_states, "endstates")
        V_lower_dash = self.get_lower_bound(belief, vectors)
        V_upper_dash = self.get_upper_bound(belief, vectors)
        V_hat = self.predict(belief, V_upper_dash, V_lower_dash) #TODO

        for state in belief:
            if state in self.pomdp.end_states:
                print(state, self.pomdp.end_states)
                return
            
        if V_hat <= L and V_upper_dash <=  max(U, V_lower_dash + epi * pow(gamma, -t)):
            return
        if t > 15:
            print(t)
            return
        Q_lo = float("-inf")
        Q_hi = float("-inf")
        a_prime = -1

        for actionIndex, action in enumerate(self.pomdp.actions):
            Q_ba_lo = Q_ba_hi = self.get_reward_belief_action(belief, actionIndex)

            obs = self.expand_belief_action(belief, actionIndex)

            for observation in obs:
                total_prob = 0
                for nxt_state in obs[observation]:
                    prob = obs[observation][nxt_state]
                    total_prob += prob
                for nxt_state in obs[observation]:
                    obs[observation][nxt_state] /= total_prob

                nxt_belief = obs[observation]

                Q_ba_lo += gamma * total_prob * self.get_lower_bound(nxt_belief, vectors)
                Q_ba_hi += gamma * total_prob * self.get_upper_bound(nxt_belief, vectors)

            Q_lo = max(Q_lo, Q_ba_lo)
            if Q_ba_hi > Q_hi:
                Q_hi = max(Q_hi, Q_ba_hi)
                a_prime = actionIndex

        L_prime = max(L, Q_lo)

        U_prime = max(U, Q_lo, + epi * pow(gamma, -t))

        # compute o prime
        o_prime = -1
        diff_max = float("-inf")
        obs = self.expand_belief_action(belief, a_prime)

        for observation in obs:
            total_prob = 0
            for nxt_state in obs[observation]:
                prob = obs[observation][nxt_state]
                total_prob += prob
            for nxt_state in obs[observation]:
                obs[observation][nxt_state] /= total_prob
            nxt_belief = obs[observation]
            diff = total_prob * (self.get_upper_bound(nxt_belief, vectors) - self.get_lower_bound(nxt_belief, vectors))
            if diff > diff_max:
                diff_max = max(diff_max, diff)
                o_prime = observation

        r = self.get_reward_belief_action(belief, a_prime)
        L_prime = (L_prime - r) / gamma
        U_prime = (U_prime - r) / gamma
        
        belief_prime = {}
        obs = self.expand_belief_action(belief, a_prime)
        p = 1
        # print("o_prime", o_prime,  a_prime)
        print(obs,)
        for observation in obs:
            print("o", observation)
            total_prob = 0
            for nxt_state in obs[observation]:
                prob = obs[observation][nxt_state]
                total_prob += prob
            for nxt_state in obs[observation]:
                obs[observation][nxt_state] /= total_prob
                nxt_belief = obs[observation]
            
            if observation != o_prime:
                L_prime -= total_prob * self.get_lower_bound(nxt_belief, vectors)
                U_prime -= total_prob * self.get_upper_bound(nxt_belief, vectors)
            else:
                p = total_prob      
                belief_prime = nxt_belief          

        L_prime /= p
        U_prime /= p

        #insert(belief_prime)
        root.children[a_prime][o_prime] = BeliefNode(belief_prime) 

        self.samplepoints(root.children[a_prime][o_prime], vectors, belief_prime, L_prime, U_prime, epi, t + 1)

    def choose(self, root):
        # print(root)
        nodes = []
        queue = deque([(root, 0)])
        while queue:
            node, level = queue.popleft()
            nodes.append(node)
            # print(node.belief, level)
            for ac in node.children:
                for ob in node.children[ac]:
                    nxt = node.children[ac][ob]
                    # print(f"\t {level + 1} | action = {ac} | observation = {ob} |  belief = {nxt.belief}")
                    queue.append((nxt, level + 1))
        return nodes

    def backup(self, root, vectors, node, gamma): #TODO
        belief = node.belief

        alpha_a_o = {}
        for actionIndex, action in enumerate(self.pomdp.actions):
            obs = self.expand_belief_action(belief, actionIndex)
            for observation in obs:
                total_prob = 0
                for nxt_state in obs[observation]:
                    prob = obs[observation][nxt_state]
                    total_prob += prob
                for nxt_state in obs[observation]:
                    obs[observation][nxt_state] /= total_prob
                nxt_belief = obs[observation]

                v = float("-inf")
                
                for vector in vectors:
                    val = vector.get_product(nxt_belief)
                    if val > v:
                        v = max(val, v)
                        alpha_a_o[(actionIndex, observation)] = vector
        
        alpha_a = defaultdict(dict)
        for actionIndex, action in enumerate(self.pomdp.actions):
            for state in self.pomdp.robot_nodes:
                v = self.pomdp.state_action_reward_map[state]
                for nxt_state, nxt_state_prob in self.pomdp.robot_state_action_map[state][actionIndex].items():
                    nxt_observation = self.pomdp.state_observation_map[nxt_state]
                    v += gamma * nxt_state_prob * alpha_a_o.get((actionIndex, nxt_observation), 0)
                if v > 0:
                    alpha_a[actionIndex][state] = v
        
        best_vector = None
        best_val = float("-inf")
        for actionIndex in alpha_a:
            val = 0
            for state, prob in belief.items():
                val += alpha_a[actionIndex].get(state, 0) * prob
            if val > best_val:
                best_vector = alpha_a[actionIndex]
                
        if best_vector:
            vectors.add(best_vector)
        

    def prune(self, root, vectors): #TODO
        return

    def solve(self):
        gamma = self.gamma
        vectors = set()
        states = self.pomdp.robot_nodes
        actions = self.pomdp.actions
        init_val_lo = float("-inf")
        init_val_hi = float("inf")
        for action in actions:
            for val in [init_val_hi, init_val_lo]:
                values = {state:val for state in states}
                alpha = Alphavector(values, action)
                vectors.add(alpha)
        initial_belief = self.pomdp.initial_belief # {state: prob}
        root = BeliefNode(initial_belief)
        
        while True:
            self.sample(root, vectors)

            nodes = self.choose(root)
            for node in nodes:
                self.backup(root, vectors, node, gamma)

            self.prune(root, vectors)

            termination = True #TODO what is termination condition
            if termination:
                break
        return vectors

def main():
    pomdp = create_scenario("ETH")
    # print(pomdp.initial_belief)
    s = SARSOP(pomdp)
    s.solve()
    return

if __name__ == "__main__":
    main()