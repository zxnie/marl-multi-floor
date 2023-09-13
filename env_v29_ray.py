import copy as cp

import gym
import networkx as nx
import numpy as np
from scipy.special import softmax
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# import matplotlib.pyplot as plt


class GridWorldAGV(MultiAgentEnv):
    def __init__(self, config):
        super(GridWorldAGV, self).__init__()
        self._rng = np.random.default_rng(config['seed'])
        '''
        Basic simulation variables
        '''
        # Operation
        self.num_agent = config['num_agent']
        self.t_global = 0  # Global time
        self.max_ep = config['max_ep']
        self.ep_count = {}
        # Map
        self.dim_x = config['dim_x']
        self.dim_y = config['dim_y']
        self.n_floors = config['n_floors']
        self.weight_horizontal = 1
        self.weight_vertical = 2
        self.weight_crossfloor = 3
        self.edge_sets = {'left': [], 'top': [], 'right': [], 'bottom': []}
        self.node_mapping, self.map_origin = self.generate_map_nfloors()
        '''
        Tasks: {[source, destination, due time], ...}
        '''
        self.t_task = {}
        self.task_pool = []
        self.task_pool_total = {}
        self.task_pool_record = []
        self.reconf_timestep = config['reconf_timestep']
        self.task_pool_reset_ep = {}
        self.task_margin = 0.20
        self.tasks_finished = {}  # [src, dst, shortest_time_delivery, shortest_time_pickup, due_time_pickup, due_time_deliver, due_time, time_used_pickup, time_used]
        self.paths = {}
        self.paths_last = {}
        self.performance = {}

        '''
        RL
        '''
        # Agent state
        # 0: current node, 1: task source, 2: task destination, 3: task remaining time
        # 4: operation flag(1: pickup, 2: deliver), 5: intersection conflict
        self.agents = {}
        self._agent_ids = set([str(i) for i in range(config["num_agent"])])
        self.agent_states = {}
        self.dones = set()

        num_nodes = self.n_floors * self.dim_x * self.dim_y
        self.observation_space = gym.spaces.Box(low=0, high=num_nodes, shape=(10,))
        self.action_space = gym.spaces.Discrete(4)

    def get_state_local(self, i):
        current_node = self.agents[i][0]
        current_loc = self.map_origin.nodes[current_node]['name']
        if self.agents[i][4] == 1:
            # Pickup
            target_node = self.agents[i][1]
        elif self.agents[i][4] == 2:
            # deliver
            target_node = self.agents[i][2]
        target_loc = self.map_origin.nodes[target_node]['name']
        state_local = np.zeros(5)
        if current_loc[1] < target_loc[1]:
            state_local[1] = 1
        if current_loc[1] > target_loc[1]:
            state_local[3] = 1
        if current_loc[2] < target_loc[2]:
            state_local[2] = 1
        if current_loc[1] > target_loc[1]:
            state_local[0] = 1
        if current_loc[0] != target_loc[0]:  # cross layer
            state_local[4] = 1
        return state_local

    def _get_obs(self):
        obs = {}
        for i in self._agent_ids:
            obs[i] = np.zeros(10)
            obs[i][:3] = self.agents[i][:3]
            obs[i][3] = self.tasks_finished[i][-1][6]
            obs[i][4] = self.agents[i][3]
            obs[i][5:] = self.get_state_local(i)
        return obs

    def _get_info(self):
        info = {}
        delay_list = []
        energy_list = []
        for i in self._agent_ids:
            delay_list += self.performance[i]['delay']
            energy_list += self.performance[i]['energy']
        for i in self._agent_ids:
            info[i] = {
                'delay_avg': np.mean(delay_list),
                'energy_avg': np.mean(energy_list),
            }
        return info

    def get_agent_ids(self):
        return self._agent_ids

    def reset(self):
        # Reset variables
        self.t_global = 0  # Global time
        self.ep_count = {}
        self.t_task = {}
        self.task_pool = []
        self.task_pool_record = []
        self.tasks_finished = {}
        self.paths = {}
        self.paths_last = {}
        self.performance = {}
        self.agents = {}
        self.dones = set()
        # Initialize Map
        ## Random blocks, not implemented yet
        # Initialize tasks
        self.init_task_pool()
        self.reset_task_pool()
        # Initialize Agents & load 1st task
        for i in self._agent_ids:
            self.ep_count[i] = 0
            self.agents[i] = np.zeros(6)  # internal state
            init_node = self._rng.integers(0, self.dim_x * self.dim_y - 1)
            self.agents[i][0] = init_node  # current location
            task = self.task_assignment(i)
            self.t_task[i] = 0
            self.agents[i][1:3] = task[:2]  # src, dst, due time
            self.agents[i][3] = task[4]
            self.tasks_finished[i] = [cp.deepcopy(task)]
            if task[0] == init_node:
                self.agents[i][4] = 2
                self.tasks_finished[i][-1].append(0)
            else:
                self.agents[i][4] = 1
            self.task_pool_reset_ep[i] = []

            self.paths[i] = [init_node]
            self.performance[i] = {
                'delay': [],
                'energy': [],
            }
        obs = self._get_obs()
        return obs

    def generate_map_nfloors(self):
        nodemapping = np.zeros((self.n_floors, self.dim_x, self.dim_y), dtype=int)
        for i in range(self.n_floors):
            for j in range(self.dim_y):
                nodemapping[i, :, j] = np.arange(j * self.dim_x, j * self.dim_x + self.dim_x) + i * (self.dim_x * self.dim_y)
        # for i in range(2):
        #     for j in range(self.dim_x):
        #         for k in range(self.dim_y):
        #             print(nodemapping[i, j, k])
        G = nx.Graph()
        for i in range(self.n_floors):
            for j in range(self.dim_x):
                for k in range(self.dim_y):
                    G.add_node(nodemapping[i, j, k], name=(i, j, k))
                    # print(nodemapping[i, j, k], 'name', (i, j, k))
        num_total = self.n_floors * self.dim_x * self.dim_y
        for i in range(num_total):
            for j in range(num_total):
                loc_i = G.nodes[i]["name"]
                loc_j = G.nodes[j]["name"]
                adj = self.adjacency_nfloors(i, j, loc_i, loc_j)
                if adj[0]:
                    if j not in G[i]:
                        G.add_edge(i, j, weight=adj[1])
        # nx.draw(G, pos=nx.spring_layout(G, iterations=1000), with_labels=True)
        # plt.show()
        return nodemapping, G

    def adjacency_nfloors(self, i, j, loc_i, loc_j):
        adj = (False, False)
        if loc_i == loc_j:
            adj = (False, False)
        else:
            if loc_i[0] == loc_j[0]:  # same floor
                if loc_i[1] == loc_j[1]:
                    if loc_i[2] == loc_j[2] + 1:
                        adj = (True, self.weight_horizontal)
                    elif loc_i[2] == loc_j[2] - 1:
                        adj = (True, self.weight_horizontal)
                elif loc_i[2] == loc_j[2]:
                    if loc_i[1] == loc_j[1] + 1:
                        adj = (True, self.weight_vertical)
                    elif loc_i[1] == loc_j[1] - 1:
                        adj = (True, self.weight_vertical)
            else:  # cross floor
                if loc_i[1] in (0, self.dim_x - 1) or loc_i[2] in (0, self.dim_y - 1):
                    if loc_j[1] in (0, self.dim_x - 1) or loc_j[2] in (0, self.dim_y - 1):
                        if loc_i[1] == loc_j[1] and loc_i[2] == loc_j[2]:
                            if loc_i[0] in (loc_j[0] - 1, loc_j[0] + 1):
                                adj = (True, self.weight_crossfloor)
                                if loc_i[2] == 0:
                                    self.edge_sets['left'].append((i, j))
                                elif loc_i[2] == self.dim_y - 1:
                                    self.edge_sets['right'].append((i, j))
                                elif loc_i[1] == 0:
                                    self.edge_sets['top'].append((i, j))
                                elif loc_i[1] == self.dim_x - 1:
                                    self.edge_sets['bottom'].append((i, j))
        return adj

    def reset_task_pool(self):
        pool_size = self.num_agent * 2
        num_per_item = int(np.ceil(pool_size / len(self.task_pool_item_time)))
        self.task_pool = []
        self.task_pool_record = [0] * pool_size
        ## Task pool with 2 times of agent number
        while len(self.task_pool) < pool_size:
            for t in self.task_pool_item_time:
                task_pool_list = self.task_pool_total[t]
                src_dst_list = self._rng.choice(task_pool_list, num_per_item)
                for src_dst in src_dst_list:
                    if len(self.task_pool) >= pool_size:
                        break
                    task = [src_dst[0], src_dst[1], int(t)]
                    self.task_pool.append(task)

    def init_task_pool(self):
        task_pool_total = {}
        for i in list(self.map_origin.nodes()):
            for j in list(self.map_origin.nodes()):
                time = nx.astar_path_length(self.map_origin, i, j, weight='1')
                if time in task_pool_total:
                    if (i, j) not in task_pool_total[time]:
                        task_pool_total[time].append((i, j))
                    if (j, i) not in task_pool_total[time]:
                        task_pool_total[time].append((j, i))
                else:
                    task_pool_total[time] = [(i, j), (j, i)]
        self.task_pool_total = task_pool_total
        # pool_item_time = []
        # for t, src_dst in self.task_pool_total.items():
        #     if t > np.ceil(self.size / 2) and len(src_dst) >= 100:
        #         pool_item_time.append(t)
        self.task_pool_item_time = [6]

    def task_assignment(self, i):
        reverse_record = np.max(self.task_pool_record) - np.array(self.task_pool_record)
        prob = softmax(reverse_record)
        task = list(self._rng.choice(self.task_pool, 1, p=prob)[0])
        t_deliver = task[2]
        # revise due time according to agent location
        t_pickup = nx.astar_path_length(self.map_origin, self.agents[i][0], task[0], weight='1')
        task.append(t_pickup)  # shortest_t_deliver
        task.append(int(np.ceil(t_pickup * (1 + self.task_margin))))  # due_t_pickup
        task.append(int(np.ceil(t_deliver * (1 + self.task_margin))))  # due_t_deliver
        task.append(int(np.ceil((t_pickup + task[2]) * (1 + self.task_margin))))  # due_t
        return task

    def get_node_location(self, n):
        if isinstance(n, tuple):
            location = (None, n[0], n[1])
        else:
            location = self.map_origin.nodes[n]['name']
        if location[1] == 0:
            if location[2] == 0:
                loc = 'left', 'top'
            elif location[2] == self.dim_y - 1:
                loc = 'right', 'top'
            else:
                loc = 'middle', 'top'
        elif location[1] == self.dim_x - 1:
            if location[2] == 0:
                loc = 'left', 'bottom'
            elif location[2] == self.dim_y - 1:
                loc = 'right', 'bottom'
            else:
                loc = 'middle', 'bottom'
        else:
            if location[2] == 0:
                loc = 'left', 'middle'
            elif location[2] == self.dim_y - 1:
                loc = 'right', 'middle'
            else:
                loc = 'middle', 'middle'
        return loc

    def get_nodes_at_sides(self, src_loc, dst_loc, side):
        # Does not include the side nodes in the same floor with src or dst
        result_nodes = []
        for n in range(self.n_floors * self.dim_x * self.dim_y):
            loc = self.map_origin.nodes[n]['name']
            if loc[0] in (src_loc[0], dst_loc[0]):  # the floors that will be reserved
                continue
            elif loc[0] > src_loc[0] and loc[0] < dst_loc[0]:
                # reserve between src and dst (src < dst, going up floor)
                loc1 = self.get_node_location(n)
                if side == loc1[0] or side == loc1[1]:
                    result_nodes.append(n)
            elif loc[0] < src_loc[0] and loc[0] > dst_loc[0]:
                # reserve between src and dst (src > dst, going down floor)
                loc1 = self.get_node_location(n)
                if side == loc1[0] or side == loc1[1]:
                    result_nodes.append(n)
            else:
                continue
        return result_nodes

    def take1floor(self, n_floor):
        Map = cp.deepcopy(self.map_origin)
        n_floors = self.n_floors
        node_mapping = self.node_mapping
        remove_nodes = []
        for i in range(n_floors):
            if i != n_floor:
                remove_nodes += list(node_mapping[i, :, :].flatten())
        Map.remove_nodes_from(remove_nodes)
        # nx.draw(Map, pos=nx.spring_layout(Map), with_labels=True)
        # plt.show()
        return Map

    def unfold_map(self, src_loc, dst_loc):
        map_set = []
        for side in ['left', 'top', 'right', 'bottom']:
            Map = cp.deepcopy(self.map_origin)
            # first remove nodes
            remove_nodes = []
            reserve_nodes = set(self.get_nodes_at_sides(src_loc, dst_loc, side))
            for i in range(self.n_floors):
                if i not in (src_loc[0], dst_loc[0]):
                    remove_list = set(self.node_mapping[i, :, :].flatten())
                    remove_nodes += list(remove_list - reserve_nodes)
            Map.remove_nodes_from(remove_nodes)
            # then remove edges (when src and dst are two adjacent floors)
            edges_remove = []
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    loc1 = self.get_node_location((x, y))
                    if side not in loc1:
                        edges_remove.append((self.node_mapping[src_loc[0], x, y], self.node_mapping[dst_loc[0], x, y]))
            Map.remove_edges_from(edges_remove)
            map_set.append(Map)
            # nx.draw(Map, pos=nx.spring_layout(Map, iterations=500), with_labels=True)
            # plt.show()
        return map_set

    def get_valid_actions_nfloors(self, node):
        # Walls and random blocks, do not consider other AGVs
        ## Walls only now, to consider blocks, replace to self.map_current
        neighbors = self.map_origin.neighbors(node)
        valid_actions = [4]
        for n in neighbors:
            a = self.path2action_nfloors(node, n)
            if a is False:
                continue
            else:
                valid_actions.append(a)
        return valid_actions

    def path2action_nfloors(self, src, dst):
        src_loc = self.map_origin.nodes[src]['name']
        dst_loc = self.map_origin.nodes[dst]['name']
        src_loc1 = self.get_node_location(src)
        dst_loc1 = self.get_node_location(dst)
        a = False
        if src_loc[0] == dst_loc[0]:  # same floor
            if src_loc[1] == dst_loc[1]:  # left or right
                if src_loc[2] - 1 == dst_loc[2]:
                    a = 0  # left
                elif src_loc[2] + 1 == dst_loc[2]:
                    a = 2  # right
            elif src_loc[2] == dst_loc[2]:  # up or down
                if src_loc[1] + 1 == dst_loc[1]:
                    a = 1  # down
                elif src_loc[1] - 1 == dst_loc[1]:
                    a = 3  # up
        else:  # cross floor
            if src_loc1 == dst_loc1 == ('middle', 'middle'):
                print('Error: only nodes at sides can cross floor.')
            else:
                if src_loc[1] == dst_loc[1] and src_loc[2] == dst_loc[2]:  # aligned
                    if src_loc[0] == dst_loc[0] - 1:
                        a = 5  # up floor
                    elif src_loc[0] == dst_loc[0] + 1:
                        a = 6  # down floor
                    else:
                        print('Error: cross more than 1 floor.')
                else:
                    print('Error: 2 nodes are not vertically aligned.')
        return a

    def ideal_next_node_nfloors(self, current_node, a):
        if a == 4:  # next_loc = current_loc
            return current_node
        else:
            current_loc = list(cp.copy(self.map_origin.nodes[current_node]['name']))
            next_loc = list(cp.copy(current_loc))
            current_loc1 = self.get_node_location(current_node)
            if a in (0, 1, 2, 3):  # same floor action
                next_loc = self.ideal_next_node_same_floor(current_loc, a)
            else:  # cross floor action
                if current_loc1[0] != ('middle', 'middle'):
                    if a == 5:
                        next_loc[0] += 1
                    elif a == 6:
                        next_loc[0] -= 1
                    else:
                        print('Error: action out of action space (0, 1, 2, 3, 4, 5, 6)')
                else:
                    print('Error: only nodes at sides can cross floor.')
                if next_loc[0] < 0 or next_loc[0] >= self.n_floors:
                    print('Error: target floor does not exist.')
            next_node = int(self.node_mapping[tuple(next_loc)])  # Ideal next node number
            return next_node

    def ideal_next_node_same_floor(self, current_loc, a):
        next_loc = cp.copy(current_loc)
        if a == 0:  # left
            next_loc = current_loc
            next_loc[2] -= 1
        elif a == 1:  # down
            next_loc = current_loc
            next_loc[1] += 1
        elif a == 2:  # right
            next_loc = current_loc
            next_loc[2] += 1
        elif a == 3:  # up
            next_loc = current_loc
            next_loc[1] -= 1
        if next_loc[1] < 0:
            next_loc = False
        elif next_loc[1] >= self.dim_x:
            next_loc = False
        if next_loc[2] < 0:
            next_loc = False
        elif next_loc[2] >= self.dim_y:
            next_loc = False
        return next_loc

    def cal_reward_local(self, action_dict_next, action_dict, RL_agents):
        agent_map, rewards_local, auctioneers_dict = {}, {}, {}
        # Gathering information
        for i in self._agent_ids:
            if i in self.dones:
                continue
            node = self.agents[i][0]
            if node in agent_map:
                agent_map[node].append(i)
            else:
                agent_map[node] = [i]
            rewards_local[i] = 0
        for node, agents in agent_map.items():
            # Basic -1 reward for potential conflict
            # if len(agents) >= 2:
            #     for agent in agents:
            #         rewards_local[agent] -= 1
            # -1 reward for conflict of 2 agents
            if len(agents) == 2:
                if action_dict_next[agents[0]] == action_dict_next[agents[1]]:
                    auctioneers_dict[node] = agents
                    for agent in agents:
                        rewards_local[agent] -= 1
            if len(agents) == 3:
                actions_node = []
                for i in agents:
                    actions_node.append(action_dict_next[i])
                if len(set(actions_node)) != 3:  # have conflict?
                    auctioneers_dict[node] = agents  # all need be in auction
                if action_dict_next[agents[0]] == action_dict_next[agents[1]] == action_dict_next[agents[2]]:
                    rewards_local[agents[0]] -= 1
                    rewards_local[agents[1]] -= 1
                    rewards_local[agents[2]] -= 1
                elif action_dict_next[agents[0]] == action_dict_next[agents[1]]:
                    rewards_local[agents[0]] -= 1
                    rewards_local[agents[1]] -= 1
                elif action_dict_next[agents[0]] == action_dict_next[agents[2]]:
                    rewards_local[agents[0]] -= 1
                    rewards_local[agents[2]] -= 1
                elif action_dict_next[agents[1]] == action_dict_next[agents[2]]:
                    rewards_local[agents[1]] -= 1
                    rewards_local[agents[2]] -= 1
            # More than 3 agents, things will get complicated even they do not conflict!
            if len(agents) > 3:
                auctioneers_dict[node] = agents
                for agent in agents:
                    rewards_local[agent] -= 1

        # Revise the actions with local games
        action_dict_next_revised, utility_dict = self.local_game(
            action_dict_next,
            action_dict,
            auctioneers_dict,
            RL_agents
        )

        return rewards_local, action_dict_next_revised, auctioneers_dict, utility_dict

    def local_game(self, action_dict_next, action_dict, auctioneers_dict, RL_agents):
        utility_dict = {}
        if len(auctioneers_dict) > 0:
            for location, agents in auctioneers_dict.items():
                auctioneers = agents
                valid_actions = self.get_valid_actions_nfloors(location)
                action_dict_next_revision, utilities = self.auction_alter(
                    auctioneers,
                    RL_agents,
                    action_dict_next,
                    action_dict,
                    valid_actions,
                )
                for i in auctioneers:
                    action_dict_next[i] = action_dict_next_revision[i]
                    utility_dict[i] = utilities[i]
        return action_dict_next, utility_dict

    def auction_alter(self, auctioneers, RL_agents, action_dict_next, action_dict, valid_actions):
        token_dict = {}
        value_matrices = {}
        available_resources = cp.copy(valid_actions)
        for i in auctioneers:
            token_dict[i] = RL_agents[i].token
            if i in action_dict:
                action_last = action_dict[i]
            else:
                action_last = 4
            value_matrix = RL_agents[i].actor_planning_auction(action_dict_next[i], action_last, self.map_origin, self.agents[i])
            value_matrices[i] = value_matrix
        assignment = {}
        auctioneers_ordered = dict(sorted(token_dict.items(), key=lambda item: item[1], reverse=True))
        for auctioneer, token in auctioneers_ordered.items():
            action_preference = np.flip(np.argsort(value_matrices[auctioneer]))
            for a in action_preference:
                if a in available_resources:
                    assignment[auctioneer] = a
                    available_resources.remove(a)
                    break
        action_revision = {}
        utilities = {}
        for i in auctioneers:
            if i not in assignment:  # don't have assignment, stay
                action = 4
                cost = 0
            else:
                action = assignment[i]
                cost = value_matrices[i][action]
                if cost <= 1:  # assigned bad action, stay
                    action = 4
                    cost = 0
            # Token exchange
            for j in auctioneers:
                if i == j:
                    RL_agents[j].token -= cost
                else:
                    RL_agents[j].token += (cost / (len(auctioneers) - 1))
            action_revision[i] = action
            utilities[i] = value_matrices[i][action]
        # print(auctioneers, 'in an auction')
        # print('Actions after auction', actions)
        return action_revision, utilities

    def remove_duplicate(self, input_list):
        output = []
        for e in input_list:
            if e not in output:
                output.append(e)
            elif e != output[-1]:
                output.append(e)
        return output

    def reset_ep(self, i):
        '''
        Join a task and its next task
        :param i: agent id
        :return:
        '''
        self.performance[i]['delay'].append(self.t_task[i])
        path = self.remove_duplicate(self.paths[i])
        self.performance[i]['energy'].append(nx.path_weight(self.map_origin, path, 'weight'))
        if self.ep_count[i] == self.max_ep:
            done = True
            return done
        else:
            done = False
        # Check out current task
        if self.tasks_finished[i][-1][:3] in self.task_pool:
            task_pool_ind = self.task_pool.index(list(self.tasks_finished[i][-1][:3]))
            self.task_pool_record[task_pool_ind] += 1
        self.tasks_finished[i][-1].append(self.t_task[i])
        # print('Task finished', self.tasks_finished[i][-1])
        # print('Path', self.paths[i])
        self.ep_count[i] += 1
        # Reset Agent state
        init_node = int(self.agents[i][0])
        self.agents[i] = np.zeros(6)
        self.agents[i][0] = init_node
        task = self.task_assignment(i)
        self.t_task[i] = 0
        self.agents[i][1:3] = task[:2]
        self.agents[i][3] = task[4]
        self.tasks_finished[i].append(cp.deepcopy(task))
        if task[0] == init_node:
            self.agents[i][4] = 2
            self.tasks_finished[i][-1].append(0)
        else:
            self.agents[i][4] = 1

        self.paths_last[i] = cp.deepcopy(self.paths[i])
        self.paths[i] = [init_node]
        return done

    def transition(self, i, action):
        # Movement (change indices of 0, 3)
        valid_actions = self.get_valid_actions_nfloors(self.agents[i][0])
        # print(f'Agent-{i}, action {action}')
        if type(action) == np.ndarray:
            action = int(np.argmax(action))
        if action in valid_actions:
            next_node = self.ideal_next_node_nfloors(self.agents[i][0], action)
        else:
            next_node = self.agents[i][0]
        self.agents[i][0] = next_node
        self.paths[i].append(int(next_node))
        if self.agents[i][3] > 0:
            self.agents[i][3] -= 1
        self.t_task[i] += 1

        # Task (change indices of 1,2,3,4)
        src = self.agents[i][1]
        dst = self.agents[i][2]
        done_ep = False
        ## done task?
        if (src in self.paths[i]) and (self.paths[i][-1] == dst):
            done_ep = True
        elif self.t_task[i] >= self.max_ep:
            done_ep = True
            self.tasks_finished[i][-1].append(self.t_task[i])
        # done pickup?
        elif self.paths[i][-1] == src:
            if self.agents[i][4] == 1:
                self.agents[i][4] = 2
                self.agents[i][3] = self.tasks_finished[i][-1][5]
        if done_ep:
            done = self.reset_ep(i)
        else:
            done = False
        return done_ep, done

    def cal_reward(self, i, done_ep, ob_old):
        if done_ep:
            if (self.tasks_finished[i][-2][8] - self.tasks_finished[i][-2][7]) <= self.tasks_finished[i][-2][5]:
                reward = 10  # in time
            else:
                reward = 5
        elif (ob_old[4] == 1) and (self.agents[i][4] == 2):
            self.tasks_finished[i][-1].append(self.t_task[i])
            if self.tasks_finished[i][-1][7] <= self.tasks_finished[i][-1][4]:
                reward = 10  # in time
            else:
                reward = 5
        else:
            reward = 0
        # Bonus for delay optimization
        distance_min_old = nx.astar_path_length(self.map_origin, ob_old[0], ob_old[int(ob_old[4])])
        distance_min_new = nx.astar_path_length(self.map_origin, self.agents[i][0],
                                                self.agents[i][int(self.agents[i][4])])
        if distance_min_new < distance_min_old:
            reward += 0.1
        # Bonus for energy optimization
        # node_last = int(ob_old[0])
        # node_now = int(self.agents[i][0])
        # if node_last == node_now:
        #     weight = 0
        # else:
        #     weight = self.map_origin.edges[node_last, node_now]['weight']
        # if weight < 2:
        #     reward += 0.5
        return reward

    def step(self, action_dict):
        rewards_global, dones_ep, dones = {}, {}, {}
        for i, action in action_dict.items():
            ob_old = np.copy(self.agents[i])
            dones_ep[i], dones[i] = self.transition(i, action)
            rewards_global[i] = self.cal_reward(i, dones_ep[i], ob_old)
            if dones[i]:
                self.dones.add(i)

        dones["__all__"] = len(self.dones) == len(self.agents)
        self.t_global += 1

        # Reconfigurations
        if self.t_global in self.reconf_timestep:
            self.reset_task_pool()
            for i in self._agent_ids:
                self.task_pool_reset_ep[i].append(self.ep_count[i])
        info = self._get_info()
        return self._get_obs(), rewards_global, dones, info

    def close(self):
        return
