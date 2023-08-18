# Import Statements
import gymnasium as gym
import gym_examples
import random
import torch
from datetime import datetime
import os
import shutil
from tensorboardX import SummaryWriter
from utilities.data_structures.Replay_Buffer import Replay_Buffer
import numpy as np
from nn_builder.pytorch.NN import NN
from torch.optim import Adam
import torch.nn.functional as F
import pprint
import time
from utilities.Utility_Functions import create_actor_distribution

TRAINING_EPISODES_PER_EVAL_EPISODE = 10

class SAC_Discrete(object):
    def __init__(self, config):
        print("Agent SAC-D:")
        pprint.pprint(config)

        self.config = config
        self.hyperparameters = self.config["Hyperparameters"] # Store Common Hyperparameters
        self.hyperparameters_actor = self.hyperparameters["Actor"] # Store Actor Specific Hyperparameters
        self.hyperparameters_critic = self.hyperparameters["Critic"] # Store Critic Specific Hyperparameters
        self.set_random_seeds(config["seed"])
        self.environment = gym.make("gym_examples/Tetris-Binary-v0", width = self.config["board_width"], height = self.config["board_height"], reward_type = self.config["reward"]) # Tetris Enviornment
        self.action_size = int(self.environment.action_space.n)
        self.config["action_size"] = self.action_size     
        self.state_size =  self.environment.reset()[0].size
        self.action_list = []
        self.debug_mode = False

        self.cleared_lines_list = []
        self.episode_rewards_list = []
        self.episode_lengths_list = []     
        self.average_score_required_to_win = float("inf")
        self.rolling_score_window = 100
        self.cleared_lines = 0
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.global_step_number = 0
        self.turn_off_exploration = False

        self.device = "cuda:0" if self.config["use_GPU"] else "cpu"

        gym.logger.set_level(40)  # stops it from printing an unnecessary warning
        self.log_interval = 1000
        self.log_interval_distribution = 10000 

        self.critic_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Critic", override_seed=self.config["seed"] + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters_critic["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters_critic["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
        self.copy_model_over(self.critic_local, self.critic_target)
        self.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters_critic["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config["seed"], device=self.device)

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters_actor["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * self.hyperparameters["entropy_target"] #0.6 #0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters_actor["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        

        # Writer
        timenow = str(datetime.now())#[0:-10]
        timenow = ' ' + timenow[0:19].replace(':', '_')
        writepath = 'SAC-D-masked-10-10-board-multiple-1/SAC-D-masked-10-10-multiple_{}'.format("tetris") + timenow + "_" +str(self.config["trial_number"])+str(self.config["iteration_number"])
        if os.path.exists(writepath): shutil.rmtree(writepath)
        self.writer = SummaryWriter(log_dir=writepath)

    # Setting Seeds
    def set_random_seeds(self, random_seed):
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
    
    # Function to Create Actor and Critic Network
    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]
        if override_seed: seed = override_seed
        else: seed = self.config["seed"]

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}
  
        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)
    
    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
    
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config["seed"])
        self.state, _ = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.cleared_lines = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []

    def track_episodes_data(self):
        """Saves the data from the recent episodes"""
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config["num_episodes_to_run"]
        self.timesteps = self.config["num_timesteps_to_run"]
        start = time.time()
        while (self.episode_number < num_episodes) and (self.global_step_number < self.timesteps) :
            self.reset_game()
            self.step()
            if self.global_step_number % 1000 == 0:
                if self.config.save_model: self.locally_save_policy()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken
    
    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        self.environment._max_episode_steps = 30000
        episode_len = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.action_list.append(self.action)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep: self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
            episode_len += 1
        self.cleared_lines_list.append(self.cleared_lines)
        self.episode_rewards_list.append(self.total_episode_score_so_far)
        self.episode_lengths_list.append(episode_len)

        # Plots
        t = self.global_step_number 
        
        if t > 0 and self.global_step_number % self.log_interval == 0:
            # Reward
            self.writer.add_scalar('timesteps/episode_reward', self.total_episode_score_so_far, t)
            # Lines cleared
            self.writer.add_scalar('timesteps/episode_lines_cleared', self.cleared_lines, t)
            # Episode length
            self.writer.add_scalar('timesteps/episode_length', episode_len, t)

            # Average episodic lines cleared
            avg_lines_cleared = np.mean(self.cleared_lines_list)
            self.writer.add_scalar("timesteps-average/average_episode_lines_cleared", avg_lines_cleared, t)
            # Average episodic reward
            avg_ep_reward = np.mean(self.episode_rewards_list)
            self.writer.add_scalar("timesteps-average/average_episode_reward", avg_ep_reward, t)
            # Average episodic length
            avg_ep_len = np.mean(self.episode_lengths_list)
            self.writer.add_scalar("timesteps-average/average_episode_length", avg_ep_len, t)

            # Training Stability - Variance
            # if len(self.episode_rewards_list) % self.log_interval == 0:
            training_stability = np.var(self.episode_rewards_list[-100:])
            self.writer.add_scalar("timesteps/training_stability", training_stability, t)

        if t > 0 and len(self.episode_rewards_list) % self.log_interval_distribution == 0:
            # if t > 0 and len(self.episode_rewards_list) % 100 == 0:
            rewards_hist = (self.episode_rewards_list)
            self.writer.add_histogram("reward_distribution", rewards_hist, t)

            # Optionally, log action distribution data to TensorBoard every 100 episodes
            # if t > 0 and len(self.episode_rewards_list) % 100 == 0:
            episode_actions = (self.action_list)
            self.writer.add_histogram("action_distribution", episode_actions, t)
        
        # End Plots
        if eval_ep: self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _, self.info = self.environment.step(action)
        self.total_episode_score_so_far += self.reward
        self.cleared_lines += self.info["cleared_lines"]
        if self.hyperparameters["clip_rewards"]: self.reward =  max(min(self.reward, 1.0), -1.0)
    
    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution("DISCRETE", action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        q_values_1 = self.critic_local(state_batch)
        q_values_2 = self.critic_local_2(state_batch)
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            # Log the Q-values as scalar values
            for action_idx in range(40):
                avg_q_value = torch.mean(q_values_1[:, action_idx])
                self.writer.add_scalar(f"Q-Values-1/Action_{action_idx}/Average_Q_Value", avg_q_value, self.global_step_number)
                self.writer.add_scalar(f"Q-Values-1/Action_{action_idx}/Max_Q_Value", torch.max(q_values_1[:, action_idx]), self.global_step_number)
                self.writer.add_scalar(f"Q-Values-1/Action_{action_idx}/Min_Q_Value", torch.min(q_values_1[:, action_idx]), self.global_step_number)

            # Log the Q-values as scalar values
            for action_idx in range(40):
                avg_q_value_2 = torch.mean(q_values_2[:, action_idx])
                self.writer.add_scalar(f"Q-Values-2/Action_{action_idx}/Average_Q_Value", avg_q_value_2, self.global_step_number)
                self.writer.add_scalar(f"Q-Values-2/Action_{action_idx}/Max_Q_Value", torch.max(q_values_2[:, action_idx]), self.global_step_number)
                self.writer.add_scalar(f"Q-Values-2/Action_{action_idx}/Min_Q_Value", torch.min(q_values_2[:, action_idx]), self.global_step_number)

        if self.global_step_number > 0 and len(self.episode_rewards_list) % self.log_interval_distribution == 0:
            # Log the Q-values as histograms
            for action_idx in range(40):
                self.writer.add_histogram(f"Q-Values-1/Action_{action_idx}/Histogram", q_values_1[:, action_idx], self.global_step_number)
            # Log the Q-values as histograms
            for action_idx in range(40):
                self.writer.add_histogram(f"Q-Values-2/Action_{action_idx}/Histogram", q_values_2[:, action_idx], self.global_step_number)
        
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        
        # Assuming log_action_probabilities is a NumPy array of shape (512,)
        action_probabilities_1 = torch.exp(log_action_probabilities)

        # Calculate the entropy
        entropy = -torch.sum(action_probabilities_1 * torch.log(action_probabilities_1))
        # print("entropy", entropy)
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar("rollout/exploration_rate", entropy, self.global_step_number)

        return policy_loss, log_action_probabilities

    def locally_save_policy(self):
        """Saves the policy"""
        # File paths for saving
        critic_local_path = "{}_{}_critic_local_network_10_2.pt".format(str(self.config.trial_number), str(self.config.iteration_number))
        critic_local_2_path = "{}_{}_critic_local_2_network_10_2.pt".format(str(self.config.trial_number), str(self.config.iteration_number))
        critic_target_path = "{}_{}_critic_target_network_10_2.pt".format(str(self.config.trial_number), str(self.config.iteration_number))
        critic_target_2_path = "{}_{}_critic_target_2_local_network_10_2.pt".format(str(self.config.trial_number), str(self.config.iteration_number))
        actor_local_path = "{}_{}_actor_local_network_10_2.pt".format(str(self.config.trial_number), str(self.config.iteration_number))
        
        # Delete existing files if they exist
        for file_path in [critic_local_path, critic_local_2_path, critic_target_path, critic_target_2_path, actor_local_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")
        
        # Save new versions of the model
        torch.save(self.critic_local.state_dict(), critic_local_path)
        torch.save(self.critic_local_2.state_dict(), critic_local_2_path)
        torch.save(self.critic_target.state_dict(), critic_target_path)
        torch.save(self.critic_target_2.state_dict(), critic_target_2_path)
        torch.save(self.actor_local.state_dict(), actor_local_path)
        print("New model versions saved successfully.")
    
    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None: state = self.state
        if eval_ep: action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            _, masks, _ = self.environment.get_next_states()
            action = self.environment.action_space.sample(masks) # masks, sample(mask: ndarray | None = None) → int
            print("Picking random action ", action, masks)
        else: action = self.actor_pick_action(state=state)
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False: action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar('loss/critic_1_loss', qf1_loss, self.global_step_number)
            self.writer.add_scalar('loss/critic_2_loss', qf2_loss, self.global_step_number)

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar('loss/actor_loss', policy_loss, self.global_step_number)
        if self.automatic_entropy_tuning: alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else: alpha_loss = None
        if self.global_step_number > 0 and self.global_step_number % self.log_interval == 0:
            self.writer.add_scalar('loss/alpha_loss', alpha_loss, self.global_step_number)
        self.update_actor_parameters(policy_loss, alpha_loss)

    def sample_experiences(self):
        return  self.memory.sample()

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both critics"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()
    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        #self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]
      
    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Step number", self.global_step_number)
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("Episode number {} ".format(self.episode_number))

        # Added Writer
        print("Cleared Lines {} ".format(self.cleared_lines))
        print("averge_cleared_lines {}".format(len(self.cleared_lines_list)))
        
        print("----------------------------")
         
