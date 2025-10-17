# settings
no_gui = True
enable_cuda = True
sumo_config = "./env/env-ny/ny/osm.sumocfg.xml"
end = None
interval = 90
verbose = True
time_limit = 10
idle_time = 180                  # ablation (0)
save_interval = 100
version = "default"
ncav = 3
iteration = 1

x_min = 348.01
y_min = 0.00
x_max = 4469.70
y_max = 7477.67

# opt hyperparameters
cost_type = "time"
waiting_time = 300              # ablation (1)
travel_delay = 420              # ablation (2)

# encoder hyperparameters
Kt = 2
Ks = 1
mean = 98.123
std = 124.659
cluster_num = 13
n_hist = 12                     # ablation (3)
n_pred = 1                      # ablation (4)
data_config = (384, 48, 48)
adj_dataset = "data/adj.npz"
blocks = [[1, 32, 64], [64, 32, 128]]
act_func = 'glu'
c_passengers = 3

lr = 1e-2
weight_detach = 1e-3
step_size = 10

# decoder hyperparameters
obs_dim = 21
act_dim = 2
lr_actor = 1e-3
lr_critic = 1e-3
gamma = 0.99
tau = 0.005
policy_noise = 5
noise_clip = 5
policy_delay = 2
replay_size = 10
batch_size = 4
eps_decay_strategy = 'cosine'

# dqn hyperparameters
dqn_act_dim = 13
dqn_lr = 1e-3
dqn_update_interval = 10

# qmix hyperparameters
n_agents = 5
state_dim = 21

epochs = 100