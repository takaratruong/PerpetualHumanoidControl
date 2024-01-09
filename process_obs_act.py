import numpy as np 
import ipdb

# gt = [8788, 8788, 7606, 7606, 7170, 7135, 7065, 6933, 3944, 3921, 3921, 3763,
#         3720, 3700, 3689, 3009, 2983, 2890, 2832, 2751, 2689, 2669, 2537, 2514,
#         2500, 2500, 2480, 2453, 2451, 2445, 2403, 2318, 2256, 2240, 2231, 2225,
#         2153, 1982, 1982, 1928, 1893, 1833, 1787, 1768, 1476, 1476, 1432, 1293,
#         1293, 1138]

if __name__ == '__main__':
    data = np.load('raw_data.npz')

    observations = data['obs']
    actions = data['act']
    ep_lens = data['ep_len']

    num_envs = observations.shape[0]

    concatenated_observations = []
    concatenated_actions = []
    episode_boundaries = []

    # Process each environment
    for env_index in range(num_envs):
        # ipdb.set_trace()
        ep_len = ep_lens[env_index] - 1
        # print(ep_len)
        
        concatenated_observations.extend(observations[env_index][:ep_len])
        # ipdb.set_trace()
        concatenated_actions.extend(actions[env_index][:ep_len])
        
        episode_boundary = len(concatenated_observations)-1 
        episode_boundaries.append(episode_boundary)
    # Convert the concatenated lists to numpy arrays
    concatenated_observations = np.array(concatenated_observations)
    concatenated_actions = np.array(concatenated_actions)
    episode_boundaries = np.array(episode_boundaries)

    # ipdb.set_trace()
    np.save('states.npy', concatenated_observations)
    np.save('actions.npy', concatenated_actions)
    np.save('ep_ends.npy', episode_boundaries)

    # print(concatenated_observations.shape)
    # print(concatenated_actions.shape)
    # print(episode_boundaries)
