actiondist = {'not_in_action_set': 4482, 'not_in_humanml3d': 2330, 'kick': 224, 'raise': 589, 'dance': 122, 'throw': 291, 'play': 72, 'jump': 576, 'rotate': 36, 'wave': 245, 'squat': 68, 'stumble': 99, 'lift': 282, 'bend': 293, 'sidestep': 50, 'jog': 202, 'put': 344, 'perform': 43, 'swing': 116, 'lean': 146, 'lower': 98, 'sit': 275, 'run': 360, 'crawl': 87, 'climb': 26, 'balance': 62, 'punch': 28, 'kneel': 46, 'wash': 34}

# get distribution of actions from counts dictionary 
action_counts = list(actiondist.values())

total_actions = sum(actiondist.values())
action_prob_dist = {action: count / total_actions for action, count in actiondist.items()}

# perform (handstand/cartwheels) = 43 counts 
# lets say 1 handstand needs ~10 duplicates  
# then we need 43*10 = 430 motion samples from perform 
# prob of perform is  0.37% 
# so we need 116,260 total in the upsampled dataset  
print(action_prob_dist['perform']  )
print(430 / action_prob_dist['perform']  )

