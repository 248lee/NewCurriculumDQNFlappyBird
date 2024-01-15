from deep_q_network import trainNetwork

# Config your training process here!
stage = 1
num_of_actions = 2
lockmode = 0
is_simple_actions_locked = False
is_activate_boss_memory = False
max_steps = 100000
is_sweet_boss = True
learning_rate = 3e-6
# End Config

def read_last_old_time():
    # Read the integer from the file 'last_old_time.txt'
    try:
        with open('last_old_time.txt', 'r') as file:
            return int(file.read())
    except (FileNotFoundError, ValueError):
        return 0

last_steps = read_last_old_time()
training_param_history_file = open('training_history.txt', 'a')
training_param_history_file.write(f"LAST STEPS:\t{last_steps}-----------------------------\n")
training_param_history_file.write(f"stage:\t{stage}\nnum of actions:\t{num_of_actions}\nlock mode:\t{lockmode}\nis simple action unlock:\t{is_simple_actions_locked}\nis activate boss memory:\t{is_activate_boss_memory}\nis sweet boss:\t{is_sweet_boss}\nlearning rate:\t{learning_rate}\n")
training_param_history_file.write('-----------------------------')
training_param_history_file.close()
trainNetwork(stage, num_of_actions, lockmode, is_simple_actions_locked, is_activate_boss_memory, max_steps, is_sweet_boss, resume_Adam=False, learning_rate=learning_rate, event=None, is_colab=True)

