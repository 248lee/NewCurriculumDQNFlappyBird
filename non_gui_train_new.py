import os
if os.path.exists("results.txt"):
    os.remove("results.txt")
if os.path.exists("last_old_time.txt"):
    os.remove("last_old_time.txt")
if os.path.exists("model/FlappyBird.h5"):
    os.remove("model/FlappyBird.h5")
if os.path.exists("scores_training.txt"):
    os.remove("scores_training.txt")
for i in range(3):
    if os.path.exists("Qvalues/Q"+str(i)+".txt"):
        os.remove("Qvalues/Q"+str(i)+".txt")
now_stage_file = open('now_stage.txt', 'w')
now_stage_file.write("1")
now_stage_file.close()
now_num_of_ac_file = open('now_num_of_actions.txt', 'w')
now_num_of_ac_file.write('2')
now_num_of_ac_file.close()
training_param_history_file = open('training_history.txt', 'w')
training_param_history_file.write('-----------------------------')
training_param_history_file.close()