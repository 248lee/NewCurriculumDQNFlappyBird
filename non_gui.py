from deep_q_network import trainNetwork
trainNetwork(2, 3, lock_mode=0, is_simple_actions_locked=False, is_activate_boss_memory=False, max_steps=2010, resume_Adam=False, learning_rate=3e-6, event=None, is_colab=True)
