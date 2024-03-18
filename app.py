import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QLineEdit, QMessageBox, QRadioButton, QButtonGroup
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QTimer, QRect, QSize
import os
from time import sleep
import matplotlib.pyplot as plt

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_stage = None
    
    def initUI(self):
        self.setWindowTitle('Network Training App')
        self.setGeometry(100, 100, 800, 600)

        # Create widgets
        self.stage_label = QLabel('Select Stage:')
        # Create three radio buttons for stages
        self.stage_buttons = []
        for stage in range(1, 4):
            radio_button = QRadioButton(f'Stage {stage}')
            self.stage_buttons.append(radio_button)        
        buttons_layout = QHBoxLayout()
        stage_group = QButtonGroup(self)
        for button in self.stage_buttons:
            buttons_layout.addWidget(button)
            stage_group.addButton(button)
        # Set the default checked state for the first radio button
        self.stage_buttons[0].setChecked(True)
        self.selected_stage = 1  

        self.num_actions_label = QLabel('Select Num of Actions:')
        self.num_actions_combo = QComboBox()
        self.num_actions_combo.addItem('2')
        self.num_actions_combo.addItem('3')

        # Create three radio buttons for lockmode
        self.lockmode_label = QLabel('Lock Mode:')
        self.lock_buttons = []
        self.lock_buttons.append(QRadioButton('Lock Pretrained'))
        self.lock_buttons.append(QRadioButton('Lock Everything Except FC'))
        self.lock_buttons.append(QRadioButton('Unlock everything'))
        locks_layout = QHBoxLayout()
        lock_group = QButtonGroup(self)
        for button in self.lock_buttons:
            locks_layout.addWidget(button)
            lock_group.addButton(button)

        self.lock_buttons[0].setChecked(True)
        self.lockmode = 0

        self.lock_simple_actions_checkbox = QCheckBox('Is Simple Actions Locked')
        self.activate_boss_mem_checkbox = QCheckBox('Is Activating Boss Memory')
        self.sweet_boss_checkbox = QCheckBox('Is Sweet Boss')
        self.inherit_checkpoint_checkbox = QCheckBox('Is Checkpoint Inheritted')
        self.resume_RB_checkbox = QCheckBox('Is Previous Stage Replay Buffer Resumed')
        self.brute_exploring_checkbox = QCheckBox('Is Brute Exploring')

        self.train_button = QPushButton('Start Training')
        self.train_new_button = QPushButton('Train New Network')

        self.max_steps_label = QLabel('Max Steps When Not Stopped:')
        self.max_steps_input = QLineEdit()
        self.max_steps_input.setText(str(100000))

        self.learning_rate_label = QLabel('Learning rate: (If you check the inherit checkpoint, you can set learning rate <= 0 to inherit the lr from the stored adam optimizer.)')
        self.learning_rate_input = QLineEdit()
        self.learning_rate_input.setText(str(0.000001))

        # Image display area
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap('not_ready.png')
        pixmap = pixmap.scaled(100, 100)
        self.image_label.setPixmap(pixmap)

        # GIF display
        '''self.gif_label = QLabel(self)
        self.gif_label.setAlignment(Qt.AlignCenter)
        #self.gif_label.setGeometry(QRect(0, 0, 200, 200))
        #self.gif_label.setMinimumSize(QSize(640, 360))
        #self.gif_label.setMaximumSize(QSize(640, 360))
        self.gif_label.setScaledContents(True) 
        self.gif_label.setObjectName('/media/caotun8plus9/linux_drive/output.gif')
        self.movie = QMovie('/media/caotun8plus9/linux_drive/output.gif')
        #self.gif_label.setStyleSheet('position: absolute;left: 500;border: 3px solid green;padding: 10px;')
        self.gif_label.setMovie(self.movie)
        self.movie.start()'''
        

        # Display the integer from the file
        self.last_old_time_label = QLabel(f'# of Already Trained Steps (Updated per 100 secs): {self.read_last_old_time()}')
        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label) 

        layout.addWidget(self.stage_label)
        layout.addLayout(buttons_layout)

        layout.addWidget(self.num_actions_label)
        layout.addWidget(self.num_actions_combo)

        layout.addWidget(self.lockmode_label)
        layout.addLayout(locks_layout)

        checkbox_layout = QGridLayout()
        checkbox_layout.addWidget(self.lock_simple_actions_checkbox, 0, 0)
        checkbox_layout.addWidget(self.activate_boss_mem_checkbox, 1, 0)
        checkbox_layout.addWidget(self.brute_exploring_checkbox, 2, 0)
        checkbox_layout.addWidget(self.sweet_boss_checkbox, 0, 1)
        checkbox_layout.addWidget(self.inherit_checkpoint_checkbox, 1, 1)
        checkbox_layout.addWidget(self.resume_RB_checkbox, 2, 1)
        layout.addLayout(checkbox_layout)

        layout.addWidget(self.max_steps_label)
        layout.addWidget(self.max_steps_input)
        layout.addWidget(self.learning_rate_label)
        layout.addWidget(self.learning_rate_input)
        layout.addWidget(self.train_button)
        layout.addWidget(self.train_new_button)
        layout.addWidget(self.last_old_time_label)
        #layout.addWidget(self.gif_label)
        #self.gif_label.move(self.gif_label.x() + 500, self.gif_label.y() - 500)
        self.setLayout(layout)
        self.update_stage_button()
        # Connect the button click event to the function
        for button in self.stage_buttons:
            button.toggled.connect(self.update_stage_button)
        for button in self.lock_buttons:
            button.toggled.connect(self.update_lockmode_button)
        self.train_button.clicked.connect(self.toggle_train_network)
        self.train_new_button.clicked.connect(self.confirm_train_new_network)

        # Thread and flag for controlling training task
        self.task_thread = None
        self.is_training = False

        # Timer for checking and updating the image
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.check_image_modification)
        self.update_timer.start(1000000)  # Check every 1000 second (adjust as needed)

        # Timer for checking and update the last_old_time
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.updateLastOldTime)
        self.update_timer.start(10000)  # Check every 10 second (adjust as needed)

        # Initialize the image path and modification time
        self.image_path = 'reward_plot.png'  # Replace with the path to your image
        self.time_path = 'last_old_time.txt'
        self.oldTime_mtime = 0

    def updateLastOldTime(self):
        self.last_old_time_label.setText(f'# of Already Trained Steps (Updated per 100 secs): {self.read_last_old_time()}')

    def toggle_train_network(self):
        if not self.is_training:
            self.update_stage_button()
            self.update_lockmode_button()
            # Start training
            stage = self.selected_stage
            num_of_actions = int(self.num_actions_combo.currentText())
            print("stage: ", stage)
            now_stage_file = open('now_stage.txt', 'r')
            now_stage = int(now_stage_file.readline())
            if now_stage > stage:
                self.wrong_stage_window(now_stage, stage)
                return
            now_stage_file.close()
            now_num_of_ac_file = open('now_num_of_actions.txt', 'r')
            now_num_of_ac = int(now_num_of_ac_file.readline())
            if num_of_actions < now_num_of_ac:
                self.wrong_num_of_ac_window(now_num_of_ac, num_of_actions)
                return
            now_num_of_ac_file.close()
            is_simple_locked = self.lock_simple_actions_checkbox.isChecked()
            is_activate_boss_memory = self.activate_boss_mem_checkbox.isChecked()
            is_sweet_boss = self.sweet_boss_checkbox.isChecked()
            is_inherit_checkpoint = self.inherit_checkpoint_checkbox.isChecked()
            is_RB_resumed = self.resume_RB_checkbox.isChecked()
            is_brute_exploring = self.brute_exploring_checkbox.isChecked()
            try:
                max_steps = int(self.max_steps_input.text())
            except ValueError:
                print("Please type an int in the max step input box!")
                return
            try:
                lr = float(self.learning_rate_input.text())
            except:
                print("Please type an float in the learning rate input box!")
                return
            
            self.train_button.setText('Stop Training')
            self.is_training = True
            self.train_new_button.setEnabled(False)

            self.training_event = threading.Event()
            self.training_thread = threading.Thread(target=self.run_train_network, args=(stage, num_of_actions, self.lockmode, is_simple_locked, is_activate_boss_memory, is_inherit_checkpoint, is_sweet_boss, max_steps, is_RB_resumed, is_brute_exploring, lr, self.training_event))
            self.training_thread.start()
            #self.run_train_network(stage, is_pretrained_unlock, max_steps, self.training_event)
        else:
            # Stop training
            self.train_new_button.setEnabled(True)
            self.is_training = False
            self.train_button.setText('Start Training')
            if self.training_thread:
                self.training_event.set()

    def run_train_network(self, stage, num_of_actions, lockmode, is_simple_locked, is_activate_boss_memory, is_inherit_checkpoint, is_sweet_boss, max_steps, is_RB_resumed, is_brute_exploring, lr, event : threading.Event):
        self.check_image_modification()
        from deep_q_network import trainNetwork
        print(f"Training Network with stage={stage}, is_simple_locked={is_simple_locked}")
        last_steps = self.read_last_old_time()
        training_param_history_file = open('training_history.txt', 'a')
        training_param_history_file.write(f"LAST STEPS:\t{last_steps}-----------------------------\n")
        training_param_history_file.write(f"stage:\t{stage}\nnum of actions:\t{num_of_actions}\nlock mode:\t{lockmode}\nis simple action locked:\t{is_simple_locked}\nis activate boss memory:\t{is_activate_boss_memory}\nis sweet boss:\t{is_sweet_boss}\nis RB resumed:\t{is_RB_resumed}\nis Brute Explore:\t{is_brute_exploring}\nlearning rate:\t{lr}\n")
        training_param_history_file.close()
        trainNetwork(stage, num_of_actions, lockmode, is_simple_locked, is_activate_boss_memory, is_sweet_boss, max_steps, is_inherit_checkpoint, is_RB_resumed, is_brute_exploring, lr, event)
        self.toggle_train_network()

    def confirm_train_new_network(self):
        # Show a confirmation dialog before proceeding with "Train New Network"
        confirmation = QMessageBox.question(self, 'Confirm Action', '阿如果你要訓練新的 Network你舊的模型就會被刪掉喔, 你要繼續嗎?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirmation == QMessageBox.Yes:
            self.train_new_network()  # Perform the action
        else:
            pass  # Do nothing if the user cancels

    def wrong_stage_window(self, now_stage, desired_stage):
        QMessageBox.information(None, 'Warning', f'你現在練到 stage{now_stage} 了, 結果你還要回去練 stage{desired_stage} ? 麻煩你選後面一點的 stage, 要不然就按 Train New Network重練一個')

    def wrong_num_of_ac_window(self, now_num_ac, desired_num_ac):
        QMessageBox.information(None, 'Warning', f'你現在練到 {now_num_ac}個 actions了, 結果你還要回去練 {desired_num_ac}個 actions? 麻煩你選3個actions, 要不然就按 Train New Network重練一個')

    def wrong_learning_rate_window(self):
        QMessageBox.information(None, 'Warning', 'Learning rate should be larger than 0 unless you check the \"inherit checkpoint\".')

    def train_new_network(self):
        if os.path.exists("results.txt"):
            os.remove("results.txt")
        if os.path.exists("last_old_time.txt"):
            os.remove("last_old_time.txt")
        if os.path.exists("model/FlappyBird.h5"):
            os.remove("model/FlappyBird.h5")
        if os.path.exists("scores_training.txt"):
            os.remove("scores_training.txt")
        if os.path.exists("running_scores_avg.txt"):
            os.remove("running_scores_avg.txt")
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
        sleep(5)
        self.toggle_train_network()
    
    def update_stage_button(self):
        self.selected_stage = None
        for index, button in enumerate(self.stage_buttons):
            if button.isChecked():
                self.selected_stage = index + 1
                break
        # Update the validity of the "Train New Network" button based on the selected stage
        if self.selected_stage is not None and self.selected_stage != 1:
            self.train_new_button.setEnabled(False)
        else:
            self.train_new_button.setEnabled(True)

    def update_lockmode_button(self):
        self.lockmode = None
        for index, button in enumerate(self.lock_buttons):
            if button.isChecked():
                self.lockmode = index
                break

    def read_last_old_time(self):
        # Read the integer from the file 'last_old_time.txt'
        try:
            with open('last_old_time.txt', 'r') as file:
                return int(file.read())
        except (FileNotFoundError, ValueError):
            return 0

    def check_image_modification(self):
        # Check if the image file has been modified
        if os.path.exists(self.time_path):
            #current_mtime = os.path.getmtime(self.time_path)
            #if current_mtime != self.oldTime_mtime:
            self.drawReward()
        
    def drawReward(self):
        # Open the file for reading
        ctr = 0
        lines = []
        lines_sparse = []
        if os.path.exists('results.txt'):
            file = open('results.txt', 'r')
            if os.path.getsize('results.txt'):
            # Read all lines from the file and convert them to floats
                for line in file:
                  lines.append(float(line.strip()))
                  if ctr % 1000 == 0:
                    lines_sparse.append(float(line.strip()))
                  ctr += 1

                plt.plot(range(len(lines)), lines)
                plt.savefig("reward_plot.png")
                #plt.plot(range(len(lines_sparse)), lines_sparse)
                #plt.savefig("reward_plot_sparse.png")
                pixmap = QPixmap(self.image_path)
                self.image_label.setPixmap(pixmap)
            else:
                pixmap = QPixmap('not_ready.png')
                self.image_label.setPixmap(pixmap)
        else:
            pixmap = QPixmap('not_ready.png')
            self.image_label.setPixmap(pixmap)

    
def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
