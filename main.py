import data_exploration
import model_training
import model_evaluation
import subprocess

subprocess.call(data_exploration)
subprocess.call(model_evaluation)
subprocess.call(model_training)
