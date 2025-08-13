import torch
import time
import os 
import torch.nn as nn
import ultralytics as yolo

# Path where model files (.pt) are stored
file_path = "../pytorch/"

# Load a YOLO model from disk
def load_model(file_name):
    model_path = os.path.join(file_path, f"{file_name}.pt")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return None
    model = yolo.YOLO(model_path)
    print(f"YOLO model '{file_name}' loaded successfully!")
    return model

# Train a YOLO model briefly and save weights
def save_model(file_name):
    model = yolo.YOLO("../pytorch/yolov8n.pt")
    model.train(data="../RoboFlow/data.yaml", epochs=10)
    save_path = os.path.join(file_path, f"{file_name}.pt")
    torch.save(model.model.state_dict(), save_path)
    print(f"YOLO model saved at {save_path}")

# Main interactive menu
while True:
    print("-------------------")
    print("ROBOSUB YOLO MODEL CREATOR")
    print("-------------------")

    prompt = input("What would you like to do?> ")
    while prompt not in ("save", "load", "help", "quit"):
        print("Please enter a valid command")
        time.sleep(3)
        os.system("clear")
        prompt = input("What would you like to do?> ")

    # Save a new model
    if prompt == "save":
        while True:
            file = input("Enter file name> ")
            confirm = input("Are you sure?> ")
            while confirm not in ("yes", "no", "y", "n"):
                print("Please enter a valid command")
                time.sleep(3)
                os.system("clear")
                confirm = input("Are you sure?> ")

            if confirm in ("yes", "y"):
                save_model(file)
                print("File has been saved!")
                time.sleep(3)
                break

            os.system("clear")
            print("File not saved")
            break

    # Load an existing model
    elif prompt == "load":
        file = input("Enter file name to load > ").strip()
        try:
            load_model(file)
        except FileNotFoundError:
            print(f"File '{file}.pt' not found.")

    # Quit program
    elif prompt == "quit":
        print("-----------------")
        print("Program Terminated")
        print("-----------------")
        break

    # Help menu
    else:
        print("1. save: Create a new model")
        print("2. load: Load a model")
    
    time.sleep(3)
    os.system("clear")
