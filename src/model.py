import torch
import time
import os 
import torch.nn as nn
import ultralytics as yolo


file_path = "../pytorch/"
# this functions saves a pytorch file 
def load_model(file_name):
    model_path = os.path.join(file_path, f"{file_name}.pt" )

    if not os.path.exists(model_path):      # catch an error in case path is messed up 
        print(f"Error: {model_path} not found.")
        return None

    model = yolo.YOLO(model_path)
    print(f"YOLO model '{file_name}' loaded successfully!")
    return model


# this functions loads a pytorch file and evaluates it to ensure its integrity
def save_model(file_name):
    model = yolo.YOLO("../pytorch/yolov8n.pt")
   
    model.train(data="../RoboFlow/data.yaml", epochs=10) # to fine-tune the dataset a lil' 

    save_path = os.path.join(file_path, f"{file_name}.pt")
    torch.save(model.model.state_dict(), save_path)
    print(f"YOLO model saved at {save_path}")


# ui for creating, saving, and loading pytorch models 
while True:
    print("-------------------")
    print("ROBOSUB YOLO MODEL CREATOR")
    print("-------------------")
    prompt = input("What would you like to do?> ")
    while prompt != "save" and prompt != "load" and prompt != "help" and prompt != "quit":
        print("Please enter a valid command")
        time.sleep(3)
        os.system("clear")
        prompt = input("What would you like to do?> ")


    if prompt == "save":
        while True:
            file = input("Enter file name>")
            prompt = input("Are you sure?>")
            while prompt != "yes" and prompt != "no" and prompt != "y" and prompt != "n":
                print("Please enter a valid command")
                time.sleep(3)
                os.system("clear")
                prompt = input("Are you sure?>")

            if prompt == "yes" or prompt == "y":
                save_model(file)
                print("File has been saved!")
                time.sleep(3)
                break

            os.system("clear")
            print("File not saved")
            break
        

    elif prompt == "load":
        file = input("Enter file name to load > ").strip()
        try:
            load_model(file)
        except FileNotFoundError:
            print(f"File '{file}.pt' not found.")
    
    elif prompt == "quit":
        print("-----------------")
        print("Program Terminated")
        print("-----------------")
        break

    else:
        print("1. save: Create a new model")
        print("2. load: Load a model")
    
    time.sleep(3)
    os.system("clear")