import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import elastic_for_gui
import threading
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
from sklearn.model_selection import train_test_split
import pandas as pd

TRAIN_PATH = ""
TEST_PATH = ""

def choose_dataset():
    clear()
    file_path = filedialog.askopenfilename(title="Select Dataset")
    if file_path:
        dataset_label.config(text=f"Dataset: {os.path.basename(file_path)}")
        dataset_label.full_path = file_path

def choose_train_test_files():
    clear()
    global TRAIN_PATH, TEST_PATH
    train_file = filedialog.askopenfilename(title="Select Train File")
    if train_file:
        test_file = filedialog.askopenfilename(title="Select Test File")
        if test_file:
            train_label.config(text=f"Train File: {os.path.basename(train_file)}")
            TRAIN_PATH = train_file
            test_label.config(text=f"Test File: {os.path.basename(test_file)}")
            TEST_PATH = test_file

def run():
    global TRAIN_PATH, TEST_PATH
    num_features = num_features_entry.get()
    percentage = percentage_entry.get()
    if not num_features:
        messagebox.showerror("Error", "Please enter the number of features.")
        return
    if not percentage:
        messagebox.showerror("Error", "Please enter the train-test split percentage.")
        return

    if dataset_label.cget("text") != "Dataset: None":
        dataset_path = dataset_label.full_path
        
        # Split the dataset into train and test 
        df = pd.read_csv(dataset_path)
        train, test = train_test_split(df, test_size=int(percentage)/100)

        # remove the file extension
        dataset_path = dataset_path.split(".")[0]

        train.to_csv(dataset_path+"_train.csv", index=False)
        test.to_csv(dataset_path+"_test.csv", index=False)
        TRAIN_PATH = dataset_path+"_train.csv"
        TEST_PATH = dataset_path+"_test.csv"
        print(f"Running with Dataset: {dataset_path}, Train-Test Split: {percentage}%, Number of Features: {num_features}")
    
    if TRAIN_PATH and TEST_PATH: # if both train and test files are selected, or dataset is selected,and train-test split percentage is entered
        print(f"Running with Train File: {TRAIN_PATH}, Test File: {TEST_PATH}, Number of Features: {num_features}")
        try:
            elastic_for_gui.delete_index(elastic_for_gui.ds)
        except:
            pass
        elastic_for_gui.create_index(elastic_for_gui.ds, int(num_features))
        thread = threading.Thread(target=insert_and_test)
        thread.start()
    else:
        messagebox.showerror("Error", "Please choose a dataset or train/test files first.")
        return

    # Show learning frame
    learning_frame.pack(pady=20)
    progress_label.config(text="Learning in progress...")

def insert_and_test():
    elastic_for_gui.insert(TRAIN_PATH, int(num_features_entry.get()), update_progress)
    update_progress(100)
    progress_label.config(text="Learning completed! Now testing...")
    predictions = elastic_for_gui.test(TEST_PATH, int(num_features_entry.get()), update_test_progress)
    update_test_progress(100)
    show_results(predictions)

def update_progress(progress):
    progress_label.config(text=f"Learning in progress... {progress:.2f}%")

def update_test_progress(progress):
    progress_label.config(text=f"Testing in progress... {progress:.2f}%")

def show_results(predictions):
    y_true_class1 = [pred[3] for pred in predictions]
    y_pred_class1 = [pred[1] for pred in predictions]
    y_true_class2 = [pred[4] for pred in predictions]
    y_pred_class2 = [pred[2] for pred in predictions]

    accuracy_class1 = accuracy_score(y_true_class1, y_pred_class1)
    recall_class1 = recall_score(y_true_class1, y_pred_class1, average='macro')
    precision_class1 = precision_score(y_true_class1, y_pred_class1, average='macro')
    f1_class1 = f1_score(y_true_class1, y_pred_class1, average='macro')

    accuracy_class2 = accuracy_score(y_true_class2, y_pred_class2)
    recall_class2 = recall_score(y_true_class2, y_pred_class2, average='macro')
    precision_class2 = precision_score(y_true_class2, y_pred_class2, average='macro')
    f1_class2 = f1_score(y_true_class2, y_pred_class2, average='macro')

    overall_accuracy = (accuracy_class1 + accuracy_class2) / 2

    results_frame.pack(pady=20)
    results_text.delete(1.0, tk.END)
    results_text.insert(tk.END, "Results:\n\n")
    results_text.insert(tk.END, "Class 1 Metrics:\n", "bold")
    results_text.insert(tk.END, f"Accuracy: {accuracy_class1:.3f}\n")
    results_text.insert(tk.END, f"Recall: {recall_class1:.3f}\n")
    results_text.insert(tk.END, f"Precision: {precision_class1:.3f}\n")
    results_text.insert(tk.END, f"F1 Score: {f1_class1:.3f}\n\n")
    results_text.insert(tk.END, "Class 2 Metrics:\n", "bold")
    results_text.insert(tk.END, f"Accuracy: {accuracy_class2:.3f}\n")
    results_text.insert(tk.END, f"Recall: {recall_class2:.3f}\n")
    results_text.insert(tk.END, f"Precision: {precision_class2:.3f}\n")
    results_text.insert(tk.END, f"F1 Score: {f1_class2:.3f}\n\n")
    results_text.insert(tk.END, f"Overall Accuracy: {overall_accuracy:.3f}\n", "bold")

    results_text.tag_configure("bold", font=("Helvetica", 12, "bold"))

def clear():
    global TRAIN_PATH, TEST_PATH
    dataset_label.config(text="Dataset: None")
    train_label.config(text="Train File: None")
    test_label.config(text="Test File: None")
    num_features_entry.delete(0, tk.END)
    percentage_entry.delete(0, tk.END)
    percentage_entry.insert(0, "30")
    progress_label.config(text="")
    results_text.delete(1.0, tk.END)
    TRAIN_PATH = ""
    TEST_PATH = ""
    
def main():
    global dataset_label, percentage_label, train_label, test_label, num_features_entry, percentage_entry, learning_frame, progress_label, results_frame, results_text, root

    root = tk.Tk()
    root.title("Attack Detection Project")
    root.geometry("400x600")
    root.configure(bg="#f0f0f0")

    title_font = ("Helvetica", 16, "bold")
    button_font = ("Helvetica", 12)

    tk.Label(root, text="Choose an option:", font=title_font, bg="#f0f0f0").pack(pady=20)

    button_frame = tk.Frame(root, bg="#f0f0f0")
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Option 1: Choose Dataset and Train-Test Split", font=button_font, command=choose_dataset).pack(pady=5, padx=20, fill=tk.X)
    
    dataset_label = tk.Label(root, text="Dataset: None", bg="#f0f0f0")
    dataset_label.pack(pady=5)
    percentage_label = tk.Label(root, text="Train-Test Split:", bg="#f0f0f0")
    percentage_label.pack(pady=5)
    percentage_entry = tk.Entry(root)
    percentage_entry.insert(0, "30")
    percentage_entry.pack(pady=5)

    tk.Button(button_frame, text="Option 2: Choose Train and Test Files", font=button_font, command=choose_train_test_files).pack(pady=5, padx=20, fill=tk.X)

    train_label = tk.Label(root, text="Train File: None", bg="#f0f0f0")
    train_label.pack(pady=5)
    test_label = tk.Label(root, text="Test File: None", bg="#f0f0f0")
    test_label.pack(pady=5)

    tk.Label(root, text="Number of Features:", bg="#f0f0f0").pack(pady=5)
    num_features_entry = tk.Entry(root)
    num_features_entry.pack(pady=5)

    button_frame2 = tk.Frame(root, bg="#f0f0f0")
    button_frame2.pack(pady=10)

    tk.Button(button_frame2, text="Run", font=button_font, command=run).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame2, text="Clear", font=button_font, command=clear).pack(side=tk.LEFT, padx=10)

    learning_frame = tk.Frame(root, bg="#f0f0f0")
    progress_label = tk.Label(learning_frame, text="", bg="#f0f0f0", font=button_font)
    progress_label.pack()

    results_frame = tk.Frame(root, bg="#f0f0f0")
    results_text = tk.Text(results_frame, height=10, width=50)
    results_text.pack()

    root.mainloop()

if __name__ == "__main__":
    main()