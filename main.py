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
    num_classes = num_classes_entry.get()
    if not num_features:
        messagebox.showerror("Error", "Please enter the number of features.")
        return
    if not percentage:
        messagebox.showerror("Error", "Please enter the train-test split percentage.")
        return
    if not num_classes:
        messagebox.showerror("Error", "Please enter the number of classes.")
        return

    if dataset_label.cget("text") != "Dataset: None":
        dataset_path = dataset_label.full_path
        split_dataset(dataset_path, percentage)
    
    if TRAIN_PATH and TEST_PATH:
        print(f"Running with Train File: {TRAIN_PATH}, Test File: {TEST_PATH}, Number of Features: {num_features}, Number of Classes: {num_classes}")
        try:
            elastic_for_gui.delete_index(elastic_for_gui.ds)
        except:
            pass
        elastic_for_gui.create_index(elastic_for_gui.ds, int(num_features), int(num_classes))
        thread = threading.Thread(target=insert_and_test, args=(int(num_classes),))
        thread.start()
    else:
        messagebox.showerror("Error", "Please choose a dataset or train/test files first.")
        return

    learning_frame.pack(pady=20)
    progress_label.config(text="Learning in progress...")

def split_dataset(dataset_path, percentage):
    global TRAIN_PATH, TEST_PATH
    df = pd.read_csv(dataset_path)
    train, test = train_test_split(df, test_size=int(percentage)/100)
    dataset_path = dataset_path.split(".")[0]
    train.to_csv(dataset_path+"_train.csv", index=False)
    test.to_csv(dataset_path+"_test.csv", index=False)
    TRAIN_PATH = dataset_path+"_train.csv"
    TEST_PATH = dataset_path+"_test.csv"
    print(f"Running with Dataset: {dataset_path}, Train-Test Split: {percentage}%")

def insert_and_test(num_classes):
    elastic_for_gui.insert(TRAIN_PATH, int(num_features_entry.get()), int(num_classes), update_progress)
    update_progress(100)
    progress_label.config(text="Learning completed! Now testing...")
    predictions = elastic_for_gui.test(TEST_PATH, int(num_features_entry.get()), int(num_classes), update_test_progress)
    update_test_progress(100)
    show_results(predictions, num_classes)

def update_progress(progress):
    progress_label.config(text=f"Learning in progress... {progress:.2f}%")

def update_test_progress(progress):
    progress_label.config(text=f"Testing in progress... {progress:.2f}%")

def show_results(predictions, num_classes):
    results_frame.pack(pady=20)
    results_text.delete(1.0, tk.END)
    results_text.insert(tk.END, "Results:\n\n")
    overall_accuracy = 0
    for i in range(num_classes):
        y_true = [pred[i] for pred in predictions]
        y_pred = [pred[num_classes+ i] for pred in predictions]
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        overall_accuracy += accuracy

        results_text.insert(tk.END, f"Class {i + 1} Metrics:\n", "bold")
        results_text.insert(tk.END, f"Accuracy: {accuracy:.3f}\n")
        results_text.insert(tk.END, f"Recall: {recall:.3f}\n")
        results_text.insert(tk.END, f"Precision: {precision:.3f}\n")
        results_text.insert(tk.END, f"F1 Score: {f1:.3f}\n\n")

    overall_accuracy /= num_classes
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
    num_classes_entry.delete(0, tk.END)
    progress_label.config(text="")
    results_text.delete(1.0, tk.END)
    TRAIN_PATH = ""
    TEST_PATH = ""

def init_GUI():
    global root, dataset_label, train_label, test_label, num_features_entry, percentage_entry, num_classes_entry, progress_label, results_text, results_frame, learning_frame
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

    tk.Label(root, text="Number of Classes:", bg="#f0f0f0").pack(pady=5)
    num_classes_entry = tk.Entry(root)
    num_classes_entry.pack(pady=5)

    button_frame2 = tk.Frame(root, bg="#f0f0f0")
    button_frame2.pack(pady=10)

    tk.Button(button_frame2, text="Run", font=button_font, command=run).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame2, text="Clear", font=button_font, command=clear).pack(side=tk.LEFT, padx=10)

    learning_frame = tk.Frame(root, bg="#f0f0f0")
    progress_label = tk.Label(learning_frame, text="", bg="#f0f0f0", font=button_font)
    progress_label.pack()

    results_frame = tk.Frame(root, bg="#f0f0f0")
    results_text = tk.Text(results_frame, height=25, width=50)
    results_text.pack()

def main():
    init_GUI()
    root.mainloop()

if __name__ == "__main__":
    main()