import os

main_directory = "F:\\Università\\ecg-fitness_raw-v1.0"
target_file_name = "c920-2.avi"

for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file == target_file_name:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
