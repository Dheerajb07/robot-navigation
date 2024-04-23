import csv

def load_csv(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

file_path = 'trajectory_data.csv'
data = load_csv(file_path)
print(data[0])