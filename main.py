import filemanager
import dataproccessing

print("Select File")
read_data = filemanager.read_csv_file()

def show_main_menu():
    print("Select a Process")
    print("1. Random Forest")
    print("2. Decision Tree")
    print("3. Neural Network")
    while True:
        choice =int(input("Enter your selection"))
        process_main_menu(choice)

def process_main_menu(choice):
    if choice == 1:
        print("[Start] Random Forest")
        dataproccessing.classification(read_data, 'rf')
        print("[End] Random Forest")
    if choice == 2:
        print("[Start] Decision Tree")
        dataproccessing.classification(read_data, 'dt')
        print("[End] Decision Tree")
    if choice == 3:
        print("[Start] Neural Network")
        dataproccessing.classification(read_data, 'nn')
        print("[End] Neural Network")
    else:
        print("Invalid selection, please try again")

    show_main_menu()

if __name__ == "__main__":
    show_main_menu()
