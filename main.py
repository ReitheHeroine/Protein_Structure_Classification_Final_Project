import filemanager
import dataproccessing

print("Select File")
read_data = filemanager.read_csv_file()

def show_main_menu():
    print("Select a Process")
    print("1. Random Forest")
    while True:
        choice =int(input("Enter your selection"))
        process_main_menu(choice)

def process_main_menu(choice):
    if choice == 1:
        print("[Start] Random Forest")
        dataproccessing.classification(read_data, 'rf')
        print("[End] Random Forest")
    else:
        print("Invalid selection, please try again")

    show_main_menu()

if __name__ == "__main__":
    show_main_menu()
