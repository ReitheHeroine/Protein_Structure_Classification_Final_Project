import filemanager

print("Select File")
read_data = filemanager.read_csv_file()
print(read_data)

def show_main_menu():
    print("Select a Process")
    print("1. Classification")
    print("2. Clustering")
    while True:
        choice =int(input("Enter your selection"))
        process_main_menu(choice)

def process_main_menu(choice):
    if choice == 1:
        print("[Start] Classification")
        print("[End] Classification")
    elif choice ==2:
        print("[Start] Clustering")
        print("[End] Clustering")
    show_main_menu()

if __name__ == "__main__":
    show_main_menu()

