import numpy as np

def main():
    # Specify the file name
    file_name = 'scores.txt'

    try:
        # Read the numbers from the file
        with open(file_name, 'r') as file:
            numbers = np.loadtxt(file, dtype=float)

        # Calculate the mean
        mean = np.mean(numbers)

        # Calculate the variance using numpy
        variance = np.var(numbers)

        # Print the result
        print(f"Mean: {mean}\tVariance: {variance}")

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except ValueError:
        print("Error: The file contains non-numeric data.")

if __name__ == "__main__":
    main()
