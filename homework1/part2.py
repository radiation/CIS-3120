# pip install inflect
import inflect

'''
This application is written with lists in mind, so it can be easily expanded to take more than three integers.
We could even prompt the user to enter the number of integers they want to compare, but we'll keep it simple for now.

Sample data:

Sets where all three values are equal to the average:
    5,5,5 (Average: 5)
    10,10,10 (Average: 10)

Sets where no value is equal to the average:
    1,2,4 (Average: 2.33, none of the values is 2.33)
    3,7,11 (Average: 7, none of the values is 7)
    5,10,15 (Average: 10, none of the values is 10)

Sets where one item is equal to the average:
    3,3,6 (Average: 4, one value is 4)
    8,4,4 (Average: 5.33, one value is 4)
    12,6,6 (Average: 8, one value is 6)
    2,1,3 (Average: 2, one value is 2)
    7,14,7 (Average: 9.33, one value is 7)

'''

# Global variables
p = inflect.engine() # Create an inflect engine so we can convert cardinal numbers to ordinal words
secret_code: int = 1979 # The secret code is 1979, which is the year I was born and a dope song by Smashing Pumpkins

# Let's tell the user how this works
def introduction() -> None:
    print("This is what the program does:")
    print("1. Takes three integers as input")
    print("2. Calculates the average of the three integers")
    print("3. Compares each integer to the average")
    print("4. Counts the number of integers that are equal to the average")

# Here we take in a list of floats and return the average as a float
def find_average(numbers: list[float]) -> float:
    return round(sum(numbers)/len(numbers), 3)

# Take a list of floats and the average as a float and compare each float to the average
def compare_to_average(numbers: list[float], average: float) -> None:
    nums_equal_to_average: int = 0
    for num in numbers:
        if num > average:
            print(f"{num} is greater than the average of {average}")
        elif num < average:
            print(f"{num} is less than the average of {average}")
        else:
            print(f"{num} is equal to the average of {average}")
            nums_equal_to_average += 1
    print(f"The number of integers equal to the average is {nums_equal_to_average}")

# Should we exit the program?
def check_secret_code(num: int) -> bool:
    if num == secret_code:
        print("You found the secret code! Exiting program.")
        return True
    return False

# We use the ordinal function from the inflect engine to tell the user which integer to enter
def read_int_value(n: int) -> int:
    return int(input(f"Enter the {p.ordinal(n+1)} integer: "))

# Let's put it all together
def main():
    introduction()

    total_ints: int = 3 # User should enter an int, but we use float so average is a float
    ints_to_check: list[int] = [] # We'll store the integers the user enters here
    num_tests: int = 0 # We'll keep track of how many tests the user has run
    avg_string: str = "" # We'll use this to print the integers the user entered
    should_exit = False # We need this because we'll be in a loop in another loop

    # If the user hasn't found the secret code after 100 tests, they're not going to
    while num_tests < 100:

        # Clear previous input
        ints_to_check.clear()
        avg_string = ""

        # We use a for loop to get "ints_to_check" integers
        for i in range(0, total_ints):
            ints_to_check.append(read_int_value(i))
            if check_secret_code(ints_to_check[i]):
                should_exit = True
                break
            avg_string += str(ints_to_check[i]) + ", " if i < total_ints - 1 else "and " + str(ints_to_check[i])

        # This is a little clumsy, but it works
        if should_exit:
            break

        # We're using a list instead of three separate variables in case we want to expand in the future
        average: float = find_average(ints_to_check)

        # Could we change this to print the list? Sure. But we're keeping it simple for now.
        print(f"The average of {avg_string} is {average = :.3f}")

        # Again, we pass a list of floats instead of explicitly passing each float
        compare_to_average(ints_to_check, average)

        # We increment the number of tests run
        num_tests += 1

        # We only want to give the user a hint after 10 tests
        if num_tests == 10:
            print("Hint: The secret code is a year that is important to me.")

    print(f"The number of tests run was {num_tests}")

if __name__ == '__main__':
    main()
