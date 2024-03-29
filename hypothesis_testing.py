import math

def open_file(file_path):
    """
    Open a file and return its second column as a list of floats
    """
    with open(file_path, "r") as file:
        return [float(line.split(",")[1]) for line in file.readlines()]
    

def calculate_mean(data):
    """
    Calculate the mean of a list of numbers

    :param data: (list) A list of numbers
    """
    return sum(data) / len(data)


def calculate_std_dev(data):
    """
    Calculate the standard deviation of a list of numbers

    :param data: (list) A list of numbers
    """
    mean = calculate_mean(data)
    return math.sqrt(sum([(x - mean) ** 2 for x in data]) / len(data))


def z_test(mean_space_invaders, mean_test_game, standard_deviation_test_game, sample_size):
    """
    Perform a z-test to compare the means of two samples

    :param mean_space_invaders: (float) The mean of the Space Invaders sample
    :param mean_test_game: (float) The mean of the test game sample
    :param standard_deviation_test_game: (float) The standard deviation of the test game sample
    :param sample_size: (int) The number of samples in each sample
    """
    return (mean_test_game - mean_space_invaders) / (standard_deviation_test_game / math.sqrt(sample_size))



def main():
    space_invaders_rewards = open_file("test2.csv")
    test_game_rewards = open_file("test.csv")

    mean_space_invaders = calculate_mean(space_invaders_rewards)
    mean_test_game = calculate_mean(test_game_rewards)
    standard_deviation_test_game = calculate_std_dev(test_game_rewards)
    sample_size = len(space_invaders_rewards)

    z_value = z_test(mean_space_invaders, mean_test_game, standard_deviation_test_game, sample_size)
    print(z_value)
    if z_value > 1.645:
        print("The test game has a significantly higher mean reward than Space Invaders")

    else:
        print("The test game does not have a significantly higher mean reward than Space Invaders")


main()