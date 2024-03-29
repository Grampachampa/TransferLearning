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

def chi_squared_test(data1, data2):
    """
    Perform a chi-squared test to compare the means of two samples

    :param data1: (list) A list of numbers
    :param data2: (list) A list of numbers
    """
    mean1 = calculate_mean(data1)
    mean2 = calculate_mean(data2)
    std_dev1 = calculate_std_dev(data1)
    std_dev2 = calculate_std_dev(data2)
    n1 = len(data1)
    n2 = len(data2)

    return ((n1 - 1) * (std_dev1 ** 2) + (n2 - 1) * (std_dev2 ** 2)) / (n1 + n2 - 2) * ((1 / n1) + (1 / n2)) * ((mean1 - mean2) ** 2)


def t_test(data1, data2):
    """
    Perform a t-test to compare the means of two samples

    :param data1: (list) A list of numbers
    :param data2: (list) A list of numbers
    """
    mean1 = calculate_mean(data1)
    mean2 = calculate_mean(data2)
    std_dev1 = calculate_std_dev(data1)
    std_dev2 = calculate_std_dev(data2)
    n1 = len(data1)
    n2 = len(data2)

    return (mean1 - mean2) / math.sqrt((std_dev1 ** 2 / n1) + (std_dev2 ** 2 / n2))

def f_test(data1, data2):
    """
    Perform an F-test to compare the variances of two samples

    :param data1: (list) A list of numbers
    :param data2: (list) A list of numbers
    """
    std_dev1 = calculate_std_dev(data1)
    std_dev2 = calculate_std_dev(data2)

    return (std_dev1 ** 2) / (std_dev2 ** 2)


def manova_test(data1, data2):
    """
    Perform a MANOVA test to compare the means of two samples

    :param data1: (list) A list of numbers
    :param data2: (list) A list of numbers
    """
    mean1 = calculate_mean(data1)
    mean2 = calculate_mean(data2)
    std_dev1 = calculate_std_dev(data1)
    std_dev2 = calculate_std_dev(data2)
    n1 = len(data1)
    n2 = len(data2)

    return ((n1 - 1) * (std_dev1 ** 2) + (n2 - 1) * (std_dev2 ** 2)) / (n1 + n2 - 2) * ((1 / n1) + (1 / n2)) * ((mean1 - mean2) ** 2)


def anova_test(data1, data2, data3, data4):

    mean1 = calculate_mean(data1)
    mean2 = calculate_mean(data2)
    mean3 = calculate_mean(data3)
    mean4 = calculate_mean(data4)

    n1 = len(data1)
    n2 = len(data2)
    n3 = len(data3)
    n4 = len(data4)

    grand_mean = (mean1 + mean2 + mean3 + mean4) / 4

    sst = sum([(x - grand_mean) ** 2 for x in data1 + data2 + data3 + data4])

    ssw = sum([(x - mean1) ** 2 for x in data1]) + sum([(x - mean2) ** 2 for x in data2]) + sum([(x - mean3) ** 2 for x in data3]) + sum([(x - mean4) ** 2 for x in data4])

    return (sst / 3) / (ssw / 12)


def regression_analysis(data1, data2):
    """
    Perform a regression analysis to compare the means of two samples

    :param data1: (list) A list of numbers
    :param data2: (list) A list of numbers
    """
    mean1 = calculate_mean(data1)
    mean2 = calculate_mean(data2)
    std_dev1 = calculate_std_dev(data1)
    std_dev2 = calculate_std_dev(data2)
    n1 = len(data1)
    n2 = len(data2)

    return (mean1 - mean2) / math.sqrt((std_dev1 ** 2 / n1) + (std_dev2 ** 2 / n2))

def main():
    space_invaders_rewards = open_file("test2.csv")
    air_raid = open_file("test.csv")
    carnival = open_file("test3.csv")
    demonattack = open_file("test4.csv")

    mean_space_invaders = calculate_mean(space_invaders_rewards)
    mean_test_game = calculate_mean(test_game_rewards)
    standard_deviation_test_game = calculate_std_dev(test_game_rewards)
    sample_size = len(space_invaders_rewards)

    z_value = z_test(mean_space_invaders, mean_test_game, standard_deviation_test_game, sample_size)

    t_test_value = t_test(space_invaders_rewards, test_game_rewards)

    f_test_value = f_test(space_invaders_rewards, test_game_rewards)

    chi_squared_test_value = chi_squared_test(space_invaders_rewards, test_game_rewards)

    anova_test_value = anova_test(space_invaders_rewards, air_raid, carnival, demonattack)

    manova_test_value = manova_test(space_invaders_rewards, test_game_rewards)

    regression_analysis_value = regression_analysis(space_invaders_rewards, test_game_rewards)


    if z_value > 1.645:
        print("The test game has a significantly higher mean reward than Space Invaders in the z-test")

    else:
        print("The test game does not have a significantly higher mean reward than Space Invaders in the z-test")

    if t_test_value > 1.645:
        print("The test game has a significantly higher mean reward than Space Invaders in the t-test")
    else:
        print("The test game does not have a significantly higher mean reward than Space Invaders in the t-test")

    if f_test_value > 1.645:
        print("The test game has a significantly higher variance than Space Invaders in the F-test")
    else:
        print("The test game does not have a significantly higher variance than Space Invaders in the F-test")

    if chi_squared_test_value > 1.645:
        print("The test game has a significantly different mean reward than Space Invaders in the chi-squared test")
    else:
        print("The test game does not have a significantly different mean reward than Space Invaders in the chi-squared test")

    if manova_test_value > 1.645:
        print("The test game has a significantly different mean reward than Space Invaders in the MANOVA test")
    else:
        print("The test game does not have a significantly different mean reward than Space Invaders in the MANOVA test")

    if anova_test_value > 1.645:
        print("The test game has a significantly different mean reward than Space Invaders in the ANOVA test")
    else:
        print("The test game does not have a significantly different mean reward than Space Invaders in the ANOVA test")

    if regression_analysis_value > 1.645:
        print("The test game has a significantly different mean reward than Space Invaders in the regression analysis")
    else:
        print("The test game does not have a significantly different mean reward than Space Invaders in the regression analysis")

    




main()