import math

x: float = -3.0

print("{:>5}   {:<7}".format("x", "y"))

while x <= 4:
    numerator = 4*x**3 - 12*x**2 - 9*x + 27
    denominator = math.sqrt(5*x**2 + 2) + (3 * abs(x-2.5))
    y = numerator/denominator
    if y == 0:
        output = "ZERO"
    elif y < 0:
        output = "NEGATIVE"
    else:
        output = "POSITIVE"
    print("{:>5}   {:<7} {:<4} {:<15}".format(x, round(y,4), "Y IS", output))
    x+=0.5
