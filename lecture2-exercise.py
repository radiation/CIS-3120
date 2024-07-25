def is_prime(n) -> bool:
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_primes(n) -> list[int]:
    primes: list[int] = [2, 3]
    current_num: int = 5

    while len(primes) < n:
        if is_prime(current_num):
            primes.append(current_num)
        current_num += 1

    return primes

def main():
    primes: list[int] = get_primes(500)
    print(*reversed(primes), sep='\n')
    print(f'sum(primes): {sum(primes)}')

if __name__ == '__main__':
    main()