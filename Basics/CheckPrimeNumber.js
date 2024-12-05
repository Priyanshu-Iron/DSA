// node CheckPrimeNumber.js

let n = 433933

function CheckPrimeNumber () {
    let count = 0

    for(let i=1; i*i<=n; i++){
        if(n%i == 0){
            count++
            if((n/i)!=i){
                count++
            }
        }

        if(count == 2){
            return true
        }else{
            return false
        }
    }
}

// console.log(CheckPrimeNumber(n));

function countPrimesLessThanN(n) {
    if (n <= 2) return 0;  // No primes less than 2

    let primeCount = 0;

    // Check all numbers less than n
    for (let i = 2; i < n; i++) {
        if (isPrime(i)) {
            primeCount++;
        }
    }

    return primeCount;
}

// Helper function to check if a number is prime
function isPrime(num) {
    if (num <= 1) return false;  // Numbers less than or equal to 1 are not prime
    for (let i = 2; i * i <= num; i++) {
        if (num % i === 0) {
            return false;  // num is divisible by i, so it's not prime
        }
    }
    return true;  // num is prime if no divisors are found
}

console.log(countPrimesLessThanN(n));


/*
var countPrimes = function(n) {
    if (n <= 2) return 0;

    // Initialize an array to mark numbers as prime
    const isPrime = new Array(n).fill(true);
    isPrime[0] = isPrime[1] = false; // 0 and 1 are not prime

    for (let i = 2; i * i < n; i++) {
        if (isPrime[i]) {
            for (let j = i * i; j < n; j += i) {
                isPrime[j] = false; // Mark multiples of i as non-prime
            }
        }
    }

    // Count the number of primes
    return isPrime.filter(Boolean).length;
};
*/