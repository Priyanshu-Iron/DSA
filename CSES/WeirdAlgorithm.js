// node WeirdAlgorithm.js

/**
    Consider an algorithm that takes as input a positive integer n. If n is even, the algorithm divides it by two, 
    and if n is odd, the algorithm multiplies it by three and adds one. The algorithm repeats this, until n is one.
    For example, the sequence for n=3 is as follows:
    3-> 10-> 5-> 16-> 8-> 4-> 2-> 1
    Your task is to simulate the execution of the algorithm for a given value of n.
**/

const input = 3

function WeirdAlgorithm (input) {
    while (input !== 1) {
        console.log(input);
        if (input % 2 === 0) {
            input = input / 2 
        } else{
            input = input*3+1
        }
    }
    console.log(input);
    return input
}

WeirdAlgorithm(input);