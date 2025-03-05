const nums = [3,2,3];

// Brute Force Method O(n^2)
function BruteForceME(array) {
    let n = array.length;

    for (let i = 0; i < array.length; i++) {
        let count = 0;

        for (let j = 0; j < array.length; j++) {
            if (array[i] === array[j]) {
                count++;
            }
        }

        if (count > n/2) {
            return array[i]
        }
    }
}

// console.log(BruteForceME(nums))

// Better Approach O(nlogn)
function BetterApproachME(array) {
    let n = array.length;
    if (n === 1) { return array[i] };

    let count = 1;
    array.sort((a,b) => a-b)

    for (let i = 0; i < n; i++) {
        if (array[i-1] === array[i]) {
            count++;
        } else{
            if (count>Math.floor(n/2)) {
                return array[i-1]
            }
            count = 1;
        }
    }

    if (count>Math.floor(n/2)) {
        return array[n-1]
    }

    return -1;
}

// console.log(BetterApproachME(nums))

// Optimal Approach O(n)
function OptimalApproachME(array) {
    const n = array.length;
    const countMap = new Map();

    for (const num of array) {
        countMap.set(num, (countMap.get(num) || 0 ) +1)

        if (countMap.get(num) > n/2) {
            return num;
        }
    }

    return -1;
}

console.log(OptimalApproachME(nums));