// Max Consecutive Ones

let arr1 = [1, 1, 0, 1, 1, 1];

function maxConsecutiveOnes(arr1) {
    let maxCount = 0; // To store the max count of consecutive ones
    let currentCount = 0; // To count the current sequence of ones

    for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] === 1) {
            currentCount++; // Increment the current count if 1 is found
        } else {
            // If 0 is found, update maxCount and reset currentCount
            maxCount = Math.max(maxCount, currentCount);
            currentCount = 0;
        }
    }

    // Handle the case where the array ends with consecutive 1s
    maxCount = Math.max(maxCount, currentCount);

    return maxCount;
}

console.log(maxConsecutiveOnes(arr1)); // Output: 3
