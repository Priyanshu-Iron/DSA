// Find Missing Number in a Array

let nums = [3, 0, 1];

function findMissingNo(nums) {
    // Sort the array
    nums.sort((a, b) => a - b);
    
    // Loop through the sorted array to find the missing number
    for (let i = 0; i <= nums.length; i++) {
        if (nums[i] !== i) {
            return i;
        }
    }
    // In case no number is missing, return -1 (though this shouldn't happen in this context)
    return -1;
}

console.log(findMissingNo(nums)); // Output: 2


// Optimal
// let n = nums.length;
//     let expectedSum = (n * (n + 1)) / 2;  // Sum of numbers from 0 to n
//     let actualSum = nums.reduce((acc, num) => acc + num, 0);  // Sum of array elements
//     return expectedSum - actualSum;